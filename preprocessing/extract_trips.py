"""Cut station-to-station trips of a planned agent out of Bi3 into compact, room-centred samples.

Robot trips: a trip starts when the robot is given a new goal station and ends when it is given
the next one (on reaching the goal). Each trip names its planned agent and keeps a short history
before the start, every agent's track (the robot and the participants) with its labels from
annotate_goals.py, and the Bi3 motion metrics of the planned agent's real path, so generated
trajectories can be compared with what the controller did. All tracks and labels
continue POST_S past the end of the trip (as far as the recording goes), so a
generated robot that is slower than the real one still has people to be replayed
against; end_index marks where the real trip ends and the metrics stop.

Tracks are cleaned (repeated mocap samples dropped, 0.5 s moving average on a BASE_HZ grid,
once per recording) and resampled to --hz. Coordinates are room-centred: the origin is the centre of the
area people and the robot actually occupy (about 3.5 x 3.5 m at both sites), and
LAAS is rotated 180 degrees so both sites have the shared two-station table on the
+x wall.

Run from the repository root: python -m preprocessing.extract_trips

Outputs (mocap-derived, kept outside this repository; read with bi3_derived.py):
  <out>/site_frames.json   room frame, room size, robot footprint and stations per site
  <out>/trips.json         list of trips
  <out>/trips_summary.csv  one row per trip with the real path's metrics
  <out>/manifest.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from metrics.bi3_metrics import bi3_metrics
from preprocessing.bi3_derived import DERIVED, SITES, TRIP_HISTORY_S, TRIP_HZ, load_frames, load_stations, moving_average, write_manifest


OUT = DERIVED / "trips"
# Bi3's own train/val/test split by trial (fluentrobotics/Bi3 create_data_npys.py).
TEST_TRIALS = {"laas": ["laas18", "laas19", "laas20"], "um": ["um16", "um17"]}
# um13, not a UM test trial: um16/um17 have no overhead video, and the camera-input experiments need one.
# The VLM planners are not trained on Bi3, so the split matters only when comparing with learned models.
DEFAULT_TRIALS = ["laas/laas18", "um/um13"]
CONTROLLERS = ["blind", "cohan", "cv", "hst", "static"]
SITE_FRAMES = {
    # origin = centre of the occupied area; rotation (deg) applied after shifting
    "laas": {"origin": [4.88, 15.62], "rotation_deg": 180.0, "room_m": 3.5,
             "robot": "PR2", "footprint_m": [0.668, 0.668]},
    "um": {"origin": [-0.23, 0.01], "rotation_deg": 0.0, "room_m": 3.5,
           "robot": "Stretch", "footprint_m": [0.330, 0.340]},
}
SMOOTH_S = 0.5          # moving-average window before resampling
BASE_HZ = 10.0          # cleaning grid
POST_S = 15.0           # replay padding after the trip end: generated robots may be slower, the humans must keep moving


def to_room(site: str, xy: np.ndarray) -> np.ndarray:
    frame = SITE_FRAMES[site]
    a = np.radians(frame["rotation_deg"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xy, float) - frame["origin"]) @ rot.T


def to_room_theta(site: str, theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(theta) + np.radians(SITE_FRAMES[site]["rotation_deg"]))))


def clean_track(time: np.ndarray, xyt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop repeated samples and smooth on a BASE_HZ grid: the grid (n,) and the smoothed [x, y, unwrapped theta] on it (n, 3)."""
    keep = np.r_[True, np.any(np.diff(xyt[:, :2], axis=0) != 0, axis=1)]
    t, p = time[keep], xyt[keep]
    base = np.arange(time[0], time[-1] + 1e-9, 1 / BASE_HZ)
    cols = [np.interp(base, t, p[:, 0]), np.interp(base, t, p[:, 1]), np.interp(base, t, np.unwrap(p[:, 2]))]
    return base, moving_average(np.stack(cols, axis=1), max(1, int(round(SMOOTH_S * BASE_HZ))))


def labels_at(time: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> list:
    idx = np.clip(np.searchsorted(time, grid, side="right") - 1, 0, len(time) - 1)
    return [str(labels[i]) for i in idx]


def robot_trips(site: str, trial: str, controller: str, stations: dict, hz: float, history_s: float) -> list[dict]:
    """Trips of the robot in one recording, cut at its goal switches, with stations the site's from load_stations."""
    frames = load_frames(site, trial, controller)
    cleaned = {agent: clean_track(f["time"], np.c_[f["x"], f["y"], f["theta"]]) for agent, f in frames.items()}
    robot = frames["robot"]
    time, goal = robot["time"], robot["goal"]
    switches = np.flatnonzero(goal[1:] != goal[:-1]) + 1
    starts = np.r_[0, switches]
    trips = []
    for k, (s, e) in enumerate(zip(starts[:-1], starts[1:])):  # the last goal is never reached
        t0, t1 = time[s], time[e]
        grid = np.arange(max(time[0], t0 - history_s), min(time[-1], t1 + POST_S) + 1e-9, 1 / hz)
        i0 = int(np.searchsorted(grid, t0 - 1e-9))
        i1 = int(np.searchsorted(grid, t1 + 1e-9)) - 1  # last sample not after t_end
        tracks = {}
        for agent, (base, smoothed) in cleaned.items():
            xyt = np.stack([np.interp(grid, base, c) for c in smoothed.T], axis=1)
            tracks[agent] = np.c_[to_room(site, xyt[:, :2]), to_room_theta(site, np.angle(np.exp(1j * xyt[:, 2])))]
        path = slice(i0, i1 + 1)
        others = np.stack([track[path, :2] for agent, track in tracks.items() if agent != "robot"], axis=1)
        metrics = bi3_metrics(grid[path], tracks["robot"][path, :2], others)
        g = str(goal[s])
        trips.append({
            "id": f"{site}_{trial}_{controller}_robot_{k:02d}",
            "site": site, "trial": trial, "controller": controller, "agent": "robot", "trip": k,
            "t_start": round(float(t0), 3), "t_end": round(float(t1), 3), "duration_s": round(float(t1 - t0), 3),
            "start_index": i0, "end_index": i1,
            "goal": g, "goal_xy": to_room(site, stations[g]["xy"]).round(3).tolist(), "next_goal": str(robot["next_goal"][s]),
            "time": np.round(grid - t0, 3).tolist(),
            "tracks": {agent: np.round(track, 3).tolist() for agent, track in tracks.items()},
            "labels": {agent: {key: labels_at(f["time"], f[key], grid) for key in ("state", "goal", "next_goal")} for agent, f in frames.items()},
            "real_metrics": {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in metrics.items()},
        })
    return trips


def site_frames(stations: dict) -> dict:
    """Room frame, robot footprint and room-frame stations per site, from stations mapping each site to load_stations(site)."""
    out = {}
    for site in SITES:
        frame = SITE_FRAMES[site]
        room_stations = {k: {"members": v["members"], "xy": to_room(site, v["xy"]).round(3).tolist()} for k, v in stations[site].items()}
        out[site] = {**frame, "stations": room_stations}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", default=DEFAULT_TRIALS, help="<site>/<trial>, e.g. laas/laas18")
    parser.add_argument("--controllers", nargs="*", default=CONTROLLERS)
    parser.add_argument("--hz", type=float, default=TRIP_HZ)
    parser.add_argument("--history-s", type=float, default=TRIP_HISTORY_S)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    for item in args.trials:
        site, trial = item.split("/")
        if trial not in TEST_TRIALS[site]:
            print(f"note: {item} is not in Bi3's test split {TEST_TRIALS[site]}")
    stations = {site: load_stations(site) for site in SITES}
    trips = [t for item in args.trials for site, trial in [item.split("/")] for c in args.controllers
             for t in robot_trips(site, trial, c, stations[site], args.hz, args.history_s)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "site_frames.json").write_text(json.dumps(site_frames(stations), indent=2) + "\n")
    (args.out / "trips.json").write_text(json.dumps(trips) + "\n")
    summary = [{k: t[k] for k in ("id", "site", "trial", "controller", "agent", "trip", "t_start", "duration_s", "goal", "next_goal")}
               | {k: v for k, v in t["real_metrics"].items() if not isinstance(v, list)} for t in trips]
    with open(args.out / "trips_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    params = {"HZ": args.hz, "HISTORY_S": args.history_s, "SMOOTH_S": SMOOTH_S, "BASE_HZ": BASE_HZ, "POST_S": POST_S,
              "SITE_FRAMES": SITE_FRAMES, "trials": args.trials, "controllers": args.controllers}
    write_manifest(args.out / "manifest.json", params, {
        "trips": len(trips),
        "fields": "agent is the planned agent of the trip; goal, goal_xy and next_goal are its goal label at the start, the goal's room-frame position and its next goal label; "
                  "time is seconds from trip start (negative = history); start_index and end_index are the first and last (inclusive) sample of the trip, samples after end_index are POST_S replay padding; "
                  "tracks[agent id] rows are [x, y, theta] in the room frame for every agent (robot, human0, ...); labels[agent id] holds per-sample state, goal and next_goal labels "
                  "(goals are keys of site_frames stations; the robot's state is empty); real_metrics are bi3_metrics of the planned agent's real path against all other agents from start_index to end_index",
    })
    print(f"{len(trips)} trips from {len(args.trials)} trials x {len(args.controllers)} controllers -> {args.out}")


if __name__ == "__main__":
    main()
