"""Cut station-to-station trips of a planned agent out of Bi3 into compact, room-centred samples.

Robot trips: a trip starts when the robot is given a new goal station and ends when it is given
the next one (on reaching the goal). Person trips (--agents human): a trip starts when the person
leaves a station visit and ends at the turnaround of their next visit (human_trips). Each trip names its planned agent and keeps a short history
before the start, every agent's track (the robot and the participants) with its labels from
annotate_goals.py, and the Bi3 motion metrics of the planned agent's real path, so generated
trajectories can be compared with what the controller did. All tracks and labels
continue POST_S past the end of the trip (as far as the recording goes), so a
generated agent that is slower than the real one still has the others to be replayed
against. end_index marks where the real trip ends and the metrics stop.

Tracks are cleaned (repeated mocap samples dropped, 0.5 s moving average on a BASE_HZ grid,
once per recording) and resampled to --hz on a grid anchored at the trip start, so the start state, the
camera frame at the start and the real-path metrics share one time (time 0). The 0.5 s average suits station-visit
detection (annotate_goals.py), where the window comes from, and these tracks are what the metrics score.
Each trip also holds a lighter model-input series, input_tracks_10hz: every agent at INPUT_HZ on the same grid
anchored at the trip start and over the same span, each sample the mean of the raw samples (repeated ones dropped)
in a centred INPUT_BIN_S bin, the heading a circular mean, with no further smoothing (binned_track). No sample
draws on a raw sample INPUT_BIN_S / 2 or more after its time. Every INPUT_HZ / hz-th sample falls on a track sample. Coordinates are room-centred: the origin is the centre of the
area people and the robot actually occupy. room_m is the traversable workspace the Bi3 paper
states, 3.5 x 3.5 m (Stratton et al., ICRA 2026, Sec. IV-A). The workstations stand along its
perimeter, so people working at them stand at or just beyond its edge. LAAS is rotated 180 degrees so both sites have the shared two-station table on the
+x wall.

Run from the repository root: python -m preprocessing.extract_trips
(person trips of every trial: python -m preprocessing.extract_trips --agents human --all-trials --out <folder>)

Outputs (mocap-derived, kept outside this repository, read with bi3_derived.py):
  <out>/site_frames.json   room frame, room size, robot footprint and stations per site (robot stopping point and where people stand)
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
from preprocessing.bi3_derived import DERIVED, SITES, TRIP_HISTORY_S, TRIP_HZ, goal_recordings, load_frames, load_stations, load_visits, moving_average, write_manifest


OUT = DERIVED / "trips_v2"
# Bi3's own train/val/test split by trial (fluentrobotics/Bi3 create_data_npys.py).
TEST_TRIALS = {"laas": ["laas18", "laas19", "laas20"], "um": ["um16", "um17"]}
# um13, not a UM test trial: um16/um17 have no overhead video, and the camera-input experiments need one.
# The VLM planners are not trained on Bi3, so the split matters only when comparing with learned models.
DEFAULT_TRIALS = ["laas/laas18", "um/um13"]
CONTROLLERS = ["blind", "cohan", "cv", "hst", "static"]
SITE_FRAMES = {
    # origin is the centre of the occupied area, and rotation (deg) is applied after shifting.
    "laas": {"origin": [4.88, 15.62], "rotation_deg": 180.0, "room_m": 3.5,
             "robot": "PR2", "footprint_m": [0.668, 0.668]},
    "um": {"origin": [-0.23, 0.01], "rotation_deg": 0.0, "room_m": 3.5,
           "robot": "Stretch", "footprint_m": [0.330, 0.340]},
}
SMOOTH_S = 0.5          # moving-average window before resampling
BASE_HZ = 10.0          # cleaning grid
POST_S = 15.0           # replay padding after the trip end: generated robots may be slower, the humans must keep moving
INPUT_HZ = 10.0         # rate of the model-input series (input_tracks_10hz)
INPUT_BIN_S = 0.1       # each model-input sample is the mean of the raw samples in a centred bin this wide


def to_room(site: str, xy: np.ndarray) -> np.ndarray:
    frame = SITE_FRAMES[site]
    a = np.radians(frame["rotation_deg"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xy, float) - frame["origin"]) @ rot.T


def to_room_theta(site: str, theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(theta) + np.radians(SITE_FRAMES[site]["rotation_deg"]))))


def drop_repeated(time: np.ndarray, xyt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Samples whose position differs from the previous sample's: mocap holds the last position between updates."""
    keep = np.r_[True, np.any(np.diff(xyt[:, :2], axis=0) != 0, axis=1)]
    return time[keep], xyt[keep]


def clean_track(time: np.ndarray, xyt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop repeated samples and smooth on a BASE_HZ grid: the grid (n,) and the smoothed [x, y, unwrapped theta] on it (n, 3)."""
    t, p = drop_repeated(time, xyt)
    base = np.arange(time[0], time[-1] + 1e-9, 1 / BASE_HZ)
    cols = [np.interp(base, t, p[:, 0]), np.interp(base, t, p[:, 1]), np.interp(base, t, np.unwrap(p[:, 2]))]
    return base, moving_average(np.stack(cols, axis=1), max(1, int(round(SMOOTH_S * BASE_HZ))))


def binned_track(time: np.ndarray, xyt: np.ndarray, grid: np.ndarray, bin_s: float = INPUT_BIN_S) -> tuple[np.ndarray, np.ndarray]:
    """[x, y, theta] at each grid time (n, 3) from raw samples (time (T,) increasing, xyt (T, 3), repeated samples already dropped),
    and which grid times had an empty bin (n,).

    A sample is the mean of the raw samples with time in [t - bin_s / 2, t + bin_s / 2), the heading their circular mean. An empty
    bin takes the last raw sample before its end: the latest position known at that time, which is what the mocap log holds there.
    So no sample uses a raw sample bin_s / 2 or more after its time. Raises ValueError for a grid time with no raw sample before
    its bin's end."""
    lo = np.searchsorted(time, grid - bin_s / 2, side="left")
    hi = np.searchsorted(time, grid + bin_s / 2, side="left")
    if np.any(hi == 0):
        raise ValueError(f"grid times before the first raw sample's bin: {grid[hi == 0][:3]}")
    values = np.c_[xyt[:, 0], xyt[:, 1], np.cos(xyt[:, 2]), np.sin(xyt[:, 2])]
    sums = np.vstack([np.zeros(4), np.cumsum(values, axis=0)])
    count = hi - lo
    empty = count == 0
    mean = (sums[hi] - sums[lo]) / np.maximum(count, 1)[:, None]
    mean[empty] = values[hi[empty] - 1]
    return np.c_[mean[:, :2], np.arctan2(mean[:, 3], mean[:, 2])], empty


def labels_at(time: np.ndarray, labels: np.ndarray, grid: np.ndarray) -> list:
    idx = np.clip(np.searchsorted(time, grid, side="right") - 1, 0, len(time) - 1)
    return [str(labels[i]) for i in idx]


def trip_grid(t0: float, t1: float, t_first: float, t_last: float, hz: float, history_s: float) -> tuple[np.ndarray, int, int]:
    """Sample times of a trip from t0 to t1 in a recording spanning t_first..t_last: the grid t0 + k / hz, from up to history_s before t0
    (not before t_first) to POST_S after t1 (not after t_last), the index of t0 and the index of the last sample not after t1."""
    k_first = -min(int(np.floor(history_s * hz + 1e-9)), int(np.floor((t0 - t_first) * hz + 1e-9)))
    k_last = int(np.floor((min(t_last, t1 + POST_S) - t0) * hz + 1e-9))
    i0 = -k_first
    return t0 + np.arange(k_first, k_last + 1) / hz, i0, i0 + int(np.floor((t1 - t0) * hz + 1e-9))


def trip_record(site: str, trial: str, controller: str, frames: dict, cleaned: dict, raw: dict, agent: str, k: int, t0: float, t1: float,
                start_goal: str, goal: str, goal_xy: list[float], next_goal: str, hz: float, history_s: float) -> dict:
    """One trip of the planned agent from t0 to t1 of a recording, with frames from load_frames, cleaned the clean_track of each agent
    and raw the drop_repeated samples of each agent (recording_tracks)."""
    time = frames[agent]["time"]
    grid, i0, i1 = trip_grid(t0, t1, time[0], time[-1], hz, history_s)
    tracks = {}
    for other, (base, smoothed) in cleaned.items():
        xyt = np.stack([np.interp(grid, base, c) for c in smoothed.T], axis=1)
        tracks[other] = np.c_[to_room(site, xyt[:, :2]), to_room_theta(site, np.angle(np.exp(1j * xyt[:, 2])))]
    path = slice(i0, i1 + 1)
    others = np.stack([track[path, :2] for other, track in tracks.items() if other != agent], axis=1)
    metrics = bi3_metrics(grid[path], tracks[agent][path, :2], others)
    dense, j0, j1 = trip_grid(t0, t1, time[0], time[-1], INPUT_HZ, history_s)
    inside = (dense >= grid[0] - 1e-6) & (dense <= grid[-1] + 1e-6)  # cut to the track grid's span, whose times it then contains
    first = int(np.argmax(inside))
    dense, j0, j1 = dense[inside], j0 - first, j1 - first
    binned = {other: binned_track(t, xyt, dense) for other, (t, xyt) in raw.items()}
    return {
        "id": f"{site}_{trial}_{controller}_{agent}_{k:02d}",
        "site": site, "trial": trial, "controller": controller, "agent": agent, "trip": k,
        "t_start": round(float(t0), 3), "t_end": round(float(t1), 3), "duration_s": round(float(t1 - t0), 3),
        "start_index": i0, "end_index": i1,
        "start_goal": start_goal, "goal": goal, "goal_xy": goal_xy, "next_goal": next_goal,
        "time": np.round(grid - t0, 3).tolist(),
        "tracks": {other: np.round(track, 3).tolist() for other, track in tracks.items()},
        "labels": {other: {key: labels_at(f["time"], f[key], grid) for key in ("state", "goal", "next_goal")} for other, f in frames.items()},
        "real_metrics": {name: (round(v, 4) if isinstance(v, float) else v) for name, v in metrics.items()},
        "input_tracks_10hz": {
            "time": np.round(dense - t0, 3).tolist(), "start_index": j0, "end_index": j1,
            "tracks": {other: np.round(np.c_[to_room(site, xyt[:, :2]), to_room_theta(site, xyt[:, 2])], 3).tolist() for other, (xyt, _) in binned.items()},
            "empty_bins": {other: np.flatnonzero(empty).tolist() for other, (_, empty) in binned.items()},
        },
    }


def recording_tracks(frames: dict) -> tuple[dict, dict]:
    """Per agent of a recording, once: its clean_track and its drop_repeated raw samples, as trip_record takes them."""
    xyt = {agent: np.c_[f["x"], f["y"], f["theta"]] for agent, f in frames.items()}
    return ({agent: clean_track(frames[agent]["time"], v) for agent, v in xyt.items()},
            {agent: drop_repeated(frames[agent]["time"], v) for agent, v in xyt.items()})


def robot_trips(site: str, trial: str, controller: str, stations: dict, hz: float, history_s: float) -> list[dict]:
    """Trips of the robot in one recording, cut at its goal switches, with stations the site's from load_stations."""
    frames = load_frames(site, trial, controller)
    cleaned, raw = recording_tracks(frames)
    robot = frames["robot"]
    time, goal = robot["time"], robot["goal"]
    switches = np.flatnonzero(goal[1:] != goal[:-1]) + 1
    starts = np.r_[0, switches]
    trips = []
    for k, (s, e) in enumerate(zip(starts[:-1], starts[1:])):  # the last goal is never reached
        g = str(goal[s])
        trips.append(trip_record(site, trial, controller, frames, cleaned, raw, "robot", k, time[s], time[e], str(goal[s - 1]) if s else "", g,
                                 to_room(site, stations[g]["xy"]).round(3).tolist(), str(robot["next_goal"][s]), hz, history_s))
    return trips


def human_trips(site: str, trial: str, controller: str, human_xy: dict, hz: float, history_s: float) -> list[dict]:
    """Trips of each person in one recording, with human_xy the site's goal points from human_goal_points.

    A trip runs from leaving visit k (t_leave, where the person's goal label switches to the next station) to the
    turnaround of visit k + 1 (t_turn), and k is the trip number. Pairs of visits that overlap in time and pairs at the
    same station are not trips: neither has a goal-to-goal move between them. A trip's next goal is the label of the
    first later visit with a different label than its goal, as `label_from_visits` takes it.
    """
    frames = load_frames(site, trial, controller)
    cleaned, raw = recording_tracks(frames)
    visits = load_visits(site, trial, controller)
    trips = []
    for agent in frames:
        if agent == "robot":
            continue
        own = sorted((v for v in visits if v["agent"] == agent), key=lambda v: v["visit"])
        for k, (here, there) in enumerate(zip(own[:-1], own[1:])):
            if there["t_arrive"] <= here["t_leave"] or there["station"] == here["station"]:
                continue
            next_goal = next((v["label"] for v in own[k + 2 :] if v["label"] != there["label"]), "")
            trips.append(trip_record(site, trial, controller, frames, cleaned, raw, agent, here["visit"], here["t_leave"], there["t_turn"],
                                     here["label"], there["label"], human_xy[there["label"]], next_goal, hz, history_s))
    return trips


def human_goal_points(stations: dict) -> dict:
    """{site: {label: [x, y]}}: where people stand at each station, in the room frame (m), with stations mapping each site to load_stations(site).

    A single station's point is the median position at the turnaround (t_turn) over all non-inferred visits of people with that
    label in every recording of the site (an inferred visit's t_turn is the far point of a missed loop, not a turnaround). A shared
    table's point is the midpoint of its members' points.
    """
    turns = {site: {} for site in SITES}
    for site, trial, controller in goal_recordings():
        frames = load_frames(site, trial, controller)
        for v in load_visits(site, trial, controller):
            if v["agent"] != "robot" and not v["inferred"]:
                f = frames[v["agent"]]
                turns[site].setdefault(v["label"], []).append([np.interp(v["t_turn"], f["time"], f["x"]), np.interp(v["t_turn"], f["time"], f["y"])])
    points = {}
    for site in SITES:
        single = {label: np.median(to_room(site, turns[site][label]), axis=0) for label, s in stations[site].items() if len(s["members"]) == 1}
        points[site] = {label: np.mean([single[m] for m in s["members"]], axis=0).round(3).tolist() for label, s in stations[site].items()}
    return points


def site_frames(stations: dict, human_xy: dict) -> dict:
    """Room frame, robot footprint and room-frame stations per site, from stations mapping each site to load_stations(site)
    and human_xy from human_goal_points. Each station has its robot stopping point (xy) and where people stand (human_xy)."""
    out = {}
    for site in SITES:
        frame = SITE_FRAMES[site]
        room_stations = {k: {"members": v["members"], "xy": to_room(site, v["xy"]).round(3).tolist(), "human_xy": human_xy[site][k]}
                         for k, v in stations[site].items()}
        out[site] = {**frame, "stations": room_stations}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", nargs="*", default=DEFAULT_TRIALS, help="<site>/<trial>, e.g. laas/laas18")
    parser.add_argument("--all-trials", action="store_true", help="every trial with goal labels, instead of --trials")
    parser.add_argument("--controllers", nargs="*", default=CONTROLLERS)
    parser.add_argument("--agents", nargs="+", choices=["robot", "human"], default=["robot"], help="planned agents to cut trips for")
    parser.add_argument("--hz", type=float, default=TRIP_HZ)
    parser.add_argument("--history-s", type=float, default=TRIP_HISTORY_S)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.all_trials:
        args.trials = list(dict.fromkeys(f"{site}/{trial}" for site, trial, _ in goal_recordings()))
    for item in args.trials:
        site, trial = item.split("/")
        if trial not in TEST_TRIALS[site]:
            print(f"note: {item} is not in Bi3's test split {TEST_TRIALS[site]}")
    stations = {site: load_stations(site) for site in SITES}
    human_xy = human_goal_points(stations)
    trips = []
    for item in args.trials:
        site, trial = item.split("/")
        for c in args.controllers:
            if "robot" in args.agents:
                trips += robot_trips(site, trial, c, stations[site], args.hz, args.history_s)
            if "human" in args.agents:
                trips += human_trips(site, trial, c, human_xy[site], args.hz, args.history_s)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "site_frames.json").write_text(json.dumps(site_frames(stations, human_xy), indent=2) + "\n")
    (args.out / "trips.json").write_text(json.dumps(trips) + "\n")
    summary = [{k: t[k] for k in ("id", "site", "trial", "controller", "agent", "trip", "t_start", "duration_s", "start_goal", "goal", "next_goal")}
               | {k: v for k, v in t["real_metrics"].items() if not isinstance(v, list)} for t in trips]
    with open(args.out / "trips_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    empty_bins = {}  # {site: {"person" | "robot": {"bins", "empty"}}} over all trips and agents
    for t in trips:
        series = t["input_tracks_10hz"]
        for agent, empty in series["empty_bins"].items():
            count = empty_bins.setdefault(t["site"], {}).setdefault("robot" if agent == "robot" else "person", {"bins": 0, "empty": 0})
            count["bins"] += len(series["time"])
            count["empty"] += len(empty)
    params = {"HZ": args.hz, "HISTORY_S": args.history_s, "SMOOTH_S": SMOOTH_S, "BASE_HZ": BASE_HZ, "POST_S": POST_S,
              "INPUT_HZ": INPUT_HZ, "INPUT_BIN_S": INPUT_BIN_S,
              "SITE_FRAMES": SITE_FRAMES, "trials": args.trials, "controllers": args.controllers, "agents": args.agents}
    write_manifest(args.out / "manifest.json", params, {
        "trips": len(trips),
        "fields": "agent is the planned agent of the trip; start_goal is the label of the station it leaves (empty for a robot's first trip); "
                  "goal, goal_xy and next_goal are its goal label at the start, the goal's room-frame position (a station's xy for the robot, its human_xy for a person) and its next goal label; "
                  "time is seconds from trip start (negative = history); start_index and end_index are the first and last (inclusive) sample of the trip, samples after end_index are POST_S replay padding; "
                  "tracks[agent id] rows are [x, y, theta] in the room frame for every agent (robot, human0, ...); labels[agent id] holds per-sample state, goal and next_goal labels "
                  "(goals are keys of site_frames stations; the robot's state is empty); real_metrics are bi3_metrics of the planned agent's real path against all other agents from start_index to end_index",
        "input_tracks_10hz": "model-input series of every agent, not smoothed like tracks: time is seconds from trip start on an INPUT_HZ grid anchored at the trip start "
                             "over the same span as tracks, so its sample (INPUT_HZ / HZ) * i is at time[i]; start_index and end_index as for tracks; tracks[agent id] rows are "
                             "[x, y, theta] in the room frame, each the mean of the raw samples (repeated samples dropped) with time in [t - INPUT_BIN_S / 2, t + INPUT_BIN_S / 2), "
                             "theta their circular mean; empty_bins[agent id] lists the samples whose bin held no raw sample, which take the last raw sample before the bin's end",
        "input_empty_bins": empty_bins,
    })
    print(f"{len(trips)} trips of {args.agents} from {len(args.trials)} trials x {len(args.controllers)} controllers -> {args.out}")


if __name__ == "__main__":
    main()
