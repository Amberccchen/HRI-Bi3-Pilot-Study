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

Once per recording, every agent gets a series on an INPUT_HZ grid from the recording's first sample (recording_series): each
sample the mean of the raw samples (repeated ones dropped) in a centred INPUT_BIN_S bin, the heading a circular mean, with no
further smoothing, and an empty bin filled by PCHIP through the raw samples GAP_CONTEXT_S either side of its gap (binned_track).
A trip's start and end snap to the nearest grid sample. Its scored tracks are every SCORED_STEP-th series sample counted from the
start, at TRIP_HZ, so the start state and the real-path metrics share one time (time 0); the series itself is the model input.
The scored end (end_index) is the scored sample nearest the end (bi3_derived.scored_end_index).
Coordinates are room-centred: the origin is the centre of the
area people and the robot actually occupy. room_m is the traversable workspace the Bi3 paper
states, 3.5 x 3.5 m (Stratton et al., ICRA 2026, Sec. IV-A). The workstations stand along its
perimeter, so people working at them stand at or just beyond its edge. LAAS is rotated 180 degrees so both sites have the shared two-station table on the
+x wall.

Run from the repository root: python -m preprocessing.extract_trips
(person trips of every trial: python -m preprocessing.extract_trips --agents human --all-trials --out <folder>)

Outputs (mocap-derived, kept outside this repository, read with bi3_derived.load_trips):
  <out>/site_frames.json   room frame, room size, robot footprint and stations per site (robot stopping point and where people stand)
  <out>/recordings.npz     per recording: its grid, and every agent's series, empty-bin flags and labels on it
  <out>/trips.json         list of trips: metadata, real-path metrics and the trip's index range on its recording's grid
  <out>/trips_summary.csv  one row per trip with the real path's metrics
  <out>/manifest.json
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from metrics.bi3_metrics import bi3_metrics
from preprocessing.bi3_derived import (DERIVED, INPUT_HZ, LABEL_KEYS, RECORDINGS_FILE, SCORED_STEP, SERIES_FIELDS, SITES, TRIP_HISTORY_S, TRIP_HZ,
                                       goal_recordings, load_frames, load_stations, load_visits, recording_key, trip_arrays, write_manifest)


OUT = DERIVED / "trips_v3"
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
POST_S = 15.0           # replay padding after the trip end: generated robots may be slower, the humans must keep moving
INPUT_BIN_S = 0.1       # each series sample is the mean of the raw samples in a centred bin this wide
GAP_CONTEXT_S = 0.4     # an empty bin is filled from the raw samples this far either side of its gap


def to_room(site: str, xy: np.ndarray) -> np.ndarray:
    frame = SITE_FRAMES[site]
    a = np.radians(frame["rotation_deg"])
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (np.asarray(xy, float) - frame["origin"]) @ rot.T


def to_room_theta(site: str, theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(theta) + np.radians(SITE_FRAMES[site]["rotation_deg"]))))


def binned_track(time: np.ndarray, xyt: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[x, y, theta] at each grid time (n, 3) from raw samples (time (T,) strictly increasing, xyt (T, 3), repeated samples already
    dropped), and which grid times had an empty bin (n,).

    A sample is the mean of the raw samples with time in [t - INPUT_BIN_S / 2, t + INPUT_BIN_S / 2), the heading their circular mean, so
    it uses no raw sample INPUT_BIN_S / 2 or more after its time. The bins of a gap between raw samples a and b hold no sample. They take
    the PCHIP interpolant through the raw samples from GAP_CONTEXT_S before a to GAP_CONTEXT_S after b, the heading unwrapped, interpolated
    and wrapped, so inside a gap a sample uses raw samples up to GAP_CONTEXT_S after the gap's end. Bins before the first or after the last raw sample
    take that sample.
    """
    lo = np.searchsorted(time, grid - INPUT_BIN_S / 2, side="left")
    hi = np.searchsorted(time, grid + INPUT_BIN_S / 2, side="left")
    values = np.c_[xyt[:, 0], xyt[:, 1], np.cos(xyt[:, 2]), np.sin(xyt[:, 2])]
    sums = np.vstack([np.zeros(4), np.cumsum(values, axis=0)])
    count = hi - lo
    empty = count == 0
    mean = (sums[hi] - sums[lo]) / np.maximum(count, 1)[:, None]
    out = np.c_[mean[:, :2], np.arctan2(mean[:, 3], mean[:, 2])]
    for after in np.unique(hi[empty]):  # the bins of one gap share its first raw sample after it, index after
        bins = np.flatnonzero(empty & (hi == after))
        if after == 0 or after == len(time):
            out[bins] = xyt[min(after, len(time) - 1)]
            continue
        before = after - 1
        context = slice(np.searchsorted(time, time[before] - GAP_CONTEXT_S, side="left"), np.searchsorted(time, time[after] + GAP_CONTEXT_S, side="right"))
        path = np.c_[xyt[context, :2], np.unwrap(xyt[context, 2])]
        filled = PchipInterpolator(time[context], path)(grid[bins])
        out[bins] = np.c_[filled[:, :2], np.angle(np.exp(1j * filled[:, 2]))]
    return out, empty


def recording_series(site: str, frames: dict) -> tuple[np.ndarray, dict]:
    """A recording's INPUT_HZ grid (n,) of mocap times, from its first sample to its last, and per agent (frames from load_frames, in its
    order) the series fields of SERIES_FIELDS on it: xyt (n, 3) float32 [x, y, theta] in the room frame from binned_track with repeated
    samples dropped, empty (n,) and the labels of the last raw sample at or before each grid time."""
    t_first = min(f["time"][0] for f in frames.values())
    t_last = max(f["time"][-1] for f in frames.values())
    grid = t_first + np.arange(int(np.floor((t_last - t_first) * INPUT_HZ + 1e-9)) + 1) / INPUT_HZ
    series = {}
    for agent, f in frames.items():
        raw = np.c_[f["x"], f["y"], f["theta"]]
        keep = np.r_[True, np.any(np.diff(raw[:, :2], axis=0) != 0, axis=1)]  # mocap holds the last position between updates
        xyt, empty = binned_track(f["time"][keep], raw[keep], grid)
        last = np.clip(np.searchsorted(f["time"], grid, side="right") - 1, 0, len(f["time"]) - 1)
        series[agent] = {"xyt": np.c_[to_room(site, xyt[:, :2]), to_room_theta(site, xyt[:, 2])].astype(np.float32), "empty": empty,
                         **{key: f[key][last].astype(str) for key in LABEL_KEYS}}
    return grid, series


def recording_arrays(name: str, grid: np.ndarray, series: dict) -> dict[str, np.ndarray]:
    """The RECORDINGS_FILE arrays of one recording, named by recording_key, from recording_series."""
    arrays = {recording_key(name, "time"): grid, recording_key(name, "agents"): np.array(list(series))}
    return arrays | {recording_key(name, agent, field): fields[field] for agent, fields in series.items() for field in SERIES_FIELDS}


def trip_span(t0: float, t1: float, grid: np.ndarray, history_s: float) -> list[int]:
    """Indices on a recording's grid of a trip from t0 to t1: [first sample with history, start, end, last sample with padding].

    Start and end are the grid samples nearest t0 and t1. History and padding are whole SCORED_STEP steps from the start, up to history_s
    before it and up to POST_S after the end, within the recording."""
    start, end = (int(np.clip(np.round((t - grid[0]) * INPUT_HZ), 0, len(grid) - 1)) for t in (t0, t1))
    history = min(int(np.floor(history_s * TRIP_HZ + 1e-9)), start // SCORED_STEP)
    padding = min((end - start + int(round(POST_S * INPUT_HZ))) // SCORED_STEP, (len(grid) - 1 - start) // SCORED_STEP)
    return [start - SCORED_STEP * history, start, end, start + SCORED_STEP * padding]


def trip_record(site: str, trial: str, controller: str, grid: np.ndarray, series: dict, agent: str, k: int, t0: float, t1: float,
                start_goal: str, goal: str, goal_xy: list[float], next_goal: str, history_s: float) -> dict:
    """The trips.json entry of one trip of the planned agent from t0 to t1 of a recording, with grid and series from recording_series."""
    indices = trip_span(t0, t1, grid, history_s)
    t_start, t_end = grid[indices[1]], grid[indices[2]]
    arrays = trip_arrays({"recording_indices": indices}, series)
    path = slice(arrays["start_index"], arrays["end_index"] + 1)
    tracks = {other: np.asarray(track, float) for other, track in arrays["tracks"].items()}
    others = np.stack([track[path, :2] for other, track in tracks.items() if other != agent], axis=1)
    metrics = bi3_metrics(arrays["time"][path], tracks[agent][path, :2], others)
    return {
        "id": f"{site}_{trial}_{controller}_{agent}_{k:02d}", "site": site, "trial": trial, "controller": controller, "agent": agent, "trip": k,
        "recording": f"{site}_{trial}_{controller}", "recording_indices": indices,
        "t_start": round(float(t_start), 3), "t_end": round(float(t_end), 3), "duration_s": round(float(t_end - t_start), 3),
        "start_goal": start_goal, "goal": goal, "goal_xy": goal_xy, "next_goal": next_goal,
        "real_metrics": {name: (round(v, 4) if isinstance(v, float) else v) for name, v in metrics.items()},
    }


def robot_trips(site: str, trial: str, controller: str, frames: dict, grid: np.ndarray, series: dict, stations: dict, history_s: float) -> list[dict]:
    """Trips of the robot in one recording, cut at its goal switches, with frames from load_frames, grid and series from
    recording_series and stations the site's from load_stations."""
    robot = frames["robot"]
    time, goal = robot["time"], robot["goal"]
    switches = np.flatnonzero(goal[1:] != goal[:-1]) + 1
    starts = np.r_[0, switches]
    trips = []
    for k, (s, e) in enumerate(zip(starts[:-1], starts[1:])):  # the last goal is never reached
        g = str(goal[s])
        trips.append(trip_record(site, trial, controller, grid, series, "robot", k, time[s], time[e], str(goal[s - 1]) if s else "", g,
                                 to_room(site, stations[g]["xy"]).round(3).tolist(), str(robot["next_goal"][s]), history_s))
    return trips


def human_trips(site: str, trial: str, controller: str, frames: dict, grid: np.ndarray, series: dict, human_xy: dict, history_s: float) -> list[dict]:
    """Trips of each person in one recording, with frames from load_frames, grid and series from recording_series and human_xy the
    site's goal points from human_goal_points.

    A trip runs from leaving visit k (t_leave, where the person's goal label switches to the next station) to the
    turnaround of visit k + 1 (t_turn), and k is the trip number. Pairs of visits that overlap in time and pairs at the
    same station are not trips: neither has a goal-to-goal move between them. A trip's next goal is the label of the
    first later visit with a different label than its goal, as `label_from_visits` takes it.
    """
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
            trips.append(trip_record(site, trial, controller, grid, series, agent, here["visit"], here["t_leave"], there["t_turn"],
                                     here["label"], there["label"], human_xy[there["label"]], next_goal, history_s))
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
    trips, archive, empty_bins = [], {}, {}
    for item in args.trials:
        site, trial = item.split("/")
        for c in args.controllers:
            frames = load_frames(site, trial, c)
            grid, series = recording_series(site, frames)
            if "robot" in args.agents:
                trips += robot_trips(site, trial, c, frames, grid, series, stations[site], args.history_s)
            if "human" in args.agents:
                trips += human_trips(site, trial, c, frames, grid, series, human_xy[site], args.history_s)
            archive |= recording_arrays(f"{site}_{trial}_{c}", grid, series)
            for agent, fields in series.items():
                count = empty_bins.setdefault(site, {}).setdefault("robot" if agent == "robot" else "person", {"bins": 0, "empty": 0})
                count["bins"] += len(grid)
                count["empty"] += int(fields["empty"].sum())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "site_frames.json").write_text(json.dumps(site_frames(stations, human_xy), indent=2) + "\n")
    np.savez_compressed(args.out / RECORDINGS_FILE, **archive)
    (args.out / "trips.json").write_text(json.dumps(trips) + "\n")
    summary = [{k: t[k] for k in ("id", "site", "trial", "controller", "agent", "trip", "t_start", "duration_s", "start_goal", "goal", "next_goal")}
               | {k: v for k, v in t["real_metrics"].items() if not isinstance(v, list)} for t in trips]
    with open(args.out / "trips_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    params = {"TRIP_HZ": TRIP_HZ, "INPUT_HZ": INPUT_HZ, "INPUT_BIN_S": INPUT_BIN_S, "GAP_CONTEXT_S": GAP_CONTEXT_S, "HISTORY_S": args.history_s, "POST_S": POST_S,
              "SITE_FRAMES": SITE_FRAMES, "trials": args.trials, "controllers": args.controllers, "agents": args.agents}
    write_manifest(args.out / "manifest.json", params, {
        "trips": len(trips),
        "fields": "trips.json: agent is the planned agent of the trip; start_goal is the label of the station it leaves (empty for a robot's first trip); "
                  "goal, goal_xy and next_goal are its goal label at the start, the goal's room-frame position (a station's xy for the robot, its human_xy for a person) and its next goal label; "
                  "recording is the trip's recording in recordings.npz and recording_indices its [first sample with history, start, end, last sample with padding] on that recording's grid; "
                  "t_start and t_end are the mocap times of the start and end samples; real_metrics are bi3_metrics of the planned agent's scored track against all other agents from start_index to end_index. "
                  "recordings.npz: <recording>/time is the recording's INPUT_HZ grid of mocap times, <recording>/agents its agent ids in order, and <recording>/<agent>/xyt, empty, state, goal, next_goal "
                  "the agent's series on the grid: [x, y, theta] in the room frame (float32), each the mean of the raw samples (repeated samples dropped) with time in [t - INPUT_BIN_S / 2, t + INPUT_BIN_S / 2) "
                  "and theta their circular mean; empty marks bins with no raw sample, filled by PCHIP through the raw samples from GAP_CONTEXT_S before to GAP_CONTEXT_S after the gap "
                  "(the nearest sample at a recording edge); the labels are those of the last raw sample at or before each grid time. "
                  "bi3_derived.load_trips adds time (seconds from trip start, negative = history), start_index, end_index, tracks and labels at TRIP_HZ "
                  "(every SCORED_STEP-th series sample from the start) and input_tracks_10hz (the series over the same span, with its own time, start_index, end_index, tracks and empty_bins)",
        "empty_bins": empty_bins,
    })
    print(f"{len(trips)} trips of {args.agents} from {len(args.trials)} trials x {len(args.controllers)} controllers -> {args.out}")


if __name__ == "__main__":
    main()
