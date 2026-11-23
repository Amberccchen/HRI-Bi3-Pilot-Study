"""Read and write the shared Bi3 derived files (video sync offsets, goal labels), and read the raw recordings.

The derived files live outside this repository, under BI3_DERIVED (default
/persistent/data/Bi3_derived), the raw Bi3 data under BI3_RAW (default
<repository>/Bi3/Bi3, where README step 1 places the dataset).
Each output folder carries a manifest.json with the parameters and the git state of the code
that produced it, so everyone can check they are using the same labels instead of rerunning
the scripts.

    # from the repository root
    from preprocessing.bi3_derived import load_video_sync, video_time, load_stations, load_visits, load_frames
    video_time("laas", "laas1", "cohan", 63.2)        # seconds into laas1/cohan.mp4
    load_stations("laas")["0/1"]                        # {"members": ["0", "1"], "xy": [...]}
    frames = load_frames("laas", "laas1", "cohan")["human0"]
    frames["goal"][1000], frames["next_goal"][1000]     # e.g. "5", "0/1"
"""

import csv
import datetime
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("BI3_RAW", REPO / "Bi3" / "Bi3"))
DERIVED = Path(os.environ.get("BI3_DERIVED", "/persistent/data/Bi3_derived"))
SITES = ("laas", "um")
SYNC_FILE = "video_sync.csv"
GOALS_DIR = "goals"
TRIP_HZ = 2.5           # sample rate of the extracted trips' scored tracks
INPUT_HZ = 10.0         # sample rate of the per-recording series the trips are cut from (input_tracks_10hz)
SCORED_STEP = round(INPUT_HZ / TRIP_HZ)  # every SCORED_STEP-th series sample, counted from the trip start, is a scored track sample
RECORDINGS_FILE = "recordings.npz"  # per-recording series and labels in a trips folder
LABEL_KEYS = ("state", "goal", "next_goal")
SERIES_FIELDS = ("xyt", "empty", *LABEL_KEYS)  # per agent of a recording in RECORDINGS_FILE
TRIP_HISTORY_S = 3.0    # history kept before each trip starts
FRAME_TYPES = {"step": int, "time": float, "x": float, "y": float, "theta": float}
VISIT_TYPES = {"visit": int, "station": int, "t_arrive": float, "t_turn": float, "t_leave": float,
               "min_dist": float, "margin": float}


def raw_json_path(site: str, trial: str, controller: str) -> Path:
    return RAW / "jsons" / site / trial / f"{controller}.json"


def video_path(site: str, trial: str, controller: str) -> Path:
    return RAW / "videos" / site / trial / f"{controller}.mp4"


def load_recording(path: Path) -> dict:
    """One raw mocap JSON as arrays: time (T,), robot (T, 3), humans (T, H, 3), robot_goal (T, 2)."""
    steps = json.loads(path.read_text())
    return {
        "time": np.array([s["time"] for s in steps]),
        "robot": np.array([s["robot_state"] for s in steps]),
        "humans": np.array([s["agent_states"] for s in steps]),
        "robot_goal": np.array([s["robot_goal"] for s in steps]),
    }


def moving_average(values: np.ndarray, n: int) -> np.ndarray:
    """Box average of n samples along axis 0 of a (T, C) array, with the edges padded by their end values."""
    padded = np.pad(values, ((n // 2, n - 1 - n // 2), (0, 0)), mode="edge")
    kernel = np.ones(n) / n
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(values.shape[1])], axis=1)


def label_members(label: str, stations: dict) -> list[str]:
    """Single-station ids a goal label stands for: [] for no goal, one id for a station, two for a shared table."""
    return stations[label]["members"] if label else []


@lru_cache(maxsize=None)
def provenance() -> dict:
    """Git commit of the code and whether its directories have uncommitted changes."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "preprocessing", "metrics", "traj_gen"))}


def write_manifest(path: Path, params: dict, extra: dict) -> None:
    """Record what produced a derived output: the code's git state and the parameters."""
    manifest = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), **provenance(), "params": params, **extra}
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def _read_csv(path: Path, types: dict) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, cast in types.items():
            row[key] = cast(row[key])
    return rows


@lru_cache(maxsize=None)
def load_video_sync() -> dict:
    """{(site, trial, controller): offset_s} with video_time = mocap_time + offset_s."""
    rows = _read_csv(DERIVED / SYNC_FILE, {"offset_s": float})
    return {(r["site"], r["trial"], r["controller"]): r["offset_s"] for r in rows}


def video_time(site: str, trial: str, controller: str, mocap_time):
    """Convert mocap time (s, as in the JSONs) to time in the matching overhead video."""
    return np.asarray(mocap_time) + load_video_sync()[(site, trial, controller)]


def load_stations(site: str) -> dict:
    """Allowed goal labels for a site: single stations ("0") and shared-table pairs ("0/1")."""
    return json.loads((DERIVED / GOALS_DIR / "stations.json").read_text())[site]


def goal_recordings() -> list[tuple[str, str, str]]:
    """(site, trial, controller) of every recording with goal labels, in site, trial and controller order."""
    return [(site, path.parent.name, path.name.removesuffix("_visits.csv"))
            for site in SITES for path in sorted((DERIVED / GOALS_DIR / site).glob("*/*_visits.csv"))]


def load_visits(site: str, trial: str, controller: str) -> list[dict]:
    rows = _read_csv(DERIVED / GOALS_DIR / site / trial / f"{controller}_visits.csv", VISIT_TYPES)
    for r in rows:
        for key in ("inferred", "corrected"):
            r[key] = r[key] == "True"
    return rows


def load_frames(site: str, trial: str, controller: str) -> dict[str, dict]:
    """Per-sample positions and labels of every agent of a recording: {agent id: column arrays}, agent ids ("robot", "human0", ...) in file order."""
    rows = _read_csv(DERIVED / GOALS_DIR / site / trial / f"{controller}_frames.csv", FRAME_TYPES)
    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)
    return {agent: {key: np.array([r[key] for r in agent_rows]) for key in agent_rows[0]} for agent, agent_rows in by_agent.items()}


def load_site_frames(trips_dir: Path) -> dict:
    """Room frame, robot footprint and room-frame stations per site, from the trips folder written by extract_trips.py."""
    return json.loads((trips_dir / "site_frames.json").read_text())


def recording_key(recording: str, *parts: str) -> str:
    """Name of an array of a recording in RECORDINGS_FILE: <recording>/time, <recording>/agents and <recording>/<agent>/<field>."""
    return "/".join((recording, *parts))


def scored_end_index(first: int, end: int, last: int) -> int:
    """Index among a trip's scored samples (every SCORED_STEP-th grid sample from first) of the real end: the scored sample nearest the end
    grid sample, the later one on a tie, and never past the last scored sample (last). Rounding down instead cut up to 0.3 s off the end."""
    return min((end - first + SCORED_STEP // 2) // SCORED_STEP, (last - first) // SCORED_STEP)


def trip_arrays(trip: dict, recording: dict[str, dict[str, np.ndarray]]) -> dict:
    """The sample arrays of a trip, cut from its recording's series ({agent id: {field: (n,) or (n, 3) array}} per SERIES_FIELDS, agents in
    file order) at trip["recording_indices"] (first sample with history, start, end, last sample with padding, on the recording's grid).

    Returns time (seconds from the trip start), start_index, end_index (scored_end_index), tracks ({agent: (T, 3) [x, y, theta]}) and labels
    ({agent: {state, goal, next_goal: (T,)}}) at TRIP_HZ, every SCORED_STEP-th series sample from the start, and input_tracks_10hz:
    the INPUT_HZ series over the same span with its own time, start_index, end_index, tracks and empty_bins (indices of samples whose
    bin held no raw sample).
    """
    first, start, end, last = trip["recording_indices"]
    dense, scored = slice(first, last + 1), slice(first, last + 1, SCORED_STEP)
    time = np.round((np.arange(first, last + 1) - start) / INPUT_HZ, 3)
    return {
        "time": time[::SCORED_STEP], "start_index": (start - first) // SCORED_STEP, "end_index": scored_end_index(first, end, last),
        "tracks": {agent: series["xyt"][scored] for agent, series in recording.items()},
        "labels": {agent: {key: series[key][scored] for key in LABEL_KEYS} for agent, series in recording.items()},
        "input_tracks_10hz": {
            "time": time, "start_index": start - first, "end_index": end - first,
            "tracks": {agent: series["xyt"][dense] for agent, series in recording.items()},
            "empty_bins": {agent: np.flatnonzero(series["empty"][dense]) for agent, series in recording.items()},
        },
    }


def load_trips(trips_dir: Path, **match) -> list[dict]:
    """Trips (each for one planned agent, trip["agent"]) from the trips folder written by extract_trips.py, optionally filtered, e.g.
    load_trips(trips_dir, agent="robot", site="um"): each the metadata of trips.json with its trip_arrays (numpy arrays)."""
    trips = [t for t in json.loads((trips_dir / "trips.json").read_text()) if all(t[k] == v for k, v in match.items())]
    recordings = {}
    with np.load(trips_dir / RECORDINGS_FILE) as archive:
        for name in dict.fromkeys(t["recording"] for t in trips):
            recordings[name] = {agent: {field: archive[recording_key(name, agent, field)] for field in SERIES_FIELDS}
                                for agent in map(str, archive[recording_key(name, "agents")])}
    return [{**t, **trip_arrays(t, recordings[t["recording"]])} for t in trips]
