"""Read and write the shared Bi3 derived files (video sync offsets, goal labels).

The files live outside this repository, under BI3_DERIVED (default
/persistent/data/Bi3_derived). Each output folder carries a manifest.json with
the parameters and a fingerprint of the script that produced it, so everyone can
check they are using the same labels instead of rerunning the scripts.

    # with preprocessing/ on sys.path (or PYTHONPATH)
    from bi3_derived import load_video_sync, video_time, load_stations, load_visits, load_frames
    video_time("laas", "laas1", "cohan", 63.2)        # seconds into laas1/cohan.mp4
    load_stations("laas")["0/1"]                        # {"members": ["0", "1"], "xy": [...]}
    frames = load_frames("laas", "laas1", "cohan", agent="human0")
    frames["goal"][1000], frames["next_goal"][1000]     # e.g. "5", "0/1"
"""

import csv
import datetime
import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


DERIVED = Path(os.environ.get("BI3_DERIVED", "/persistent/data/Bi3_derived"))
SYNC_FILE = "video_sync.csv"
GOALS_DIR = "goals"
FRAME_TYPES = {"step": int, "time": float, "x": float, "y": float, "theta": float}
VISIT_TYPES = {"visit": int, "station": int, "t_arrive": float, "t_turn": float, "t_leave": float,
               "min_dist": float, "margin": float}


def write_manifest(path: Path, script: Path, params: dict, extra: dict | None = None) -> None:
    """Record what produced a derived output: script hash, git state and parameters."""
    root = script.resolve().parents[1]
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    manifest = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": str(script.resolve().relative_to(root)),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "script_uncommitted": bool(git("status", "--porcelain", "--", str(script.resolve()))),
        "params": params,
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")


def _read_csv(path: Path, types: dict) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate it with the preprocessing scripts (see README)")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, cast in types.items():
            if row.get(key, "") != "":
                row[key] = cast(row[key])
    return rows


@lru_cache(maxsize=None)
def load_video_sync(root: Path = DERIVED) -> dict:
    """{(site, trial, controller): offset_s} with video_time = mocap_time + offset_s."""
    rows = _read_csv(root / SYNC_FILE, {"offset_s": float})
    return {(r["site"], r["trial"], r["controller"]): r["offset_s"] for r in rows}


def video_time(site: str, trial: str, controller: str, mocap_time, root: Path = DERIVED):
    """Convert mocap time (s, as in the JSONs) to time in the matching overhead video."""
    return np.asarray(mocap_time) + load_video_sync(root)[(site, trial, controller)]


def load_stations(site: str, root: Path = DERIVED) -> dict:
    """Allowed goal labels for a site: single stations ("0") and shared-table pairs ("0/1")."""
    return json.loads((root / GOALS_DIR / "stations.json").read_text())[site]


def load_visits(site: str, trial: str, controller: str, root: Path = DERIVED) -> list[dict]:
    rows = _read_csv(root / GOALS_DIR / site / trial / f"{controller}_visits.csv", VISIT_TYPES)
    for r in rows:
        for key in ("inferred", "corrected"):
            r[key] = r[key] == "True"
    return rows


def load_frames(site: str, trial: str, controller: str, agent: str | None = None, root: Path = DERIVED) -> dict:
    """Per-sample labels as column arrays; agent is "robot", "human0" or "human1"."""
    rows = _read_csv(root / GOALS_DIR / site / trial / f"{controller}_frames.csv", FRAME_TYPES)
    if agent is not None:
        rows = [r for r in rows if r["agent"] == agent]
    return {key: np.array([r[key] for r in rows]) for key in rows[0]} if rows else {}


TRIPS_DIR = "trips"


def load_site_frames(root: Path = DERIVED) -> dict:
    """Room frame, robot footprint/radius and room-frame stations per site (from extract_trips.py)."""
    return json.loads((root / TRIPS_DIR / "site_frames.json").read_text())


def load_trips(root: Path = DERIVED, **match) -> list[dict]:
    """Robot trips from extract_trips.py, optionally filtered, e.g. load_trips(site="um", controller="cohan")."""
    path = root / TRIPS_DIR / "trips.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; generate it with preprocessing/extract_trips.py")
    trips = json.loads(path.read_text())
    return [t for t in trips if all(t.get(k) == v for k, v in match.items())]
