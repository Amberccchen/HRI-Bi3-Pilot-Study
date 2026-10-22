"""Estimate the time offset between each Bi3 overhead video and its mocap JSON.

Frame-to-frame pixel change in the video is cross-correlated with the summed
speed of the robot and both participants from mocap. The best lag gives
video_time = mocap_time + offset_s. The peak correlation and how far it stands
above the other lags (peak_z) indicate how much to trust each offset.

Output: <out>/video_sync.csv and <out>/video_sync.manifest.json. Read offsets with
bi3_derived.load_video_sync() or bi3_derived.video_time().
"""

import argparse
import csv
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bi3_derived import DERIVED, SYNC_FILE, write_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Bi3" / "Bi3"
OUT = DERIVED
FPS = 10                # both signals are compared at this rate
MAX_LAG_S = 20.0        # search offsets within +/- this
FRAME_SIZE = (160, 120)  # video is downscaled to this before differencing


def video_motion(path: Path) -> np.ndarray:
    w, h = FRAME_SIZE
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={FPS},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))


def mocap_speed(path: Path) -> np.ndarray:
    steps = json.loads(path.read_text())
    time = np.array([s["time"] for s in steps])
    tracks = np.concatenate(
        [np.array([s["robot_state"][:2] for s in steps])[:, None], np.array([[a[:2] for a in s["agent_states"]] for s in steps])], axis=1
    )
    grid = np.arange(time[0], time[-1], 1 / FPS)
    resampled = np.stack([np.stack([np.interp(grid, time, tracks[:, k, j]) for j in range(2)], -1) for k in range(tracks.shape[1])], 1)
    return np.linalg.norm(np.diff(resampled, axis=0), axis=-1).sum(axis=1) * FPS


def process(job: tuple[str, Path, Path]) -> dict:
    site, video, mocap = job
    v, m = video_motion(video), mocap_speed(mocap)
    lags = np.arange(-int(MAX_LAG_S * FPS), int(MAX_LAG_S * FPS) + 1)
    corr = []
    for lag in lags:
        a, b = (v[lag:], m) if lag >= 0 else (v, m[-lag:])
        n = min(len(a), len(b))
        corr.append(np.corrcoef(a[:n], b[:n])[0, 1])
    corr = np.array(corr)
    best = int(np.nanargmax(corr))
    return {
        "site": site, "trial": video.parent.name, "controller": video.stem,
        "video_s": round(len(v) / FPS, 1), "mocap_s": round(len(m) / FPS, 1),
        "offset_s": round(lags[best] / FPS, 2), "peak_corr": round(float(corr[best]), 3),
        "peak_z": round(float((corr[best] - np.nanmean(corr)) / np.nanstd(corr)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    jobs = []
    for site in ["laas", "um"]:
        for video in sorted((args.data / "videos" / site).glob("*/*.mp4")):
            mocap = args.data / "jsons" / site / video.parent.name / f"{video.stem}.json"
            if mocap.exists():
                jobs.append((site, video, mocap))
    with Pool(args.workers) as pool:
        rows = pool.map(process, jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / SYNC_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    params = {"FPS": FPS, "MAX_LAG_S": MAX_LAG_S, "FRAME_SIZE": FRAME_SIZE}
    write_manifest(args.out / "video_sync.manifest.json", Path(__file__), params, {
        "videos": len(rows),
        "columns": "site, trial, controller, video_s, mocap_s, offset_s (video_time = mocap_time + offset_s), "
                   "peak_corr, peak_z (peak height above other lags, in standard deviations)",
    })
    print(f"{len(rows)} videos -> {args.out / SYNC_FILE}")


if __name__ == "__main__":
    main()
