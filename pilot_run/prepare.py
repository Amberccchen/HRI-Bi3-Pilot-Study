"""Prepare blinded, time-stamped contact sheets for a Bi3 smoke test."""

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Bi3" / "Bi3"
OUT = ROOT / "pilot_run" / "laas3"
SESSION = "laas3"
ALGORITHMS = ["blind", "cv", "cohan"]
FRAME_PERIOD_S = 5
FRAME_WIDTH = 480
FRAMES_PER_PAGE = 15

sys.path.insert(0, str(ROOT / "pilot_tools"))
import imageio_ffmpeg  # noqa: E402


def duration_s(ffmpeg: str, path: Path) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not read duration for {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def make_sheets(frame_dir: Path, label: str) -> int:
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {frame_dir}")
    with Image.open(frames[0]) as first:
        frame_height = first.height
    columns = 3
    rows = (FRAMES_PER_PAGE + columns - 1) // columns
    header = 38
    caption = 26
    for page, offset in enumerate(range(0, len(frames), FRAMES_PER_PAGE), start=1):
        sheet = Image.new(
            "RGB", (columns * FRAME_WIDTH, rows * (frame_height + caption) + header), "white"
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), f"Trial {label} | {SESSION} | page {page} | one frame every {FRAME_PERIOD_S}s", fill="black")
        for slot, frame_path in enumerate(frames[offset : offset + FRAMES_PER_PAGE]):
            with Image.open(frame_path) as frame:
                x = (slot % columns) * FRAME_WIDTH
                y = header + (slot // columns) * (frame_height + caption)
                sheet.paste(frame.convert("RGB"), (x, y))
                draw.text((x + 8, y + frame_height + 4), f"t ~ {(offset + slot) * FRAME_PERIOD_S}s", fill="black")
        sheet.save(OUT / f"trial_{label}_page{page}.jpg", quality=90)
    return len(frames)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mapping_path = OUT / "sealed_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        shuffled = ALGORITHMS.copy()
        random.SystemRandom().shuffle(shuffled)
        mapping = dict(zip("ABC", shuffled))
        mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    report = {}
    for label in "ABC":
        algorithm = mapping[label]
        video = DATA / "videos" / "laas" / SESSION / f"{algorithm}.mp4"
        if not video.exists():
            raise FileNotFoundError(video)
        frame_dir = OUT / f"frames_{FRAME_PERIOD_S}s_{label}"
        frame_dir.mkdir(exist_ok=True)
        if not list(frame_dir.glob("*.jpg")):
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-vf",
                    f"fps=1/{FRAME_PERIOD_S},scale={FRAME_WIDTH}:-2",
                    "-q:v",
                    "3",
                    str(frame_dir / "%03d.jpg"),
                ],
                check=True,
            )
        count = make_sheets(frame_dir, label)
        report[label] = {"duration_s": duration_s(ffmpeg, video), "frames": count}
    (OUT / "media_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
