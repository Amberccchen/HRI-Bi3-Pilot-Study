"""Prepare one blinded LAAS session for an incremental perception pilot."""

import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "Bi3" / "Bi3"
SESSION = "laas10"
sys.path.insert(0, str(ROOT / "pilot_tools"))
import imageio_ffmpeg  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)


def duration(video):
    result = subprocess.run([FFMPEG, "-hide_banner", "-i", str(video)], capture_output=True, text=True)
    h, m, s = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def frame(video, second, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{second:04d}.jpg"
    if not path.exists():
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", str(second), "-i", str(video), "-frames:v", "1", "-vf", "scale=480:-2", "-y", str(path)], check=True)
    with Image.open(path) as im:
        return im.convert("RGB")


def contact_sheet(label, video, times, suffix, columns):
    cache = OUT / "frames" / label
    first = frame(video, times[0], cache)
    w, h = first.size
    rows = math.ceil(len(times) / columns)
    sheet = Image.new("RGB", (w * columns, (h + 30) * rows + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"Trial {label}: {suffix}", font=FONT, fill="black")
    for i, t in enumerate(times):
        x, y = (i % columns) * w, 40 + (i // columns) * (h + 30)
        sheet.paste(frame(video, t, cache), (x, y))
        draw.text((x + 8, y + h + 3), f"t={t}s", font=FONT, fill="black")
    sheet.save(OUT / f"{label}_{suffix}.jpg", quality=88)


def main():
    OUT.mkdir(exist_ok=True)
    mapping_path = OUT / "sealed_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
    else:
        algorithms = ["blind", "cv", "cohan"]
        random.SystemRandom().shuffle(algorithms)
        mapping = {f"{SESSION}_{letter}": algorithm for letter, algorithm in zip("ABC", algorithms)}
        mapping_path.write_text(json.dumps(mapping, indent=2) + "\n")
    truth = {}
    for label, algorithm in mapping.items():
        video = DATA / "videos" / "laas" / SESSION / f"{algorithm}.mp4"
        log_path = DATA / "jsons" / "laas" / SESSION / f"{algorithm}.json"
        log = json.loads(log_path.read_text())
        sampled = {}
        for entry in log:
            sampled.setdefault(int(entry["time"]), entry)
        entries = [sampled[k] for k in sorted(sampled)]
        distances = [min(math.dist(e["robot_state"][:2], a[:2]) for a in e["agent_states"][:2]) for e in entries]
        closest_index = min(range(len(entries)), key=lambda i: distances[i])
        closest_second = int(entries[closest_index]["time"])
        total = int(duration(video))
        overview_times = list(range(0, total, 15))
        window_start = max(0, min(total - 11, closest_second - 5))
        window_times = list(range(window_start, window_start + 11))
        contact_sheet(label, video, overview_times, "overview_15s", 5)
        contact_sheet(label, video, window_times, "closest_1s", 4)
        truth[label] = {"duration_s": total, "closest_second": closest_second, "minimum_center_distance_m": round(distances[closest_index], 3), "overview_times": overview_times, "window_times": window_times}
    (OUT / "sealed_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print("Prepared three anonymous trials. Do not open sealed_mapping.json or sealed_truth.json before visual predictions.")


if __name__ == "__main__":
    main()
