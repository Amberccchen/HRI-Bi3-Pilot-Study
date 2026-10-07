"""Create blinded visual and motion summaries for four LAAS pairs."""

import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Bi3" / "Bi3"
OUT = Path(__file__).resolve().parent
SESSIONS = ["laas4", "laas5", "laas6", "laas7"]
ALGORITHMS = ["blind", "cv", "cohan"]
TIMES = [0, 40, 80, 120, 160, 200]
WIDTH, HEIGHT = 1440, 1120
sys.path.insert(0, str(ROOT / "pilot_tools"))
import imageio_ffmpeg  # noqa: E402


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


FONT = font(22)
SMALL = font(17)
TITLE = font(27)


def video_duration(ffmpeg, path):
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise ValueError(f"No duration for {path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def frame_at(ffmpeg, video, time_s, path):
    if not path.exists():
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(time_s), "-i", str(video),
             "-frames:v", "1", "-vf", "scale=480:-2", "-y", str(path)],
            check=True,
        )
    with Image.open(path) as im:
        return im.convert("RGB")


def second_samples(log):
    sampled = []
    last_second = -1
    for entry in log:
        second = int(entry["time"])
        if second > last_second:
            sampled.append(entry)
            last_second = second
    return sampled


def stats(samples):
    duration = samples[-1]["time"] - samples[0]["time"]
    robot = [entry["robot_state"][:2] for entry in samples]
    human0 = [entry["agent_states"][0][:2] for entry in samples]
    human1 = [entry["agent_states"][1][:2] for entry in samples]
    nearest = [min(math.dist(r, h0), math.dist(r, h1)) for r, h0, h1 in zip(robot, human0, human1)]
    steps = [math.dist(a, b) for a, b in zip(robot, robot[1:])]
    return {
        "duration_s": round(duration, 1),
        "min_center_distance_m": round(min(nearest), 2),
        "seconds_center_distance_below_1m": sum(d < 1.0 for d in nearest),
        "robot_path_m": round(sum(steps), 1),
        "robot_mean_speed_m_s": round(sum(steps) / duration, 2),
        "robot_stationary_fraction": round(sum(step < 0.05 for step in steps) / len(steps), 2),
    }, robot, human0, human1, nearest


def draw_series(draw, points, rect, bounds, color, marker_every=30):
    left, top, right, bottom = rect
    xmin, xmax, ymin, ymax = bounds
    xy = [
        (left + (p[0] - xmin) / (xmax - xmin) * (right - left),
         bottom - (p[1] - ymin) / (ymax - ymin) * (bottom - top))
        for p in points
    ]
    draw.line(xy, fill=color, width=4)
    for index in range(0, len(xy), marker_every):
        x, y = xy[index]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def make_summary(label, video, log_path, ffmpeg):
    with log_path.open(encoding="utf-8") as handle:
        log = json.load(handle)
    samples = second_samples(log)
    values, robot, human0, human1, nearest = stats(samples)
    duration = video_duration(ffmpeg, video)
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((20, 8), f"Trial {label}  |  {duration:.0f} s total", font=TITLE, fill="#17202a")
    frames_dir = OUT / "frames" / label
    frames_dir.mkdir(parents=True, exist_ok=True)
    for index, requested in enumerate(TIMES):
        actual = min(requested, max(0, int(duration) - 1))
        frame = frame_at(ffmpeg, video, actual, frames_dir / f"{index:02d}.png")
        x = (index % 3) * 480
        y = 44 + (index // 3) * 292
        image.paste(frame, (x, y))
        draw.text((x + 8, y + frame.height + 2), f"t={actual}s", font=SMALL, fill="#17202a")
    draw.text((25, 652), "Full-trial movement tracks (1-second samples)", font=FONT, fill="#17202a")
    rect = (40, 705, 700, 1075)
    draw.rectangle(rect, outline="#a4adb5", width=2)
    all_points = robot + human0 + human1
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    margin = 0.25
    bounds = (min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin)
    draw_series(draw, human0, rect, bounds, "#2776b5")
    draw_series(draw, human1, rect, bounds, "#218f62")
    draw_series(draw, robot, rect, bounds, "#bf3333")
    draw.text((50, 1079), "Robot", font=SMALL, fill="#bf3333")
    draw.text((135, 1079), "Person 1", font=SMALL, fill="#2776b5")
    draw.text((255, 1079), "Person 2", font=SMALL, fill="#218f62")
    draw.text((750, 652), "Robot-to-nearest-person center distance", font=FONT, fill="#17202a")
    graph = (755, 705, 1395, 895)
    draw.rectangle(graph, outline="#a4adb5", width=2)
    max_dist = max(2.0, max(nearest))
    threshold_y = graph[3] - 1.0 / max_dist * (graph[3] - graph[1])
    draw.line((graph[0], threshold_y, graph[2], threshold_y), fill="#a8a8a8", width=2)
    draw.text((graph[0] + 5, threshold_y - 20), "1.0 m", font=SMALL, fill="#666666")
    line = [
        (graph[0] + i / max(1, len(nearest) - 1) * (graph[2] - graph[0]),
         graph[3] - d / max_dist * (graph[3] - graph[1]))
        for i, d in enumerate(nearest)
    ]
    draw.line(line, fill="#7a3db5", width=3)
    draw.text((760, 913), "Center distance does not account for body radii.", font=SMALL, fill="#40464d")
    draw.text((760, 945), f"Minimum center distance: {values['min_center_distance_m']} m", font=SMALL, fill="#17202a")
    draw.text((760, 970), f"Time below 1.0 m: {values['seconds_center_distance_below_1m']} s", font=SMALL, fill="#17202a")
    draw.text((760, 995), f"Robot path length: {values['robot_path_m']} m", font=SMALL, fill="#17202a")
    draw.text((760, 1020), f"Mean speed: {values['robot_mean_speed_m_s']} m/s", font=SMALL, fill="#17202a")
    draw.text((760, 1045), f"Stationary time fraction: {values['robot_stationary_fraction']}", font=SMALL, fill="#17202a")
    image.save(OUT / f"{label}.jpg", quality=90)
    return values


def main():
    mapping_path = OUT / "sealed_mapping.json"
    if mapping_path.exists():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    else:
        mapping = {}
        rng = random.SystemRandom()
        for session in SESSIONS:
            shuffled = ALGORITHMS.copy()
            rng.shuffle(shuffled)
            for anonymous, algorithm in zip("ABC", shuffled):
                mapping[f"{session}_{anonymous}"] = {"session": session, "algorithm": algorithm}
        mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    metrics = {}
    for label, source in mapping.items():
        session = source["session"]
        algorithm = source["algorithm"]
        video = DATA / "videos" / "laas" / session / f"{algorithm}.mp4"
        log = DATA / "jsons" / "laas" / session / f"{algorithm}.json"
        metrics[label] = make_summary(label, video, log, ffmpeg)
    (OUT / "sealed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Prepared {len(metrics)} blinded trial summaries for {len(SESSIONS)} participant pairs.")


if __name__ == "__main__":
    main()
