"""Render anonymized mocap views after frame-stage predictions are frozen."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "Bi3" / "Bi3" / "jsons" / "laas" / "laas11"
mapping = json.loads((OUT / "sealed_mapping.json").read_text())
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)


def line_plot(draw, series, rect, maximum, color, label, threshold=None, mark=None):
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, outline="#707070", width=2)
    if threshold is not None:
        y = y1 - threshold / maximum * (y1 - y0)
        draw.line((x0, y, x1, y), fill="#999999", width=2)
        draw.text((x0 + 5, y - 21), f"{threshold:g} m", font=SMALL, fill="#555555")
    xy = [(x0 + i / max(1, len(series)-1) * (x1-x0), y1 - min(maximum, v)/maximum * (y1-y0)) for i, v in enumerate(series)]
    draw.line(xy, fill=color, width=3)
    if mark is not None:
        x = x0 + mark / max(1, len(series)-1) * (x1-x0)
        draw.line((x, y0, x, y1), fill="#aa3333", width=2)
    draw.text((x0, y0-34), label, font=FONT, fill="#222222")
    draw.text((x0, y1+7), "start", font=SMALL, fill="#555555")
    draw.text((x1-40, y1+7), "end", font=SMALL, fill="#555555")


for label, algorithm in mapping.items():
    log = json.loads((DATA / f"{algorithm}.json").read_text())
    samples = {}
    for e in log:
        samples.setdefault(int(e["time"]), e)
    entries = [samples[k] for k in sorted(samples)]
    distances = [min(math.dist(e["robot_state"][:2], a[:2]) for a in e["agent_states"][:2]) for e in entries]
    closest = min(range(len(entries)), key=lambda i: distances[i])
    speeds = [0.0] + [math.dist(entries[i]["robot_state"][:2], entries[i-1]["robot_state"][:2]) for i in range(1, len(entries))]
    window = entries[max(0,closest-15):min(len(entries),closest+16)]
    paths = [[e["robot_state"][:2] for e in window]] + [[e["agent_states"][p][:2] for e in window] for p in (0,1)]
    xs = [p[0] for path in paths for p in path]
    ys = [p[1] for path in paths for p in path]
    xmin, xmax = min(xs)-0.4, max(xs)+0.4
    ymin, ymax = min(ys)-0.4, max(ys)+0.4
    canvas = Image.new("RGB", (1400, 850), "#fafafa")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 18), f"Trial {label}: mocap map and full-trial time series", font=FONT, fill="#222222")
    rect=(60,100,690,730)
    draw.rectangle(rect, outline="#777777", width=2)
    colors=["#ba3030", "#2878b4", "#268a55"]
    for path,color,name in zip(paths,colors,["Robot","Person 1","Person 2"]):
        xy=[(rect[0]+(x-xmin)/(xmax-xmin)*(rect[2]-rect[0]), rect[3]-(y-ymin)/(ymax-ymin)*(rect[3]-rect[1])) for x,y in path]
        draw.line(xy, fill=color, width=5)
        for pos in xy[::5]:
            draw.ellipse((pos[0]-5,pos[1]-5,pos[0]+5,pos[1]+5),fill=color)
        draw.text((65+(["Robot","Person 1","Person 2"].index(name))*190,755),name,font=FONT,fill=color)
    draw.text((60,65), "Tracks in 30 s around closest recorded approach", font=FONT, fill="#222222")
    line_plot(draw, distances, (760,140,1350,375), max(2.5,max(distances)), "#7440ae", "Nearest person center distance (m)", threshold=1.0, mark=closest)
    line_plot(draw, speeds, (760,495,1350,730), max(0.5,max(speeds)), "#ca7525", "Robot displacement per second (m)", mark=closest)
    canvas.save(OUT / f"{label}_motion_map.jpg", quality=90)
print("Rendered three anonymized mocap maps.")

