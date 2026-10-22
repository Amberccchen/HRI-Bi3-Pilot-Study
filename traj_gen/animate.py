"""Time views of finished episodes, for when paths cross or double back and a static path plot hides who was where when.

- animation.gif: one frame per time step on the BEV axes. The robot and the people are disks to
  scale with short fading trails; the real controller's robot at the same time is a dashed ghost
  disk; the plan the robot is currently following (rolling episodes replan) is a thin line; the
  frame title gives the time and the current robot-person clearance.
- timeline.png: the whole episode on one map with colour = time (same colour scale for the robot
  and the people), so a crossing point shows whether both were there at the same moment.

Both are written into the episode folder, next to result.png. Read only the episode's own files
and the trip it was run on, so they work on any finished episode (model or baseline).
"""

import io
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from PIL import Image

from bev import EXTENT_M, LABEL_PX, PERSON, ROBOT, _axes, _draw_room
from bi3_derived import load_site_frames, load_trips
from scene import DT
from traj_gen_metrics import HUMAN_RADIUS

TRAIL_S = 3.0      # fading trail length: about the history the model sees (2.8 s)
GIF_PX = 640       # frames are downscaled to this side: readable, and a 100-frame GIF stays a few MB
CMAP = "viridis"   # perceptually uniform, so equal colour steps are equal time steps


def _load(folder: Path) -> dict:
    """Everything a time view needs from an episode folder and its trip."""
    manifest = json.loads((folder / "manifest.json").read_text())
    trip = load_trips(id=manifest["trip"])[0]
    data = np.loadtxt(folder / "trajectory.csv", delimiter=",", skiprows=1, ndmin=2)
    a = trip["start_index"]
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_text().splitlines()]
    return {"manifest": manifest, "site": load_site_frames()[trip["site"]], "goal": np.asarray(trip["robot_goal_xy"], float),
            "t": data[:, 0], "robot": data[:, 1:3], "humans": [data[:, 3 + 2 * h: 5 + 2 * h] for h in range((data.shape[1] - 3) // 2)],
            "real": np.asarray(trip["robot"], float)[a: trip["end_index"] + 1, :2], "plans": plans}


def _title(ep: dict) -> str:
    m = ep["manifest"]
    return " ".join(str(v) for v in (m["experiment"], m["framing"], m["thinking"], m["trip"], f"s{m['sample']}") if v is not None)


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    return buf.getvalue()


def _frame(ep: dict, k: int) -> bytes:
    fig, ax, _ = _axes(False)
    site, radius = ep["site"], ep["site"]["robot_radius_m"]
    _draw_room(ax, site, ep["goal"])
    ax.plot(*ep["real"].T, ls="--", color="#555555", lw=1.5, alpha=0.5, zorder=3)
    if k < len(ep["real"]):
        ax.add_patch(Circle(ep["real"][k], radius, fill=False, ec="#555555", ls="--", lw=2, zorder=4))
    t = ep["t"][k]
    current = [p for p in ep["plans"] if p["time_s"] <= t + 1e-6]
    if current and current[-1]["positions_m"]:
        q = np.searchsorted(ep["t"], current[-1]["time_s"] - 1e-6)
        plan = np.vstack([ep["robot"][q], current[-1]["positions_m"]])
        ax.plot(*plan.T, ":", color=ROBOT, lw=1.5, alpha=0.8, zorder=3)
    lo = max(0, k - round(TRAIL_S / DT))
    for track, color, r in [(ep["robot"], ROBOT, radius)] + [(h, PERSON, HUMAN_RADIUS) for h in ep["humans"]]:
        seg = track[lo: k + 1]
        for j in range(1, len(seg)):  # older segments fainter
            ax.plot(*seg[j - 1: j + 1].T, "-", color=color, lw=3, alpha=0.15 + 0.85 * j / len(seg), zorder=4)
        ax.add_patch(Circle(track[k], r, fc=color, ec="k", alpha=0.55, zorder=5))
    for h, track in enumerate(ep["humans"]):
        ax.text(*track[k], f"P{h + 1}", ha="center", va="center", fontsize=LABEL_PX * 0.6, zorder=6)
    clearance = min(np.linalg.norm(h[k] - ep["robot"][k]) for h in ep["humans"]) - radius - HUMAN_RADIUS
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"{_title(ep)}\nt = {t:.1f} s   clearance {clearance:+.2f} m\ndashed ring: real robot now, dotted: current plan",
            ha="left", va="top", fontsize=LABEL_PX * 0.6, zorder=7)
    return _png(fig)


def animate(folder: Path, speed: float) -> Path:
    """Write <folder>/animation.gif, played `speed` times faster than real time; the last frame is held."""
    ep = _load(folder)
    frames = [Image.open(io.BytesIO(_frame(ep, k))).convert("RGB") for k in range(len(ep["t"]))]
    for f in frames:
        f.thumbnail((GIF_PX, GIF_PX), Image.Resampling.LANCZOS)
    ms = round(1000 * DT / speed)
    out = folder / "animation.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=[ms] * (len(frames) - 1) + [2000], loop=0)
    return out


def timeline(folder: Path) -> Path:
    """Write <folder>/timeline.png: robot and people paths coloured by time on one scale."""
    ep = _load(folder)
    fig, ax, _ = _axes(False)
    _draw_room(ax, ep["site"], ep["goal"])
    ax.plot(*ep["real"].T, ls="--", color="#555555", lw=1.5, alpha=0.6, zorder=3)
    cmap, t = colormaps[CMAP], ep["t"]
    norm = (t - t[0]) / max(t[-1] - t[0], DT)
    for track, lw, marker in [(ep["robot"], 5, "o")] + [(h, 2.5, "^") for h in ep["humans"]]:
        ax.add_collection(LineCollection(np.stack([track[:-1], track[1:]], axis=1), colors=cmap(norm[:-1]), lw=lw, zorder=4))
        ax.scatter(*track.T, c=cmap(norm), marker=marker, s=30 if marker == "o" else 22, ec="k", lw=0.3, zorder=5)
    for h, track in enumerate(ep["humans"]):
        ax.text(*track[0] + [0, 0.12], f"P{h + 1}", ha="center", fontsize=LABEL_PX * 0.7, color="#7a3300", zorder=7)
    ax.plot(*ep["robot"][0], marker="s", color="k", ms=8, zorder=6)
    cax = fig.add_axes([0.665, 0.924, 0.245, 0.012])  # x 0.6..1.7 m, y 1.86..1.92 m: the strip above the room's top wall, right of the title
    cbar = fig.colorbar(ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_ticks([0, 1], labels=[f"{t[0]:.1f} s", f"{t[-1]:.1f} s"])
    cbar.ax.tick_params(labelsize=LABEL_PX * 0.55)
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"{_title(ep)}\ncolour = time; circles: robot, triangles: people; dashed: real path", ha="left", va="top", fontsize=LABEL_PX * 0.6, zorder=7)
    out = folder / "timeline.png"
    out.write_bytes(_png(fig))
    return out
