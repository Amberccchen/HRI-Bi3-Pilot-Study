"""Top-down map (BEV) of a Scene: the image the vision model sees and the pixel <-> metre transform.

Fixed extent (the 3.5 m room plus a margin), so a metre is always the same number of pixels
and a drawn mark can be mapped back exactly. Drawn from the Scene only (leakage rule).
Pixel coordinates are continuous image coordinates: (0, 0) is the top-left corner of the
image, u to the right, v downward.
"""

import io
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
from scene import Scene  # noqa: E402
from traj_gen_metrics import GOAL_TOL, HUMAN_RADIUS  # noqa: E402


# 896 = 32 * 28: Qwen's vision tower works on 28 px blocks (one token each), so this size is
# not resized (1024 tokens per map) and is big enough that a person (0.25 m) is 50 px wide.
SIZE_PX = 896
DPI = 100            # only fixes the figure size in inches; pixels are what matter
EXTENT_M = 4.0       # drawn extent per side: the 3.5 m room plus 0.25 m margin each side for people at the walls
PLOT_PX = 800        # axes size, so 1 m = 200 px exactly and a pixel is 5 mm
AXES_LEFT_PX, AXES_BOTTOM_PX = 76, 56  # room for the axis label and tick labels on the left and bottom; 76 + 800 + 20 = 896
GRID_M = 0.5         # grid spacing in metre mode
GRID_PX = 100        # grid spacing in pixel mode: round numbers for the model to read (0.5 m = 100 px)
STATION_M = 0.2      # drawn station box size: the station position is the robot's goal point in front of the table and real robots stop 0.49-0.60 m from it, so a bigger box would overlap the robot
LABEL_OFFSET_M = 0.28  # station labels sit this far outward (away from the room centre), as the box is too small to hold one
TABLE_PAD_M = 0.12   # shared-table outline around its two stations
LABEL_PX = 20        # label font size; readable at 896 px
ROBOT, PERSON, GOAL, GRID, WALL, BOX = "#1f77b4", "#e66101", "#2ca02c", "#d9d9d9", "#222222", "#b8b8b8"


@dataclass(frozen=True)
class BevTransform:
    """u = u0 + scale * x, v = v0 - scale * y (metres of the room frame <-> image pixels)."""

    scale: float
    u0: float
    v0: float

    @classmethod
    def from_axes(cls, ax, height_px: int) -> "BevTransform":
        (ux, vy), (ux1, _) = ax.transData.transform([(0, 0), (1, 0)])  # matplotlib display y is up from the bottom
        return cls(scale=float(ux1 - ux), u0=float(ux), v0=float(height_px - vy))

    def to_pixel(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        return np.stack([self.u0 + self.scale * xy[..., 0], self.v0 - self.scale * xy[..., 1]], axis=-1)

    def to_metre(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float)
        return np.stack([(uv[..., 0] - self.u0) / self.scale, (self.v0 - uv[..., 1]) / self.scale], axis=-1)


def _label(label: str) -> str:
    """Station label as drawn: S3, S0/S1, none."""
    return "none" if label == "" else "/".join(f"S{k}" for k in label.split("/"))


def _axes(pixel_ticks: bool):
    fig = Figure(figsize=(SIZE_PX / DPI, SIZE_PX / DPI), dpi=DPI)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([AXES_LEFT_PX / SIZE_PX, AXES_BOTTOM_PX / SIZE_PX, PLOT_PX / SIZE_PX, PLOT_PX / SIZE_PX])
    ax.set_xlim(-EXTENT_M / 2, EXTENT_M / 2)
    ax.set_ylim(-EXTENT_M / 2, EXTENT_M / 2)
    fig.canvas.draw()  # the transform below is read from the finished axes
    transform = BevTransform.from_axes(ax, SIZE_PX)
    if pixel_ticks:
        u = np.arange(GRID_PX, SIZE_PX, GRID_PX)
        xs, ys = transform.to_metre(np.stack([u, u], axis=-1)).T
        ax.set_xticks(xs, [str(k) for k in u])
        ax.set_yticks(ys, [str(k) for k in u])  # v labels run downward: ys is in the same order as u
    else:
        ticks = np.arange(-EXTENT_M / 2, EXTENT_M / 2 + 1e-9, GRID_M)
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.grid(color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=14)
    ax.set_xlabel("u (px)" if pixel_ticks else "x (m)", fontsize=16)
    ax.set_ylabel("v (px)" if pixel_ticks else "y (m)", fontsize=16)
    ax.yaxis.set_label_coords(-0.065, 0.5)  # the default position is clipped by the image edge
    return fig, ax, transform


def _trail(ax, history: np.ndarray, now: np.ndarray, color: str) -> None:
    pts = np.vstack([history, now])
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=color, lw=2, ms=4, alpha=0.8, zorder=3)


def render_bev(scene: Scene, pixel_ticks: bool = False) -> tuple[bytes, BevTransform]:
    """PNG bytes and the exact transform. pixel_ticks labels the grid in image pixels (for the pixel output mode)."""
    fig, ax, transform = _axes(pixel_ticks)
    room = scene.site["room_m"]
    ax.add_patch(Rectangle((-room / 2,) * 2, room, room, fill=False, ec=WALL, lw=3, zorder=2))
    stations = scene.site["stations"]
    for k, s in stations.items():
        if "/" in k:
            pts = np.array([stations[m]["xy"] for m in s["members"]])
            lo, hi = pts.min(axis=0) - (STATION_M / 2 + TABLE_PAD_M), pts.max(axis=0) + (STATION_M / 2 + TABLE_PAD_M)
            ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=BOX, lw=2, ls="--", zorder=2))
            ax.text(max(room / 2, hi[0]) + 0.03, (lo[1] + hi[1]) / 2, "shared table", ha="left", va="center", rotation=90, fontsize=LABEL_PX * 0.6, color="#555555")  # just outside the wall the table is on (+x at both sites)
        else:
            x, y = s["xy"]
            ax.add_patch(Rectangle((x - STATION_M / 2, y - STATION_M / 2), STATION_M, STATION_M, fc=BOX, ec="#777777", zorder=2))
            out = np.array([x, y]) * (1 + LABEL_OFFSET_M / np.hypot(x, y))
            ax.text(*out, _label(k), ha="center", va="center", fontsize=LABEL_PX * 0.8, zorder=4)
    ax.add_patch(Circle(scene.goal_xy, GOAL_TOL, fill=False, ec=GOAL, lw=2, ls="--", zorder=3))
    ax.plot(*scene.goal_xy, marker="*", color=GOAL, ms=18, zorder=4)
    _trail(ax, scene.robot_history, scene.robot[-1], ROBOT)
    ax.add_patch(Circle(scene.robot[-1], scene.site["robot_radius_m"], fc=ROBOT, ec="k", alpha=0.45, zorder=5))
    ax.plot(*scene.robot[-1], marker="+", color="k", ms=10, zorder=6)
    for p in scene.people:
        _trail(ax, p["history"], p["xy"], PERSON)
        ax.add_patch(Circle(p["xy"], HUMAN_RADIUS, fc=PERSON, ec="k", alpha=0.6, zorder=5))
        ax.text(p["xy"][0], p["xy"][1] + HUMAN_RADIUS + 0.06, f"{p['name']} → {_label(p['goal'])} then {_label(p['next_goal'])}",
                ha="center", va="bottom", fontsize=LABEL_PX * 0.7, color="#7a3300", zorder=7,
                bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1})
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"t = {scene.time_s:.1f} s", ha="left", va="top", fontsize=LABEL_PX * 0.8, zorder=7)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)  # no bbox_inches="tight": it would shift the pixels
    return buf.getvalue(), BevTransform.from_axes(ax, SIZE_PX)


@lru_cache(maxsize=None)
def bev_transform() -> BevTransform:
    """The transform without a scene (it only depends on the fixed layout), e.g. to build Pixels units."""
    return _axes(False)[2]
