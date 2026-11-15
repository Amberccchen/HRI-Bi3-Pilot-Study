"""Top-down map (BEV) of a Scene: the image the vision model sees and the pixel <-> metre transform.

Fixed extent (the 3.5 m room plus a margin), so a metre is always the same number of pixels
and a drawn mark can be mapped back exactly. Drawn from the Scene only (leakage rule).
Pixel coordinates are continuous image coordinates: (0, 0) is the top-left corner of the
image, u to the right, v downward.
"""

import io
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from preprocessing.bi3_derived import label_members
from traj_gen.agents import ARRIVAL_RADIUS_M, agent_kind, body_radius, prompt_radius
from traj_gen.scene import DT, Scene, person_name


# 896 = 32 * 28: Qwen's vision tower works on 28 px blocks (one token each), so this size is
# not resized (1024 tokens per map) and is big enough that a person (0.25 m) is 50 px wide.
SIZE_PX = 896
DPI = 100            # only fixes the figure size in inches. Pixels are what matter
EXTENT_M = 4.0       # drawn extent per side: the 3.5 m room plus 0.25 m margin each side for people at the walls
PLOT_PX = 800        # axes size, so 1 m = 200 px exactly and a pixel is 5 mm
AXES_LEFT_PX, AXES_BOTTOM_PX = 76, 56  # room for the axis label and tick labels on the left and bottom, 76 + 800 + 20 = 896
GRID_M = 0.5         # grid spacing in metre mode
GRID_PX = 100        # grid spacing in pixel mode: round numbers for the model to read (0.5 m = 100 px)
STATION_BOX_M = 0.2   # side of a station's box as drawn on the map: 40 px, wide enough for an S<k> label at 0.8 * LABEL_PX
LABEL_PX = 20        # label font size, readable at 896 px
LABEL_EVERY_S = 2.0  # result figure: time mark on the executed path every 2 s (one rolling execution window)
ROBOT, PERSON, GOAL, GRID, WALL, BOX = "#1f77b4", "#e66101", "#2ca02c", "#d9d9d9", "#222222", "#b8b8b8"
STRAIGHT_LINE, REAL_PATH, PERSON_LABEL, PLANNED_PERSON = "#777777", "#006d2c", "#7a3300", "#7b3294"  # the real path is a darker green than the goal marks
KIND_COLOUR = {"robot": ROBOT, "human": PERSON}  # colour of a logged agent of a kind
PLANNED_COLOUR = {"robot": ROBOT, "human": PLANNED_PERSON}  # the planned agent on the model's map, by kind: a planned person stands apart from the other person
# Result and overlay figures: the colour says who controls an agent (the VLM or the log), the marker shape what the agent is.
VLM_COLOUR = PLANNED_PERSON
KIND_MARKER = {"robot": "s", "human": "^"}
KIND_NAME = {"robot": "robot", "human": "person"}
START_MARKER = "X"
REAL_PATH_LABEL = {"robot": "real robot's path", "human": "real person's path"}  # green dashed line, by the planned agent's kind


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


def _label(label: str, stations: dict) -> str:
    """Station label as drawn: S3, S0/S1, none."""
    members = label_members(label, stations)
    return "/".join(f"S{k}" for k in members) if members else "none"


def figure_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)  # no bbox_inches="tight": it would shift the pixels
    return buf.getvalue()


def bev_axes(pixel_ticks: bool):
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


def draw_room(ax, site: dict, agent: str, goal_xy: np.ndarray) -> None:
    """Walls, stations (a shared table is one box spanning its two stations) and the planned agent's goal with its arrival radius: the part of every map that does not change in time.
    Stations are drawn at the points where people stand, for every planned agent. Only the goal is at the planned agent's own point."""
    room = site["room_m"]
    ax.add_patch(Rectangle((-room / 2,) * 2, room, room, fill=False, ec=WALL, lw=3, zorder=2))
    stations = site["stations"]
    # Each station is centred on the room border: the standing point moved perpendicular onto the nearest side.
    on_border = {}
    for k, s in stations.items():
        p = np.array(s["human_xy"], float)
        side = int(np.argmax(np.abs(p)))
        on_border[k] = p.copy()
        on_border[k][side] = np.sign(p[side]) * room / 2
    tabled = {m for s in stations.values() if len(s["members"]) > 1 for m in s["members"]}
    for k, s in stations.items():
        if len(s["members"]) > 1:  # one box over the table's stations, labelled by the stations' own entries
            pts = np.array([on_border[m] for m in s["members"]])
            lo, hi = pts.min(axis=0) - STATION_BOX_M / 2, pts.max(axis=0) + STATION_BOX_M / 2
            ax.add_patch(Rectangle(lo, *(hi - lo), fc=BOX, ec="#777777", zorder=2))
        else:
            if k not in tabled:
                ax.add_patch(Rectangle(on_border[k] - STATION_BOX_M / 2, STATION_BOX_M, STATION_BOX_M, fc=BOX, ec="#777777", zorder=2))
            ax.text(*on_border[k], _label(k, stations), ha="center", va="center", fontsize=LABEL_PX * 0.8, color=WALL, zorder=4)
    ax.add_patch(Circle(goal_xy, ARRIVAL_RADIUS_M[agent_kind(agent)], fill=False, ec=GOAL, lw=2, ls="--", zorder=3))
    ax.plot(*goal_xy, marker="*", color=GOAL, ms=18, zorder=4)


def render_bev(scene: Scene, pixel_ticks: bool, predict: bool) -> tuple[bytes, BevTransform]:
    """PNG bytes and the exact transform. pixel_ticks labels the grid in image pixels (for the pixel output mode). predict marks each person's constant-velocity positions at PREDICT_S."""
    fig, ax, transform = bev_axes(pixel_ticks)
    site, kind = scene.site, agent_kind(scene.agent)
    colour, radius = PLANNED_COLOUR[kind], prompt_radius(scene.agent, site)
    draw_room(ax, site, scene.agent, scene.goal_xy)
    _trail(ax, scene.history, scene.track[-1], colour)
    ax.add_patch(Circle(scene.track[-1], radius, fc=colour, ec="k", alpha=0.45, zorder=5))
    ax.plot(*scene.track[-1], marker="+", color="k", ms=10, zorder=6)
    if kind == "human":
        ax.text(scene.track[-1][0], scene.track[-1][1] + radius + 0.06, "you", ha="center", va="bottom", fontsize=LABEL_PX * 0.7, color=PLANNED_PERSON,
                zorder=7, bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1})
    for p in scene.others:
        r, other_colour = prompt_radius(p["id"], site), KIND_COLOUR[p["kind"]]
        _trail(ax, p["history"], p["xy"], other_colour)
        ax.add_patch(Circle(p["xy"], r, fc=other_colour, ec="k", alpha=0.6, zorder=5))
        if predict:
            ax.plot(*np.vstack([p["xy"], p["predicted"]]).T, ls="--", color=other_colour, lw=1, zorder=3)
            for q in p["predicted"]:
                ax.add_patch(Circle(q, r, fill=False, ec=other_colour, lw=2, zorder=5))
        if p["kind"] == "human":  # the map shows a person's goals only to a planned robot
            label = f"{person_name(p['id'])} → {_label(p['goal'], site['stations'])} then {_label(p['next_goal'], site['stations'])}" if kind == "robot" else person_name(p["id"])
            ax.text(p["xy"][0], p["xy"][1] + r + 0.06, label, ha="center", va="bottom", fontsize=LABEL_PX * 0.7, color=PERSON_LABEL, zorder=7,
                    bbox={"fc": "white", "ec": "none", "alpha": 0.75, "pad": 1})
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"t = {scene.time_s:.1f} s", ha="left", va="top", fontsize=LABEL_PX * 0.8, zorder=7)
    return figure_png(fig), BevTransform.from_axes(ax, SIZE_PX)


@lru_cache(maxsize=None)
def bev_transform() -> BevTransform:
    """The transform without a scene (it only depends on the fixed layout), e.g. to build Pixels units."""
    return bev_axes(False)[2]


def draw_others(ax, site: dict, others: dict[str, np.ndarray]) -> None:
    """Tracks of the logged agents ({agent id: (T, 2)}): trail with the kind's marker, final disk at the last point, at the scoring radius,, and a P<k> label at a person's start."""
    for other, track in others.items():
        kind = agent_kind(other)
        ax.plot(*track.T, "-", marker=KIND_MARKER[kind], color=KIND_COLOUR[kind], lw=2, ms=4, alpha=0.7, zorder=3)
        ax.add_patch(Circle(track[-1], body_radius(other, site), fc=KIND_COLOUR[kind], ec="k", alpha=0.4, zorder=5))
        ax.plot(*track[-1], marker=KIND_MARKER[kind], color=KIND_COLOUR[kind], mec="k", ms=7, zorder=6)
        if kind == "human":
            ax.text(*track[0] + [0, 0.12], person_name(other), ha="center", fontsize=LABEL_PX * 0.7, color=PERSON_LABEL, zorder=7)


def draw_reference(ax, goal_xy: np.ndarray, start: np.ndarray, real: np.ndarray) -> None:
    """The straight line from the start to the goal (below the executed paths) and the real path of the planned agent (above them, so it stays visible where they overlap)."""
    ax.plot(*np.array([start, goal_xy]).T, ls=":", color=STRAIGHT_LINE, lw=2, zorder=3)
    ax.plot(*real.T, ls="--", color=REAL_PATH, lw=2, zorder=6)
    ax.plot(*start, marker=START_MARKER, color="k", ms=9, zorder=7)


def trajectory_legend(agent: str, others: list[str]) -> list:
    """Legend entries of the result and overlay figures, for a two-column legend (ncol=2, columns filled top to bottom). The left column holds
    the colour roles, then the final-position disk, goal and real path. The right column holds the shapes, then the arrival circle, start and straight line."""
    kinds = list(dict.fromkeys([agent_kind(agent), *map(agent_kind, others)]))
    logged = [k for k in KIND_COLOUR if k in {agent_kind(o) for o in others}]
    colours = [Line2D([], [], color=VLM_COLOUR, lw=2.5, label="VLM-controlled"),
               *[Line2D([], [], color=KIND_COLOUR[k], lw=2, label=f"logged {KIND_NAME[k]}") for k in logged]]
    shapes = [Line2D([], [], color="#777777", marker=KIND_MARKER[k], ls="", ms=7, label=KIND_NAME[k]) for k in kinds]
    shapes += [Line2D([], [], ls="", label=" ") for _ in range(len(colours) - len(shapes))]
    left = [Line2D([], [], color="#888888", marker="o", ls="", ms=11, alpha=0.5, mec="k", label="final pos with collision disk"),
            Line2D([], [], color=GOAL, marker="*", ls="", ms=11, label="goal"),
            Line2D([], [], color=REAL_PATH, ls="--", lw=2, label=REAL_PATH_LABEL[agent_kind(agent)])]
    right = [Line2D([], [], color=GOAL, marker="o", ls="", ms=11, mfc="none", mew=1.5, label=f"goal arrival thresh ({round(ARRIVAL_RADIUS_M[agent_kind(agent)], 2):g} m)"),
             Line2D([], [], color="k", marker=START_MARKER, ls="", ms=8, label="start"),
             Line2D([], [], color=STRAIGHT_LINE, ls=":", lw=2, label="straight line")]
    return colours + left + shapes + right


def render_result(site: dict, agent: str, goal_xy: np.ndarray, executed: np.ndarray, real: np.ndarray, others: dict[str, np.ndarray], title: str) -> bytes:
    """Figure of a whole episode of the planned agent on the BEV axes: executed path (time marks), the real path, the straight line and the
    logged agents' trails. Final positions are drawn at each agent's scoring radius (body_radius), the disk contact is scored with."""
    fig, ax, _ = bev_axes(False)
    draw_room(ax, site, agent, goal_xy)
    draw_reference(ax, goal_xy, executed[0], real)
    draw_others(ax, site, others)
    marker = KIND_MARKER[agent_kind(agent)]
    ax.plot(*executed.T, "-", marker=marker, color=VLM_COLOUR, lw=2.5, ms=4, alpha=0.8, zorder=4)
    ax.add_patch(Circle(executed[-1], body_radius(agent, site), fc=VLM_COLOUR, ec="k", alpha=0.45, zorder=5))
    ax.plot(*executed[-1], marker=marker, color=VLM_COLOUR, mec="k", ms=7, zorder=6)
    for k in range(0, len(executed), round(LABEL_EVERY_S / DT)):
        ax.text(*executed[k] + [0.05, -0.1], f"{k * DT:.0f}s", fontsize=LABEL_PX * 0.6, color=VLM_COLOUR, zorder=7)
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, title, ha="left", va="top", fontsize=LABEL_PX * 0.7, zorder=7)
    ax.legend(handles=trajectory_legend(agent, list(others)), ncol=2, loc="lower left", fontsize=LABEL_PX * 0.45, framealpha=0.85)
    return figure_png(fig)


def render_overlay(site: dict, agent: str, goal_xy: np.ndarray, samples: list[np.ndarray], real: np.ndarray, others: dict[str, np.ndarray], title: str) -> bytes:
    """Several executed paths of the planned agent (e.g. the samples of one condition) of one trip on the same axes, without time marks or
    legend (the caller adds trajectory_legend). Final positions are drawn at each agent's scoring radius (body_radius)."""
    fig, ax, _ = bev_axes(False)
    draw_room(ax, site, agent, goal_xy)
    draw_reference(ax, goal_xy, real[0], real)
    draw_others(ax, site, others)
    marker = KIND_MARKER[agent_kind(agent)]
    for path in samples:
        ax.plot(*path.T, "-", marker=marker, color=VLM_COLOUR, lw=2.5, ms=4, alpha=0.8, zorder=4)
        ax.add_patch(Circle(path[-1], body_radius(agent, site), fc=VLM_COLOUR, ec="k", alpha=0.35, zorder=5))
        ax.plot(*path[-1], marker=marker, color=VLM_COLOUR, mec="k", ms=7, zorder=6)
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, title, ha="left", va="top", fontsize=LABEL_PX * 0.7, zorder=7)
    return figure_png(fig)
