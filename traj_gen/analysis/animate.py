"""Time views of finished episodes, for when paths cross or double back and a static path plot hides who was where when.

- animation.gif: one frame per time step on the BEV axes. The planned agent and the other agents are
  body disks to scale with short fading trails. The planned agent's real path at the same time is a
  dashed ghost disk, and the plan it is currently following (rolling episodes replan) is a thin line.
  The frame title gives the time and the current clearance to the nearest other agent.
- animation.gif of a condition folder: the samples of one condition on one trip, drawn as in the overlay figures (purple VLM-controlled
  paths all drawn the same way, logged agents blue and orange, square robot and triangle person), one frame per time
  step. A sample that has ended holds its final position while the others continue.
- timeline.png: the whole episode on one map with colour = time (same colour scale for every agent),
  so a crossing point shows whether both were there at the same moment.

The single-episode views are written into the episode folder, next to result.png. They read only the episode's own files
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

from metrics.traj_gen_metrics import clearance_gap
from traj_gen.agents import PERSONAL_SPACE_RADIUS_M, agent_kind, body_radii, body_radius
from traj_gen.harness.episodes import load_trajectory
from traj_gen.outputs.baselines import real_path
from traj_gen.observation.bev import (EXTENT_M, KIND_COLOUR, KIND_MARKER, LABEL_PX, PERSON_LABEL, REAL_PATH, VLM_COLOUR, bev_axes, draw_others, draw_reference, draw_room,
                          figure_png, trajectory_legend)
from traj_gen.observation.scene import DT, person_name

TRAIL_S = 3.0      # fading trail length: about the history the model sees (2.8 s)
GIF_PX = 640       # frames are downscaled to this side: readable, and a 100-frame GIF stays a few MB
CMAP = "viridis"   # perceptually uniform, so equal colour steps are equal time steps


def load_episode(folder: Path, trips: dict, sites: dict) -> dict:
    """Everything a time view needs from an episode folder and its trip (trips and sites as returned by run.load_run).
    track is the planned agent's (T, 2) executed track, others {agent id: (T, 2)}, gap the (T,) clearance to the nearest other agent."""
    manifest = json.loads((folder / "manifest.json").read_text())
    trip = trips[manifest["trip"]]
    agent = manifest["agent"]
    t, positions = load_trajectory(folder)
    track = positions.pop(agent)
    site = sites[trip["site"]]
    plans = [json.loads(line) for line in (folder / "plans.jsonl").read_text().splitlines()]
    gap = clearance_gap(track, np.stack(list(positions.values()), axis=1), body_radius(agent, site), body_radii(positions, site))
    return {"manifest": manifest, "site": site, "agent": agent, "goal": np.asarray(trip["goal_xy"], float), "t": t, "track": track,
            "others": positions, "gap": gap, "real": real_path(trip), "plans": plans}


def _title(ep: dict) -> str:
    m = ep["manifest"]
    return " ".join(str(v) for v in (m["experiment"], m["framing"], m["thinking"], m["trip"], f"s{m['sample']}") if v is not None)


def _frame(ep: dict, k: int) -> bytes:
    fig, ax, _ = bev_axes(False)
    site, agent = ep["site"], ep["agent"]
    radius, colour = body_radius(agent, site), VLM_COLOUR  # the disks the metrics use
    draw_room(ax, site, agent, ep["goal"])
    ax.plot(*ep["real"].T, ls="--", color=REAL_PATH, lw=1.5, alpha=0.5, zorder=3)
    if k < len(ep["real"]):
        ax.add_patch(Circle(ep["real"][k], radius, fill=False, ec=REAL_PATH, ls="--", lw=2, zorder=4))
    t = ep["t"][k]
    current = [p for p in ep["plans"] if p["time_s"] <= t + 1e-6]
    if current and current[-1]["positions_m"]:
        q = np.searchsorted(ep["t"], current[-1]["time_s"] - 1e-6)
        plan = np.vstack([ep["track"][q], current[-1]["positions_m"]])
        ax.plot(*plan.T, ":", color=colour, lw=1.5, alpha=0.8, zorder=3)
    lo = max(0, k - round(TRAIL_S / DT))
    drawn = [(agent, ep["track"], radius, colour)] + [(other, track, body_radius(other, site), KIND_COLOUR[agent_kind(other)]) for other, track in ep["others"].items()]
    for agent_id, track, r, agent_colour in drawn:
        kind = agent_kind(agent_id)
        seg = track[lo: k + 1]
        for j in range(1, len(seg)):  # older segments fainter
            ax.plot(*seg[j - 1: j + 1].T, "-", color=agent_colour, lw=3, alpha=0.15 + 0.85 * j / len(seg), zorder=4)
        ax.add_patch(Circle(track[k], r, fc=agent_colour, ec="k", alpha=0.55, zorder=5))
        if agent_id != agent and kind in PERSONAL_SPACE_RADIUS_M:
            ax.add_patch(Circle(track[k], PERSONAL_SPACE_RADIUS_M[kind], fill=False, ec=agent_colour, ls=":", lw=1.5, zorder=4))
    for other, track in ep["others"].items():
        if agent_kind(other) == "human":
            ax.text(*track[k], person_name(other), ha="center", va="center", fontsize=LABEL_PX * 0.6, zorder=6)
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"{_title(ep)}\nt = {t:.1f} s   clearance {ep['gap'][k]:+.2f} m\ndashed ring: real {agent_kind(agent)} now, dotted line: current plan, dotted ring: personal space",
            ha="left", va="top", fontsize=LABEL_PX * 0.6, zorder=7)
    return figure_png(fig)


def _write_gif(frames: list[bytes], out: Path, speed: float) -> Path:
    """Write PNG frames as a GIF at `speed` times real time, holding the last frame."""
    images = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    for image in images:
        image.thumbnail((GIF_PX, GIF_PX), Image.Resampling.LANCZOS)
    ms = round(1000 * DT / speed)
    images[0].save(out, save_all=True, append_images=images[1:], duration=[ms] * (len(images) - 1) + [2000], loop=0)
    return out


def animate(folder: Path, ep: dict, speed: float) -> Path:
    """Write <folder>/animation.gif from load_episode(folder), played `speed` times faster than real time, holding the last frame."""
    return _write_gif([_frame(ep, k) for k in range(len(ep["t"]))], folder / "animation.gif", speed)


def animate_samples(folder: Path, eps: list[dict], speed: float) -> Path:
    """Write <folder>/animation.gif from the load_episode of every sample of one condition on one trip: one frame per time step, until the
    longest sample ends. The logged agents are drawn once, from the longest sample's log (the same in every sample)."""
    longest = max(eps, key=lambda ep: len(ep["t"]))
    site, agent, kind = longest["site"], longest["agent"], agent_kind(longest["agent"])
    others = longest["others"]
    frames = []
    for k in range(len(longest["t"])):
        fig, ax, _ = bev_axes(False)
        draw_room(ax, site, agent, longest["goal"])
        draw_reference(ax, longest["goal"], longest["real"][0], longest["real"])
        real = longest["real"]  # the real agent now, as a dashed ring at body size (held at its last logged position)
        ax.add_patch(Circle(real[min(k, len(real) - 1)], body_radius(agent, site), fill=False, ec=REAL_PATH, ls="--", lw=2, zorder=6))
        draw_others(ax, site, {other: track[: k + 1] for other, track in others.items()})  # the logged agents up to this step
        for ep in eps:
            head = min(k, len(ep["t"]) - 1)  # a finished sample holds its final position
            path = ep["track"][: head + 1]
            ax.plot(*path.T, marker=KIND_MARKER[kind], color=VLM_COLOUR, lw=2.5, ms=4, alpha=0.8, zorder=4)
            ax.add_patch(Circle(path[-1], body_radius(agent, site), fc=VLM_COLOUR, ec="k", alpha=0.35, zorder=5))
            ax.plot(*path[-1], marker=KIND_MARKER[kind], color=VLM_COLOUR, mec="k", ms=7, zorder=6)
        m = longest["manifest"]
        ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"{m['experiment']} {m['framing']} {m['thinking']}\n{m['trip']}\nt = {k * DT:.1f} s",
                ha="left", va="top", fontsize=LABEL_PX * 0.6, zorder=7)
        ax.legend(handles=trajectory_legend(agent, list(others)), ncol=2, loc="lower left", fontsize=LABEL_PX * 0.6, framealpha=0.85)
        frames.append(figure_png(fig))
    return _write_gif(frames, folder / "animation.gif", speed)


def timeline(folder: Path, ep: dict) -> Path:
    """Write <folder>/timeline.png from load_episode(folder): the paths of the planned agent and the others coloured by time on one scale."""
    fig, ax, _ = bev_axes(False)
    draw_room(ax, ep["site"], ep["agent"], ep["goal"])
    ax.plot(*ep["real"].T, ls="--", color=REAL_PATH, lw=1.5, alpha=0.6, zorder=3)
    cmap, t = colormaps[CMAP], ep["t"]
    norm = (t - t[0]) / max(t[-1] - t[0], DT)
    for track, lw, marker in [(ep["track"], 5, "o")] + [(other, 2.5, "^") for other in ep["others"].values()]:
        ax.add_collection(LineCollection(np.stack([track[:-1], track[1:]], axis=1), colors=cmap(norm[:-1]), lw=lw, zorder=4))
        ax.scatter(*track.T, c=cmap(norm), marker=marker, s=30 if marker == "o" else 22, ec="k", lw=0.3, zorder=5)
    for other, track in ep["others"].items():
        if agent_kind(other) == "human":
            ax.text(*track[0] + [0, 0.12], person_name(other), ha="center", fontsize=LABEL_PX * 0.7, color=PERSON_LABEL, zorder=7)
    ax.plot(*ep["track"][0], marker="s", color="k", ms=8, zorder=6)
    cax = fig.add_axes([0.665, 0.924, 0.245, 0.012])  # x 0.6..1.7 m, y 1.86..1.92 m: the strip above the room's top wall, right of the title
    cbar = fig.colorbar(ScalarMappable(cmap=cmap), cax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_ticks([0, 1], labels=[f"{t[0]:.1f} s", f"{t[-1]:.1f} s"])
    cbar.ax.tick_params(labelsize=LABEL_PX * 0.55)
    ax.text(-EXTENT_M / 2 + 0.05, EXTENT_M / 2 - 0.05, f"{_title(ep)}\ncolour = time; circles: planned agent, triangles: other agents; dashed: real path", ha="left", va="top", fontsize=LABEL_PX * 0.6, zorder=7)
    out = folder / "timeline.png"
    out.write_bytes(figure_png(fig))
    return out
