# coding-standard: noqa (write-up figures of the v1 run, not part of the pipeline)
"""Figures for the write-up of the robot episodes of a run, with one list of rows per panel. Each panel answers one question, fixes the
other factors and shows only the compared conditions.

The panels are stacked as the rows of one figure and share one layout: a row per condition, a column per metric. Dots are the mean
over trips with a 95% bootstrap interval (hollow = thinking off, filled = on). The solid line is the straight line and the dashed
line the real controllers. All values are absolute.
- Panel A, perception ladder: single-shot waypoints, social prompt. Each row adds input to the one above.
- Panel B, output type: the map as input, social prompt, rolling replanning. The rows are waypoints, pixels and velocities.
- Panel C, replanning horizon: the map as input, social prompt. The rows are single shot, rolling, every step (velocity only) and rolling in one conversation.
- Panel D, prompt: the four framings, pooled over PROMPT_EXPERIMENTS (ss_wp_bev, rh_wp_bev, rh_vel1_bev).
Goal reached and speed are not drawn.

Writes into <run>/report: figure_ladder_output_horizon_prompt_contact_time.png and ..._contact_events.png (each also as PDF), one
per way of counting contact, and personal_space_sensitivity.csv, the robot's personal-space overlap at other person radii.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from metrics.traj_gen_metrics import clearance_metrics
from traj_gen.agents import PERSONAL_SPACE_RADIUS_M, agent_kind, body_radii, body_radius
from traj_gen.analysis.analyze import INK, ci, per_trip_means, scored, style
from traj_gen.harness.episodes import ANALYSIS_DIR, REPORT_DIR, TEST_TRIPS, load_trajectory
from traj_gen.harness.experiments import BASELINES, LEGACY_IDS
from traj_gen.run import load_run

STRAIGHT, REPLAY = "straight", "replay"  # experiment ids of the baselines
MODEL_COLOUR = "#2a78d6"   # validated categorical slot 1: one hue, since the panels compare conditions of one model
REAL_COLOUR = "#52514e"    # secondary ink grey: the reference, not a series
THINKING = [("nothink", "thinking off", False), ("think", "thinking on", True)]  # (key, label, filled)
# (metric, axis label, direction hint, scale). All are absolute values, since the straight line's own values vary by trip.
CONTACT = {"time": ("contact_time", "contact\n(% of episode time)", "← less contact", 100), "events": ("contact_events", "contacts\n(per episode)", "← fewer contacts", 1)}
OTHER_METRICS = [("personal_space_overlap", "personal-space overlap\n(% of episode time)", "← less intrusion", 100), ("path_efficiency", "path efficiency\n(1 = straight)", "straighter →", 1)]
DOT_OFFSET = 0.17   # half the vertical gap between the thinking-off and thinking-on dots of a row (rows are 1 apart)
ROW_IN = 0.34       # figure height per row, inches
PANEL_IN = 0.35     # figure height per panel for its title and gap, inches
FIXED_IN = 1.05     # figure height for the axis labels, direction hints and legend, inches

LADDER = [("text only", ["ss_wp_text"], "social"), ("+ map", ["ss_wp_bev"], "social"), ("+ camera photo", ["ss_wp_bev_cam"], "social"),
          ("+ predicted pos", ["ss_wp_bev_constvel_future"], "social")]
# output type: rolling replanning every 2 s is the one horizon with waypoints, pixels and velocities
OUTPUT_TYPES = [("waypoints", ["rh_wp_bev"], "social"), ("pixels", ["rh_px_bev"], "social"), ("* velocities", ["rh_velseq_bev"], "social")]
# Row marker, put before the label and explained in the caption: * velocity outputs have no single-shot experiment (only rolling every 2 s, every step
# and in one conversation), and every step exists only for velocities.
# horizon: waypoints, except the per-step row, which only exists for velocities
HORIZONS = [("single shot", ["ss_wp_bev"], "social"), ("rolling", ["rh_wp_bev"], "social"), ("* one step (vel)", ["rh_vel1_bev"], "social"),
            ("rolling, convo", ["rh_wp_bev_cam_conv"], "social")]
PROMPT_EXPERIMENTS = ["ss_wp_bev", "rh_wp_bev", "rh_vel1_bev"]  # the experiments pooled in the prompt panel, each needs all four framings
PROMPTS = [(label, PROMPT_EXPERIMENTS, framing) for label, framing in
           [("neutral", "neutral"), ("social", "social"), ("social + size", "social_body"), ("cautious", "cautious")]]


def trips(by_cond: dict, experiments: list[str], framing: str, thinking: str, metric: str) -> pd.Series:
    """Per-trip value of a metric: the mean over the experiments' per-trip means."""
    return pd.concat([by_cond[(e, framing, thinking)][metric] for e in experiments], axis=1).mean(axis=1)


def draw(axes, rows: list[tuple[str, list[str], str]], by_cond: dict, metrics: list, hints: bool) -> None:
    """One panel into a row of axes, one per metric."""
    straight, real = by_cond[(STRAIGHT, "-", "-")], by_cond[(REPLAY, "-", "-")]
    for ax, (metric, label, direction, scale) in zip(axes, metrics):
        ref = {"straight": scale * straight[metric].mean(), "real": scale * real[metric].mean()}
        ax.axvline(ref["straight"], color=INK, lw=1, zorder=2)
        ax.axvline(ref["real"], color=REAL_COLOUR, ls="--", lw=1.5, zorder=2)
        ax.plot(list(ref.values()), [0, 0], alpha=0)  # keeps the reference lines inside the axis limits
        for r, (_, experiments, framing) in enumerate(rows):
            for (th, _, filled), off in zip(THINKING, (-DOT_OFFSET, DOT_OFFSET)):
                m, lo, hi = ci(scale * trips(by_cond, experiments, framing, th, metric))
                ax.plot([lo, hi], [r + off] * 2, color=MODEL_COLOUR, lw=1.5, zorder=3)
                ax.plot(m, r + off, "o", ms=6, mfc=MODEL_COLOUR if filled else "white", mec=MODEL_COLOUR, mew=1.5, zorder=4)
        ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
        style(ax, label)
        ax.xaxis.label.set_fontsize(8)
        ax.tick_params(labelsize=8)
        if hints:
            ax.annotate(direction, (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom", fontsize=8, color=REAL_COLOUR)
    axes[0].invert_yaxis()


def legend_handles() -> list:
    handles = [Line2D([], [], marker="o", ls="", ms=6, color=MODEL_COLOUR, mfc=MODEL_COLOUR if f else "white", mew=1.5, label=l) for _, l, f in THINKING]
    handles += [Line2D([], [], color=INK, lw=1, label="straight line"), Line2D([], [], color=REAL_COLOUR, ls="--", lw=1.5, label="real controllers")]
    return handles


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))


def stacked(by_cond: dict, panels: list[tuple[str, list]], metrics: list, path: Path) -> None:
    """The panels as the rows of one figure. A metric's x axis is shared down the column, so the panels read on one scale."""
    heights = [len(rows) for _, rows in panels]
    fig = Figure(figsize=(7.2, FIXED_IN + PANEL_IN * len(panels) + ROW_IN * sum(heights)), dpi=200, layout="constrained")
    grid = fig.subplots(len(panels), len(metrics), sharex="col", sharey="row", gridspec_kw={"height_ratios": heights})
    for i, (axes, (title, rows)) in enumerate(zip(grid, panels)):
        draw(axes, rows, by_cond, metrics, hints=i == 0)
        for ax in axes:
            ax.tick_params(labelbottom=True)  # the scale is shared, but each panel keeps its own tick labels
            ax.set_xlabel(ax.get_xlabel() if i == len(panels) - 1 else "")  # axis labels once, under the last row
        axes[0].set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold", pad=16 if i == 0 else 6)  # the first row's title clears the direction hints
    handles = legend_handles()
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    save(fig, path)


SENSITIVITY_RADII = (0.15, 0.20, 0.25)  # person radius for personal-space overlap: 0.15 is contact, 0.25 the one in agents.py, 0.20 the in-between


def sensitivity(run: Path, sites: dict) -> None:
    """Personal-space overlap (% of episode time, mean over trips) of the robot at other person radii, for the baselines and the pooled model episodes."""
    episodes = pd.read_csv(run / ANALYSIS_DIR / "episodes.csv")
    episodes = episodes[(episodes.status == "ok") & (episodes.kind == "robot")]
    groups = {"straight line": episodes.experiment == STRAIGHT, "real controllers": episodes.experiment == REPLAY,
              **{f"Qwen, {label}": ~episodes.experiment.isin(BASELINES) & (episodes.thinking == th) for th, label, _ in THINKING}}
    rows = []
    for name, mask in groups.items():
        for radius in SENSITIVITY_RADII:
            overlap = []
            for _, ep in episodes[mask].iterrows():
                _, positions = load_trajectory(run / ep.folder)
                track, site = positions.pop(ep.agent), sites[ep.site]
                others = np.stack(list(positions.values()), axis=1)
                with_space = [k for k, o in enumerate(positions) if agent_kind(o) in PERSONAL_SPACE_RADIUS_M]  # the swept radius replaces their personal-space radius
                overlap.append(clearance_metrics(track, others, body_radius(ep.agent, site), body_radii(positions, site),
                                                 others[:, with_space], [radius] * len(with_space))["personal_space_overlap"])
            per_trip = pd.Series(overlap, index=episodes[mask].trip.values).groupby(level=0).mean()
            rows.append({"group": name, "person_radius_m": radius, "overlap_pct": 100 * per_trip.mean()})
    pd.DataFrame(rows).pivot(index="group", columns="person_radius_m", values="overlap_pct").round(0).to_csv(run / REPORT_DIR / "personal_space_sensitivity.csv")


def report(run: Path) -> None:
    _, _, _, sites = load_run(run, run / TEST_TRIPS)
    episodes = pd.read_csv(run / ANALYSIS_DIR / "episodes.csv")
    episodes["experiment"] = episodes.experiment.replace(LEGACY_IDS)  # episodes.csv written before the derived ids
    by_cond = per_trip_means(scored(episodes[episodes.kind == "robot"]))  # the write-up covers robot generation
    panels = [("(a) Perception ladder", LADDER), ("(b) Output type", OUTPUT_TYPES), ("(c) Replanning horizon", HORIZONS), ("(d) Prompt", PROMPTS)]
    for contact, first in CONTACT.items():  # the two ways to count contact, as separate figures to compare
        stacked(by_cond, [(title, rows) for title, rows in panels], [first, *OTHER_METRICS], run / REPORT_DIR / f"figure_ladder_output_horizon_prompt_contact_{contact}.png")
    sensitivity(run, sites)
    print(f"-> {run / REPORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the write-up figures of a run into <run>/report (needs <run>/analysis/episodes.csv from the analyze command).")
    parser.add_argument("--run", type=Path, required=True, help="run folder")
    report(parser.parse_args().run)


if __name__ == "__main__":
    main()
