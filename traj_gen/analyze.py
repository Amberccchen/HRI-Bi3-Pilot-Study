"""Analysis of a trajectory generation run: tables, comparisons and figures (design: PLAN.md).

Reads every finished episode of the test trips under a run folder (experiments in the registry
only, so smoke/ and _superseded*/ are ignored; a run still in progress is fine, missing
conditions are simply absent) and writes <run>/analysis/:

  episodes.csv          one row per episode: condition, trip, status, metrics, queries, tokens, latency
  summary.csv           per condition (experiment x framing x thinking, plus the baselines) and site
                        (all / laas / um): status counts and episode-level metric means
  paired_baselines.csv  every condition minus replay and minus straight on the same trips
  comparisons.csv       the PLAN comparisons (neutral vs social, nothink vs think, ...) as differences
  figures/              conditions_<thinking>.png, speed_violation.png, trip_<id>.png (overlays)

Scoring rules. Path metrics (clearance, time to goal, ...) use status-ok episodes only; goal_reached
counts every episode, so an invalid or failed one counts as not reached. Paired differences and
comparisons work on per-trip means (over the samples of the condition), so a trip is one
observation and the intervals are percentile bootstrap intervals over trips (few trips: read them
as rough). time_to_goal only exists for episodes that reached the goal.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D

from bev import render_overlay
from bi3_derived import load_site_frames, load_trips
from episodes import ConversationEpisode, RollingEpisode, SingleShotEpisode
from experiments import EXPERIMENTS
from planners import THINKING
from prompting import FRAMINGS

BASELINES = ("replay", "straight")
KEY = ["experiment", "framing", "thinking"]
N_BOOT = 2000   # bootstrap resamples of the trips: the interval's edges are stable to about 0.01 SD
SEED = 20261022  # fixed so the intervals are reproducible; the run date, no other meaning
INFO = ["trip", "site", "controller", "stratum", "sample", "status", "queries", "samples", "retries", "invalid_replies",
        "forced_answers", "prompt_tokens", "completion_tokens", "query_latency_s", "wall_s"]
SKIP = {"min_distance_per_person", "min_distance_index", "min_distance_person"}  # per-person detail, not an episode-level number
RATES = ["goal_reached", "footprint_overlap", "out_of_room", "station_crossing"]
MEANS = ["min_distance", "time_to_goal", "path_efficiency", "velocity_mean", "acceleration_mean", "speed_violation_rate", "raw_speed_max", "final_distance_to_goal", "station_distance_min"]
PAIRED = ["goal_reached", "footprint_overlap", "station_crossing", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
COMPARED = ["goal_reached", "footprint_overlap", "station_crossing", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
OVERLAY = ["ss_wp_bev", "rh_wp_bev", "rh_velseq_bev"]  # default experiments in the per-trip overlays: one per horizon/output family
FAMILY = {SingleShotEpisode: "single shot", RollingEpisode: "rolling", ConversationEpisode: "conversation"}
# validated categorical slots 1-3 (dataviz reference palette, light surface); baselines use neutral grey and violet
FAMILY_COLOUR = {"single shot": "#2a78d6", "rolling": "#eb6834", "conversation": "#1baf7a"}
BASELINE_STYLE = {"replay": ("#52514e", "--"), "straight": ("#6250d6", ":")}
INK, GRID = "#0b0b0b", "#e4e3df"


def collect(run: Path) -> pd.DataFrame:
    """One row per finished episode of the test trips."""
    strata = {c["id"]: c["stratum"] for c in json.loads((run / "test_trips.json").read_text())}
    rows = []
    for exp in EXPERIMENTS:
        for path in sorted((run / exp).glob("**/manifest.json")):
            m = json.loads(path.read_text())
            if m["trip"] in strata:
                metrics = {k: v for k, v in (m["metrics"] or {}).items() if k not in SKIP}
                rows.append({"experiment": exp, "framing": m["framing"] or "-", "thinking": m["thinking"] or "-", "folder": str(path.parent.relative_to(run)),
                             **{k: m.get(k) for k in INFO if k != "stratum"}, "stratum": strata[m["trip"]], **metrics})
    return pd.DataFrame(rows)


def scored(episodes: pd.DataFrame) -> pd.DataFrame:
    """Episodes as scored: metrics of episodes that are not ok are dropped, and they count as goal not reached."""
    df = episodes.copy()
    metric_cols = [c for c in df.columns if c in RATES + MEANS + ["clearance_min"]]
    df.loc[df.status != "ok", metric_cols] = np.nan
    df[RATES] = df[RATES].astype(float)
    df["goal_reached"] = df.goal_reached.fillna(0.0)
    return df


def ci(values) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval over trips (NaN trips dropped)."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    means = v[np.random.default_rng(SEED).integers(0, v.size, (N_BOOT, v.size))].mean(axis=1)
    return float(v.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    def one(g: pd.DataFrame) -> pd.Series:
        calls = g.samples.sum()
        return pd.Series({
            "n": len(g), **{s: (g.status == s).sum() for s in ("ok", "invalid", "failed", "error")},
            **{c: g[c].mean() for c in RATES}, "clearance_min_mean": g.clearance_min.mean(), "clearance_min_median": g.clearance_min.median(),
            **{c: g[c].mean() for c in MEANS}, "queries_per_episode": g.queries.mean(), "retries": g.retries.sum(), "forced_answers": g.forced_answers.sum(),
            "tokens_per_query": g.completion_tokens.sum() / calls if calls else np.nan, "latency_s_per_query": g.query_latency_s.sum() / calls if calls else np.nan})

    parts = [df.groupby(KEY, sort=False).apply(one, include_groups=False).assign(site="all")]
    parts += [d.groupby(KEY, sort=False).apply(one, include_groups=False).assign(site=s) for s, d in df.groupby("site")]
    return pd.concat(parts).reset_index().sort_values(["site", "experiment", "framing", "thinking"], kind="stable")


def per_trip_means(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """condition -> table (trip x metric) of per-trip means over the samples; site as a column."""
    means = df.groupby(KEY + ["trip"])[PAIRED].mean()
    sites = df.groupby("trip").site.first()
    return {k: g.droplevel(KEY).assign(site=sites) for k, g in means.groupby(level=KEY)}


def paired_baselines(by_cond: dict) -> pd.DataFrame:
    rows = []
    for cond, table in by_cond.items():
        if cond[0] in BASELINES:
            continue
        for base in BASELINES:
            if (base, "-", "-") not in by_cond:
                continue
            ref = by_cond[(base, "-", "-")]
            for site in ("all", "laas", "um"):
                for metric in PAIRED:
                    d = (table[metric] - ref[metric])[table.site.index if site == "all" else table.index[table.site == site]]
                    rows.append({"experiment": cond[0], "framing": cond[1], "thinking": cond[2], "baseline": base, "site": site, "metric": metric,
                                 "n_trips": int(d.notna().sum()), **dict(zip(["diff", "ci_lo", "ci_hi"], ci(d)))})
    return pd.DataFrame(rows)


def comparison_pairs() -> list[tuple[str, str, tuple, tuple]]:
    """(comparison, scope, A condition, B condition); the difference reported is A - B. Scope = the factor the pairs are split by."""
    models = [e for e in EXPERIMENTS if e not in BASELINES]
    # the conversation partner has the same images and output as rh_wp_bev_cam / rh_velseq_bev
    by_experiment = {
        "bev - text input (waypoints, single shot)": [("ss_wp_bev", "ss_wp_text")],
        "pixel - metric output (bev)": [("ss_px_bev", "ss_wp_bev"), ("rh_px_bev", "rh_wp_bev")],
        "cam + bev - bev only (waypoints)": [("ss_wp_bev_cam", "ss_wp_bev"), ("rh_wp_bev_cam", "rh_wp_bev")],
        "rolling - single shot (bev)": [("rh_wp_bev", "ss_wp_bev"), ("rh_px_bev", "ss_px_bev")],
        "conversation - fresh query each time": [("rh_wp_bev_conv", "rh_wp_bev_cam"), ("rh_velseq_bev_conv", "rh_velseq_bev")],
        "velocity sequence - waypoints (rolling)": [("rh_velseq_bev", "rh_wp_bev")],
        "single velocity step - waypoints (rolling)": [("rh_vel1_bev", "rh_wp_bev")],
    }
    pairs = []
    for e in models:
        for th in THINKING:
            pairs.append(("social - neutral prompt", th, (e, "social", th), (e, "neutral", th)))
        for fr in FRAMINGS:
            pairs.append(("think - nothink", fr, (e, fr, "think"), (e, fr, "nothink")))
    for name, experiments in by_experiment.items():
        for a, b in experiments:
            for fr in FRAMINGS:
                for th in THINKING:
                    pairs.append((name, th, (a, fr, th), (b, fr, th)))
    return pairs


def comparisons(by_cond: dict) -> pd.DataFrame:
    rows = []
    pairs = [p for p in comparison_pairs() if p[2] in by_cond and p[3] in by_cond]
    for name in dict.fromkeys(p[0] for p in pairs):
        for scope in ["all"] + sorted({p[1] for p in pairs if p[0] == name}):
            chosen = [p for p in pairs if p[0] == name and scope in ("all", p[1])]
            for metric in COMPARED:
                d = pd.concat([by_cond[a][metric] - by_cond[b][metric] for _, _, a, b in chosen], axis=1).mean(axis=1)  # per trip, over the pairs
                rows.append({"comparison": name, "scope": scope, "metric": metric, "n_pairs": len(chosen), "n_trips": int(d.notna().sum()), **dict(zip(["diff", "ci_lo", "ci_hi"], ci(d)))})
    return pd.DataFrame(rows)


def style(ax, xlabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.tick_params(labelsize=9, colors=INK, length=0)
    ax.grid(axis="x", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def dot_plot(path: Path, panels: dict[str, str], rows: list[str], offsets: dict, by_cond: dict, key, marker, title: str, legend: list) -> None:
    """Rows = experiments, one panel per metric; one dot with a bootstrap interval per (offset label); baselines as vertical lines."""
    fig = Figure(figsize=(max(3.4 * len(panels) + 1.6, 7.5), 0.36 * len(rows) * len(offsets) + 2.6), dpi=150)
    axes = fig.subplots(1, len(panels), sharey=True, squeeze=False)[0]
    for ax, (metric, label) in zip(axes, panels.items()):
        for r, exp in enumerate(rows):
            for off, o in zip(np.linspace(-0.2, 0.2, len(offsets)) if len(offsets) > 1 else [0.0], offsets):
                cond = key(exp, o)
                if cond not in by_cond:
                    continue
                m, lo, hi = ci(by_cond[cond][metric])
                colour = FAMILY_COLOUR[FAMILY[EXPERIMENTS[exp].episode]]
                filled = marker(o)
                ax.plot([lo, hi], [r + off] * 2, color=colour, lw=1.5, zorder=3)
                ax.plot(m, r + off, "o", ms=7, mfc=colour if filled else "white", mec=colour, mew=1.5, zorder=4)
        for base, (colour, ls) in BASELINE_STYLE.items():
            if (base, "-", "-") in by_cond:
                ax.axvline(by_cond[(base, "-", "-")][metric].mean(), color=colour, ls=ls, lw=1.5, zorder=2)
        ax.set_yticks(range(len(rows)), rows)
        style(ax, label)
    axes[0].invert_yaxis()  # shared y: once, first experiment on top
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.legend(handles=legend, loc="lower center", ncol=min(len(legend), 4), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(path)


def present(by_cond: dict, **fixed) -> list[str]:
    """Experiments (registry order) that have a condition with the given framing/thinking."""
    return [e for e in EXPERIMENTS if e not in BASELINES and any(c[0] == e and all(c[KEY.index(k)] == v for k, v in fixed.items()) for c in by_cond)]


def legend_handles(baselines: bool, markers: list[tuple[str, bool]]) -> list:
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=f) for f, c in FAMILY_COLOUR.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=7, color=INK, mfc=INK if filled else "white", label=label) for label, filled in markers]
    if baselines:
        handles += [Line2D([], [], color=c, ls=ls, lw=1.5, label=f"{b} (mean)") for b, (c, ls) in BASELINE_STYLE.items()]
    return handles


def condition_figures(by_cond: dict, fig_dir: Path) -> None:
    panels = {"goal_reached": "goal reached (rate)", "clearance_min": "min clearance (m, higher = more room)", "time_to_goal": "time to goal (s)", "path_efficiency": "path efficiency (1 = straight)"}
    baselines = any((b, "-", "-") in by_cond for b in BASELINES)
    for th in THINKING:
        rows = present(by_cond, thinking=th)
        if rows:
            dot_plot(fig_dir / f"conditions_{th}.png", panels, rows, {"neutral": 0, "social": 1}, by_cond, lambda e, f: (e, f, th), lambda f: f == "social",
                     f"Thinking {'off' if th == 'nothink' else 'on'}: mean over trips with 95% bootstrap interval", legend_handles(baselines, [("neutral prompt", False), ("social prompt", True)]))
    rows = [e for e in present(by_cond) if any(by_cond[c]["speed_violation_rate"].notna().any() for c in by_cond if c[0] == e)]
    if rows:
        dot_plot(fig_dir / "speed_violation.png", {"speed_violation_rate": "share of plan steps above 0.3 m/s"}, rows, {"nothink": 0, "think": 1}, by_cond,
                 lambda e, th: (e, "neutral", th), lambda th: th == "think", "Speed-cap violations before limiting, neutral prompt", legend_handles(False, [("thinking off", False), ("thinking on", True)]))


def overlay_figures(df: pd.DataFrame, run: Path, experiments: list[str], fig_dir: Path) -> None:
    trips = {t["id"]: t for t in load_trips()}
    sites = load_site_frames()
    legend = [Line2D([], [], color="#555555", ls="--", label="real path"), Line2D([], [], color="#2ca02c", ls=":", label="straight line"),
              Line2D([], [], color="#e66101", marker="o", ms=4, label="people"), Line2D([], [], color="#1f77b4", marker="o", ms=4, label="model samples (disk = final robot position)")]
    for trip_id, d in df[df.experiment.isin(experiments)].groupby("trip"):
        trip = trips[trip_id]
        a, goal = trip["start_index"], np.asarray(trip["robot_goal_xy"], float)
        real = np.asarray(trip["robot"], float)[a: trip["end_index"] + 1, :2]
        shown = [e for e in experiments if e in set(d.experiment)]
        cols = [c for c in sorted({(r.thinking, r.framing) for r in d.itertuples()}, key=lambda c: (THINKING.index(c[0]), FRAMINGS.index(c[1])))]
        fig = Figure(figsize=(4.2 * len(cols), 4.2 * len(shown) + 0.9), dpi=110)
        axes = fig.subplots(len(shown), len(cols), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, exp in enumerate(shown):
            for j, (th, fr) in enumerate(cols):
                g = d[(d.experiment == exp) & (d.thinking == th) & (d.framing == fr)]
                if g.empty:
                    continue
                paths = [np.loadtxt(run / f / "trajectory.csv", delimiter=",", skiprows=1, ndmin=2)[:, 1:3] for f in g.folder]
                end = max(len(real), *map(len, paths))
                humans = [np.asarray(h, float)[a: a + end, :2] for h in trip["humans"]]
                title = f"{exp} {fr} {th}: {int((g.status == 'ok').sum())}/{len(g)} ok"
                png = render_overlay(sites[trip["site"]], goal, paths, real, humans, title)
                axes[i, j].imshow(imread(io.BytesIO(png), format="png"))
        fig.suptitle(f"{trip_id} ({trip['controller']}, {d.stratum.iloc[0]})", fontsize=11, x=0.01, ha="left")
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(fig_dir / f"trip_{trip_id}.png")


def analyze(run: Path, overlay: list[str]) -> None:
    out = run / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    episodes = collect(run)
    episodes.to_csv(out / "episodes.csv", index=False)
    df = scored(episodes)
    summary(df).to_csv(out / "summary.csv", index=False)
    by_cond = per_trip_means(df)
    paired_baselines(by_cond).to_csv(out / "paired_baselines.csv", index=False)
    comparisons(by_cond).to_csv(out / "comparisons.csv", index=False)
    condition_figures(by_cond, out / "figures")
    overlay_figures(df, run, overlay, out / "figures")
    print(f"{len(episodes)} episodes, {len(by_cond)} conditions -> {out}")
