"""Analysis of a trajectory generation run: tables, comparisons and figures.

Reads every finished episode of the test trips under a run folder. Only folders of experiments in
the registry are read, so other folders such as smoke/ are ignored. A run still in progress is fine,
and missing conditions are simply absent. Writes into <run>/analysis/:

  episodes.csv          one row per episode: condition, trip, planned agent and its kind, status, metrics, queries, tokens, latency
  summary.csv           per planned-agent kind, condition (experiment x framing x thinking, plus the baselines) and
                        site (all / laas / um): status counts and episode-level rates and means, grouped by
                        metric class (invalid, contact, efficiency)
  paired_baselines.csv  every condition minus replay and minus straight on the same trips, per kind
  comparisons.csv       the condition comparisons (social - neutral prompt, think - nothink, bev - text input, ...) as differences, per kind
  figures/              conditions_<kind>_<thinking>.png, speed_violation_<kind>.png, trip_<id>.png (overlays)

Robot and person episodes are never pooled: every table and figure is per planned-agent kind ("robot" or "human").

Scoring rules. Path metrics (clearance, time to goal, ...) use status-ok episodes only. goal_reached
counts every episode, so one without a usable path counts as not reached. An episode is invalid when
its reply is unusable, the server failed (status error), or its plan exceeds the speed cap in any step,
does not reach the goal, leaves the room or crosses a station. The invalid rate and the rate of each of
these parts are shares of all episodes of the condition. An episode without a path (invalid_reply or
error) counts only in its status part. The speed-cap part does not apply to the baselines, which have no
model reply, and is NaN for them. Paired differences and comparisons work on per-trip means (over
the samples of the condition), so a trip is one observation and the intervals are percentile
bootstrap intervals over trips (few trips: read them as rough). time_to_goal only exists for episodes that reached the goal.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D

from metrics.traj_gen_metrics import METRIC_CLASS
from preprocessing.bi3_derived import SITES
from traj_gen.agents import ARRIVAL_RADIUS_M, SPEED_CAP_M_S, agent_kind, prompt_radius
from traj_gen.bev import GOAL, KIND_COLOUR, KIND_LEGEND, REAL_PATH, render_overlay
from traj_gen.episodes import ANALYSIS_DIR, TEST_TRIPS, ConversationEpisode, RollingEpisode, SingleShotEpisode, load_trajectory, real_path
from traj_gen.experiments import BASELINES, EXPERIMENTS
from traj_gen.planners import THINKING
from traj_gen.prompting import FRAMINGS

COND = ["experiment", "framing", "thinking"]  # a condition within one planned-agent kind
KEY = ["kind", *COND]                         # a condition across kinds: robot and person episodes are never pooled
N_BOOT = 2000   # bootstrap resamples of the trips: the interval's edges are stable to about 0.01 SD
SEED = 20261022  # fixed so the intervals are reproducible. The value is a date with no other meaning
INFO = ["trip", "site", "controller", "agent", "stratum", "sample", "status", "queries", "samples", "retries", "invalid_replies",
        "forced_answers", "prompt_tokens", "completion_tokens", "query_latency_s", "wall_s"]
STATUSES = ("ok", "invalid_reply", "error")
SKIP = {"min_distance_per_other", "min_distance_index", "min_distance_other"}  # per-agent detail, not an episode-level number
METRICS = [m for m in METRIC_CLASS if m not in SKIP]  # episode-level metric columns, present (NaN when absent) in every episode table
RATES = ["goal_reached", "contact_time", "personal_space_overlap", "out_of_room", "station_crossing"]
# Parts of the invalid rate, 0/1 per episode over all episodes (NaN where the part does not apply).
INVALID_PARTS = ["invalid_unusable_reply", "invalid_error", "invalid_speed_cap", "invalid_goal", "invalid_room", "invalid_station"]
MEANS = ["contact_events", "min_distance", "time_to_goal", "path_efficiency", "velocity_mean", "acceleration_mean", "speed_violation_rate", "raw_speed_max", "final_distance_to_goal", "station_distance_min"]
CLASS_ORDER = ("invalid", "contact", "efficiency")
PAIRED = ["goal_reached", "contact_time", "contact_events", "personal_space_overlap", "station_crossing", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
COMPARED = ["goal_reached", "contact_time", "contact_events", "personal_space_overlap", "station_crossing", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
OVERLAY = ["ss_wp_bev", "rh_wp_bev", "rh_velseq_bev"]  # default experiments in the per-trip overlays: one per horizon/output family
OVERLAY_PANEL_IN = 4.2         # side of one panel of a per-trip overlay figure, inches
OVERLAY_LEGEND_COLUMN_IN = 2.6  # width one legend column of the overlay figures needs at font size 9, inches
REAL_PATH_LABEL = {"robot": "real controller's path", "human": "real person's path"}  # grey dashed line, by the planned agent's kind
PROMPT_DISK = {"robot": "circumscribed disk", "human": "disk"}  # the planned agent's disk in the prompts, by kind
FRAMING_MARKER = {"neutral": ("o", False), "social": ("o", True), "social_body": ("s", True), "cautious": ("D", True)}  # (shape, filled) per prompt framing
FAMILY = {SingleShotEpisode: "single shot", RollingEpisode: "rolling", ConversationEpisode: "conversation"}
# validated categorical slots 1-3 (dataviz reference palette, light surface). The baselines use neutral grey and violet.
FAMILY_COLOUR = {"single shot": "#2a78d6", "rolling": "#eb6834", "conversation": "#1baf7a"}
BASELINE_STYLE = {"replay": ("#52514e", "--"), "straight": ("#6250d6", ":")}
INK, GRID = "#0b0b0b", "#e4e3df"


def collect(run: Path) -> pd.DataFrame:
    """One row per finished episode of the test trips."""
    strata = {c["id"]: c["stratum"] for c in json.loads((run / TEST_TRIPS).read_text())}
    rows = []
    for exp in EXPERIMENTS:
        for path in sorted((run / exp).glob("**/manifest.json")):
            m = json.loads(path.read_text())
            if m["trip"] in strata:
                unknown = set(m["metrics"] or {}) - set(METRIC_CLASS)
                if unknown:
                    raise ValueError(f"{path}: metrics {sorted(unknown)} have no class in METRIC_CLASS")
                metrics = {k: v for k, v in (m["metrics"] or {}).items() if k not in SKIP}
                rows.append({"kind": agent_kind(m["agent"]), "experiment": exp, "framing": m["framing"] or "-", "thinking": m["thinking"] or "-",
                             "folder": str(path.parent.relative_to(run)), **{k: m[k] for k in INFO if k != "stratum"}, "stratum": strata[m["trip"]], **metrics})
    return pd.DataFrame(rows, columns=[*KEY, "folder", *INFO, *METRICS])


def scored(episodes: pd.DataFrame) -> pd.DataFrame:
    """Episodes as scored: metrics of episodes that are not ok are dropped, and they count as goal not reached.
    Adds the INVALID_PARTS and invalid (any part), 0/1 per episode. An episode without a path is 0 in every part but its
    status part. invalid_speed_cap is NaN for experiments without a model, whose plans have no raw speeds."""
    df = episodes.copy()
    df[RATES] = df[RATES].astype(float)  # before the NaN assignment: pandas refuses NaN into a bool column
    ok = df.status == "ok"
    df.loc[~ok, RATES + MEANS + ["clearance_min"]] = np.nan
    df["goal_reached"] = df.goal_reached.fillna(0.0)
    df["invalid_unusable_reply"] = (df.status == "invalid_reply").astype(float)
    df["invalid_error"] = (df.status == "error").astype(float)
    df["invalid_speed_cap"] = (df.speed_violation_rate > 0).astype(float).where(df.experiment.map(lambda e: EXPERIMENTS[e].uses_model))
    df["invalid_goal"] = (ok & (df.goal_reached == 0)).astype(float)
    df["invalid_room"] = (df.out_of_room == 1).astype(float)
    df["invalid_station"] = (df.station_crossing == 1).astype(float)
    df["invalid"] = (df[INVALID_PARTS] == 1).any(axis=1).astype(float)
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
        by_class = {cls: {} for cls in CLASS_ORDER}
        by_class["invalid"] |= {c: g[c].mean() for c in ["invalid", *INVALID_PARTS]}  # NaN only where a part applies to no episode
        for c in RATES + MEANS:
            if c not in ("out_of_room", "station_crossing"):  # reported as their invalid parts, over all episodes
                by_class[METRIC_CLASS[c]][c] = g[c].mean()
        by_class["contact"] |= {"clearance_min_mean": g.clearance_min.mean(), "clearance_min_median": g.clearance_min.median()}
        return pd.Series({
            "n": len(g), **{s: (g.status == s).sum() for s in STATUSES}, **{k: v for cls in CLASS_ORDER for k, v in by_class[cls].items()},
            "queries_per_episode": g.queries.mean(), "retries": g.retries.sum(), "forced_answers": g.forced_answers.sum(),
            "tokens_per_query": g.completion_tokens.sum() / calls if calls else np.nan, "latency_s_per_query": g.query_latency_s.sum() / calls if calls else np.nan})

    parts = [df.groupby(KEY, sort=False).apply(one, include_groups=False).assign(site="all")]
    parts += [d.groupby(KEY, sort=False).apply(one, include_groups=False).assign(site=s) for s, d in df.groupby("site")]
    return pd.concat(parts).reset_index().sort_values(["kind", "site", *COND], kind="stable")


def per_trip_means(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """condition (COND) -> table (trip x metric) of per-trip means over the samples, site as a column. df holds one planned-agent kind."""
    if df.kind.nunique() > 1:
        raise ValueError(f"per-trip means pool one planned-agent kind only, got {sorted(df.kind.unique())}")
    means = df.groupby(COND + ["trip"])[PAIRED].mean()
    sites = df.groupby("trip").site.first()
    return {k: g.droplevel(COND).assign(site=sites) for k, g in means.groupby(level=COND)}


def paired_baselines(by_cond: dict) -> pd.DataFrame:
    rows = []
    for cond, table in by_cond.items():
        if cond[0] in BASELINES:
            continue
        for base in sorted(BASELINES):
            if (base, "-", "-") not in by_cond:
                continue
            ref = by_cond[(base, "-", "-")]
            for site in ("all", *SITES):
                for metric in PAIRED:
                    d = (table[metric] - ref[metric])[table.site.index if site == "all" else table.index[table.site == site]]
                    rows.append({"experiment": cond[0], "framing": cond[1], "thinking": cond[2], "baseline": base, "site": site, "metric": metric,
                                 "n_trips": int(d.notna().sum()), **dict(zip(["diff", "ci_lo", "ci_hi"], ci(d)))})
    return pd.DataFrame(rows)


# Experiment pairs (A, B) per comparison, the difference reported is A - B. The conversation partner has the same images and output as rh_wp_bev_cam / rh_velseq_bev.
EXPERIMENT_COMPARISONS = {
    "bev - text input (waypoints, single shot)": [("ss_wp_bev", "ss_wp_text")],
    "pixel - metric output (bev)": [("ss_px_bev", "ss_wp_bev"), ("rh_px_bev", "rh_wp_bev")],
    "cam + bev - bev only (waypoints)": [("ss_wp_bev_cam", "ss_wp_bev"), ("rh_wp_bev_cam", "rh_wp_bev")],
    "rolling - single shot (bev)": [("rh_wp_bev", "ss_wp_bev"), ("rh_px_bev", "ss_px_bev")],
    "conversation - fresh query each time": [("rh_wp_bev_conv", "rh_wp_bev_cam"), ("rh_velseq_bev_conv", "rh_velseq_bev")],
    "velocity sequence - waypoints (rolling)": [("rh_velseq_bev", "rh_wp_bev")],
    "single velocity step - waypoints (rolling)": [("rh_vel1_bev", "rh_wp_bev")],
}
if unknown := ({e for pairs in EXPERIMENT_COMPARISONS.values() for pair in pairs for e in pair} | set(OVERLAY)) - set(EXPERIMENTS):
    raise ValueError(f"experiment ids not in the registry: {sorted(unknown)}")


def comparison_pairs() -> list[tuple[str, str, tuple, tuple]]:
    """(comparison, scope, A condition, B condition), with the difference reported as A - B. The scope is the factor the pairs are split by."""
    models = [e for e in EXPERIMENTS if e not in BASELINES]
    by_experiment = EXPERIMENT_COMPARISONS | {"prediction - none": [(e, EXPERIMENTS[e].base) for e in models if EXPERIMENTS[e].predict]}
    pairs = []
    for e in models:
        for th in THINKING:
            pairs.append(("social - neutral prompt", th, (e, "social", th), (e, "neutral", th)))
            pairs.append(("social + body size - social prompt", th, (e, "social_body", th), (e, "social", th)))
            pairs.append(("cautious - social prompt", th, (e, "cautious", th), (e, "social", th)))
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


def dot_plot(path: Path, panels: dict[str, str], rows: list[str], offsets: list, by_cond: dict, key, marker, title: str, legend: list) -> None:
    """Rows are experiments, with one panel per metric. Each offset label gets one dot with a bootstrap interval, drawn as marker(label) = (shape, filled). Baselines are vertical lines."""
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
                shape, filled = marker(o)
                ax.plot([lo, hi], [r + off] * 2, color=colour, lw=1.5, zorder=3)
                ax.plot(m, r + off, shape, ms=7, mfc=colour if filled else "white", mec=colour, mew=1.5, zorder=4)
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
    return [e for e in EXPERIMENTS if e not in BASELINES and any(c[0] == e and all(c[COND.index(k)] == v for k, v in fixed.items()) for c in by_cond)]


def legend_handles(baselines: bool, markers: list[tuple[str, str, bool]]) -> list:
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=f) for f, c in FAMILY_COLOUR.items()]
    handles += [Line2D([], [], marker=shape, ls="", ms=7, color=INK, mfc=INK if filled else "white", label=label) for label, shape, filled in markers]
    if baselines:
        handles += [Line2D([], [], color=c, ls=ls, lw=1.5, label=f"{b} (mean)") for b, (c, ls) in BASELINE_STYLE.items()]
    return handles


def condition_figures(by_cond: dict, kind: str, fig_dir: Path) -> None:
    """Dot plots per thinking mode and of the speed-cap violations, of the episodes of one planned-agent kind (by_cond from per_trip_means)."""
    panels = {"goal_reached": "goal reached (rate)", "clearance_min": "min clearance (m, higher = more room)", "time_to_goal": "time to goal (s)", "path_efficiency": "path efficiency (1 = straight)"}
    baselines = any((b, "-", "-") in by_cond for b in BASELINES)
    for th in THINKING:
        rows = present(by_cond, thinking=th)
        if rows:
            framings = [f for f in FRAMINGS if any(c[1] == f and c[2] == th for c in by_cond)]
            dot_plot(fig_dir / f"conditions_{kind}_{th}.png", panels, rows, framings, by_cond, lambda e, f: (e, f, th), FRAMING_MARKER.get,
                     f"Thinking {'off' if th == 'nothink' else 'on'}: mean over trips with 95% bootstrap interval",
                     legend_handles(baselines, [(f"{f.replace('_', ' + ')} prompt", *FRAMING_MARKER[f]) for f in framings]))
    rows = [e for e in present(by_cond) if any(by_cond[c]["speed_violation_rate"].notna().any() for c in by_cond if c[0] == e)]
    if rows:
        dot_plot(fig_dir / f"speed_violation_{kind}.png", {"speed_violation_rate": f"share of plan steps above {SPEED_CAP_M_S[kind]:g} m/s"}, rows, ["nothink", "think"], by_cond,
                 lambda e, th: (e, "neutral", th), lambda th: ("o", th == "think"), "Speed-cap violations before limiting, neutral prompt", legend_handles(False, [("thinking off", "o", False), ("thinking on", "o", True)]))


def overlay_figures(df: pd.DataFrame, run: Path, experiments: list[str], trips: dict, sites: dict, fig_dir: Path) -> None:
    """trip_<id>.png per trip: the samples of each experiment (rows) and condition (columns) drawn by render_overlay, with a legend below that fits the figure width."""
    for trip_id, d in df[df.experiment.isin(experiments)].groupby("trip"):
        trip = trips[trip_id]
        site, agent = sites[trip["site"]], trip["agent"]
        kind, a, goal = agent_kind(agent), trip["start_index"], np.asarray(trip["goal_xy"], float)
        real = real_path(trip)
        shown = [e for e in experiments if e in set(d.experiment)]
        cols = sorted({(r.thinking, r.framing) for r in d.itertuples()}, key=lambda c: (("-", *THINKING).index(c[0]), ("-", *FRAMINGS).index(c[1])))  # "-": baselines
        width = OVERLAY_PANEL_IN * len(cols)
        fig = Figure(figsize=(width, OVERLAY_PANEL_IN * len(shown) + 0.4), dpi=110)
        axes = fig.subplots(len(shown), len(cols), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, exp in enumerate(shown):
            for j, (th, fr) in enumerate(cols):
                g = d[(d.experiment == exp) & (d.thinking == th) & (d.framing == fr)]
                if g.empty:
                    continue
                paths = [load_trajectory(run / f)[1][agent] for f in g.folder]
                end = max(len(real), *map(len, paths))
                others = {o: np.asarray(track, float)[a: a + end, :2] for o, track in trip["tracks"].items() if o != agent}
                title = f"{exp} {fr} {th}: {int((g.status == 'ok').sum())}/{len(g)} ok"
                png = render_overlay(site, agent, goal, paths, real, others, title)
                axes[i, j].imshow(imread(io.BytesIO(png), format="png"))
        other_kinds = dict.fromkeys(agent_kind(o) for o in trip["tracks"] if o != agent)
        legend = [Line2D([], [], color=REAL_PATH, ls="--", label=REAL_PATH_LABEL[kind]), Line2D([], [], color=GOAL, ls=":", label="straight line"),
                  Line2D([], [], color=GOAL, marker="*", ls="", ms=10, label="goal"),
                  Line2D([], [], color=GOAL, marker="o", ls="", ms=10, mfc="none", mew=1.5, label=f"arrival radius ({ARRIVAL_RADIUS_M[kind]:g} m)"),
                  *[Line2D([], [], color=KIND_COLOUR[k], marker="o", ms=4, label=f"{KIND_LEGEND[k]} (disk: final position)") for k in other_kinds],
                  Line2D([], [], color=KIND_COLOUR[kind], marker="o", ms=4, label="model samples"),
                  Line2D([], [], color=KIND_COLOUR[kind], marker="o", ls="", ms=10, alpha=0.3, mec="k",
                         label=f"final position ({PROMPT_DISK[kind]}, {prompt_radius(agent, site):g} m)"),
                  Line2D([], [], color="k", marker="s", ls="", ms=7, label="start")]
        fig.suptitle(f"{trip_id} ({trip['controller']}, {d.stratum.iloc[0]})", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        ncol = int(min(4, max(1, width // OVERLAY_LEGEND_COLUMN_IN)))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, fontsize=9, frameon=False)
        fig.savefig(fig_dir / f"trip_{trip_id}.png", bbox_inches="tight")  # the tight box takes in the legend below the panels and the title


def analyze(run: Path, overlay: list[str], trips: dict, sites: dict) -> None:
    """Write the tables and figures of the run's episodes into <run>/analysis, with trips and sites as returned by run.load_run."""
    out = run / ANALYSIS_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    episodes = collect(run)
    episodes.to_csv(out / "episodes.csv", index=False)
    df = scored(episodes)
    summary(df).to_csv(out / "summary.csv", index=False)
    paired, compared, n_conditions = [], [], 0
    for kind, d in df.groupby("kind"):
        by_cond = per_trip_means(d)
        for tables, table in ((paired, paired_baselines(by_cond)), (compared, comparisons(by_cond))):
            tables.append(table.assign(kind=kind)[["kind", *table.columns]])
        condition_figures(by_cond, kind, out / "figures")
        n_conditions += len(by_cond)
    pd.concat(paired).to_csv(out / "paired_baselines.csv", index=False)
    pd.concat(compared).to_csv(out / "comparisons.csv", index=False)
    overlay_figures(df, run, overlay, trips, sites, out / "figures")
    print(f"{len(episodes)} episodes, {n_conditions} conditions -> {out}")
