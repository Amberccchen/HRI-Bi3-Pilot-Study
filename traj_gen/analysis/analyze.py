"""Analysis of a trajectory generation run: tables, comparisons and figures.

Reads every finished episode under a run folder, whose trips must all be in the trip list (default <run>/test_trips.json, as for `run`). Only experiment folders
are read: those with a <id>/<framing>/harness.json, whose settings define the experiment, and the baseline folders, so other folders such as
smoke/ are ignored (harness.presets.run_experiments). A run still in progress is fine, and missing conditions are simply absent. Writes into <run>/analysis/:

  episodes.csv          one row per episode: condition, trip, planned agent and its kind, status, metrics, queries, tokens, latency
  summary.csv           per planned-agent kind, condition (experiment x framing x thinking, plus the baselines) and
                        site (all / laas / um): status counts and episode-level rates and means, grouped by
                        metric class (invalid, contact, efficiency)
  paired_baselines.csv  every condition minus replay and minus straight on the same trips, per kind
  comparisons.csv       the condition comparisons (think - nothink, bev - text input, ...) as differences, per kind
  similarity.csv        planned people only: per condition, site and feature, the W1 of the generated moves to real moves
                        (human_similarity.feature_distances), against the site's pool of real moves and against the real
                        moves of the same test trips
  similarity_pool.csv   planned people only: per site, the person trips in the trips folder, the trips in the real pool, and the trips left
                        out because they start inside the arrival radius or have no move
  figures/              conditions_<kind>.png, speed_violation_<kind>.png (condition_figure), trip_<id>.png (overlays)

Robot and person episodes are never pooled: every table and figure is per planned-agent kind ("robot" or "human").

Scoring rules. Path metrics (clearance, time to goal, ...) use status-ok episodes only. goal_reached
counts every episode, so one without a usable path counts as not reached. An episode is invalid when
its reply is unusable, the server failed (status error), or its plan exceeds the speed cap in any step,
does not reach the goal or leaves the room (beyond the agent kind's ROOM_MARGIN_M). The invalid rate and the rate of each of
these parts are shares of all episodes of the condition. An episode without a path (invalid_reply or
error) counts only in its status part. The speed-cap part does not apply to the baselines, which have no
model reply, and is NaN for them. Paired differences and comparisons work on per-trip means (over
the samples of the condition), so a trip is one observation and the intervals are percentile
bootstrap intervals over trips (few trips: read them as rough). time_to_goal only exists for episodes that reached the goal.

Similarity. A move is the part of a path from the trip start to move_end_index (traj_gen_metrics.move_end_index), for the
generated paths of ok episodes and for the real paths alike, so a replay episode's move is its trip's real move. Episodes
without a move (not ok, or the goal not reached) are left out and counted. The real moves are every person trip of the site in
the run's trips folder that reaches its goal and does not start within the arrival radius (as select requires), stratified by
(start station, goal station). A trial is one participant pair, which ran all five controllers. Against the site pool, the real
set is the site's moves from trials other than the generated moves' trials. Against the same trips, it is the test trips' real
moves. In both, the reference draws are REFERENCE_DRAWS_PER_TRIAL matched subsets of each of the site's other trials (feature_distances), and their numbers are reported with each
row. similarity_pool.csv lists per site the trips in the pool and the person trips left out of it.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D

from metrics.human_similarity import MIN_PERCENTILE_DRAWS, feature_distances, trajectory_features
from metrics.traj_gen_metrics import METRIC_CLASS, move_end_index
from preprocessing.bi3_derived import SITES
from traj_gen.agents import ARRIVAL_RADIUS_M, SPEED_CAP_M_S, agent_kind
from traj_gen.harness.episodes import ANALYSIS_DIR, ConversationEpisode, RollingEpisode, SingleShotEpisode, load_trajectory
from traj_gen.harness.experiments import BASELINES, Experiment
from traj_gen.harness.presets import PRESETS, run_experiments
from traj_gen.observation.bev import render_overlay, trajectory_legend
from traj_gen.observation.prompting import FRAMINGS
from traj_gen.outputs.baselines import real_path
from traj_gen.outputs.planner import THINKING

COND = ["experiment", "framing", "thinking"]  # a condition within one planned-agent kind
KEY = ["kind", *COND]                         # a condition across kinds: robot and person episodes are never pooled
N_BOOT = 2000   # bootstrap resamples of the trips: the interval's edges are stable to about 0.01 SD
SEED = 20261022  # fixed so the intervals are reproducible. The value is a date with no other meaning
INFO = ["trip", "site", "controller", "agent", "stratum", "sample", "status", "queries", "samples", "retries", "invalid_replies",
        "forced_answers", "prompt_tokens", "completion_tokens", "query_latency_s", "wall_s", "move_end_index"]
STATUSES = ("ok", "invalid_reply", "error")
SKIP = {"min_distance_per_other", "min_distance_index", "min_distance_other",  # per-agent detail, not an episode-level number
        "station_crossing", "station_distance_min"}  # no longer computed: skipped when an older run folder's manifest still holds them
METRICS = [m for m in METRIC_CLASS if m not in SKIP]  # episode-level metric columns, present (NaN when absent) in every episode table
RATES = ["goal_reached", "contact_time", "personal_space_overlap", "out_of_room"]
# Parts of the invalid rate, 0/1 per episode over all episodes (NaN where the part does not apply).
INVALID_PARTS = ["invalid_unusable_reply", "invalid_error", "invalid_speed_cap", "invalid_goal", "invalid_room"]
MEANS = ["contact_events", "min_distance", "time_to_goal", "path_efficiency", "velocity_mean", "acceleration_mean", "speed_violation_rate", "raw_speed_max", "final_distance_to_goal"]
CLASS_ORDER = ("invalid", "contact", "efficiency")
PAIRED = ["goal_reached", "contact_time", "contact_events", "personal_space_overlap", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
COMPARED = ["goal_reached", "contact_time", "contact_events", "personal_space_overlap", "clearance_min", "time_to_goal", "path_efficiency", "speed_violation_rate"]
OVERLAY = list(dict.fromkeys(preset.harness("human").id for preset in PRESETS.values()))  # default experiments in the per-trip overlays: the presets
OVERLAY_PANEL_IN = 4.2         # side of one panel of a per-trip overlay figure, inches
THINKING_DOTS = (("nothink", False), ("think", True))  # (thinking, filled) of the two dots of a row of condition_figure
CONDITION_PANELS = {"goal_reached": "goal reached (rate)", "clearance_min": "min clearance (m, higher = more room)", "time_to_goal": "time to goal (s)",
                    "path_efficiency": "path efficiency (1 = straight)"}
TURN_MIN_DISPLACEMENT_M = 0.15  # turning angles need both 0.8 s displacements at least this long (human_similarity.trajectory_features)
SIMILARITY_DRAWS = 1000         # bootstrap draws of each feature distance
REFERENCE_DRAWS_PER_TRIAL = 20  # matched subsets per reference trial, pooled: 300-400 reference draws per site from 16-19 trials
FAMILY = {SingleShotEpisode: "single shot", RollingEpisode: "rolling", ConversationEpisode: "conversation"}
# validated categorical slots 1-3 (dataviz reference palette, light surface). The baselines use neutral grey and violet.
FAMILY_COLOUR = {"single shot": "#2a78d6", "rolling": "#eb6834", "conversation": "#1baf7a"}
BASELINE_STYLE = {"replay": ("#52514e", "--"), "straight": ("#6250d6", ":")}
INK, GRID = "#0b0b0b", "#e4e3df"


def collect(run: Path, chosen: list[dict], folders: dict[str, Experiment]) -> pd.DataFrame:
    """One row per finished episode of the chosen trips (id and stratum each), from the experiment folders (folder name -> Experiment, as
    harness.presets.run_experiments returns them). The experiment column is the experiment's id. Raises ValueError when the run folder holds
    episodes of other trips."""
    strata = {c["id"]: c["stratum"] for c in chosen}
    rows, other_trips = [], set()
    for folder, experiment in folders.items():
        exp = experiment.id
        for path in sorted((run / folder).glob("**/manifest.json")):
            m = json.loads(path.read_text())
            if m["trip"] not in strata:
                other_trips.add(m["trip"])
                continue
            unknown = set(m["metrics"] or {}) - set(METRIC_CLASS) - SKIP
            if unknown:
                raise ValueError(f"{path}: metrics {sorted(unknown)} have no class in METRIC_CLASS")
            metrics = {k: v for k, v in (m["metrics"] or {}).items() if k not in SKIP}
            rows.append({"kind": agent_kind(m["agent"]), "experiment": exp, "framing": m["framing"] or "-", "thinking": m["thinking"] or "-",
                         "folder": str(path.parent.relative_to(run)), **{k: m[k] for k in INFO if k != "stratum"}, "stratum": strata[m["trip"]], **metrics})
    if other_trips:
        raise ValueError(f"{run} holds episodes of trips that are not in the trip list: {sorted(other_trips)}. Pass the trip list the run was made with (--trips)")
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
    df["invalid_speed_cap"] = (df.speed_violation_rate > 0).astype(float).where(~df.experiment.isin(list(BASELINES)))
    df["invalid_goal"] = (ok & (df.goal_reached == 0)).astype(float)
    df["invalid_room"] = (df.out_of_room == 1).astype(float)
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
            if c != "out_of_room":  # reported as its invalid part, over all episodes
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


# Experiment pairs (A, B) per comparison. The difference reported is A - B. The conversation partner has the same images and output as
# rh_positions_m_bev_cam / rh_positions_px_bev / rh_velseq_m_bev.
EXPERIMENT_COMPARISONS = {
    "bev - text input (positions in metres, single shot)": [("ss_positions_m_bev", "ss_positions_m_text")],
    "pixel - metre units (positions, bev)": [("ss_positions_px_bev", "ss_positions_m_bev"), ("rh_positions_px_bev", "rh_positions_m_bev")],
    "cam + bev - bev only (positions in metres)": [("ss_positions_m_bev_cam", "ss_positions_m_bev"), ("rh_positions_m_bev_cam", "rh_positions_m_bev")],
    "rolling - single shot (bev)": [("rh_positions_m_bev", "ss_positions_m_bev"), ("rh_positions_px_bev", "ss_positions_px_bev")],
    "conversation - fresh query each time": [("rh_positions_m_bev_cam_conv", "rh_positions_m_bev_cam"), ("rh_positions_px_bev_conv", "rh_positions_px_bev"),
                                             ("rh_velseq_m_bev_conv", "rh_velseq_m_bev")],
    "velocity sequence - positions (rolling, metres)": [("rh_velseq_m_bev", "rh_positions_m_bev")],
    "single velocity step - positions (rolling, metres)": [("rh_vel1_m_bev", "rh_positions_m_bev")],
}


def comparison_pairs(experiments: dict[str, Experiment]) -> list[tuple[str, str, tuple, tuple]]:
    """(comparison, scope, A condition, B condition) over the experiments (id -> Experiment), with the difference reported as A - B. The scope is the factor the pairs are split by."""
    models = [e for e in experiments if e not in BASELINES]
    by_experiment = EXPERIMENT_COMPARISONS | {"prediction - none": [(e, experiments[e].base) for e in models if experiments[e].predict]}
    pairs = []
    for e in models:
        for fr in FRAMINGS:
            pairs.append(("think - nothink", fr, (e, fr, "think"), (e, fr, "nothink")))
    for name, experiments in by_experiment.items():
        for a, b in experiments:
            for fr in FRAMINGS:
                for th in THINKING:
                    pairs.append((name, th, (a, fr, th), (b, fr, th)))
    return pairs


def comparisons(by_cond: dict, experiments: dict[str, Experiment]) -> pd.DataFrame:
    rows = []
    pairs = [p for p in comparison_pairs(experiments) if p[2] in by_cond and p[3] in by_cond]
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


def condition_figure(path: Path, panels: list[tuple[str, list[tuple[str, list[str], str]]]], metrics: dict[str, str], by_cond: dict,
                     experiments: dict[str, Experiment], title: str) -> None:
    """Panels stacked as the rows of one figure, with a row per condition and a column per metric (metrics: metric -> axis label), the x
    axis shared down each column. A panel is (title, rows) and a row (label, experiment ids pooled per trip, framing). Each row has a dot
    for thinking off (hollow) and on (filled): the mean over trips of the row's per-trip values (the mean over its experiments of their
    per-trip means) with its 95% bootstrap interval, coloured by the episode family of its first experiment. A row missing one of its
    experiments for a thinking mode gets no dot for it. The baselines (straight line, replay) are vertical lines at their mean over trips.
    Any set of conditions of one planned-agent kind can be drawn this way (by_cond from per_trip_means, experiments id -> Experiment)."""
    heights = [len(rows) for _, rows in panels]
    fig = Figure(figsize=(max(3.0 * len(metrics) + 2.0, 7.5), 1.2 + 0.35 * len(panels) + 0.42 * sum(heights)), dpi=150, layout="constrained")
    grid = fig.subplots(len(panels), len(metrics), sharex="col", sharey="row", squeeze=False, gridspec_kw={"height_ratios": heights})
    for i, (axes, (panel_title, rows)) in enumerate(zip(grid, panels)):
        for ax, (metric, label) in zip(axes, metrics.items()):
            for base, (colour, ls) in BASELINE_STYLE.items():
                if (base, "-", "-") in by_cond:
                    ax.axvline(by_cond[(base, "-", "-")][metric].mean(), color=colour, ls=ls, lw=1.5, zorder=2)
            for r, (_, pooled, framing) in enumerate(rows):
                colour = FAMILY_COLOUR[FAMILY[experiments[pooled[0]].episode]]
                for (thinking, filled), off in zip(THINKING_DOTS, (-0.17, 0.17)):
                    conds = [(e, framing, thinking) for e in pooled]
                    if any(c not in by_cond for c in conds):
                        continue
                    m, lo, hi = ci(pd.concat([by_cond[c][metric] for c in conds], axis=1).mean(axis=1))
                    ax.plot([lo, hi], [r + off] * 2, color=colour, lw=1.5, zorder=3)
                    ax.plot(m, r + off, "o", ms=6, mfc=colour if filled else "white", mec=colour, mew=1.5, zorder=4)
            ax.set_yticks(range(len(rows)), [name for name, _, _ in rows])
            ax.tick_params(labelbottom=True)  # the scale is shared, but each panel keeps its own tick labels
            style(ax, label if i == len(panels) - 1 else "")
        axes[0].set_ylim(len(rows) - 0.5, -0.5)  # shared y: once per panel, first row on top
        if panel_title:
            axes[0].set_title(panel_title, loc="left", fontsize=10, color=INK, fontweight="bold")
    handles = [Line2D([], [], marker="o", ls="", ms=7, color=c, label=f) for f, c in FAMILY_COLOUR.items()]
    handles += [Line2D([], [], marker="o", ls="", ms=7, color=INK, mfc=INK if filled else "white", label=f"thinking {'on' if filled else 'off'}")
                for _, filled in THINKING_DOTS]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.5, label=f"{b} (mean)") for b, (c, ls) in BASELINE_STYLE.items() if (b, "-", "-") in by_cond]
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.legend(handles=handles, loc="outside lower center", ncol=min(len(handles), 4), fontsize=9, frameon=False)
    fig.savefig(path)


def condition_figures(by_cond: dict, kind: str, fig_dir: Path, experiments: dict[str, Experiment]) -> None:
    """condition_figure of every model condition (a row per experiment and framing) and of the speed-cap violations, of the episodes of one
    planned-agent kind (by_cond from per_trip_means, experiments id -> Experiment)."""
    rows = [(f"{e} {f}", [e], f) for e, f in dict.fromkeys((c[0], c[1]) for c in by_cond if c[0] not in BASELINES)]
    rows.sort(key=lambda row: (list(experiments).index(row[1][0]), list(FRAMINGS).index(row[2])))
    if rows:
        condition_figure(fig_dir / f"conditions_{kind}.png", [("", rows)], CONDITION_PANELS, by_cond, experiments, "Mean over trips with 95% bootstrap interval")
    with_raw = [row for row in rows if any(by_cond[c]["speed_violation_rate"].notna().any() for c in by_cond if c[:2] == (row[1][0], row[2]))]
    if with_raw:
        condition_figure(fig_dir / f"speed_violation_{kind}.png", [("", with_raw)], {"speed_violation_rate": f"share of plan steps above {SPEED_CAP_M_S[kind]:g} m/s"},
                         by_cond, experiments, "Speed-cap violations before limiting")


def overlay_figures(df: pd.DataFrame, run: Path, experiments: list[str], trips: dict, sites: dict, fig_dir: Path) -> None:
    """trip_<id>.png per trip, drawn by render_overlay: one row per model experiment and a column per model condition (thinking x
    framing), and a two-column legend below. Baselines get no panel: every panel draws the straight line and the real path."""
    for trip_id, d in df[df.experiment.isin(experiments) & ~df.experiment.isin(list(BASELINES))].groupby("trip"):
        trip = trips[trip_id]
        site, agent = sites[trip["site"]], trip["agent"]
        kind, a, goal = agent_kind(agent), trip["start_index"], np.asarray(trip["goal_xy"], float)
        real = real_path(trip)
        shown = [e for e in experiments if e in set(d.experiment)]
        cols = sorted({(r.thinking, r.framing) for r in d.itertuples()}, key=lambda c: (THINKING.index(c[0]), list(FRAMINGS).index(c[1])))
        panels = [(i, j, g) for i, e in enumerate(shown) for j, (th, fr) in enumerate(cols)
                  for g in [d[(d.experiment == e) & (d.thinking == th) & (d.framing == fr)]] if not g.empty]
        width = OVERLAY_PANEL_IN * len(cols)
        fig = Figure(figsize=(width, OVERLAY_PANEL_IN * len(shown) + 0.4), dpi=110)
        axes = fig.subplots(len(shown), len(cols), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, j, g in panels:
            paths = [load_trajectory(run / f)[1][agent] for f in g.folder]
            end = max(len(real), *map(len, paths))
            others = {o: np.asarray(track, float)[a: a + end, :2] for o, track in trip["tracks"].items() if o != agent}
            r = g.iloc[0]
            title = f"{r.experiment} {r.framing} {r.thinking}: {int((g.status == 'ok').sum())}/{len(g)} ok"
            axes[i, j].imshow(imread(io.BytesIO(render_overlay(site, agent, goal, paths, real, others, title)), format="png"))
        legend = trajectory_legend(agent, [o for o in trip["tracks"] if o != agent])
        fig.suptitle(f"{trip_id} ({trip['controller']}, {d.stratum.iloc[0]})", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)
        fig.savefig(fig_dir / f"trip_{trip_id}.png", bbox_inches="tight")  # the tight box takes in the legend below the panels and the title


def move_features(trip: dict, positions: dict[str, np.ndarray], end: int) -> dict:
    """trajectory_features of a person's move: positions ({agent id: (T, 2)} from the trip start) up to index end, stratified by (start station, goal station)."""
    agent = trip["agent"]
    other = next(a for a in positions if a not in (agent, "robot"))
    return trajectory_features(positions[agent][: end + 1], positions["robot"][: end + 1], positions[other][: end + 1], (trip["start_goal"], trip["goal"]),
                               (trip["site"], trip["trial"]), min_displacement_m=TURN_MIN_DISPLACEMENT_M)


def real_pool(trips: dict, sites: dict) -> tuple[dict, pd.DataFrame]:
    """The real moves of the person trips that reach their goal and do not start within the arrival radius, as ({site: {trip id: move features}}, per-site counts of the trips pooled and left out)."""
    radius = ARRIVAL_RADIUS_M["human"]
    pool = {site: {} for site in SITES}
    counts = {site: {"person_trips": 0, "start_inside_arrival_radius": 0, "without_move": 0} for site in SITES}
    for trip in trips.values():
        if agent_kind(trip["agent"]) != "human":
            continue
        counts[trip["site"]]["person_trips"] += 1
        a, b = trip["start_index"], trip["end_index"] + 1
        positions = {agent: np.asarray(track, float)[a:b, :2] for agent, track in trip["tracks"].items()}
        goal = np.asarray(trip["goal_xy"], float)
        if np.linalg.norm(positions[trip["agent"]][0] - goal) <= radius:
            counts[trip["site"]]["start_inside_arrival_radius"] += 1
            continue
        end = move_end_index(positions[trip["agent"]], goal, radius, sites[trip["site"]], trip["goal"])
        if end is None:
            counts[trip["site"]]["without_move"] += 1
        else:
            pool[trip["site"]][trip["id"]] = move_features(trip, positions, end)
    table = pd.DataFrame([{"site": site, **c, "pooled_trips": len(pool[site]), "pooled_trials": len({m["trial"] for m in pool[site].values()})}
                          for site, c in counts.items()])
    return pool, table


def similarity(df: pd.DataFrame, run: Path, trips: dict, pool: dict) -> pd.DataFrame:
    """One row per person condition, site, comparison and feature: the FeatureDistance of the generated moves to the real moves, with the real pool from real_pool."""
    rng = np.random.default_rng(SEED)
    rows = []
    for (*cond, site), d in df[df.kind == "human"].groupby([*COND, "site"], sort=False):
        ok = d[(d.status == "ok") & d.move_end_index.notna()]
        generated = [move_features(trips[r.trip], load_trajectory(run / r.folder)[1], int(r.move_end_index)) for r in ok.itertuples()]
        if not generated:
            continue
        generated_trials = {m["trial"] for m in generated}
        moves = list(pool[site].values())
        comparisons = {"site pool": [m for m in moves if m["trial"] not in generated_trials],
                       "same trips": [pool[site][t] for t in sorted(set(d.trip) & set(pool[site]))]}
        for comparison, real in comparisons.items():
            distances = feature_distances(generated, real, moves, n_bootstrap=SIMILARITY_DRAWS, rng=rng,
                                          reference_draws_per_trial=REFERENCE_DRAWS_PER_TRIAL, reference_seed=SEED)
            for feature, fd in distances.items():
                rows.append({**dict(zip(COND, cond)), "site": site, "comparison": comparison, "feature": feature, "w1": fd.w1,
                             "w1_lo": fd.w1_interval[0], "w1_hi": fd.w1_interval[1], "reference_median": fd.reference_median,
                             "reference_lo": fd.reference_interval[0], "reference_hi": fd.reference_interval[1],
                             "reference_interval": "percentiles 2.5-97.5" if fd.n_reference_used >= MIN_PERCENTILE_DRAWS else "min-max",
                             "n_reference_draws": fd.n_reference_used, "n_reference_trials": fd.n_reference_trials, "reference_moves_median": fd.reference_moves_median,
                             "n_bootstrap_used": fd.n_bootstrap_used, "n_generated": fd.n_generated, "n_real": fd.n_real,
                             "n_episodes": len(d), "n_without_move": len(d) - len(ok)})
    return pd.DataFrame(rows)


def analyze(run: Path, overlay: list[str], chosen: list[dict], trips: dict, sites: dict) -> None:
    """Write the tables and figures of the run's episodes into <run>/analysis, with the trip list, trips and sites as returned by run.load_run."""
    out = run / ANALYSIS_DIR
    (out / "figures").mkdir(parents=True, exist_ok=True)
    folders = run_experiments(run)
    by_id = {exp.id: exp for exp in folders.values()}
    experiments = {e: by_id[e] for e in sorted(by_id)}
    episodes = collect(run, chosen, folders)
    episodes.to_csv(out / "episodes.csv", index=False)
    df = scored(episodes)
    summary(df).to_csv(out / "summary.csv", index=False)
    paired, compared, n_conditions = [], [], 0
    for kind, d in df.groupby("kind"):
        by_cond = per_trip_means(d)
        for tables, table in ((paired, paired_baselines(by_cond)), (compared, comparisons(by_cond, experiments))):
            tables.append(table.assign(kind=kind)[["kind", *table.columns]])
        condition_figures(by_cond, kind, out / "figures", experiments)
        n_conditions += len(by_cond)
    pd.concat(paired).to_csv(out / "paired_baselines.csv", index=False)
    pd.concat(compared).to_csv(out / "comparisons.csv", index=False)
    if (df.kind == "human").any():
        pool, pool_counts = real_pool(trips, sites)
        pool_counts.to_csv(out / "similarity_pool.csv", index=False)
        similarity(df, run, trips, pool).to_csv(out / "similarity.csv", index=False)
    overlay_figures(df, run, overlay, trips, sites, out / "figures")
    print(f"{len(episodes)} episodes, {n_conditions} conditions -> {out}")
