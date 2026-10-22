"""Annotate each Bi3 participant's current and next workstation over time.

Participants shuttle blocks between perimeter workstations, so a station visit
shows up as a turnaround close to a station. Visits are found per person, then
one backward pass labels every in-transit sample with the station it is heading
to. The robot is labelled from its recorded goals and also run through the same
detector, which checks the detector against ground truth.

Goal labels are text: a station id ("0".."5"), or a shared-table pair ("0/1" at
LAAS, "4/5" at UM) when a participant's visits there could not be resolved to one
of the two stations. stations.json lists every allowed label per site.

Outputs (mocap-derived, so kept outside this repository; read them with bi3_derived.py):
  <out>/manifest.json                        parameters and script fingerprint
  <out>/stations.json                        allowed goal labels and coordinates per site
  <out>/<site>/<trial>/<controller>_frames.csv  per-sample labels for robot and humans
  <out>/<site>/<trial>/<controller>_visits.csv  detected visits
  <out>/robot_check.csv                      detector vs recorded robot goals
"""

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bi3_derived import DERIVED, GOALS_DIR, write_manifest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Bi3" / "Bi3"
OUT = DERIVED / GOALS_DIR

# Workstation coordinates are the robot's goal points (the only six values
# robot_goal takes per site). People stand 0.3-0.6 m further out than these.
STATIONS = {
    "laas": [(3.66, 15.0), (3.66, 16.0), (4.85, 14.45), (4.85, 16.9), (5.8, 14.5), (5.8, 16.9)],
    "um": [(-1.55, 1.45), (-1.5, -1.4), (-0.1, -1.45), (-0.05, 1.45), (1.33, 0.48), (1.38, -0.4)],
}
# Station pairs on one table where people stand close together: LAAS 0/1 (about
# 0.35 m apart where people stand) and UM 4/5 (one table, two tags). Other pairs
# can have close goal points (LAAS 2/4, 3/5 are 0.95 m) but are reached from different sides.
SHARED_TABLES = {"laas": [(0, 1)], "um": [(4, 5)]}
SMOOTH_S = 0.5          # moving-average window on positions
TURN_LOOK_M = 0.4       # path length before/after a sample used to measure turning
TURN_MIN_DEG = 100      # a visit reverses direction at least this much
VISIT_RADIUS_M = 1.0    # turnaround must be this close to a station
AT_STATION_SLACK_M = 0.15  # "at station" while within closest approach + slack
MERGE_S = 2.0           # repeated turnarounds at one station closer than this are one visit
LOOP_M = 1.0            # leaving a station by more than this and returning implies a visit in between
PHASE_MIN_SHARE = 0.7   # relabel a phase's table visits only if its majority holds this share of the vote
GOAL_TOL_M = 0.05       # recorded robot goal matches a station within this
ROBOT_LAG_S = (1.0, 4.0)  # robot turnaround window after its goal switches (before, after)


def load(path: Path) -> dict:
    steps = json.loads(path.read_text())
    return {
        "time": np.array([s["time"] for s in steps]),
        "robot": np.array([s["robot_state"] for s in steps]),
        "humans": np.array([s["agent_states"] for s in steps]),
        "robot_goal": np.array([s["robot_goal"] for s in steps]),
    }


def smooth(p: np.ndarray, time: np.ndarray) -> np.ndarray:
    n = max(1, int(round(SMOOTH_S / np.median(np.diff(time)))))
    pad = np.pad(p, ((n // 2, n - 1 - n // 2), (0, 0)), mode="edge")
    kernel = np.ones(n) / n
    return np.stack([np.convolve(pad[:, k], kernel, mode="valid") for k in range(p.shape[1])], axis=1)


def turn_angles(p: np.ndarray) -> np.ndarray:
    """Direction change (deg) between the previous and next TURN_LOOK_M of path."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    back = np.searchsorted(s, s - TURN_LOOK_M, side="right") - 1
    fwd = np.searchsorted(s, s + TURN_LOOK_M, side="left")
    ok = (back >= 0) & (fwd < len(p))
    back, fwd = np.clip(back, 0, len(p) - 1), np.clip(fwd, 0, len(p) - 1)
    u, v = p - p[back], p[fwd] - p
    cos = np.sum(u * v, axis=1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    angle[~ok] = 0.0
    return angle


def detect_visits(p: np.ndarray, time: np.ndarray, stations: np.ndarray, shared_tables: list = ()) -> list[dict]:
    """Visits as turnarounds near a station, with arrive/leave times.

    Pass shared_tables for participants to relabel side-by-side stations by tower phase.
    """
    ps = smooth(p, time)
    dist = np.linalg.norm(ps[:, None] - stations[None], axis=-1)
    nearest, dmin = dist.argmin(1), dist.min(1)
    turning = (turn_angles(ps) >= TURN_MIN_DEG) & (dmin < VISIT_RADIUS_M)
    visits = []
    # each contiguous run of turning samples is one turnaround; keep its closest approach
    edges = np.flatnonzero(np.diff(np.concatenate([[0], turning.astype(int), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        i = start + int(np.argmin(dmin[start:stop]))
        st = int(nearest[i])
        if visits and visits[-1]["station"] == st and time[i] - visits[-1]["t_leave"] < MERGE_S:
            visits[-1]["t_leave"] = time[i]
            continue
        visits.append({"station": st, "i": i, "t_turn": time[i], "t_leave": time[i], "inferred": False})
    # A wide U-turn around a station can stay under TURN_MIN_DEG. If someone goes
    # between visits at the same or neighbouring stations (e.g. 0 -> 0, or LAAS
    # 0 -> 1 next door) via a point far from both, the far end was a missed visit.
    filled = []
    for a, b in zip(visits, visits[1:] + [None]):
        filled.append(a)
        if b is None or np.linalg.norm(stations[a["station"]] - stations[b["station"]]) > LOOP_M + 0.05:
            continue
        away = np.minimum(dist[a["i"] : b["i"], a["station"]], dist[a["i"] : b["i"], b["station"]])
        far = a["i"] + int(np.argmax(away))
        if away.max() > LOOP_M and nearest[far] not in (a["station"], b["station"]) and dmin[far] < VISIT_RADIUS_M:
            filled.append({"station": int(nearest[far]), "i": far, "t_turn": time[far], "t_leave": time[far], "inferred": True})
    visits = filled
    for v in visits:
        order = np.argsort(dist[v["i"]])
        v["runner_up"] = int(order[1]) if order[0] == v["station"] else int(order[0])
        v["margin"] = dist[v["i"], v["runner_up"]] - dist[v["i"], v["station"]]
        v["corrected"] = False
        v["label"] = str(v["station"])
    if shared_tables:
        visits = correct_by_phase(visits, len(stations), shared_tables)
    for v in visits:
        d, i = dist[:, v["station"]], v["i"]
        v["min_dist"] = d[i]
        near = d <= v["min_dist"] + AT_STATION_SLACK_M
        a = i
        while a > 0 and near[a - 1]:
            a -= 1
        b = max(i, int(np.searchsorted(time, v["t_leave"])))
        while b < len(d) - 1 and near[b + 1]:
            b += 1
        v["i_arrive"], v["i_leave"] = a, b
        v["t_arrive"], v["t_leave"] = time[a], time[b]
    return visits


def correct_by_phase(visits: list[dict], n_stations: int, shared_tables: list) -> list[dict]:
    """Give each tower phase one station per shared table.

    Each participant builds two towers in turn, shuttling between one pile and one
    tower per phase, so within a phase they keep returning to the same spot. On the
    tables holding two stations 1 m apart, position alone is ambiguous: people
    stand at the inner ends, about 0.35 m apart. The sequence is split into the two
    phases where the station pairs change (a shared table counted as one station),
    then each phase's visits to a shared table take the margin-weighted majority
    station if it holds PHASE_MIN_SHARE of the vote. Otherwise they are labelled
    with the pair, e.g. "0/1".
    """
    table = np.arange(n_stations)
    for pair in shared_tables:
        table[list(pair)] = min(pair)
    groups = [int(table[v["station"]]) for v in visits]

    def purity(seq: list[int]) -> int:
        return sum(sorted(np.bincount(seq, minlength=n_stations))[-2:]) if seq else 0

    n = len(visits)
    cuts = range(4, n - 3)
    split = max(cuts, key=lambda c: purity(groups[:c]) + purity(groups[c:])) if len(cuts) else n
    for phase, (lo, hi) in enumerate([(0, split), (split, n)]):
        for v in visits[lo:hi]:
            v["phase"], v["phase_share"] = phase, 1.0
        for pair in shared_tables:
            members = [v for v in visits[lo:hi] if v["station"] in pair]
            if not members:
                continue
            weight = {}
            for v in members:
                weight[v["station"]] = weight.get(v["station"], 0.0) + 0.05 + min(v["margin"], 0.5)
            winner = max(weight, key=weight.get)
            share = weight[winner] / sum(weight.values())
            for v in members:
                v["phase_share"] = round(share, 3)
                if share >= PHASE_MIN_SHARE and v["station"] != winner:
                    v["station"], v["corrected"] = winner, True
                v["label"] = str(v["station"]) if share >= PHASE_MIN_SHARE else shared_label(pair)
    return visits


def shared_label(pair: tuple) -> str:
    return "/".join(str(k) for k in sorted(pair))


def label_from_visits(n: int, visits: list[dict]) -> dict:
    """One backward pass: state, current station, goal and the goal after it ("" for none)."""
    state = np.full(n, "transit", dtype=object)
    station = np.full(n, "", dtype=object)
    goal, next_goal = np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    for v in visits:
        state[v["i_arrive"] : v["i_leave"] + 1] = "at_station"
        station[v["i_arrive"] : v["i_leave"] + 1] = v["label"]
    if visits:
        state[visits[-1]["i_leave"] + 1 :] = "after_last_visit"
    upcoming, after = "", ""
    for i in range(n - 1, -1, -1):
        if station[i] and station[i] != upcoming:
            upcoming, after = station[i], upcoming
        goal[i], next_goal[i] = upcoming, after
        if state[i] == "after_last_visit":
            goal[i] = next_goal[i] = ""
    return {"state": state, "station": station, "goal": goal, "next_goal": next_goal}


def robot_labels(robot_goal: np.ndarray, stations: np.ndarray) -> dict:
    """Recorded robot goals as station ids, with the goal after each one."""
    d = np.linalg.norm(robot_goal[:, None] - stations[None], axis=-1)
    goal = np.where(d.min(1) < GOAL_TOL_M, d.argmin(1), -1)
    next_goal = np.full(len(goal), -1)
    after = -1
    for i in range(len(goal) - 2, -1, -1):
        if goal[i + 1] != goal[i]:
            after = goal[i + 1]
        next_goal[i] = after
    return {"goal": goal, "next_goal": next_goal}


def robot_check(goal: np.ndarray, time: np.ndarray, visits: list[dict]) -> dict:
    """Detector vs recorded robot goals.

    The recorded goal switches as soon as the robot is within ~0.5 m of it, and the
    robot turns around shortly after, so a visit matches a goal whose segment ended
    within (-ROBOT_LAG_S[0], ROBOT_LAG_S[1]) s before its turnaround. The final goal
    segment is never completed and is not counted.
    """
    changes = np.flatnonzero(np.diff(goal)) + 1
    completed = [(goal[c - 1], time[c]) for c in changes]
    def matches(v: dict, g: int, t_end: float) -> bool:
        return v["station"] == g and -ROBOT_LAG_S[0] <= v["t_turn"] - t_end <= ROBOT_LAG_S[1]
    found = sum(any(matches(v, g, t) for v in visits) for g, t in completed)
    matched = sum(any(matches(v, g, t) for g, t in completed) for v in visits)
    return {"completed_goals": len(completed), "goals_found": found, "detected_visits": len(visits), "visits_matching_a_goal": matched}


def process(job: tuple[str, Path, Path]) -> dict:
    site, path, out = job
    trial, controller = path.parent.name, path.stem
    stations = np.array(STATIONS[site])
    data = load(path)
    time, n = data["time"], len(data["time"])
    rows, visit_rows = [], []
    tracks = [("robot", data["robot"])] + [(f"human{h}", data["humans"][:, h]) for h in range(data["humans"].shape[1])]
    robot_visits = None
    for agent, track in tracks:
        visits = detect_visits(track[:, :2], time, stations, SHARED_TABLES[site] if agent != "robot" else ())
        if agent == "robot":
            robot_visits = visits
            labels = {k: np.array([str(g) if g >= 0 else "" for g in ids], dtype=object)
                      for k, ids in robot_labels(data["robot_goal"], stations).items()}
            labels["state"] = np.full(n, "", dtype=object)
            labels["station"] = np.full(n, "", dtype=object)
        else:
            labels = label_from_visits(n, visits)
        for k, v in enumerate(visits):
            visit_rows.append({"agent": agent, "visit": k, "station": v["station"], "label": v["label"], "t_arrive": round(v["t_arrive"], 3),
                               "t_turn": round(v["t_turn"], 3), "t_leave": round(v["t_leave"], 3), "min_dist": round(v["min_dist"], 3),
                               "margin": round(v["margin"], 3), "inferred": v["inferred"], "corrected": v["corrected"],
                               "phase": v.get("phase", ""), "phase_share": v.get("phase_share", "")})
        for i in range(n):
            rows.append({"step": i, "time": round(time[i], 3), "agent": agent,
                         "x": round(track[i, 0], 4), "y": round(track[i, 1], 4), "theta": round(track[i, 2], 4),
                         "state": labels["state"][i], "station": labels["station"][i],
                         "goal": labels["goal"][i], "next_goal": labels["next_goal"][i]})
    dest = out / site / trial
    dest.mkdir(parents=True, exist_ok=True)
    for name, table in [("frames", rows), ("visits", visit_rows)]:
        with open(dest / f"{controller}_{name}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    check = robot_check(robot_labels(data["robot_goal"], stations)["goal"], time, robot_visits)
    human_visits = [sum(r["agent"] == f"human{h}" for r in visit_rows) for h in range(data["humans"].shape[1])]
    return {"site": site, "trial": trial, "controller": controller, "duration_s": round(time[-1] - time[0], 1),
            **check, "human0_visits": human_visits[0], "human1_visits": human_visits[1]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only", nargs="*", help="limit to <site>/<trial>/<controller> or <site>/<trial>, e.g. laas/laas1/cohan")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    jobs = []
    for site in STATIONS:
        for path in sorted((args.data / "jsons" / site).glob("*/*.json")):
            key = f"{site}/{path.parent.name}/{path.stem}"
            if args.only and not any(key == o or key.startswith(o.rstrip("/") + "/") for o in args.only):
                continue
            jobs.append((site, path, args.out))
    args.out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for site, xys in STATIONS.items():
        labels[site] = {str(k): {"members": [str(k)], "xy": list(xy)} for k, xy in enumerate(xys)}
        for pair in SHARED_TABLES[site]:
            labels[site][shared_label(pair)] = {"members": [str(k) for k in sorted(pair)],
                                                "xy": np.mean([xys[k] for k in pair], axis=0).round(3).tolist()}
    (args.out / "stations.json").write_text(json.dumps(labels, indent=2) + "\n")
    with Pool(args.workers) as pool:
        summary = pool.map(process, jobs)
    with open(args.out / "robot_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    segs = sum(s["completed_goals"] for s in summary)
    found = sum(s["goals_found"] for s in summary)
    det = sum(s["detected_visits"] for s in summary)
    ok = sum(s["visits_matching_a_goal"] for s in summary)
    params = {k: v for k, v in globals().items() if k.isupper() and k not in ("ROOT", "DATA", "OUT", "DERIVED", "GOALS_DIR")}
    write_manifest(args.out / "manifest.json", Path(__file__), params, {
        "files": len(summary), "only": args.only,
        "robot_check": {"completed_goals": segs, "goals_found": found, "detected_visits": det, "visits_matching_a_goal": ok},
        "columns": {
            "frames": "step, time (s, mocap clock), agent (robot/human0/human1), x, y, theta, state (at_station/transit/after_last_visit; "
                      "empty for robot), station (label while at one), goal (station being visited or headed to), next_goal (the one after); "
                      "labels are text keys of stations.json, empty when none",
            "visits": "agent, visit, station (best single station id), label (goal label, may be a pair), t_arrive/t_turn/t_leave (s), "
                      "min_dist (m to labelled station), margin (m, nearest vs second-nearest before correction), inferred, corrected, "
                      "phase (0/1, people only), phase_share (majority share of the shared-table vote in that phase)",
        },
    })
    print(f"{len(summary)} files -> {args.out}")
    print(f"robot check: found {found}/{segs} recorded goals ({found / segs:.0%}); {ok}/{det} detected visits match a goal ({ok / max(det, 1):.0%})")


if __name__ == "__main__":
    main()
