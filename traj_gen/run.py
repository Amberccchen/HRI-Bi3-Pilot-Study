"""Command line for the Bi3 VLM trajectory generation experiments (design: PLAN.md).

  select   choose the 16 test trips and write <run>/test_trips.json
  run      run experiments (resumable; one folder per episode, see episodes.py)

Run folders live outside the repository (they hold mocap-derived outputs):
/persistent/experiments/HRI/bi3_traj_gen/<run>.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "metrics"))
sys.path.insert(0, str(ROOT / "traj_gen"))
from bi3_derived import load_site_frames, load_trips  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402
from planners import THINKING  # noqa: E402
from prompting import FRAMINGS  # noqa: E402


RUN = Path("/persistent/experiments/HRI/bi3_traj_gen/20261022_v1")
SITES = ["laas", "um"]
N_CLOSE = 5          # per site: smallest real min_distance among short trips (where social behaviour shows)
N_TERCILES = 3       # per site: one trip from each duration tercile of the rest (easy, medium, long/robot-waiting)
MAX_SHORT_S = 25.0   # "short" for the close-encounter stratum; longer trips are mostly the robot waiting
MIN_POST_S = 10.0    # human replay left after the real trip ends: trips at the end of a recording leave a slower plan nothing to be scored against
MAX_PER_CONTROLLER = 2  # per site, so no controller dominates the test set
CONCURRENCY = 32     # episodes in parallel: the server gives ~780 tok/s in total (up to 64 sequences), so 32 streams get ~24 tok/s each and more would only add waiting time; nothink episodes mostly wait on the server and use less
SEED = 20261022      # fixed so the tercile picks are reproducible; the run date, no other meaning


def select_trips(trips: list[dict], rng: np.random.Generator) -> list[dict]:
    """Close-encounter and duration-tercile picks for one site, trip 0 excluded (the robot starts from rest), as are trips without MIN_POST_S of replay after them."""
    pool = [t for t in trips if t["trip"] != 0 and t["time"][-1] - t["time"][t["end_index"]] >= MIN_POST_S]
    chosen, per_controller = [], Counter()

    def take(t: dict, stratum: str) -> bool:
        if per_controller[t["controller"]] >= MAX_PER_CONTROLLER:
            return False
        per_controller[t["controller"]] += 1
        chosen.append({"id": t["id"], "stratum": stratum})
        return True

    close = sorted((t for t in pool if t["duration_s"] <= MAX_SHORT_S), key=lambda t: t["real_metrics"]["min_distance"])
    n = 0
    for t in close:
        n += take(t, "close")
        if n == N_CLOSE:
            break
    taken = {c["id"] for c in chosen}
    rest = sorted((t for t in pool if t["id"] not in taken), key=lambda t: t["duration_s"])
    for k, group in enumerate(np.array_split(np.arange(len(rest)), N_TERCILES)):
        for i in rng.permutation(group):
            if take(rest[i], f"duration_tercile_{k + 1}"):
                break
    assert len(chosen) == N_CLOSE + N_TERCILES, f"could not fill the test set: {chosen}"
    return chosen


def select(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(SEED)
    by_id = {t["id"]: t for t in load_trips()}
    chosen = [c for site in SITES for c in select_trips([t for t in by_id.values() if t["site"] == site], rng)]
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / "test_trips.json").write_text(json.dumps(chosen, indent=2) + "\n")
    for c in chosen:
        t = by_id[c["id"]]
        print(f"{c['id']:32s} {c['stratum']:20s} {t['duration_s']:6.1f} s  min_distance {t['real_metrics']['min_distance']:.2f} m")


def run(args: argparse.Namespace) -> None:
    trips = {t["id"]: t for t in load_trips()}
    sites = load_site_frames()
    episodes = [exp.episode(args.run, exp, trips[c["id"]], sites[trips[c["id"]]["site"]], framing, thinking, k)
                for e in args.experiments for exp in [EXPERIMENTS[e]] for c in json.loads((args.trips or args.run / "test_trips.json").read_text())
                for framing, thinking, k in exp.episode.conditions(args.framings, args.thinking, args.samples)]
    todo = [ep for ep in episodes if not ep.done()]
    print(f"{len(todo)} episodes to run, {len(episodes) - len(todo)} already done; concurrency {args.concurrency}", flush=True)
    pool = ThreadPoolExecutor(args.concurrency)
    futures = {pool.submit(ep.go): ep for ep in todo}
    t0 = time.perf_counter()
    try:
        for n, future in enumerate(as_completed(futures), 1):
            ep = futures[future]
            future.result()  # go() catches the episode's own exceptions; anything raised here is a bug in saving
            print(f"[{n}/{len(todo)} {time.perf_counter() - t0:.0f}s] {ep.exp.id} {ep.framing} {ep.thinking} {ep.trip['id']} s{ep.sample}: {ep.status}", flush=True)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("interrupted; finished episodes are kept, rerun to resume", flush=True)
        os._exit(130)  # running threads sit in blocking HTTP calls that can take minutes; they would hold the process open


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RUN, help="run folder")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select", help="write test_trips.json").set_defaults(func=select)
    p = sub.add_parser("run", help="run experiments")
    p.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS), help="experiment ids (default: all)")
    p.add_argument("--framings", nargs="+", choices=FRAMINGS, default=list(FRAMINGS))
    p.add_argument("--thinking", nargs="+", choices=THINKING, default=["nothink"], help="thinking off first (PLAN); run think afterwards")
    p.add_argument("--samples", type=int, default=3, help="samples per trip and condition")
    p.add_argument("--trips", type=Path, help="trip list (default: <run>/test_trips.json)")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY, help="episodes in parallel")
    p.set_defaults(func=run)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
