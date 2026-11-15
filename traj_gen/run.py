"""Command line for the Bi3 VLM trajectory generation experiments (design: PLAN.md).
Run from the repository root: python -m traj_gen.run --run <run folder> <command> ...

  select   choose the 16 test trips from --trips-dir and write <run>/test_trips.json and <run>/run.json
  run      run experiments (resumable; one folder per episode, see episodes.py)
  analyze  tables and figures of a (partial) run into <run>/analysis (see analyze.py)
  animate  animation.gif and timeline.png (colour = time) in given episode folders (see animate.py)

Run folders live outside the repository (they hold mocap-derived outputs):
/persistent/experiments/HRI/bi3_traj_gen/<run>.
"""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from preprocessing.bi3_derived import SITES, load_site_frames, load_trips, provenance
from traj_gen.analyze import OVERLAY, analyze
from traj_gen.animate import animate, load_episode, timeline
from traj_gen.episodes import RUN_CONFIG, TEST_TRIPS
from traj_gen.experiments import EXPERIMENTS
from traj_gen.planners import THINKING
from traj_gen.prompting import FRAMINGS
from traj_gen.scene import DT, PROMPT_HISTORY_STEPS


STEP_TOLERANCE_S = 2e-3  # trip times are stored to 1 ms
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


def load_run(run: Path, trip_list: Path) -> tuple[dict, list[dict], dict, dict]:
    """(run.json, trip list, trips by id, site frames) of a run folder: the trip list read from trip_list, and the trips and sites of the folder run.json names.

    Raises ValueError when a listed trip is missing, is not sampled at DT, or starts before the history the prompt reads.
    """
    config = json.loads((run / RUN_CONFIG).read_text())
    trips_dir = Path(config["trips_dir"])
    trips = {t["id"]: t for t in load_trips(trips_dir)}
    chosen = json.loads(trip_list.read_text())
    for c in chosen:
        if c["id"] not in trips:
            raise ValueError(f"trip {c['id']} of {trip_list} is not in {trips_dir / 'trips.json'}")
        trip = trips[c["id"]]
        steps = np.diff(trip["time"])
        if np.abs(steps - DT).max() > STEP_TOLERANCE_S:
            raise ValueError(f"trip {trip['id']} is not sampled at {DT} s: steps from {steps.min()} to {steps.max()} s")
        if trip["start_index"] < PROMPT_HISTORY_STEPS:
            raise ValueError(f"trip {trip['id']} starts at index {trip['start_index']}, but the prompt reads {PROMPT_HISTORY_STEPS} steps of history")
    return config, chosen, trips, load_site_frames(trips_dir)


def select(args: argparse.Namespace) -> None:
    for name in (TEST_TRIPS, RUN_CONFIG):
        if (args.run / name).exists():
            raise ValueError(f"{args.run / name} exists: select into a new run folder")
    rng = np.random.default_rng(SEED)
    by_id = {t["id"]: t for t in load_trips(args.trips_dir)}
    chosen = [c for site in SITES for c in select_trips([t for t in by_id.values() if t["site"] == site], rng)]
    args.run.mkdir(parents=True, exist_ok=True)
    (args.run / TEST_TRIPS).write_text(json.dumps(chosen, indent=2) + "\n")
    (args.run / RUN_CONFIG).write_text(json.dumps({"trips_dir": str(args.trips_dir), "git_commit": provenance()["git_commit"]}, indent=2) + "\n")
    for c in chosen:
        t = by_id[c["id"]]
        print(f"{c['id']:32s} {c['stratum']:20s} {t['duration_s']:6.1f} s  min_distance {t['real_metrics']['min_distance']:.2f} m")


def run(args: argparse.Namespace) -> None:
    if args.samples < 1 or args.concurrency < 1:
        raise ValueError(f"--samples and --concurrency must be at least 1, got {args.samples} and {args.concurrency}")
    config, chosen, trips, sites = load_run(args.run, args.trips or args.run / TEST_TRIPS)
    code = provenance()
    if code["git_dirty"]:
        raise ValueError("the code has uncommitted changes, so its commit does not identify it: commit before running episodes")
    if code["git_commit"] != config["git_commit"]:  # episodes of two code versions would mix in one run folder
        raise ValueError(f"{args.run} was started at commit {config['git_commit']} but the code is at {code['git_commit']}: start a new run folder")
    episodes = [exp.episode(args.run, exp, trips[c["id"]], sites[trips[c["id"]]["site"]], framing, thinking, k)
                for e in args.experiments for exp in [EXPERIMENTS[e]] for c in chosen
                for framing, thinking, k in exp.episode.conditions(args.framings, args.thinking, args.samples)]
    todo = [ep for ep in episodes if not ep.done()]
    print(f"{len(todo)} episodes to run, {len(episodes) - len(todo)} already done; concurrency {args.concurrency}", flush=True)
    pool = ThreadPoolExecutor(args.concurrency)
    futures = {pool.submit(ep.go): ep for ep in todo}
    t0 = time.perf_counter()
    try:
        for n, future in enumerate(as_completed(futures), 1):
            ep = futures[future]
            future.result()  # go() catches server and transport errors; anything raised here is a bug and stops the run
            print(f"[{n}/{len(todo)} {time.perf_counter() - t0:.0f}s] {ep.exp.id} {ep.framing} {ep.thinking} {ep.trip['id']} s{ep.sample}: {ep.status}", flush=True)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        print("interrupted; finished episodes are kept, rerun to resume", flush=True)
        os._exit(130)  # running threads sit in blocking HTTP calls that can take minutes; they would hold the process open


def analysis(args: argparse.Namespace) -> None:
    _, _, trips, sites = load_run(args.run, args.run / TEST_TRIPS)
    analyze(args.run, args.overlay, trips, sites)


def animate_episodes(args: argparse.Namespace) -> None:
    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")
    _, _, trips, sites = load_run(args.run, args.run / TEST_TRIPS)
    for folder in args.episodes:
        if not folder.resolve().is_relative_to(args.run.resolve()):
            raise ValueError(f"episode {folder} is not inside the run folder {args.run}")
        ep = load_episode(folder, trips, sites)
        print(animate(folder, ep, args.speed), timeline(folder, ep))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run folder")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help=f"write {TEST_TRIPS} and {RUN_CONFIG}")
    p.add_argument("--trips-dir", type=Path, required=True, help="folder with trips.json and site_frames.json (written by preprocessing.extract_trips)")
    p.set_defaults(func=select)
    p = sub.add_parser("run", help="run experiments")
    p.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS), help="experiment ids (default: all)")
    p.add_argument("--framings", nargs="+", choices=FRAMINGS, default=list(FRAMINGS))
    p.add_argument("--thinking", nargs="+", choices=THINKING, default=["nothink"], help="thinking off first (PLAN); run think afterwards")
    p.add_argument("--samples", type=int, default=3, help="samples per trip and condition")
    p.add_argument("--trips", type=Path, help=f"trip list (default: <run>/{TEST_TRIPS})")
    p.add_argument("--concurrency", type=int, default=CONCURRENCY, help="episodes in parallel")
    p.set_defaults(func=run)
    p = sub.add_parser("analyze", help="summary tables and figures")
    p.add_argument("--overlay", nargs="+", choices=list(EXPERIMENTS), default=OVERLAY, help="experiments shown in the per-trip overlay figures")
    p.set_defaults(func=analysis)
    p = sub.add_parser("animate", help="animation.gif and timeline.png for finished episodes")
    p.add_argument("episodes", nargs="+", type=Path, help="episode folders (each with a manifest.json)")
    p.add_argument("--speed", type=float, default=2.0, help="playback speed vs real time (2: a 12 s trip plays in 6 s)")
    p.set_defaults(func=animate_episodes)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
