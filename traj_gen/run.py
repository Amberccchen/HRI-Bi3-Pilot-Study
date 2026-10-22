"""Command line for the Bi3 VLM trajectory generation experiments (design: PLAN.md).

  select   choose the 16 test trips and write <run>/test_trips.json

Run folders live outside the repository (they hold mocap-derived outputs):
/persistent/experiments/HRI/bi3_traj_gen/<run>.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "metrics"))
from bi3_derived import load_trips  # noqa: E402


RUN = Path("/persistent/experiments/HRI/bi3_traj_gen/20261022_v1")
SITES = ["laas", "um"]
N_CLOSE = 5          # per site: smallest real min_distance among short trips (where social behaviour shows)
N_TERCILES = 3       # per site: one trip from each duration tercile of the rest (easy, medium, long/robot-waiting)
MAX_SHORT_S = 25.0   # "short" for the close-encounter stratum; longer trips are mostly the robot waiting
MAX_PER_CONTROLLER = 2  # per site, so no controller dominates the test set
SEED = 20261022      # fixed so the tercile picks are reproducible; the run date, no other meaning


def select_trips(trips: list[dict], rng: np.random.Generator) -> list[dict]:
    """Close-encounter and duration-tercile picks for one site, trip 0 excluded (the robot starts from rest)."""
    pool = [t for t in trips if t["trip"] != 0]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RUN, help="run folder")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("select", help="write test_trips.json").set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
