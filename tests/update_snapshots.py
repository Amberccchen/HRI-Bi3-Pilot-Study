"""Regenerate tests/snapshots/<preset>.json for every harness preset: python -m tests.update_snapshots. Run it only after an intended
change to a preset or its prompts, and review the diff before committing."""

import json

from tests.test_snapshots import SNAPSHOTS, render
from traj_gen.harnesses import PRESETS

if __name__ == "__main__":
    SNAPSHOTS.mkdir(exist_ok=True)
    for stale in set(p.stem for p in SNAPSHOTS.glob("*.json")) - set(PRESETS):
        (SNAPSHOTS / f"{stale}.json").unlink()
        print(f"removed {stale}.json")
    for name in PRESETS:
        (SNAPSHOTS / f"{name}.json").write_text(json.dumps(render(name), indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {name}.json")
