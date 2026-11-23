"""Snapshot tests of the harness presets, one per preset and planned-agent kind (tests/snapshots/<preset>_<kind>.json): the rendered prompts, the positions a fixed reply executes to, its settings and
the planner's thinking budget, compared exactly. No images and no model. Overrides are covered by unit tests (tests/test_harness_presets.py), not snapshots.

After an intended change to a preset or its prompts, regenerate the snapshots with python -m tests.update_snapshots and review the diff.
"""

import difflib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_harness import robot_trip
from tests.test_human_prompt import SITE, person_trip
from traj_gen.harness.episodes import ConversationEpisode
from traj_gen.harness.presets import PRESETS
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import Scene

SNAPSHOTS = Path(__file__).parent / "snapshots"
FOLLOWUP_STEPS = 2  # the follow-up turn is rendered this many steps after the trip start (the test trips have 3 steps of replay left)
# A fixed reply per output mode and units: positions, [turn in degrees, distance] steps, or velocities.
REPLY = {("positions", "m"): [[0.1, -0.2], [0.2, 0.1], [0.3, 0.4]], ("positions", "px"): [[460.0, 420.0], [480.0, 440.0], [500.0, 470.0]],
         ("steer", "px"): [[0.0, 20.0], [-15.0, 22.0], [30.0, 10.0]], ("steer", "m"): [[0.0, 0.1], [-15.0, 0.11], [30.0, 0.05]]}


KINDS = ("human", "robot")
CASES = [f"{name}_{kind}" for name in PRESETS for kind in KINDS]  # snapshot names


def render(case: str) -> dict:
    """The snapshot of one preset for one planned-agent kind (case <preset>_<kind>), on the fixed person trip or robot trip of the tests."""
    name, kind = case.rsplit("_", 1)
    harness = PRESETS[name].harness(kind)
    exp = harness.experiment()
    trip = person_trip() if harness.kind == "human" else robot_trip()
    track, start = np.asarray(trip["tracks"][trip["agent"]], float)[:, :2], trip["start_index"]
    planner = exp.planner(exp, trip, harness.framing, "think")
    builder = PromptBuilder(harness.framing, planner, exp.images, planner.max_steps(len(trip["time"]) - start - 1), exp.predict)
    scene = Scene(trip, SITE, start, track[: start + 1])
    later = Scene(trip, SITE, start + FOLLOWUP_STEPS, track[: start + FOLLOWUP_STEPS + 1])
    reply = np.array(REPLY[harness.output, harness.units])
    snapshot = {"preset": name, "kind": kind, "id": harness.id, "settings": harness.settings(), "think_max_tokens": planner.think_max_tokens,
                "system": builder.system(scene), "first": builder.first(scene).text,
                "followup": builder.followup(later, exp.execute_s).text if exp.episode is ConversationEpisode else None,
                "reply": reply.tolist(), "positions_m": np.round(planner.positions(reply, scene)[0], 6).tolist()}
    return json.loads(json.dumps(snapshot))


def test_every_preset_has_a_snapshot_and_no_other_snapshot_exists():
    assert sorted(p.stem for p in SNAPSHOTS.glob("*.json")) == sorted(CASES), "run python -m tests.update_snapshots after adding or removing a preset"


@pytest.mark.parametrize("name", CASES)
def test_preset_matches_its_snapshot(name):
    stored, now = json.loads((SNAPSHOTS / f"{name}.json").read_text()), render(name)
    changed = [part for part in sorted(set(stored) | set(now)) if stored.get(part) != now.get(part)]
    if changed:
        diffs = []
        for part in changed:
            a, b = (x.splitlines() if isinstance(x, str) else json.dumps(x, indent=1).splitlines() for x in (stored.get(part), now.get(part)))
            diffs.append(f"{part}:\n" + "\n".join(list(difflib.unified_diff(a, b, "snapshot", "now", lineterm="", n=1))[2:20]))
        pytest.fail(f"preset and kind {name}: {', '.join(changed)} differ from tests/snapshots/{name}.json "
                    f"(if the change is intended, run python -m tests.update_snapshots)\n" + "\n".join(diffs))
