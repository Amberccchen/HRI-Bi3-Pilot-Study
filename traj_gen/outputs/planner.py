"""The shared base of the output modes: turn a model reply into the planned agent's positions, one per time step from where it is now.

A planner is made per episode from a harness.experiments.Experiment. Model planners (QueryPlanner
subclasses) also build the prompts and render the images for the queries. Each subclass is one
output mode: its class attributes (name, prompt template, reply key, example) and
methods (to_units, max_steps, positions) say how it is prompted and how a reply becomes positions. The units (UNITS, the
experiment's units setting) are not part of the mode: every number of the prompts and the reply is written in them, and a reply is
converted back to metres through them.

- positions: points in the world frame; the speed limit keeps the path's shape and stretches its timing (outputs.positions).
- velocities: each clipped to the speed cap, then integrated (outputs.velocity).
- steer: steps [turn, distance] from the current heading, each turn and distance cut to the agent's limits, with
  the steer prompts (outputs.steer.SteerPlanner).
- steer_vel: steps [turn rate, speed], each held for the step along a circular arc, cut to the agent's limits, with the steer prompts
  (outputs.steer.SteerVelocityPlanner).
- StraightPlanner, ReplayPlanner: no model, so the baselines go through the same episode,
  metric and saving code as the model conditions (outputs.baselines).

positions() also returns the raw per-step speeds of the reply (before limiting) for the
speed-violation metric, or None for the baselines, which do not have a reply. The speed cap is the
planned agent's (trip["agent"]).
"""

import json
from functools import cached_property

import numpy as np

from traj_gen.agents import SPEED_CAP_M_S, agent_kind
from traj_gen.observation.bev import bev_transform, render_bev
from traj_gen.observation.camera import camera_frame
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import DT, Scene, person_name, track_motion
from traj_gen.observation.units import Metres, Pixels


THINKING = ("nothink", "think")  # run option of the model conditions: the model's thinking off, on
MAX_PLAN_STEPS = 75  # 30 s, about 3x the straight-line time across the room's diagonal (4.9 m) at the speed cap. A longer plan could not be scored in a replay anyway
UNITS = {"m": Metres(), "px": Pixels(bev_transform())}  # the units setting -> the coordinate system the model reads and writes
# How far ahead each turn lists the other agents' recorded positions (prediction=true_future), by the planned agent's kind. The plan check
# tests against the whole recorded futures.
TRUE_FUTURE_S = {"human": 4.0, "robot": 12.0}


def retime(start: np.ndarray, points: np.ndarray, step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Walk the path start -> points at most step_m per step, keeping its shape: (n, 2) positions and the plan's (len(points),) speeds in m/s."""
    points = np.vstack([start, points])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])  # arclength of each plan point
    reached = [0.0]  # arclength the agent has actually covered at each step: at most step_m more than the step before, never ahead of the plan
    for target in arc[1:]:
        reached.append(min(target, reached[-1] + step_m))
    while reached[-1] < arc[-1]:  # the plan has ended but the agent is behind it
        reached.append(min(arc[-1], reached[-1] + step_m))
    s = np.array(reached[1:])
    return np.stack([np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])], axis=1), seg / DT


class Planner:
    """Base: what an episode needs from a planner. name and units are None for the baselines, which have no output mode.
    speed_cap is the planned agent's (m/s), step_m the most it moves in one step."""

    name = None

    def __init__(self, exp, trip: dict, framing: str | None, thinking: str | None):
        self.exp, self.trip, self.framing, self.thinking = exp, trip, framing, thinking
        self.units = UNITS[exp.units] if exp.units else None
        self.kind = agent_kind(trip["agent"])
        self.speed_cap = SPEED_CAP_M_S[self.kind]
        self.step_m = self.speed_cap * DT

    def positions(self, value: np.ndarray | None, scene: Scene) -> tuple[np.ndarray, np.ndarray | None]:
        """(n, 2) positions of the planned agent in metres for steps 1..n after the scene's time, and the raw speeds (n_reply,) in m/s."""
        raise NotImplementedError

    def motion(self, value: np.ndarray | None, scene: Scene, track: np.ndarray) -> tuple[float, float]:
        """The heading (radians in the room frame) and speed (m/s) at the end of track, the planned agent's track after executing steps of this
        reply from the scene. Straight steps, so read from the track (track_motion); the modes that plan a heading override it."""
        return track_motion(track)

    def plan_record(self, value) -> dict:
        """The mode's fields of a plans.jsonl row for a reply (None for the baselines), after its positions() was computed."""
        return {"reply": None if value is None else value.tolist()}

    def manifest_fields(self) -> dict:
        """The mode's fields of the episode manifest beyond the shared ones."""
        return {}


class QueryPlanner(Planner):
    """Planners that ask the model. A subclass sets name, template (prompt template), key (JSON key of the reply),
    example (metres, converted by to_units, per planned-agent kind) and overrides to_units, example_text and max_steps where its mode differs.
    The reply of these modes is a JSON object holding an array of number pairs under key (parse)."""

    scored_distance_framings = False  # whether the mode goes with the scored-distance framings (observation.prompting.SCORED_DISTANCE_FRAMINGS), and only with them
    system_template = "default"
    followup_template = "default"  # prompts/followup/<name>.txt
    reply_template = "default"     # prompts/reply/<name>.txt
    scored_sizes = False  # the text gives the scored body radii instead of the drawn (enclosing) ones
    # Qwen thinking budget with thinking on: thinking and answer in one call. Medium effort needs 1-6k tokens, text-only waypoints about 8k
    # (8 of 12 are forced at 8192). Beyond this the answer is forced
    think_max_tokens = 16384

    def to_units(self, value) -> np.ndarray:
        """An example position in metres, written in the units."""
        return self.units.pos(value)

    def example_text(self) -> str:
        """The example pairs of the reply format, written in the units."""
        return ", ".join("[" + ", ".join(self.units.num(c) for c in self.to_units(p)) + "]" for p in self.example[self.kind])

    def output_fields(self) -> dict:
        """Fields of the output template beyond the shared ones."""
        return {}

    def reply_fields(self) -> dict:
        """Fields of the reply template: the JSON key and the example pairs."""
        return {"key": self.key, "example": self.example_text()}

    def extra_text(self, scene: Scene) -> str:
        """Text after the state: the other agents' recorded future positions for the next TRUE_FUTURE_S of the planned agent's kind when the
        experiment gives them, else nothing."""
        if not self.exp.true_future:
            return ""
        i, steps = len(scene.track) - 1, round(TRUE_FUTURE_S[self.kind] / DT)
        if self.kind == "robot":
            lines = ["Where each person will actually be (recorded in the study; they walk exactly this way whatever the robot does):"]
        else:  # a planned person is also given the robot's recorded future
            lines = ["Where the other participant and the robot will actually be (recorded in the study; they move exactly this way whatever you do):"]
        for other, track in self.trip["tracks"].items():
            if other == self.trip["agent"] or (self.kind == "robot" and agent_kind(other) != "human"):
                continue
            future = np.asarray(track, float)[i + 1: i + 1 + steps, :2]
            points = ", ".join(f"t+{DT * (k + 1):.1f} s {self.units.pos_text(xy)}" for k, xy in enumerate(future))
            lines.append(f"{'Robot' if agent_kind(other) == 'robot' else person_name(other)}: {points}.")
        return "\n".join(lines)

    def closing_text(self, scene: Scene) -> str:
        """Text that closes every user turn of the scene (the first turn and the follow-ups), "" for none."""
        return ""

    @property
    def prefill(self) -> str:
        """The start of a reply, to continue from when the thinking is cut off."""
        return f'{{"{self.key}": [['

    def parse(self, text: str, max_steps: int) -> np.ndarray | None:
        """The (k, 2) array under key of the JSON object in the reply, or None if the reply is not a valid plan of at most max_steps steps."""
        try:
            plan = np.atleast_2d(np.array(json.loads(text[text.index("{"): text.rindex("}") + 1])[self.key], float))  # atleast_2d: the model sometimes drops the inner brackets of a single pair
        except (ValueError, KeyError, TypeError):  # no braces, bad JSON, missing key, or entries that are not numbers
            return None
        ok = plan.ndim == 2 and plan.shape[1] == 2 and 1 <= len(plan) <= max_steps and np.isfinite(plan).all()
        return plan if ok else None

    def max_steps(self, remaining: int) -> int:
        """Longest plan asked for: the rest of the human replay, at most MAX_PLAN_STEPS."""
        return min(MAX_PLAN_STEPS, remaining)

    def prompt_builder(self, scene: Scene) -> PromptBuilder:
        """The prompts of a new conversation for this scene's remaining horizon (the builder fixes the longest plan allowed)."""
        return PromptBuilder(self.framing, self, self.exp.images, self.max_steps(len(self.trip["time"]) - len(scene.track)), self.exp.predict)

    @cached_property
    def cam(self) -> bytes:
        return camera_frame(self.trip)  # the frame is the same for every query of the episode

    def images(self, scene: Scene, kinds: list[str]) -> dict[str, bytes]:
        return {k: render_bev(scene, isinstance(self.units, Pixels), self.exp.predict)[0] if k == "bev" else self.cam for k in kinds}
