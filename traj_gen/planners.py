"""Planners: turn a model reply into the planned agent's positions, one per time step from where it is now.

A planner is made per episode from an experiments.Experiment. Model planners (QueryPlanner
subclasses) also open the chats and render the images for the queries. Each subclass is one
output mode: its class attributes (name, prompt template, reply key, example, units) and
methods (to_units, max_steps, positions) say how it is prompted and how a reply becomes positions:

- waypoints (metres): the speed limit keeps the path's shape and stretches its timing.
- pixels: the same after mapping the BEV pixels to metres.
- velocities: each clipped to the speed cap, then integrated.
- StraightPlanner, ReplayPlanner: no model, so the baselines go through the same episode,
  metric and saving code as the model conditions.

positions() also returns the raw per-step speeds of the reply (before limiting) for the
speed-violation metric; None for the baselines, which do not have a reply. The speed cap is the
planned agent's (trip["agent"]).
"""

from functools import cached_property

import numpy as np

from traj_gen.agents import SPEED_CAP_M_S, agent_kind
from traj_gen.bev import bev_transform, render_bev
from traj_gen.camera import camera_frame
from traj_gen.client import Chat, QwenClient
from traj_gen.episodes import real_path
from traj_gen.prompting import PromptBuilder
from traj_gen.scene import DT, Metres, Pixels, Scene


THINKING = ("nothink", "think")  # run option of the model conditions: Qwen thinking off, on
MAX_PLAN_STEPS = 75  # 30 s: about 3x the straight-line time across the room's diagonal (4.9 m) at the speed cap; a longer plan could not be scored in a replay anyway
METRES, PIXELS = Metres(), Pixels(bev_transform())


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
    units = None

    def __init__(self, exp, trip: dict, framing: str | None, thinking: str | None):
        self.exp, self.trip, self.framing, self.thinking = exp, trip, framing, thinking
        self.kind = agent_kind(trip["agent"])
        self.speed_cap = SPEED_CAP_M_S[self.kind]
        self.step_m = self.speed_cap * DT

    def positions(self, value: np.ndarray | None, scene: Scene) -> tuple[np.ndarray, np.ndarray | None]:
        """(n, 2) positions of the planned agent in metres for steps 1..n after the scene's time, and the raw speeds (n_reply,) in m/s."""
        raise NotImplementedError


class QueryPlanner(Planner):
    """Planners that ask the model. A subclass sets name, template (prompt template), key (JSON key of the reply),
    example (metres, converted by to_units, per planned-agent kind) and overrides units, pixel_ticks, to_units and max_steps where its mode differs."""

    units = METRES
    pixel_ticks = False  # BEV grid labelled in metres

    def to_units(self, value) -> np.ndarray:
        """An example position in metres, written in the units of the output mode."""
        return self.units.pos(value)

    def max_steps(self, remaining: int) -> int:
        """Longest plan asked for: the rest of the human replay, at most MAX_PLAN_STEPS."""
        return min(MAX_PLAN_STEPS, remaining)

    def chat(self, scene: Scene) -> Chat:
        """A new conversation for this scene's remaining horizon (the builder fixes the longest plan allowed)."""
        builder = PromptBuilder(self.framing, self, self.exp.images, self.max_steps(len(self.trip["time"]) - len(scene.track)), self.exp.predict, self.kind)
        return QwenClient(builder, self.thinking == "think").chat(builder.system(scene))

    @cached_property
    def cam(self) -> bytes:
        return camera_frame(self.trip)  # the frame is the same for every query of the episode

    def images(self, scene: Scene, kinds: list[str]) -> dict[str, bytes]:
        return {k: render_bev(scene, self.pixel_ticks, self.exp.predict)[0] if k == "bev" else self.cam for k in kinds}


class WaypointPlanner(QueryPlanner):
    """Positions in metres; the speed limit retimes the path."""

    name = "waypoints"
    template = "output_waypoints"
    key = "waypoints"  # key of the JSON object the model replies with: {key: [[a, b], ...]}
    # Example numbers in metres (converted to the units in use), in an arbitrary place so they do not hint at where to go:
    # for the robot a short slow path, ~0.1 m per step (0.25 m/s), for a person ~0.4 m per step (1.0 m/s).
    example = {"robot": [(-0.50, 0.40), (-0.40, 0.45), (-0.30, 0.50)], "human": [(-0.50, 0.40), (-0.12, 0.52), (0.26, 0.64)]}

    def positions(self, value, scene):
        return retime(scene.track[-1], value, self.step_m)


class PixelPlanner(WaypointPlanner):
    """Points on the BEV in pixels: mapped to metres, then as waypoints."""

    name = "pixels"
    template = "output_pixels"
    key = "points"
    units = PIXELS
    pixel_ticks = True

    def positions(self, value, scene):
        return super().positions(self.units.transform.to_metre(value), scene)


class VelocityPlanner(QueryPlanner):
    """One velocity per step in m/s, each scaled down to the speed cap and integrated."""

    name = "velocity_sequence"
    template = "output_velocity_seq"
    key = "velocities"
    example = {"robot": [(0.20, 0.10), (0.20, 0.15), (0.15, 0.20)]}  # m/s, below the cap

    def to_units(self, value) -> np.ndarray:
        return self.units.vec(value)

    def positions(self, value, scene):
        speed = np.linalg.norm(value, axis=1)
        clipped = value * (self.speed_cap / np.maximum(speed, self.speed_cap))[:, None]  # factor 1 at or below the cap
        return scene.track[-1] + np.cumsum(clipped * DT, axis=0), speed


class StepVelocityPlanner(VelocityPlanner):
    """A single velocity for the next step."""

    name = "velocity"
    template = "output_velocity"
    example = {"robot": [(0.20, 0.10)]}

    def max_steps(self, remaining: int) -> int:
        return 1


class StraightPlanner(Planner):
    """Baseline: straight line to the goal at the speed cap."""

    def positions(self, value, scene):
        return retime(scene.track[-1], scene.goal_xy[None], self.step_m)[0], None


class ReplayPlanner(Planner):
    """Baseline: the planned agent's real path, as recorded."""

    def positions(self, value, scene):
        return real_path(self.trip)[1:], None  # the scene is at start_index, so the real path after it
