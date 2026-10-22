"""Planners: turn a model reply into the robot's positions, one per time step from where it is now.

A planner is made per episode from an experiments.Experiment. Model planners (QueryPlanner
subclasses) also open the chats and render the images for the queries; the output mode decides
how a reply becomes positions:

- waypoints (metres): the speed limit keeps the path's shape and stretches its timing.
- pixels: the same after mapping the BEV pixels to metres.
- velocities: each clipped to the speed cap, then integrated.
- StraightPlanner, ReplayPlanner: no model, so the baselines go through the same episode,
  metric and saving code as the model conditions.

positions() also returns the raw per-step speeds of the reply (before limiting) for the
speed-violation metric; None for the baselines, which do not have a reply.
"""

from functools import cached_property

import numpy as np

from bev import render_bev
from camera import camera_frame
from client import Chat, QwenClient
from scene import DT, Scene
from traj_gen_metrics import SPEED_CAP


THINKING = ("nothink", "think")  # run option of the model conditions: Qwen thinking off, on
MAX_PLAN_STEPS = 75  # 30 s: about 3x the straight-line time across the room's diagonal (4.9 m) at the speed cap; a longer plan could not be scored in a replay anyway
STEP_M = SPEED_CAP * DT  # most the robot moves in one step


class Planner:
    """Base: what an episode needs from a planner."""

    def __init__(self, exp, trip: dict, framing: str | None, thinking: str | None):
        self.exp, self.trip, self.framing, self.thinking = exp, trip, framing, thinking

    def positions(self, value: np.ndarray | None, scene: Scene) -> tuple[np.ndarray, np.ndarray | None]:
        """(n, 2) robot positions in metres for steps 1..n after the scene's time, and the raw speeds (n_reply,) in m/s."""
        raise NotImplementedError


class QueryPlanner(Planner):
    """Planners that ask the model: the output mode (prompt builder, units) comes from the experiment."""

    pixel_ticks = False  # BEV grid labelled in metres

    def max_steps(self, remaining: int) -> int:
        """Longest plan asked for: the rest of the human replay, at most MAX_PLAN_STEPS."""
        return min(MAX_PLAN_STEPS, remaining)

    def chat(self, scene: Scene) -> Chat:
        """A new conversation for this scene's remaining horizon (the builder fixes the longest plan allowed)."""
        builder = self.exp.prompt(self.framing, self.exp.units, self.exp.images, self.max_steps(len(self.trip["time"]) - len(scene.robot)), self.exp.predict)
        return QwenClient(builder, self.thinking == "think").chat(builder.system(scene))

    @cached_property
    def cam(self) -> bytes:
        return camera_frame(self.trip)  # the frame is the same for every query of the episode

    def images(self, scene: Scene, kinds: list[str]) -> dict[str, bytes]:
        return {k: render_bev(scene, self.pixel_ticks, self.exp.predict)[0] if k == "bev" else self.cam for k in kinds}


class WaypointPlanner(QueryPlanner):
    """Positions in metres; the speed limit retimes the path."""

    def positions(self, value, scene):
        points = np.vstack([scene.robot[-1], value])
        seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])  # arclength of each plan point
        reached = [0.0]  # arclength the robot has actually covered at each step: at most STEP_M more than the step before, never ahead of the plan
        for target in arc[1:]:
            reached.append(min(target, reached[-1] + STEP_M))
        while reached[-1] < arc[-1]:  # the plan has ended but the robot is behind it
            reached.append(min(arc[-1], reached[-1] + STEP_M))
        s = np.array(reached[1:])
        return np.stack([np.interp(s, arc, points[:, 0]), np.interp(s, arc, points[:, 1])], axis=1), seg / DT


class PixelPlanner(WaypointPlanner):
    """Points on the BEV in pixels: mapped to metres, then as waypoints."""

    pixel_ticks = True

    def positions(self, value, scene):
        return super().positions(self.exp.units.transform.to_metre(value), scene)


class VelocityPlanner(QueryPlanner):
    """One velocity per step in m/s, each scaled down to the speed cap and integrated."""

    def positions(self, value, scene):
        speed = np.linalg.norm(value, axis=1)
        clipped = value * (SPEED_CAP / np.maximum(speed, SPEED_CAP))[:, None]  # factor 1 at or below the cap
        return scene.robot[-1] + np.cumsum(clipped * DT, axis=0), speed


class StepVelocityPlanner(VelocityPlanner):
    """A single velocity for the next step."""

    def max_steps(self, remaining: int) -> int:
        return 1


class StraightPlanner(Planner):
    """Baseline: straight line to the goal at the speed cap."""

    def positions(self, value, scene):
        to_goal = scene.goal_xy - scene.robot[-1]
        dist = np.linalg.norm(to_goal)
        along = np.minimum(np.arange(1, int(np.ceil(dist / STEP_M)) + 1) * STEP_M, dist)
        return scene.robot[-1] + along[:, None] * to_goal / dist, None


class ReplayPlanner(Planner):
    """Baseline: the real controller's path, as recorded."""

    def positions(self, value, scene):
        i = len(scene.robot)  # first index after the scene
        return np.asarray(self.trip["robot"], float)[i: self.trip["end_index"] + 1, :2], None
