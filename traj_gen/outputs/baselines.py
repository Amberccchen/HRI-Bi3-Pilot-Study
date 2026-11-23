"""The baselines without a model: a straight line to the goal, and the recorded path."""

import numpy as np

from traj_gen.agents import STRAIGHT_LINE_SPEED_M_S
from traj_gen.observation.scene import DT
from traj_gen.outputs.planner import Planner, retime


def real_path(trip: dict) -> np.ndarray:
    """(n, 2) positions of the trip's planned agent, as recorded, from the trip's start to its end, in the room frame."""
    return np.asarray(trip["tracks"][trip["agent"]], float)[trip["start_index"]: trip["end_index"] + 1, :2]


class StraightPlanner(Planner):
    """Baseline: straight line to the goal at the planned agent's STRAIGHT_LINE_SPEED_M_S."""

    def positions(self, value, scene):
        return retime(scene.track[-1], scene.goal_xy[None], STRAIGHT_LINE_SPEED_M_S[self.kind] * DT)[0], None


class ReplayPlanner(Planner):
    """Baseline: the planned agent's real path, as recorded."""

    def positions(self, value, scene):
        return real_path(self.trip)[1:], None  # the scene is at start_index, so the real path after it
