"""The velocity output modes, robot only: a velocity sequence, or a single velocity for the next step."""

import numpy as np

from traj_gen.observation.scene import DT
from traj_gen.outputs.planner import QueryPlanner


class VelocityPlanner(QueryPlanner):
    """One velocity per step in m/s, each scaled down to the speed cap and integrated."""

    name = "velocity_sequence"
    template = "velseq"
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
    template = "vel1"
    example = {"robot": [(0.20, 0.10)]}

    def max_steps(self, remaining: int) -> int:
        return 1
