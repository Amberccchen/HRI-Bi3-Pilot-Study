"""The steer output mode: steps [turn, distance] on the BEV from the current heading, with the harness prompts."""

import numpy as np

from traj_gen.observation.scene import DT, Scene
from traj_gen.outputs.positions import PixelPlanner


class SteerPlanner(PixelPlanner):
    """The steer mode: steps [turn in degrees, distance in pixels] on the BEV, with the harness prompts (the scored contact and personal-space
    distances, waiting allowed, the scored body sizes, the current heading each turn) and a larger thinking budget. The heading starts as the
    direction of the agent's last movement; each step first turns by its turn (positive: clockwise on the map, from +u toward +v), cut to
    the agent's largest turn, then moves its distance, cut to [0, step]. The robot's prompt states the largest distance per step; a
    person's gives only the top speed (with the stated maximum, people asked for exactly it in 75-82% of steps, run 20261117_human_m4a_f2b3123)."""

    name = "steer"
    template = "output_steer"
    key = "steps"
    harness = True
    system_template = "system_harness"
    followup_template = "followup_harness"
    scored_sizes = True
    think_max_tokens = 32768  # most harness calls ran past 16384 tokens (robot test runs, 2026-11-16)
    # Largest turn per 0.4 s step. Robot: the real robots' largest body yaw change, 35.8 deg at LAAS and 34.4 at UM (trips_v2, 144 robot trips).
    # Person: the 99th percentile of the change in walking direction between consecutive steps both above 0.2 m/s, 115.6 deg over the
    # trips_v2_human goal-to-goal moves (14292 trips; the median is 9 deg), the same percentile as the person's speed cap.
    MAX_TURN_DEG = {"robot": 35.0, "human": 115.0}
    # Example steps at about each kind's walking or driving scale: robot 20-22 px (0.10-0.11 m, near its 24 px limit), person 80 px (0.4 m, about 1.0 m/s).
    EXAMPLE = {"robot": "[0, 20], [-15, 22], [0, 0]", "human": "[0, 80], [-15, 80], [0, 0]"}

    @property
    def max_turn_deg(self) -> float:
        return self.MAX_TURN_DEG[self.kind]

    def example_text(self) -> str:
        return self.EXAMPLE[self.kind]

    def output_fields(self) -> dict:
        return {"max_turn": f"{self.max_turn_deg:g}"}

    def heading(self, scene: Scene) -> float:
        """Direction of the agent's last movement of at least 1 px, in degrees on the map (0 = +u, 90 = +v); 0 if it has not moved."""
        uv = self.units.pos(scene.track)
        moves = np.diff(uv, axis=0)
        moved = np.flatnonzero(np.linalg.norm(moves, axis=1) >= 1.0)
        return float(np.degrees(np.arctan2(*moves[moved[-1]][::-1]))) if moved.size else 0.0

    def extra_text(self, scene: Scene) -> str:
        future = super().extra_text(scene)
        who, whose = ("Robot heading", "its") if self.kind == "robot" else ("Your heading", "your")
        return (future + "\n\n" if future else "") + f"{who} now: {self.heading(scene):.0f} degrees (0 = +u, right; 90 = +v, down), the direction of {whose} last movement."

    def positions(self, value, scene):
        step_px = self.units.length(self.step_m)
        turn = np.clip(value[:, 0], -self.max_turn_deg, self.max_turn_deg)
        theta = np.radians(self.heading(scene) + np.cumsum(turn))
        d = np.clip(value[:, 1], 0.0, step_px)
        uv = self.units.pos(scene.track[-1]) + np.cumsum(d[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1), axis=0)
        return self.units.transform.to_metre(uv), np.abs(value[:, 1]) / self.units.length(1.0) / DT
