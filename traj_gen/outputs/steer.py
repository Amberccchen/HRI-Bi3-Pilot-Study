"""The steer output mode: steps [turn, distance] from the current heading, in the run's units, with the harness prompts."""

import numpy as np

from traj_gen.observation.scene import DT, Scene
from traj_gen.outputs.planner import QueryPlanner


class SteerPlanner(QueryPlanner):
    """The steer mode: steps [turn in degrees, distance in the units], with the harness prompts (the scored contact and personal-space
    distances, waiting allowed, the scored body sizes, the current heading each turn) and a larger thinking budget. The heading starts as the
    direction of the agent's last movement; each step first turns by its turn (positive: from the units' first axis toward the second,
    clockwise on the map in pixels, anticlockwise in metres), cut to the agent's largest turn, then moves its distance, cut to [0, step].
    The robot's prompt states the largest distance per step; a person's gives only the top speed (with the stated maximum, people asked for exactly it in 75-82% of steps, run 20261117_human_m4a_f2b3123)."""

    name = "steer"
    template = "steer"
    key = "steps"
    harness = True
    system_template = "harness"
    followup_template = "harness"
    scored_sizes = True
    think_max_tokens = 32768  # most harness calls ran past 16384 tokens (robot test runs, 2026-11-16)
    # Largest turn per 0.4 s step. Robot: the real robots' largest body yaw change, 35.8 deg at LAAS and 34.4 at UM (trips_v2, 144 robot trips).
    # Person: the 99th percentile of the change in walking direction between consecutive steps both above 0.2 m/s, 115.6 deg over the
    # trips_v2_human goal-to-goal moves (14292 trips; the median is 9 deg), the same percentile as the person's speed cap.
    MAX_TURN_DEG = {"robot": 35.0, "human": 115.0}
    # Example steps [turn in degrees, distance in metres] at about each kind's walking or driving scale: robot 0.10-0.11 m (near its 0.12 m
    # limit per step), person 0.4 m (about 1.0 m/s). The distances are written in the units.
    example = {"robot": [(0, 0.10), (-15, 0.11), (0, 0.0)], "human": [(0, 0.40), (-15, 0.40), (0, 0.0)]}
    HEADING_MIN_MOVE_M = 0.005  # the heading is the direction of the last movement at least this long

    @property
    def max_turn_deg(self) -> float:
        return self.MAX_TURN_DEG[self.kind]

    def example_text(self) -> str:
        return ", ".join(f"[{turn:g}, {self.units.len_text(d)}]" for turn, d in self.example[self.kind])

    def output_fields(self) -> dict:
        return {"max_turn": f"{self.max_turn_deg:g}"}

    def heading(self, scene: Scene) -> float:
        """Direction of the agent's last movement of at least HEADING_MIN_MOVE_M, in degrees in the units' frame (0 = along the first axis,
        90 = along the second); 0 if it has not moved."""
        moves = np.diff(self.units.pos(scene.track), axis=0)
        moved = np.flatnonzero(np.linalg.norm(np.diff(scene.track, axis=0), axis=1) >= self.HEADING_MIN_MOVE_M)
        return float(np.degrees(np.arctan2(*moves[moved[-1]][::-1]))) if moved.size else 0.0

    def extra_text(self, scene: Scene) -> str:
        future = super().extra_text(scene)
        who, whose = ("Robot heading", "its") if self.kind == "robot" else ("Your heading", "your")
        (c0, c1), (d0, d1) = self.units.axes, self.units.directions
        return (future + "\n\n" if future else "") + f"{who} now: {self.heading(scene):.0f} degrees (0 = +{c0}, {d0}; 90 = +{c1}, {d1}), the direction of {whose} last movement."

    def positions(self, value, scene):
        turn = np.clip(value[:, 0], -self.max_turn_deg, self.max_turn_deg)
        theta = np.radians(self.heading(scene) + np.cumsum(turn))
        d = np.clip(value[:, 1], 0.0, self.units.length(self.step_m))
        uv = self.units.pos(scene.track[-1]) + np.cumsum(d[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1), axis=0)
        return self.units.pos_to_metres(uv), self.units.length_to_metres(np.abs(value[:, 1])) / DT


class SteerVelocityPlanner(SteerPlanner):
    """The steer_vel mode: steps [turn rate in degrees per second, speed in the units per second], each held for the step, so the agent
    follows a circular arc of radius speed / turn rate (the unicycle model), straight at a turn rate of 0. The heading starts as steer's and
    continues from each step's end. The turn rate is cut to the kind's steer turn cap per step divided by the step (MAX_TURN_DEG / DT), the
    speed to [0, the kind's speed cap]. Same harness prompts, system and follow-up texts as steer."""

    name = "steer_velocity"
    template = "steer_vel"
    # Example steps [turn rate in deg/s, speed in m/s] at about each kind's walking or driving speed. The speeds are written in the units.
    example = {"robot": [(0, 0.25), (-40, 0.27), (0, 0.0)], "human": [(0, 1.0), (-40, 1.0), (0, 0.0)]}

    @property
    def max_turn_rate_deg_s(self) -> float:
        return self.max_turn_deg / DT

    def output_fields(self) -> dict:
        return {"max_turn_rate": f"{self.max_turn_rate_deg_s:g}"}

    def positions(self, value, scene):
        omega = np.radians(np.clip(value[:, 0], -self.max_turn_rate_deg_s, self.max_turn_rate_deg_s))
        speed = np.clip(value[:, 1], 0.0, self.units.length(self.speed_cap))
        theta0 = np.radians(self.heading(scene)) + np.concatenate([[0.0], np.cumsum(omega * DT)[:-1]])  # heading at each step's start
        theta1 = theta0 + omega * DT
        turning = omega != 0.0
        # Arc of radius speed / omega: the chord is (speed / omega) (sin theta1 - sin theta0, cos theta0 - cos theta1); straight when omega = 0.
        radius = np.divide(speed, omega, out=np.zeros_like(speed), where=turning)
        chord = np.where(turning[:, None], radius[:, None] * np.stack([np.sin(theta1) - np.sin(theta0), np.cos(theta0) - np.cos(theta1)], axis=1),
                         (speed * DT)[:, None] * np.stack([np.cos(theta0), np.sin(theta0)], axis=1))
        uv = self.units.pos(scene.track[-1]) + np.cumsum(chord, axis=0)
        return self.units.pos_to_metres(uv), self.units.length_to_metres(np.abs(value[:, 1]))
