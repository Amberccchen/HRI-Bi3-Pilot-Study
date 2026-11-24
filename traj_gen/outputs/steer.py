"""The steer output mode: steps [turn, distance] from the current heading, in the run's units, with the steer prompts."""

import numpy as np

from traj_gen.observation.scene import DT, Scene
from traj_gen.outputs.planner import QueryPlanner


class SteerPlanner(QueryPlanner):
    """The steer mode: steps [turn in degrees, distance in the units], with the steer prompts (the scored contact and personal-space
    distances, waiting allowed, the scored body sizes) and a larger thinking budget. The heading starts as the scene's heading (the one the
    state gives: where the last executed step ended, or the direction of the last movement before setting off). Each step first turns by its
    turn (positive: from the units' first axis toward the second, clockwise on the map in pixels, anticlockwise in metres), cut to the agent's
    largest turn, then moves its distance, cut to [0, step].
    The robot's prompt states the largest distance per step; a person's gives only the top speed (with the stated maximum, people asked for exactly it in 75-82% of steps)."""

    name = "steer"
    template = "steer"
    key = "steps"
    scored_distance_framings = True
    system_template = "steer"
    scored_sizes = True
    think_max_tokens = 32768  # most calls with the steer prompts ran past 16384 tokens (robot test runs, 2026-11-16)
    # Largest turn per 0.4 s step, rounded down to a whole degree. Robot: the real robots' largest body yaw change per step within a trip,
    # 39.0 deg at LAAS and 35.7 at UM (the 144 robot trips of trips_v3). Person: the 99th percentile of the change in walking direction
    # between consecutive steps both above 0.2 m/s, 131.5 deg over the goal-to-goal moves of the 14,292 trips of trips_v3_human (the median
    # is 11 deg), the same percentile as the person's speed cap.
    MAX_TURN_DEG = {"robot": 39.0, "human": 131.0}
    # Example steps [turn in degrees, distance in metres] at about each kind's walking or driving scale: robot 0.10-0.11 m (near its 0.12 m
    # limit per step), person 0.4 m (about 1.0 m/s). The distances are written in the units.
    example = {"robot": [(0, 0.10), (-15, 0.11), (0, 0.0)], "human": [(0, 0.40), (-15, 0.40), (0, 0.0)]}

    @property
    def max_turn_deg(self) -> float:
        return self.MAX_TURN_DEG[self.kind]

    def example_text(self) -> str:
        return ", ".join(f"[{turn:g}, {self.units.len_text(d)}]" for turn, d in self.example[self.kind])

    def output_fields(self) -> dict:
        return {"max_turn": f"{self.max_turn_deg:g}"}

    def step_ends(self, value: np.ndarray, scene: Scene) -> tuple[np.ndarray, np.ndarray]:
        """Per step of a reply: the heading at the step's end (radians in the units' frame) and the speed (the units per second), after the cuts."""
        turn = np.clip(value[:, 0], -self.max_turn_deg, self.max_turn_deg)
        d = np.clip(value[:, 1], 0.0, self.units.length(self.step_m))
        return np.radians(self.units.heading_deg(scene.heading) + np.cumsum(turn)), d / DT

    def positions(self, value, scene):
        theta, speed = self.step_ends(value, scene)
        uv = self.units.pos(scene.track[-1]) + np.cumsum((speed * DT)[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1), axis=0)
        return self.units.pos_to_metres(uv), self.units.length_to_metres(np.abs(value[:, 1])) / DT

    def motion(self, value, scene, track):
        steps = len(track) - len(scene.track)  # executed steps of this reply. Beyond its last step the agent held its place
        theta, speed = self.step_ends(value, scene)
        last = min(steps, len(theta)) - 1
        return (self.units.heading_to_metres(np.degrees(theta[last])),
                float(self.units.length_to_metres(speed[last])) if steps <= len(theta) else 0.0)


class SteerVelocityPlanner(SteerPlanner):
    """The steer_vel mode: steps [turn rate in degrees per second, speed in the units per second], each held for the step, so the agent
    follows a circular arc of radius speed / turn rate (the unicycle model), straight at a turn rate of 0. The heading starts as steer's and
    continues from each step's end. The turn rate is cut to the kind's steer turn cap per step divided by the step (MAX_TURN_DEG / DT), the
    speed to [0, the kind's speed cap]. Same framings and system texts as steer."""

    name = "steer_velocity"
    template = "steer_vel"
    # Example steps [turn rate in deg/s, speed in m/s] at about each kind's walking or driving speed. The speeds are written in the units.
    example = {"robot": [(0, 0.25), (-40, 0.27), (0, 0.0)], "human": [(0, 1.0), (-40, 1.0), (0, 0.0)]}

    @property
    def max_turn_rate_deg_s(self) -> float:
        return self.max_turn_deg / DT

    def output_fields(self) -> dict:
        return {"max_turn_rate": f"{self.max_turn_rate_deg_s:g}"}

    def step_ends(self, value, scene):
        omega = np.clip(value[:, 0], -self.max_turn_rate_deg_s, self.max_turn_rate_deg_s)
        speed = np.clip(value[:, 1], 0.0, self.units.length(self.speed_cap))
        return np.radians(self.units.heading_deg(scene.heading) + np.cumsum(omega * DT)), speed

    def positions(self, value, scene):
        theta1, speed = self.step_ends(value, scene)  # heading at each step's end
        theta0 = np.concatenate([[np.radians(self.units.heading_deg(scene.heading))], theta1[:-1]])  # and at its start
        omega = (theta1 - theta0) / DT
        turning = omega != 0.0
        # Arc of radius speed / omega: the chord is (speed / omega) (sin theta1 - sin theta0, cos theta0 - cos theta1); straight when omega = 0.
        radius = np.divide(speed, omega, out=np.zeros_like(speed), where=turning)
        chord = np.where(turning[:, None], radius[:, None] * np.stack([np.sin(theta1) - np.sin(theta0), np.cos(theta0) - np.cos(theta1)], axis=1),
                         (speed * DT)[:, None] * np.stack([np.cos(theta0), np.sin(theta0)], axis=1))
        uv = self.units.pos(scene.track[-1]) + np.cumsum(chord, axis=0)
        return self.units.pos_to_metres(uv), self.units.length_to_metres(np.abs(value[:, 1]))
