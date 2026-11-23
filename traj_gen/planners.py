"""Planners: turn a model reply into the planned agent's positions, one per time step from where it is now.

A planner is made per episode from an experiments.Experiment. Model planners (QueryPlanner
subclasses) also open the chats and render the images for the queries. Each subclass is one
output mode: its class attributes (name, prompt template, reply key, example, units) and
methods (to_units, max_steps, positions) say how it is prompted and how a reply becomes positions:

- waypoints (metres): the speed limit keeps the path's shape and stretches its timing.
- pixels: the same after mapping the BEV pixels to metres.
- velocities: each clipped to the speed cap, then integrated.
- steer: steps [turn, distance] on the BEV from the current heading, each turn and distance cut to the agent's limits, with
  the harness prompts (planners.SteerPlanner).
- StraightPlanner, ReplayPlanner: no model, so the baselines go through the same episode,
  metric and saving code as the model conditions.

positions() also returns the raw per-step speeds of the reply (before limiting) for the
speed-violation metric, or None for the baselines, which do not have a reply. The speed cap is the
planned agent's (trip["agent"]).
"""

from functools import cached_property

import numpy as np

from traj_gen.agents import SPEED_CAP_M_S, STRAIGHT_LINE_SPEED_M_S, agent_kind
from traj_gen.bev import bev_transform, render_bev
from traj_gen.camera import camera_frame
from traj_gen.client import THINK_MAX_TOKENS, Chat, make_client
from traj_gen.episodes import real_path
from traj_gen.prompting import PromptBuilder
from traj_gen.scene import DT, Metres, Pixels, Scene, person_name


THINKING = ("nothink", "think")  # run option of the model conditions: the model's thinking off, on
MAX_PLAN_STEPS = 75  # 30 s, about 3x the straight-line time across the room's diagonal (4.9 m) at the speed cap. A longer plan could not be scored in a replay anyway
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
    harness = False  # whether the mode goes with the harness framings (prompting.HARNESS_FRAMINGS)
    system_template = "system"  # + "_human" for a planned person
    followup_template = "followup"
    scored_sizes = False  # the text gives the scored body radii instead of the drawn (enclosing) ones
    think_max_tokens = THINK_MAX_TOKENS  # Qwen thinking budget with thinking on

    def to_units(self, value) -> np.ndarray:
        """An example position in metres, written in the units of the output mode."""
        return self.units.pos(value)

    def example_text(self) -> str:
        """The example pairs of the reply format, written in the output units."""
        return ", ".join("[" + ", ".join(self.units.num(c) for c in self.to_units(p)) + "]" for p in self.example[self.kind])

    def output_fields(self) -> dict:
        """Fields of the output template beyond the shared ones."""
        return {}

    def extra_text(self, scene: Scene) -> str:
        """Text after the state: the other agents' recorded future positions when the experiment gives them, else nothing."""
        if not self.exp.future:
            return ""
        i = len(scene.track) - 1
        if self.kind == "robot":
            lines = ["Where each person will actually be (recorded in the study; they walk exactly this way whatever the robot does):"]
        else:  # a planned person is also given the robot's recorded future
            lines = ["Where the other participant and the robot will actually be (recorded in the study; they move exactly this way whatever you do):"]
        for other, track in self.trip["tracks"].items():
            if other == self.trip["agent"] or (self.kind == "robot" and agent_kind(other) != "human"):
                continue
            future = np.asarray(track, float)[i + 1:, :2]
            points = ", ".join(f"t+{DT * (k + 1):.1f} s {self.units.pos_text(xy)}" for k, xy in enumerate(future))
            lines.append(f"{'Robot' if agent_kind(other) == 'robot' else person_name(other)}: {points}.")
        return "\n".join(lines)

    def max_steps(self, remaining: int) -> int:
        """Longest plan asked for: the rest of the human replay, at most MAX_PLAN_STEPS."""
        return min(MAX_PLAN_STEPS, remaining)

    def chat(self, scene: Scene) -> Chat:
        """A new conversation for this scene's remaining horizon (the builder fixes the longest plan allowed)."""
        builder = PromptBuilder(self.framing, self, self.exp.images, self.max_steps(len(self.trip["time"]) - len(scene.track)), self.exp.predict)
        return make_client(builder, self.thinking == "think").chat(builder.system(scene))

    @cached_property
    def cam(self) -> bytes:
        return camera_frame(self.trip)  # the frame is the same for every query of the episode

    def images(self, scene: Scene, kinds: list[str]) -> dict[str, bytes]:
        return {k: render_bev(scene, self.pixel_ticks, self.exp.predict)[0] if k == "bev" else self.cam for k in kinds}


class WaypointPlanner(QueryPlanner):
    """Positions in metres. The speed limit retimes the path."""

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
    """Baseline: straight line to the goal at the planned agent's STRAIGHT_LINE_SPEED_M_S."""

    def positions(self, value, scene):
        return retime(scene.track[-1], scene.goal_xy[None], STRAIGHT_LINE_SPEED_M_S[self.kind] * DT)[0], None


class ReplayPlanner(Planner):
    """Baseline: the planned agent's real path, as recorded."""

    def positions(self, value, scene):
        return real_path(self.trip)[1:], None  # the scene is at start_index, so the real path after it
