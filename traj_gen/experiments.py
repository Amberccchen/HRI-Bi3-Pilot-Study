"""The experiment registry: each id maps to a planner (output mode), the images and an episode type.

Framing (one of prompting.FRAMINGS) and thinking (nothink/think) are run options, not part of the id. The
single-shot experiments plan the whole path at the trip start. The rolling ones re-plan every
execute_s of executed motion (rh_vel1_bev every step), the `_conv` ones in one growing conversation. The `_constvel_future` ones are their
base experiment plus each person's constant-velocity prediction (text and map).

Experiment id parts, in order:
  episode type   ss = single shot (one query at the trip start); rh = rolling horizon (re-plan every execute_s of executed motion)
  output mode    wp = timed waypoints in metres; px = map positions in pixels; vel1 = one velocity step; velseq = a velocity sequence;
                 steer = a turn from the current heading and a distance per step, with the harness prompts (planners.SteerPlanner, only with the harness framings)
  images         text = no image; bev = the top-down (bird's-eye view) map; cam = plus the camera view
  conv           all queries of an episode in one growing conversation (episodes.ConversationEpisode); without it each query is a fresh chat
  true_future    the other agents' recorded future positions in the text each turn (Experiment.true_future)
  check          the plan check: each plan is tested against those futures and the first conflict is reported back for revision (Experiment.check_rounds)
  constvel_future  each person's position at t + 1 s and t + 2 s if it kept its current velocity, in the text and map (Experiment.base)
straight and replay are the baselines (no model).

The harness experiments (planners.SteerPlanner, with the harness framings only: harness for a person, robot_harness for the robot) come in three levels: the harness; the harness plus
the other agents' recorded future trajectories (true_future, a proxy for a forecaster); and that plus the plan check (check_rounds), which tests each plan against the recorded futures and reports the first conflict back for revision.
"""

from dataclasses import dataclass, replace

from traj_gen.episodes import BaselineEpisode, ConversationEpisode, RollingEpisode, SingleShotEpisode
from traj_gen.planners import SteerPlanner, PixelPlanner, QueryPlanner, ReplayPlanner, StepVelocityPlanner, StraightPlanner, VelocityPlanner, WaypointPlanner
from traj_gen.prompting import HARNESS_FRAMINGS
from traj_gen.scene import DT


@dataclass(frozen=True)
class Experiment:
    id: str
    planner: type
    images: tuple[str, ...]
    episode: type
    execute_s: float | None  # executed motion between queries (rolling only)
    base: str | None = None  # id of the experiment without the prediction (_constvel_future rows only)
    true_future: bool = False  # give the other agents' recorded future positions in the text each turn
    check_rounds: int = 0  # check each plan against the recorded futures and report the first conflict back, up to this many revisions

    def __post_init__(self):
        if issubclass(self.episode, (RollingEpisode, ConversationEpisode)):
            steps = round(self.execute_s / DT)
            if steps < 1 or abs(steps * DT - self.execute_s) >= 1e-9:
                raise ValueError(f"{self.id}: execute_s {self.execute_s} is not a positive whole number of {DT} s steps")
        elif self.execute_s is not None:
            raise ValueError(f"{self.id}: execute_s is only for rolling and conversation episodes, got {self.execute_s}")

    @property
    def predict(self) -> bool:
        """Also give each person's constant-velocity position (text and map)."""
        return self.base is not None

    @property
    def uses_model(self) -> bool:
        return issubclass(self.planner, QueryPlanner)

    def accepts(self, framing: str) -> bool:
        """Whether a run should pair this experiment with the framing: harness framings go only with the steer mode."""
        return self.uses_model and (framing in HARNESS_FRAMINGS) == self.planner.harness


_ROLLING_S = 2.0  # 5 steps, about 7 queries per trip
_experiments = [
    Experiment("ss_wp_text", WaypointPlanner, (), SingleShotEpisode, None),
    Experiment("ss_wp_bev", WaypointPlanner, ("bev",), SingleShotEpisode, None),
    Experiment("ss_wp_bev_cam", WaypointPlanner, ("bev", "cam"), SingleShotEpisode, None),
    Experiment("ss_px_bev", PixelPlanner, ("bev",), SingleShotEpisode, None),
    Experiment("rh_wp_bev", WaypointPlanner, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_cam", WaypointPlanner, ("bev", "cam"), RollingEpisode, _ROLLING_S),
    Experiment("rh_px_bev", PixelPlanner, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_vel1_bev", StepVelocityPlanner, ("bev",), RollingEpisode, 0.4),
    Experiment("rh_velseq_bev", VelocityPlanner, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_conv", WaypointPlanner, ("bev", "cam"), ConversationEpisode, _ROLLING_S),
    Experiment("rh_px_bev_conv", PixelPlanner, ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_velseq_bev_conv", VelocityPlanner, ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_steer_bev_conv", SteerPlanner, ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_steer_bev_conv_true_future", SteerPlanner, ("bev",), ConversationEpisode, _ROLLING_S, true_future=True),
    Experiment("rh_steer_bev_conv_true_future_check", SteerPlanner, ("bev",), ConversationEpisode, _ROLLING_S, true_future=True, check_rounds=2),
    Experiment("straight", StraightPlanner, (), BaselineEpisode, None),
    Experiment("replay", ReplayPlanner, (), BaselineEpisode, None),
]
_experiments += [replace(e, id=f"{e.id}_constvel_future", base=e.id) for e in _experiments if e.id in ("ss_wp_bev", "rh_wp_bev", "rh_vel1_bev")]  # single shot, rolling 2 s and the shortest rolling horizon
EXPERIMENTS = {e.id: e for e in _experiments}
BASELINES = tuple(e.id for e in _experiments if not e.uses_model)  # registry order
