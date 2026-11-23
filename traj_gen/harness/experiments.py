"""The experiment registry of the runs before harness presets: each id maps to a planner (output mode), the images and an episode type.

New runs are set up with harnesses (traj_gen/harness/presets.py), which derive the same ids from their settings and build Experiment objects
the same way. The registry stays for the baselines (straight and replay, no model) and so that the analysis can read run folders
without a harness.json. The ids here are the run folder names of the runs before the units setting: wp is the positions output in metres and
px the positions output in pixels, and the steer experiments are in pixels. The conversation experiment with waypoints, map and camera
was registered as rh_wp_bev_conv and then as rh_wp_bev_cam_conv. LEGACY_IDS maps the old folder name to it.

Framing (one of observation.prompting.FRAMINGS) and thinking (nothink/think) are run options, not part of the id. The
single-shot experiments plan the whole path at the trip start. The rolling ones re-plan every
execute_s of executed motion (rh_vel1_bev every step), the `_conv` ones in one growing conversation. The `_constvel_future` ones are their
base experiment plus each person's constant-velocity prediction (text and map). This prediction is deprecated: it is not
confirmed to be complete. The id naming key is in traj_gen/harness/presets.py.

The harness experiments (outputs.steer.SteerPlanner, with the harness framings only: harness for a person, robot_harness for the robot) come in three levels: the harness; the harness plus
the other agents' recorded future trajectories (true_future, a proxy for a forecaster); and that plus the plan check (check_rounds), which tests each plan against the recorded futures and reports the first conflict back for revision.
"""

from dataclasses import dataclass, replace

from traj_gen.harness.episodes import BaselineEpisode, ConversationEpisode, RollingEpisode, SingleShotEpisode
from traj_gen.observation.prompting import HARNESS_FRAMINGS
from traj_gen.observation.scene import DT
from traj_gen.outputs.baselines import ReplayPlanner, StraightPlanner
from traj_gen.outputs.planner import QueryPlanner
from traj_gen.outputs.positions import PositionPlanner
from traj_gen.outputs.steer import SteerPlanner
from traj_gen.outputs.velocity import StepVelocityPlanner, VelocityPlanner


@dataclass(frozen=True)
class Experiment:
    id: str
    planner: type
    units: str | None  # the units setting (outputs.planner.UNITS), None for the baselines
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
        """Also give each person's constant-velocity position (text and map): the constvel_future prediction, deprecated."""
        return self.base is not None

    @property
    def uses_model(self) -> bool:
        return issubclass(self.planner, QueryPlanner)

    def accepts(self, framing: str) -> bool:
        """Whether a run should pair this experiment with the framing: harness framings go only with the steer mode."""
        return self.uses_model and (framing in HARNESS_FRAMINGS) == self.planner.harness


_ROLLING_S = 2.0  # 5 steps, about 7 queries per trip
_experiments = [
    Experiment("ss_wp_text", PositionPlanner, "m", (), SingleShotEpisode, None),
    Experiment("ss_wp_bev", PositionPlanner, "m", ("bev",), SingleShotEpisode, None),
    Experiment("ss_wp_bev_cam", PositionPlanner, "m", ("bev", "cam"), SingleShotEpisode, None),
    Experiment("ss_px_bev", PositionPlanner, "px", ("bev",), SingleShotEpisode, None),
    Experiment("rh_wp_bev", PositionPlanner, "m", ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_cam", PositionPlanner, "m", ("bev", "cam"), RollingEpisode, _ROLLING_S),
    Experiment("rh_px_bev", PositionPlanner, "px", ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_vel1_bev", StepVelocityPlanner, "m", ("bev",), RollingEpisode, 0.4),
    Experiment("rh_velseq_bev", VelocityPlanner, "m", ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_cam_conv", PositionPlanner, "m", ("bev", "cam"), ConversationEpisode, _ROLLING_S),
    Experiment("rh_px_bev_conv", PositionPlanner, "px", ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_velseq_bev_conv", VelocityPlanner, "m", ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_steer_bev_conv", SteerPlanner, "px", ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("rh_steer_bev_conv_true_future", SteerPlanner, "px", ("bev",), ConversationEpisode, _ROLLING_S, true_future=True),
    Experiment("rh_steer_bev_conv_true_future_check", SteerPlanner, "px", ("bev",), ConversationEpisode, _ROLLING_S, true_future=True, check_rounds=2),
    Experiment("straight", StraightPlanner, None, (), BaselineEpisode, None),
    Experiment("replay", ReplayPlanner, None, (), BaselineEpisode, None),
]
_experiments += [replace(e, id=f"{e.id}_constvel_future", base=e.id) for e in _experiments if e.id in ("ss_wp_bev", "rh_wp_bev", "rh_vel1_bev")]  # single shot, rolling 2 s and the shortest rolling horizon
EXPERIMENTS = {e.id: e for e in _experiments}
LEGACY_IDS = {"rh_wp_bev_conv": "rh_wp_bev_cam_conv"}  # experiment folder name before the derived ids -> id
BASELINES = tuple(e.id for e in _experiments if not e.uses_model)  # registry order
