"""The experiment registry: each id maps to a planner (output mode), the images and an episode type.

Framing (neutral/social) and thinking (nothink/think) are run options, not part of the id. The
single-shot experiments plan the whole path at the trip start. The rolling ones re-plan every
execute_s of executed motion (rh_vel1_bev every step), the `_conv` ones in one growing conversation. The `_pred` ones are their
base experiment plus each person's constant-velocity prediction (text and map).
"""

from dataclasses import dataclass, replace

from traj_gen.episodes import BaselineEpisode, ConversationEpisode, RollingEpisode, SingleShotEpisode
from traj_gen.planners import PixelPlanner, QueryPlanner, ReplayPlanner, StepVelocityPlanner, StraightPlanner, VelocityPlanner, WaypointPlanner
from traj_gen.scene import DT


@dataclass(frozen=True)
class Experiment:
    id: str
    planner: type
    images: tuple[str, ...]
    episode: type
    execute_s: float | None  # executed motion between queries (rolling only)
    base: str | None = None  # id of the experiment without the prediction (_pred rows only)

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
    Experiment("rh_velseq_bev_conv", VelocityPlanner, ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("straight", StraightPlanner, (), BaselineEpisode, None),
    Experiment("replay", ReplayPlanner, (), BaselineEpisode, None),
]
_experiments += [replace(e, id=f"{e.id}_pred", base=e.id) for e in _experiments if e.id in ("ss_wp_bev", "rh_wp_bev", "rh_vel1_bev")]  # single shot, rolling 2 s and the shortest rolling horizon
EXPERIMENTS = {e.id: e for e in _experiments}
BASELINES = tuple(e.id for e in _experiments if not e.uses_model)  # registry order
