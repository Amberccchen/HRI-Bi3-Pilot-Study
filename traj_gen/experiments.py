"""The experiment registry (PLAN.md, Experiments): id -> planner, prompt builder, units, images, episode type.

Framing (neutral/social) and thinking (nothink/think) are run options, not part of the id. The
single-shot experiments plan the whole path at the trip start; the rolling ones re-plan every
execute_s of executed motion (rh_vel1_bev every step), `_conv` ones in one growing conversation, `_pred` ones are their base
experiment plus each person's constant-velocity prediction (text and map).
"""

from dataclasses import dataclass, replace

from bev import bev_transform
from episodes import BaselineEpisode, ConversationEpisode, RollingEpisode, SingleShotEpisode
from planners import PixelPlanner, ReplayPlanner, StepVelocityPlanner, StraightPlanner, VelocityPlanner, WaypointPlanner
from prompting import PixelPrompt, PromptBuilder, VelocityPrompt, VelocitySeqPrompt
from scene import Pixels, Units


@dataclass(frozen=True)
class Experiment:
    id: str
    planner: type
    prompt: type | None  # PromptBuilder subclass; None for the baselines
    units: Units | None
    images: tuple[str, ...]
    episode: type
    execute_s: float | None  # executed motion between queries (rolling only)
    predict: bool = False    # also give each person's constant-velocity position (text and map)


METRES, PIXELS = Units(), Pixels(bev_transform())
_ROLLING_S = 2.0  # 5 steps, about 7 queries per trip
_experiments = [
    Experiment("ss_wp_text", WaypointPlanner, PromptBuilder, METRES, (), SingleShotEpisode, None),
    Experiment("ss_wp_bev", WaypointPlanner, PromptBuilder, METRES, ("bev",), SingleShotEpisode, None),
    Experiment("ss_wp_bev_cam", WaypointPlanner, PromptBuilder, METRES, ("bev", "cam"), SingleShotEpisode, None),
    Experiment("ss_px_bev", PixelPlanner, PixelPrompt, PIXELS, ("bev",), SingleShotEpisode, None),
    Experiment("rh_wp_bev", WaypointPlanner, PromptBuilder, METRES, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_cam", WaypointPlanner, PromptBuilder, METRES, ("bev", "cam"), RollingEpisode, _ROLLING_S),
    Experiment("rh_px_bev", PixelPlanner, PixelPrompt, PIXELS, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_vel1_bev", StepVelocityPlanner, VelocityPrompt, METRES, ("bev",), RollingEpisode, 0.4),
    Experiment("rh_velseq_bev", VelocityPlanner, VelocitySeqPrompt, METRES, ("bev",), RollingEpisode, _ROLLING_S),
    Experiment("rh_wp_bev_conv", WaypointPlanner, PromptBuilder, METRES, ("bev", "cam"), ConversationEpisode, _ROLLING_S),
    Experiment("rh_velseq_bev_conv", VelocityPlanner, VelocitySeqPrompt, METRES, ("bev",), ConversationEpisode, _ROLLING_S),
    Experiment("straight", StraightPlanner, None, None, (), BaselineEpisode, None),
    Experiment("replay", ReplayPlanner, None, None, (), BaselineEpisode, None),
]
_experiments += [replace(e, id=f"{e.id}_pred", predict=True) for e in _experiments if e.id in ("ss_wp_bev", "rh_wp_bev", "rh_vel1_bev")]  # single shot, rolling 2 s and the shortest rolling horizon
EXPERIMENTS = {e.id: e for e in _experiments}
