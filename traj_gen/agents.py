"""Physical and task parameters of the agents of a trip: the planned agent and the others it is scored against.

Agent ids are those of the trips: "robot", and "human<k>" for the people. A parameter is looked up by the
agent's kind ("robot" or "human"); a kind without an entry has no value yet and the lookup raises KeyError.
"""

import numpy as np


SPEED_CAP_M_S = {"robot": 0.3}       # the limit the model is told
ARRIVAL_RADIUS_M = {"robot": 0.5}    # arrival distance given to the model; Bi3 switches to the next goal at about this
PERSON_BODY_RADIUS_M = 0.15          # person body radius for scoring; mocap tracks only give the body centre
PERSON_PROMPT_RADIUS_M = 0.25        # typical shoulder half-width, the person's size in the prompts and maps
# Personal space of an agent as an obstacle: the disk around its centre that another agent's body should stay out of.
# For a person 0.25 m, a 0.10 m buffer beyond the body radius, a modelling choice (0.20 m is the sensitivity), not a measurement.
PERSONAL_SPACE_RADIUS_M = {"human": 0.25}


def agent_kind(agent_id: str) -> str:
    """"robot" for the robot, "human" for "human<k>"."""
    if agent_id == "robot":
        return "robot"
    if agent_id.startswith("human") and agent_id[len("human"):].isdigit():
        return "human"
    raise ValueError(f"unknown agent id {agent_id!r}: expected 'robot' or 'human<k>'")


# The robot has two radii. The prompts and maps use the footprint's circumscribed disk (robot_prompt_radius), which is larger
# than the robot on purpose, so that planned paths keep a margin from people. Scoring uses the inscribed disk (robot_body_radius),
# the realistic body.
def robot_body_radius(footprint_m: list[float]) -> float:
    """Radius of the disk the robot's rectangular footprint (site_frames.json) contains: half the shorter side."""
    return min(footprint_m) / 2


def robot_prompt_radius(footprint_m: list[float]) -> float:
    """Radius of the disk that contains the robot's rectangular footprint (site_frames.json), in metres rounded to the millimetre."""
    return round(float(np.hypot(*footprint_m) / 2), 3)


def body_radius(agent_id: str, site: dict) -> float:
    """Radius (m) of the agent's body disk for scoring; site is the agent's site from site_frames.json."""
    return robot_body_radius(site["footprint_m"]) if agent_kind(agent_id) == "robot" else PERSON_BODY_RADIUS_M


def prompt_radius(agent_id: str, site: dict) -> float:
    """Radius (m) of the agent's disk in the prompts and maps; site is the agent's site from site_frames.json."""
    return robot_prompt_radius(site["footprint_m"]) if agent_kind(agent_id) == "robot" else PERSON_PROMPT_RADIUS_M
