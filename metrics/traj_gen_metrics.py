"""Footprint-aware and task metrics for the generated (or replayed) path of one planned agent.

bi3_metrics.py scores motion with centre-to-centre distances, as the Bi3 paper does.
Generated paths also need to be judged as navigation: does the agent's body hit another agent,
does it intrude on their personal space, does it reach its goal, does it stay in the room. Every
agent is a disk of a radius the caller gives (a robot's is its footprint's inscribed radius, since
the planner's heading is unknown and the corners are not modelled), so:

- clearance: centre distance minus both body radii, per other agent, minimised over the other
  agents and time. Negative means the bodies touch (contact). With a person body radius of 0.15 m,
  one real controller contact is measured, in trip laas_laas18_static_robot_02 (clearance -0.011 m). It
  is unlikely to be a real collision, because both bodies are modelled as circles. Contact is also
  reported as the share of the episode's time steps in contact (contact_time) and as the number of
  separate contacts, a run of consecutive steps counting once (contact_events; steps are 0.4 s apart,
  so a touch between two samples is missed).
- personal-space overlap: the share of the episode's time steps with the agent's body inside some
  other agent's personal space (a disk of the given radius around that agent's centre).
- goal reached: the path gets within the arrival radius + GOAL_SLACK of the goal station. Bi3
  switches goals at about the arrival radius on the raw mocap, but the cleaned room-frame real
  paths end 0.49-0.60 m from the goal, so the slack keeps the real controllers from
  being scored as failures. Time to goal is the first such sample.
- final distance to goal: distance of the last sample, also for paths that never arrive.
- out of room: any sample outside the square room of side room_m.
- stations: the agent's centre entering the box of a station that is not part of its goal (the stations
  are obstacles the planner is told about; the real controllers never do this), and the closest
  centre distance to such a station.
- speed cap: for a plan whose raw (before speed limiting) per-step speeds are given,
  the fraction of steps above the speed cap and the largest raw speed.

Inputs are one executed path as arrays: time (T,) in seconds from trip start, agent
(T, 2) and others (T, K, 2) in the room frame (metres), goal (2,), the radii, arrival radius and
speed cap per agent, and the room size and stations from site_frames.json.
"""

import numpy as np

from preprocessing.bi3_derived import label_members


GOAL_SLACK = 0.1     # smoothing/mocap offset: real paths end up to 0.60 m from the goal at the switch
STATION_BOX_M = 0.2  # side of a station's box as drawn on the map; the station position is the robot's stopping point in front of the table
# Class of every metric an episode manifest can hold. invalid: the plan breaks a hard rule of the task (an unusable reply, which has
# no metrics, is invalid too). contact: closeness to the other agents. efficiency: how the agent moves.
METRIC_CLASS = {
    **dict.fromkeys(["goal_reached", "final_distance_to_goal", "out_of_room", "station_crossing", "station_distance_min",
                     "speed_violation_rate", "raw_speed_max"], "invalid"),
    **dict.fromkeys(["clearance_min", "contact_time", "contact_events", "personal_space_overlap", "min_distance",
                     "min_distance_per_other", "min_distance_index", "min_distance_other"], "contact"),
    **dict.fromkeys(["time_to_goal", "path_efficiency", "path_length", "straight_distance", "velocity_mean", "velocity_median",
                     "velocity_max", "velocity_std", "acceleration_mean", "acceleration_max", "acceleration_tangential_mean"], "efficiency"),
}


def other_stations(site: dict, goal: str) -> np.ndarray:
    """(k, 2) positions of the single stations that are not part of the goal label (from site_frames.json)."""
    goal_members = label_members(goal, site["stations"])
    return np.array([s["xy"] for k, s in site["stations"].items() if len(s["members"]) == 1 and k not in goal_members], float)


def station_metrics(agent: np.ndarray, stations: np.ndarray) -> dict:
    """Does the agent's centre enter another station's box, and how close does it get to one."""
    offset = np.abs(np.asarray(agent, float)[:, None, :] - stations[None])  # (T, k, 2)
    return {"station_crossing": bool((offset.max(axis=-1) <= STATION_BOX_M / 2).any()),
            "station_distance_min": float(np.linalg.norm(offset, axis=-1).min())}


def clearance_gap(agent: np.ndarray, others: np.ndarray, agent_radius: float, other_radii: list[float]) -> np.ndarray:
    """(T,) smallest body-to-body gap from the agent to the other agents: centre distance minus both radii, minimised over the others."""
    centre = np.linalg.norm(np.asarray(others, float) - np.asarray(agent, float)[:, None, :], axis=-1)  # (T, K)
    return (centre - agent_radius - np.asarray(other_radii, float)).min(axis=1)


def clearance_metrics(agent: np.ndarray, others: np.ndarray, agent_radius: float, other_radii: list[float], personal_space_radii: list[float]) -> dict:
    """Minimum clearance to the other agents, the share of time steps in contact and in someone's personal space, and the number of separate contacts."""
    gap = clearance_gap(agent, others, agent_radius, other_radii)
    touching = gap < 0
    centre = np.linalg.norm(np.asarray(others, float) - np.asarray(agent, float)[:, None, :], axis=-1)  # (T, K)
    intruding = (centre - agent_radius < np.asarray(personal_space_radii, float)).any(axis=1)
    return {"clearance_min": float(gap.min()), "contact_time": float(touching.mean()),
            "contact_events": int((touching & ~np.concatenate([[False], touching[:-1]])).sum()),  # runs of consecutive contact steps
            "personal_space_overlap": float(intruding.mean())}


def traj_gen_metrics(time: np.ndarray, agent: np.ndarray, others: np.ndarray, goal: np.ndarray, agent_radius: float, other_radii: list[float],
                     personal_space_radii: list[float], arrival_radius: float, speed_cap: float, room_m: float, stations: np.ndarray,
                     raw_speed: np.ndarray | None = None) -> dict:
    """All footprint and task metrics for one path; stations from other_stations(); raw_speed (steps,) only for model plans."""
    time, agent, others = np.asarray(time, float), np.asarray(agent, float), np.asarray(others, float)
    to_goal = np.linalg.norm(agent - np.asarray(goal, float), axis=1)
    arrived = np.flatnonzero(to_goal <= arrival_radius + GOAL_SLACK)
    out = {**clearance_metrics(agent, others, agent_radius, other_radii, personal_space_radii),
           "goal_reached": bool(arrived.size), "time_to_goal": float(time[arrived[0]] - time[0]) if arrived.size else float("nan"),
           "final_distance_to_goal": float(to_goal[-1]), "out_of_room": bool((np.abs(agent) > room_m / 2).any())}
    out |= station_metrics(agent, stations)
    if raw_speed is not None:
        out |= {"speed_violation_rate": float((np.asarray(raw_speed) > speed_cap).mean()), "raw_speed_max": float(np.max(raw_speed))}
    return out
