"""Footprint-aware and task metrics for generated (or replayed) robot paths.

bi3_metrics.py scores motion with centre-to-centre distances, as the Bi3 paper does.
Generated paths also need to be judged as navigation: does the robot's body hit a
person, does it intrude on their personal space, does it reach its goal, does it stay in
the room. The robot is a disk of its footprint's inscribed radius (half the shorter side;
the planner's heading is unknown, so the corners are not modelled) and a person is a disk, so:

- clearance: centre distance - robot radius - BODY_RADIUS, minimised over time and people;
  negative means the bodies touch (contact). BODY_RADIUS is the largest round value for which no real
  controller trip collides (closest real centre distance 0.47 m LAAS, 0.36 m UM), so real
  controllers having clearance >= 0 is an assumption of the metric, not a finding. Contact is
  also reported as the share of the episode's time steps in contact (contact_time) and as the number
  of separate contacts, a run of consecutive steps counting once (contact_events; steps are 0.4 s apart,
  so a touch between two samples is missed).
- personal-space overlap: the same with PERSONAL_SPACE_RADIUS, the person's disk inflated by
  a buffer; the share of the episode's time steps with the robot inside any person's disk.
  The radius is a modelling choice (0.20 m is the sensitivity), not a measurement.
- goal reached: the path gets within GOAL_TOL + GOAL_SLACK of the goal station. Bi3
  switches goals at about GOAL_TOL on the raw mocap, but the cleaned room-frame real
  paths end 0.49-0.60 m from the goal, so the slack keeps the real controllers from
  being scored as failures. Time to goal is the first such sample.
- final distance to goal: distance of the last sample, also for paths that never arrive.
- out of room: any sample outside the square room of side room_m.
- stations: the robot's centre entering the box of a station other than its goal (the stations
  are obstacles the planner is told about; the real controllers never do this), and the closest
  centre distance to such a station.
- speed cap: for a plan whose raw (before speed limiting) per-step speeds are given,
  the fraction of steps above SPEED_CAP and the largest raw speed.

Inputs are one executed path as arrays: time (T,) in seconds from trip start, robot
(T, 2) and humans (T, H, 2) in the room frame (metres), goal (2,); footprint and room
size come from site_frames.json.
"""

import numpy as np


BODY_RADIUS = 0.15            # largest round value with no collision in the real controller trips; mocap tracks only give the body centre
PERSONAL_SPACE_RADIUS = 0.25  # typical shoulder half-width, the person's size in the prompts and maps; a 0.10 m buffer beyond BODY_RADIUS
GOAL_TOL = 0.5       # arrival distance given to the model; Bi3 switches to the next goal at about this
GOAL_SLACK = 0.1     # smoothing/mocap offset: real paths end up to 0.60 m from the goal at the switch
SPEED_CAP = 0.3      # m/s, the limit the model is told
STATION_BOX_M = 0.2  # side of a station's box as drawn on the map; the station position is the robot's stopping point in front of the table


def other_stations(site: dict, goal: str) -> np.ndarray:
    """(k, 2) positions of the single stations that are not the robot's goal (from site_frames.json)."""
    return np.array([s["xy"] for k, s in site["stations"].items() if len(s["members"]) == 1 and k != goal], float)


def station_metrics(robot: np.ndarray, stations: np.ndarray) -> dict:
    """Does the robot's centre enter another station's box, and how close does it get to one."""
    offset = np.abs(np.asarray(robot, float)[:, None, :] - stations[None])  # (T, k, 2)
    return {"station_crossing": bool((offset.max(axis=-1) <= STATION_BOX_M / 2).any()),
            "station_distance_min": float(np.linalg.norm(offset, axis=-1).min())}


def inscribed_radius(footprint_m: list[float]) -> float:
    """Radius of the disk the robot's rectangular footprint (site_frames.json) contains."""
    return min(footprint_m) / 2


def clearance_gap(robot: np.ndarray, humans: np.ndarray, footprint_m: list[float]) -> np.ndarray:
    """(T,) distance from the robot's disk to the nearest person's centre."""
    return np.linalg.norm(np.asarray(humans, float) - np.asarray(robot, float)[:, None, :], axis=-1).min(axis=1) - inscribed_radius(footprint_m)


def clearance_metrics(robot: np.ndarray, humans: np.ndarray, footprint_m: list[float], personal_space_radius: float = PERSONAL_SPACE_RADIUS) -> dict:
    """Minimum clearance to the people, the share of time steps in contact and in someone's personal space, and the number of separate contacts."""
    gap = clearance_gap(robot, humans, footprint_m)
    touching = gap < BODY_RADIUS
    return {"clearance_min": float(gap.min() - BODY_RADIUS), "contact_time": float(touching.mean()),
            "contact_events": int((touching & ~np.concatenate([[False], touching[:-1]])).sum()),  # runs of consecutive contact steps
            "personal_space_overlap": float((gap < personal_space_radius).mean())}


def traj_gen_metrics(time: np.ndarray, robot: np.ndarray, humans: np.ndarray, goal: np.ndarray,
                     footprint_m: list[float], room_m: float, stations: np.ndarray, raw_speed: np.ndarray | None = None) -> dict:
    """All footprint and task metrics for one path; stations from other_stations(); raw_speed (steps,) only for model plans."""
    time, robot, humans = np.asarray(time, float), np.asarray(robot, float), np.asarray(humans, float)
    to_goal = np.linalg.norm(robot - np.asarray(goal, float), axis=1)
    arrived = np.flatnonzero(to_goal <= GOAL_TOL + GOAL_SLACK)
    out = {**clearance_metrics(robot, humans, footprint_m),
           "goal_reached": bool(arrived.size), "time_to_goal": float(time[arrived[0]] - time[0]) if arrived.size else float("nan"),
           "final_distance_to_goal": float(to_goal[-1]), "out_of_room": bool((np.abs(robot) > room_m / 2).any())}
    out |= station_metrics(robot, stations)
    if raw_speed is not None:
        out |= {"speed_violation_rate": float((np.asarray(raw_speed) > SPEED_CAP).mean()), "raw_speed_max": float(np.max(raw_speed))}
    return out
