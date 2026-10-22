"""Footprint-aware and task metrics for generated (or replayed) robot paths.

bi3_metrics.py scores motion with centre-to-centre distances, as the Bi3 paper does.
Generated paths also need to be judged as navigation: does the robot's body hit a
person, does it reach its goal, does it stay in the room. The robot is a disk and a
person is a disk of HUMAN_RADIUS, so:

- clearance: centre distance - robot radius - HUMAN_RADIUS, minimised over time and
  people; negative means the disks overlap (footprint_overlap). The robot disk is the
  circumscribed one (conservative), so the real controllers overlap often too; read it as a
  rate relative to the replayed real path, not as contact.
- goal reached: the path gets within GOAL_TOL + GOAL_SLACK of the goal station. Bi3
  switches goals at about GOAL_TOL on the raw mocap, but the cleaned room-frame real
  paths end 0.49-0.60 m from the goal, so the slack keeps the real controllers from
  being scored as failures. Time to goal is the first such sample.
- final distance to goal: distance of the last sample, also for paths that never arrive.
- out of room: any sample outside the square room of side room_m.
- speed cap: for a plan whose raw (before speed limiting) per-step speeds are given,
  the fraction of steps above SPEED_CAP and the largest raw speed.

Inputs are one executed path as arrays: time (T,) in seconds from trip start, robot
(T, 2) and humans (T, H, 2) in the room frame (metres), goal (2,); radius and room
size come from site_frames.json.
"""

import numpy as np


HUMAN_RADIUS = 0.25  # typical shoulder half-width; mocap tracks only give the body centre
GOAL_TOL = 0.5       # arrival distance given to the model; Bi3 switches to the next goal at about this
GOAL_SLACK = 0.1     # smoothing/mocap offset: real paths end up to 0.60 m from the goal at the switch
SPEED_CAP = 0.3      # m/s, the limit the model is told


def traj_gen_metrics(time: np.ndarray, robot: np.ndarray, humans: np.ndarray, goal: np.ndarray,
                     radius: float, room_m: float, raw_speed: np.ndarray | None = None) -> dict:
    """All footprint and task metrics for one path; raw_speed (steps,) only for model plans."""
    time, robot, humans = np.asarray(time, float), np.asarray(robot, float), np.asarray(humans, float)
    clearance = np.linalg.norm(humans - robot[:, None, :], axis=-1) - radius - HUMAN_RADIUS  # (T, H)
    to_goal = np.linalg.norm(robot - np.asarray(goal, float), axis=1)
    arrived = np.flatnonzero(to_goal <= GOAL_TOL + GOAL_SLACK)
    out = {"clearance_min": float(clearance.min()), "footprint_overlap": bool((clearance < 0).any()),
           "goal_reached": bool(arrived.size), "time_to_goal": float(time[arrived[0]] - time[0]) if arrived.size else float("nan"),
           "final_distance_to_goal": float(to_goal[-1]), "out_of_room": bool((np.abs(robot) > room_m / 2).any())}
    if raw_speed is not None:
        out |= {"speed_violation_rate": float((np.asarray(raw_speed) > SPEED_CAP).mean()), "raw_speed_max": float(np.max(raw_speed))}
    return out
