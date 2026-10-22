"""Robot motion metrics from the Bi3 paper (Stratton et al., ICRA 2026, Sec. VI, Fig. 7).

The paper reports minimum distance, acceleration, velocity and path efficiency per
controller and site, but defines only path efficiency precisely. The readings used
here are:

- minimum distance: centre-to-centre distance between the robot's and each person's
  tracked position (what the mocap gives), minimised over time and people. The paper
  reports it in mm; these functions use metres.
- velocity: the paper does not say which statistic; mean, median, max and std of the
  robot's speed are all returned, with the mean as the likely match.
- acceleration: "average acceleration"; the mean magnitude of the robot's 2-D
  acceleration vector is the primary value, with the mean absolute change in speed
  (tangential acceleration) and the max also returned.
- path efficiency: straight-line start-to-end distance divided by distance travelled.

Inputs are one trajectory segment (e.g. one robot trip between stations) as arrays:
time (T,) in seconds, robot (T, 2) and humans (T, H, 2) positions in metres. Velocity
and acceleration use finite differences, so pass trajectories that are already
cleaned and resampled (raw 50 Hz mocap with repeated samples gives spiky values).
"""

import numpy as np


def min_distance(robot: np.ndarray, humans: np.ndarray) -> dict:
    """Closest centre-to-centre robot-person distance (m), overall and per person."""
    d = np.linalg.norm(humans - robot[:, None, :], axis=-1)  # (T, H)
    per_person = np.nanmin(d, axis=0)
    t_idx, h_idx = np.unravel_index(np.nanargmin(d), d.shape)
    return {"min_distance": float(per_person.min()), "min_distance_per_person": per_person.tolist(),
            "min_distance_index": int(t_idx), "min_distance_person": int(h_idx)}


def velocity(time: np.ndarray, robot: np.ndarray) -> dict:
    """Robot speed statistics (m/s)."""
    speed = np.linalg.norm(np.gradient(robot, time, axis=0), axis=1)
    return {"velocity_mean": float(speed.mean()), "velocity_median": float(np.median(speed)),
            "velocity_max": float(speed.max()), "velocity_std": float(speed.std())}


def acceleration(time: np.ndarray, robot: np.ndarray) -> dict:
    """Robot acceleration statistics (m/s^2)."""
    vel = np.gradient(robot, time, axis=0)
    acc = np.linalg.norm(np.gradient(vel, time, axis=0), axis=1)
    tangential = np.abs(np.gradient(np.linalg.norm(vel, axis=1), time))
    return {"acceleration_mean": float(acc.mean()), "acceleration_max": float(acc.max()),
            "acceleration_tangential_mean": float(tangential.mean())}


def path_efficiency(robot: np.ndarray) -> dict:
    """Straight-line start-to-end distance / cumulative path length (1 = straight)."""
    travelled = float(np.linalg.norm(np.diff(robot, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(robot[-1] - robot[0]))
    return {"path_efficiency": straight / travelled if travelled > 0 else float("nan"),
            "path_length": travelled, "straight_distance": straight}


def bi3_metrics(time: np.ndarray, robot: np.ndarray, humans: np.ndarray) -> dict:
    """All four Bi3 metrics for one trajectory segment."""
    time, robot, humans = np.asarray(time, float), np.asarray(robot, float), np.asarray(humans, float)
    return {**min_distance(robot, humans), **velocity(time, robot), **acceleration(time, robot), **path_efficiency(robot)}
