"""Motion metrics from the Bi3 paper (Stratton et al., ICRA 2026, Sec. VI, Fig. 7), for the path of one planned agent.

The paper reports minimum distance, acceleration, velocity and path efficiency of the robot per
controller and site, but defines only path efficiency precisely. The readings used here are:

- minimum distance: centre-to-centre distance between the agent's and each other agent's
  tracked position (what the mocap gives), minimised over time and the other agents. The paper
  reports it in mm; these functions use metres.
- velocity: the paper does not say which statistic; mean, median, max and std of the
  agent's speed are all returned, with the mean as the likely match.
- acceleration: "average acceleration"; the mean magnitude of the agent's 2-D
  acceleration vector is the primary value, with the mean absolute change in speed
  (tangential acceleration) and the max also returned.
- path efficiency: straight-line start-to-end distance divided by distance travelled.

Inputs are one trajectory segment (e.g. one robot trip between stations) as arrays:
time (T,) in seconds, agent (T, 2) and others (T, K, 2) positions in metres. Velocity
and acceleration use finite differences, so pass trajectories that are already
cleaned and resampled (raw 50 Hz mocap with repeated samples gives spiky values).
"""

import numpy as np


def min_distance(agent: np.ndarray, others: np.ndarray) -> dict:
    """Closest centre-to-centre distance (m) from the agent to the other agents, overall and per other agent."""
    d = np.linalg.norm(others - agent[:, None, :], axis=-1)  # (T, K)
    per_other = np.nanmin(d, axis=0)
    t_idx, k_idx = np.unravel_index(np.nanargmin(d), d.shape)
    return {"min_distance": float(per_other.min()), "min_distance_per_other": per_other.tolist(),
            "min_distance_index": int(t_idx), "min_distance_other": int(k_idx)}


def velocity(time: np.ndarray, agent: np.ndarray) -> dict:
    """Speed statistics (m/s) of the agent."""
    speed = np.linalg.norm(np.gradient(agent, time, axis=0), axis=1)
    return {"velocity_mean": float(speed.mean()), "velocity_median": float(np.median(speed)),
            "velocity_max": float(speed.max()), "velocity_std": float(speed.std())}


def acceleration(time: np.ndarray, agent: np.ndarray) -> dict:
    """Acceleration statistics (m/s^2) of the agent."""
    vel = np.gradient(agent, time, axis=0)
    acc = np.linalg.norm(np.gradient(vel, time, axis=0), axis=1)
    tangential = np.abs(np.gradient(np.linalg.norm(vel, axis=1), time))
    return {"acceleration_mean": float(acc.mean()), "acceleration_max": float(acc.max()),
            "acceleration_tangential_mean": float(tangential.mean())}


def path_efficiency(agent: np.ndarray) -> dict:
    """Straight-line start-to-end distance / cumulative path length (1 = straight)."""
    travelled = float(np.linalg.norm(np.diff(agent, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(agent[-1] - agent[0]))
    return {"path_efficiency": straight / travelled if travelled > 0 else float("nan"),
            "path_length": travelled, "straight_distance": straight}


def bi3_metrics(time: np.ndarray, agent: np.ndarray, others: np.ndarray) -> dict:
    """All four Bi3 metrics for one trajectory segment of the agent, against the other agents."""
    time, agent, others = np.asarray(time, float), np.asarray(agent, float), np.asarray(others, float)
    return {**min_distance(agent, others), **velocity(time, agent), **acceleration(time, agent), **path_efficiency(agent)}
