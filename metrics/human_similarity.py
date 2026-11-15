"""Similarity of human trajectories to real human moves, compared feature by feature.

A trajectory is one goal-to-goal move of one person, sampled every STEP_S seconds: from leaving the start
station's visit radius to entering the goal station's. Generated and real trajectories must be cut by that
same rule before their features are compared, so that both sets describe the same segment of a move.

The features follow THOR (Rudenko et al., RA-L 2020): speed per step, the absolute turning angle between
consecutive 0.8 s displacements, the trajectory's minimum distance to the robot, and its minimum distance to
the other person. Distances are between body centres.

Two sets are compared per feature by the Wasserstein-1 distance (W1) in the feature's units. Each W1 comes
with a bootstrap interval over trajectories and with a real-against-real reference: the W1 between sets of
real trajectories with the generated set's size and station mix and the same real set.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

STEP_S = 0.4  # sampling interval of every input track, s
TURN_WINDOW_STEPS = 2  # turning angles compare displacements over 2 steps, 0.8 s
FEATURE_NAMES = ("speed", "turning_angle", "min_distance_robot", "min_distance_human")


def trajectory_features(xy, robot_xy, other_human_xy, stratum, *, min_displacement_m):
    """Per-feature samples of one trajectory.

    Args:
        xy: (T, 2) positions of the person, metres, every STEP_S seconds.
        robot_xy: (T, 2) robot positions at the same times.
        other_human_xy: (T, 2) positions of the other person at the same times.
        stratum: hashable label of the move, the (start station, goal station) pair. Resampling in
            `feature_distances` stays within strata.
        min_displacement_m: a turning angle is kept only when both of its 0.8 s displacements are at least
            this long. Shorter displacements are near-stationary and their direction is noise.

    Returns:
        dict with "stratum" and one sample array per name in FEATURE_NAMES: "speed" (T-1,) in m/s,
        "turning_angle" (K,) in degrees in [0, 180] with K <= T-4, "min_distance_robot" (1,) and
        "min_distance_human" (1,) in metres.
    """
    if min_displacement_m <= 0:
        raise ValueError(f"min_displacement_m must be positive, got {min_displacement_m}")
    # A (1, 2) track of another agent would broadcast against xy without an error.
    if not len(xy) == len(robot_xy) == len(other_human_xy):
        raise ValueError(
            f"tracks differ in length: {len(xy)}, {len(robot_xy)}, {len(other_human_xy)}"
        )
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / STEP_S

    # The angle at step i is between p_i - p_{i-2} and p_{i+2} - p_i.
    window = TURN_WINDOW_STEPS
    displacement = xy[window:] - xy[:-window]
    incoming, outgoing = displacement[:-window], displacement[window:]
    moving = (np.linalg.norm(incoming, axis=1) >= min_displacement_m) & (
        np.linalg.norm(outgoing, axis=1) >= min_displacement_m
    )
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    dot = np.sum(incoming * outgoing, axis=1)
    turning_angle = np.degrees(np.abs(np.arctan2(cross, dot)))[moving]

    return {
        "stratum": stratum,
        "speed": speed,
        "turning_angle": turning_angle,
        "min_distance_robot": np.array([np.linalg.norm(xy - robot_xy, axis=1).min()]),
        "min_distance_human": np.array([np.linalg.norm(xy - other_human_xy, axis=1).min()]),
    }


@dataclass(frozen=True)
class FeatureDistance:
    """W1 between a generated and a real set for one feature, in the feature's units."""

    w1: float
    w1_interval: tuple[float, float]  # bootstrap 2.5 and 97.5 percentiles
    reference_median: float  # W1 of real sets with the generated set's size and station mix to the real set
    reference_interval: tuple[float, float]  # 2.5 and 97.5 percentiles over reference draws
    n_generated: int
    n_real: int


def feature_distances(generated, real, reference_pool, *, n_bootstrap, n_reference_draws, rng):
    """Compare a generated set of trajectories with a real set, feature by feature.

    W1 is computed between the samples of all trajectories of each set, pooled. The bootstrap resamples
    trajectories with replacement within each stratum, in both sets. Each reference draw takes, from every
    stratum of `reference_pool`, as many trajectories as the generated set has in that stratum, without
    replacement, and computes their W1 to `real`. The reference is then a set of real trajectories with the
    generated set's size and station mix, compared with the same real set. Giving every record the same
    stratum makes both resamplings unstratified.

    When the generated set is close to the real set, each bootstrap draw adds sampling noise to W1, so the
    bootstrap interval sits above W1 and can exclude it. In that range the reference interval is the
    comparison that tells a real-like set from an unlike one.

    Args:
        generated: list of records from `trajectory_features`.
        real: list of records from `trajectory_features`, the real set compared with.
        reference_pool: list of real records, disjoint from `real`, that reference sets are drawn from.
        n_bootstrap: number of bootstrap draws for the W1 interval.
        n_reference_draws: number of real-against-real draws.
        rng: numpy Generator for all draws.

    Returns:
        dict from each name in FEATURE_NAMES to its FeatureDistance.

    Raises:
        ValueError: if a stratum of the generated set has more trajectories than the same stratum of
            `reference_pool`.
    """
    if n_bootstrap < 1 or n_reference_draws < 1:
        raise ValueError(f"need at least one draw, got {n_bootstrap} and {n_reference_draws}")

    def indices_by_stratum(records):
        groups = {}
        for i, record in enumerate(records):
            groups.setdefault(record["stratum"], []).append(i)
        return {stratum: np.array(idx) for stratum, idx in groups.items()}

    generated_strata = indices_by_stratum(generated)
    real_strata = indices_by_stratum(real)
    pool_strata = indices_by_stratum(reference_pool)
    for stratum, idx in generated_strata.items():
        n_available = len(pool_strata.get(stratum, ()))
        if len(idx) > n_available:
            raise ValueError(
                f"stratum {stratum!r}: {len(idx)} generated trajectories, {n_available} in the reference pool"
            )

    def resample(strata):
        return np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata.values()])

    bootstrap_draws = [(resample(generated_strata), resample(real_strata)) for _ in range(n_bootstrap)]
    reference_draws = [
        np.concatenate(
            [rng.choice(pool_strata[stratum], size=len(idx), replace=False) for stratum, idx in generated_strata.items()]
        )
        for _ in range(n_reference_draws)
    ]

    def flatten(records, name):
        lengths = np.array([len(record[name]) for record in records])
        return np.concatenate([record[name] for record in records]), np.cumsum(lengths) - lengths, lengths

    def gather(flat, starts, lengths, indices):
        """Samples of the trajectories at `indices`, concatenated in that order (repeats allowed)."""
        selected = lengths[indices]
        output_starts = np.cumsum(selected) - selected
        return flat[np.repeat(starts[indices] - output_starts, selected) + np.arange(selected.sum())]

    all_generated, all_real = np.arange(len(generated)), np.arange(len(real))
    distances = {}
    for name in FEATURE_NAMES:
        generated_flat, real_flat, pool_flat = flatten(generated, name), flatten(real, name), flatten(reference_pool, name)

        def w1(first_flat, first_indices, real_indices):
            return wasserstein_distance(gather(*first_flat, first_indices), gather(*real_flat, real_indices))

        bootstrap = [w1(generated_flat, g, r) for g, r in bootstrap_draws]
        reference = [w1(pool_flat, subset, all_real) for subset in reference_draws]
        distances[name] = FeatureDistance(
            w1=w1(generated_flat, all_generated, all_real),
            w1_interval=tuple(float(v) for v in np.percentile(bootstrap, [2.5, 97.5])),
            reference_median=float(np.median(reference)),
            reference_interval=tuple(float(v) for v in np.percentile(reference, [2.5, 97.5])),
            n_generated=len(generated),
            n_real=len(real),
        )
    return distances
