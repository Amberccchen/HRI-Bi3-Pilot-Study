"""Similarity of human trajectories to real human moves, compared feature by feature.

A trajectory is one goal-to-goal move of one person, sampled every STEP_S seconds: from leaving the start
station's visit radius to entering the goal station's. Generated and real trajectories must be cut by that
same rule before their features are compared, so that both sets describe the same segment of a move.

The features follow THOR (Rudenko et al., RA-L 2020): speed per step, the absolute turning angle between
consecutive 0.8 s displacements, the trajectory's minimum distance to the robot, and its minimum distance to
the other person. Distances are between body centres.

Two sets are compared per feature by the Wasserstein-1 distance (W1) in the feature's units. Each W1 comes
with a bootstrap interval over trajectories and with a real-against-real reference: the W1 of the moves of one
other recording (one trial and controller, so other participants), matched to the generated set's station mix as
far as that recording allows, to the same real set, over every such recording. A generated set comes from a few
recordings, so a reference of real moves drawn independently across recordings would leave out how much people
differ between recordings.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance

from preprocessing.bi3_derived import TRIP_HZ

STEP_S = 1 / TRIP_HZ  # sampling interval of every input track, s
TURN_WINDOW_STEPS = 2  # turning angles compare displacements over 2 steps, 0.8 s
FEATURE_NAMES = ("speed", "turning_angle", "min_distance_robot", "min_distance_human")


def trajectory_features(xy, robot_xy, other_human_xy, stratum, recording, *, min_displacement_m):
    """Per-feature samples of one trajectory.

    Args:
        xy: (T, 2) positions of the person, metres, every STEP_S seconds.
        robot_xy: (T, 2) robot positions at the same times.
        other_human_xy: (T, 2) positions of the other person at the same times.
        stratum: hashable label of the move, the (start station, goal station) pair. Resampling in
            `feature_distances` stays within strata.
        recording: hashable label of the recording the move comes from (site, trial and controller), which
            `feature_distances` draws reference sets by.
        min_displacement_m: a turning angle is kept only when both of its 0.8 s displacements are at least
            this long. Shorter displacements are near-stationary and their direction is noise.

    Returns:
        dict with "stratum", "recording" and one sample array per name in FEATURE_NAMES: "speed" (T-1,) in m/s,
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
        "recording": recording,
        "speed": speed,
        "turning_angle": turning_angle,
        "min_distance_robot": np.array([np.linalg.norm(xy - robot_xy, axis=1).min()]),
        "min_distance_human": np.array([np.linalg.norm(xy - other_human_xy, axis=1).min()]),
    }


@dataclass(frozen=True)
class FeatureDistance:
    """W1 between a generated and a real set for one feature, in the feature's units."""

    w1: float  # NaN when either set has no sample of the feature
    w1_interval: tuple[float, float]  # bootstrap 2.5 and 97.5 percentiles over the draws used
    reference_median: float  # median over reference recordings of the W1 of their matched moves to the real set
    reference_interval: tuple[float, float]  # 2.5 and 97.5 percentiles over reference recordings, or min and max below MIN_PERCENTILE_DRAWS
    n_generated: int
    n_real: int
    n_bootstrap_used: int  # bootstrap draws with samples of the feature in both resampled sets
    n_reference_used: int  # reference recordings with samples of the feature in their matched moves
    reference_moves_median: float  # median number of moves in a reference recording's matched set


MIN_PERCENTILE_DRAWS = 40  # fewer reference draws than this: the interval is their min and max, as 2.5 percentiles of so few are not estimated


def feature_distances(generated, real, reference_pool, *, n_bootstrap, rng):
    """Compare a generated set of trajectories with a real set, feature by feature.

    W1 is computed between the samples of all trajectories of each set, pooled. The bootstrap resamples
    trajectories with replacement within each stratum, in both sets. The reference has one draw per recording of
    `reference_pool` that none of the generated trajectories comes from: from each stratum of the generated set,
    as many of that recording's trajectories as the generated set has there, or all it has if fewer, without
    replacement, and their W1 to `real` without that recording's own trajectories. A draw (bootstrap or
    reference) whose pooled samples of a feature are empty in either set is skipped for that feature only, and
    the draws used are counted.

    When the generated set is close to the real set, each bootstrap draw adds sampling noise to W1, so the
    bootstrap interval sits above W1 and can exclude it. In that range the reference interval is the
    comparison that tells a real-like set from an unlike one.

    Args:
        generated: list of records from `trajectory_features`.
        real: list of records from `trajectory_features`, the real set compared with.
        reference_pool: list of real records whose recordings the reference draws come from.
        n_bootstrap: number of bootstrap draws for the W1 interval.
        rng: numpy Generator for all draws.

    Returns:
        dict from each name in FEATURE_NAMES to its FeatureDistance.
    """
    if n_bootstrap < 1:
        raise ValueError(f"need at least one bootstrap draw, got {n_bootstrap}")

    def indices_by(records, key):
        groups = {}
        for i, record in enumerate(records):
            groups.setdefault(record[key], []).append(i)
        return {label: np.array(idx) for label, idx in groups.items()}

    generated_strata = indices_by(generated, "stratum")
    real_strata = indices_by(real, "stratum")

    def resample(strata):
        return np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata.values()])

    bootstrap_draws = [(resample(generated_strata), resample(real_strata)) for _ in range(n_bootstrap)]

    generated_recordings = {record["recording"] for record in generated}
    real_recording = [record["recording"] for record in real]
    reference_draws = []  # (indices into reference_pool, indices into real) per recording
    for recording, idx in indices_by(reference_pool, "recording").items():
        if recording in generated_recordings:
            continue
        own_strata = indices_by([reference_pool[i] for i in idx], "stratum")
        chosen = [idx[rng.choice(own_strata[stratum], size=min(len(want), len(own_strata[stratum])), replace=False)]
                  for stratum, want in generated_strata.items() if stratum in own_strata]
        if chosen:
            reference_draws.append((np.concatenate(chosen), np.array([i for i, r in enumerate(real_recording) if r != recording], dtype=int)))

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
            """W1 of the two sets of trajectories, or None when either has no sample."""
            first, second = gather(*first_flat, first_indices), gather(*real_flat, real_indices)
            return wasserstein_distance(first, second) if first.size and second.size else None

        bootstrap = [d for d in (w1(generated_flat, g, r) for g, r in bootstrap_draws) if d is not None]
        reference = [d for d in (w1(pool_flat, subset, rest) for subset, rest in reference_draws) if d is not None]
        point = w1(generated_flat, all_generated, all_real)
        reference_edges = np.percentile(reference, [2.5, 97.5]) if len(reference) >= MIN_PERCENTILE_DRAWS else (min(reference, default=np.nan), max(reference, default=np.nan))
        distances[name] = FeatureDistance(
            w1=float("nan") if point is None else point,
            w1_interval=tuple(float(v) for v in np.percentile(bootstrap, [2.5, 97.5])) if bootstrap else (float("nan"),) * 2,
            reference_median=float(np.median(reference)) if reference else float("nan"),
            reference_interval=tuple(float(v) for v in reference_edges),
            n_generated=len(generated),
            n_real=len(real),
            n_bootstrap_used=len(bootstrap),
            n_reference_used=len(reference),
            reference_moves_median=float(np.median([len(subset) for subset, _ in reference_draws])) if reference_draws else float("nan"),
        )
    return distances
