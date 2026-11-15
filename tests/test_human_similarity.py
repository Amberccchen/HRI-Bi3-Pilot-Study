import numpy as np
import pytest

from metrics.human_similarity import FEATURE_NAMES, STEP_S, feature_distances, trajectory_features
from traj_gen.analyze import move_features

FAR = np.array([100.0, 100.0])  # another agent too far away to matter


def features(xy, stratum="all", min_displacement_m=0.15):
    xy = np.asarray(xy, dtype=float)
    far = np.tile(FAR, (len(xy), 1))
    return trajectory_features(xy, far, far, stratum, "rec", min_displacement_m=min_displacement_m)


def record(stratum="all", trial="rec", **samples):
    """A per-trajectory record with the given samples, and one zero sample for every other feature."""
    out = {"stratum": stratum, "trial": trial}
    for name in FEATURE_NAMES:
        out[name] = np.asarray(samples.get(name, [0.0]), dtype=float)
    return out


def test_straight_line_has_zero_turning_and_one_speed():
    xy = np.c_[np.arange(10) * 1.2 * STEP_S, np.zeros(10)]
    f = features(xy)
    np.testing.assert_allclose(f["speed"], 1.2)
    np.testing.assert_allclose(f["turning_angle"], 0.0, atol=1e-9)
    assert len(f["turning_angle"]) == 10 - 4


def test_corner_turning_angles_by_step():
    # Along x for four steps, then along y. The angle at step i compares p_i - p_{i-2} with p_{i+2} - p_i.
    xy = [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0), (2, 0.5), (2, 1), (2, 1.5), (2, 2)]
    np.testing.assert_allclose(features(xy)["turning_angle"], [0, 45, 90, 45, 0], atol=1e-9)


def test_turning_skips_near_stationary_displacements():
    still_then_walk = [(0, 0), (0, 0), (0, 0), (0, 0), (0.5, 0), (1, 0), (1.5, 0), (2, 0)]
    walk_then_still = [(0, 0), (0.5, 0), (1, 0), (1.5, 0), (1.5, 0), (1.5, 0), (1.5, 0), (1.5, 0)]
    # Kept angles: steps 4 and 5 in the first path, only step 2 in the second.
    assert len(features(still_then_walk)["turning_angle"]) == 2
    assert len(features(walk_then_still)["turning_angle"]) == 1
    # A 0.1 m displacement is below 0.15 m and above 0.05 m.
    slow = np.c_[np.arange(8) * 0.05, np.zeros(8)]
    assert len(features(slow, min_displacement_m=0.15)["turning_angle"]) == 0
    assert len(features(slow, min_displacement_m=0.05)["turning_angle"]) == 4


def test_minimum_distances_come_from_the_right_agent():
    xy = np.c_[np.arange(5.0), np.zeros(5)]
    robot = np.tile([2.0, 1.0], (5, 1))  # closest at step 2, 1.0 m
    other = np.tile([4.0, 0.6], (5, 1))  # closest at step 4, 0.6 m
    f = trajectory_features(xy, robot, other, "all", "rec", min_displacement_m=0.15)
    np.testing.assert_allclose(f["min_distance_robot"], [1.0])
    np.testing.assert_allclose(f["min_distance_human"], [0.6])


def compare(generated, real, reference_pool, seed=0, n_draws=200):
    rng = np.random.default_rng(seed)
    return feature_distances(generated, real, reference_pool, n_bootstrap=n_draws, rng=rng)


def test_identical_sets_have_zero_distance():
    rng = np.random.default_rng(1)
    base = [record(speed=rng.uniform(0, 2, 5), turning_angle=rng.uniform(0, 180, 3)) for _ in range(6)]
    out = compare(base, base + base, base)
    for name in FEATURE_NAMES:
        assert out[name].w1 == pytest.approx(0.0, abs=1e-12)


def test_shifted_set_distance_equals_the_shift():
    rng = np.random.default_rng(2)
    base = [record(speed=rng.uniform(0, 2, rng.integers(1, 8))) for _ in range(6)]
    shifted = [record(speed=r["speed"] + 0.3) for r in base]
    assert compare(shifted, base, base)["speed"].w1 == pytest.approx(0.3)


def test_samples_are_pooled_not_averaged_per_trajectory():
    # Pooled, the generated samples are 3/4 at 0 and 1/4 at 1, so W1 to all-zero samples is 0.25.
    # Weighting each trajectory equally would give 0.5.
    generated = [record(speed=[0, 0, 0]), record(speed=[1])]
    real = [record(speed=[0]) for _ in range(3)]
    assert compare(generated, real, real)["speed"].w1 == pytest.approx(0.25)


def test_bootstrap_resamples_trajectories_not_samples():
    # Two generated trajectories of 100 identical samples each. Resampling trajectories gives W1 0 (one of
    # each) or 0.5 (two of one), each with probability 1/2. Resampling samples would keep W1 near 0.
    generated = [record(speed=np.zeros(100)), record(speed=np.ones(100))]
    real = [record(speed=np.r_[np.zeros(50), np.ones(50)]) for _ in range(3)]
    low, high = compare(generated, real, real, n_draws=1000)["speed"].w1_interval
    assert low == pytest.approx(0.0, abs=1e-12)
    assert high == pytest.approx(0.5)


def test_reference_draws_one_other_trial_against_the_real_set_without_it():
    # Trial A moves at 0 and B at 1, and the real set is both. Each draw compares one trial with the other,
    # so every reference W1 is 1. Comparing a trial with a real set that still holds its own moves gives 0.5.
    # Trial G, which the generated set comes from, must never be drawn: its moves at 5 would give W1 above 1.
    generated = [record(speed=[0.0], trial="G")]
    a = [record(speed=[0.0], trial=("um", "um1")) for _ in range(3)]  # labels as analyze gives them: (site, trial)
    b = [record(speed=[1.0], trial=("um", "um2")) for _ in range(3)]
    g = [record(speed=[5.0], trial="G")]
    out = compare(generated, a + b, a + b + g)["speed"]
    assert out.n_reference_used == 2
    assert out.reference_median == pytest.approx(1.0)
    assert out.reference_interval == pytest.approx((1.0, 1.0))  # min and max of two draws


def test_reference_matches_the_generated_station_mix_within_a_trial():
    # Trial A has three moves of stratum "a" at 0 and three of "b" at 5. The generated set has two moves of "a",
    # so A's draw is two of its "a" moves, which are at 0 like the real set: W1 0. Drawing A's moves regardless of
    # stratum would bring in moves at 5.
    generated = [record("a", speed=[0.0], trial="G") for _ in range(2)]
    real = [record("a", speed=[0.0], trial="R") for _ in range(4)]
    pool = [record("a", speed=[0.0], trial="A") for _ in range(3)] + [record("b", speed=[5.0], trial="A") for _ in range(3)]
    out = compare(generated, real, pool)["speed"]
    assert out.reference_median == pytest.approx(0.0, abs=1e-12)
    assert out.reference_moves_median == 2


def test_draws_without_samples_of_a_feature_are_skipped_for_that_feature_only():
    # Trial B's moves have no turning angle (too short), so B's draw has no turning sample: it is skipped for
    # turning and still used for speed. A generated set without any turning sample has no turning W1.
    generated = [record(speed=[1.0], turning_angle=[], trial="G")]
    real = [record(speed=[1.0], turning_angle=[10.0], trial="R")]
    pool = [record(speed=[1.0], turning_angle=[10.0], trial="A"), record(speed=[2.0], turning_angle=[], trial="B")]
    out = compare(generated, real, pool, n_draws=50)
    assert out["speed"].n_reference_used == 2 and out["turning_angle"].n_reference_used == 1
    assert out["speed"].reference_interval == pytest.approx((0.0, 1.0))
    assert np.isnan(out["turning_angle"].w1) and out["turning_angle"].n_bootstrap_used == 0
    assert out["speed"].n_bootstrap_used == 50


def test_moves_of_one_participant_pair_share_a_trial_label_across_controllers():
    # One trial is one participant pair that ran every controller. Labelling moves by recording (trial and controller)
    # would split the pair into several reference draws and compare a pair's moves with the same pair's other runs.
    def trip(controller):
        return {"agent": "human0", "start_goal": "0", "goal": "1", "site": "um", "trial": "um1", "controller": controller}

    positions = {"human0": np.zeros((5, 2)), "robot": np.ones((5, 2)), "human1": np.ones((5, 2))}
    assert move_features(trip("cv"), positions, 4)["trial"] == move_features(trip("hst"), positions, 4)["trial"] == ("um", "um1")
