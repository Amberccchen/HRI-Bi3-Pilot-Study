import numpy as np

from metrics.traj_gen_metrics import move_end_index


def test_move_ends_at_the_start_of_the_run_into_the_station():
    """Walk towards a station, stop beside it, step back: the move ends where the final approach run begins, not at the
    closest approach and not where the path first came near the station earlier."""
    site = {"stations": {"4": {"members": ["4"], "xy": [2.0, 0.0]}}}
    goal = np.array([2.3, 0.0])  # where people stand, beyond the station position
    path = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [1.8, 0.0], [1.95, 0.0], [2.0, 0.0], [2.05, 0.0]])
    # Arrival (0.4 m of the goal) is first at index 5 (0.35 m), so the closest approach is searched in 0..5: index 5 at 0.05 m
    # from the station, and the run within 0.05 + 0.15 = 0.2 m starts at index 4 (0.2 m). Searching the whole path would find
    # index 6 (0 m) and a run starting at index 5.
    assert move_end_index(path, goal, 0.4, site, "4") == 4


def test_pair_label_uses_the_nearer_member_and_no_arrival_gives_none():
    """A shared table: the distance is to the nearer of its two stations. A path that never comes within the arrival radius
    plus the goal slack has no move."""
    site = {"stations": {"0": {"members": ["0"], "xy": [2.0, 0.5]}, "1": {"members": ["1"], "xy": [2.0, -0.5]},
                         "0/1": {"members": ["0", "1"], "xy": [2.0, 0.0]}}}
    goal = np.array([2.3, 0.0])
    path = np.array([[0.0, -0.5], [0.8, -0.5], [1.4, -0.5], [1.75, -0.5], [2.0, -0.45]])
    # To the nearer member: 2, 1.2, 0.6, 0.25, 0.05 m. The last point is 0.541 m from the goal, outside 0.4 m but inside
    # 0.4 + 0.15 m, so the whole path is searched: closest at index 4, and index 3 is outside 0.05 + 0.15 m. With the other
    # member alone the run would start at index 3.
    assert move_end_index(path, goal, 0.4, site, "0/1") == 4
    assert move_end_index(path[:3], goal, 0.4, site, "0/1") is None
