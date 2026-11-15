import numpy as np

from preprocessing.extract_trips import POST_S, trip_grid


def test_grid_is_anchored_at_the_trip_start():
    """A trip start between grid steps of the recording: the start state must still be sampled at time 0 exactly."""
    t0, t1, hz = 12.37, 19.0, 2.5
    grid, i0, i1 = trip_grid(t0, t1, t_first=0.0, t_last=100.0, hz=hz, history_s=3.0)
    time = np.round(grid - t0, 3)  # as stored in the trips
    assert time[i0] == 0
    np.testing.assert_allclose(np.diff(grid), 1 / hz)
    assert i0 == 7  # floor(3.0 * 2.5) history samples
    assert grid[i1] <= t1 < grid[i1] + 1 / hz
    assert grid[-1] <= t1 + POST_S < grid[-1] + 1 / hz


def test_history_and_padding_stop_at_the_recording_edges():
    """Near the recording's start and end the grid is cut there, and t0 stays on it."""
    t0, t1, hz = 1.3, 9.9, 2.5
    grid, i0, i1 = trip_grid(t0, t1, t_first=0.0, t_last=10.0, hz=hz, history_s=3.0)
    assert grid[0] >= 0.0 and grid[-1] <= 10.0
    assert i0 == 3 and np.round(grid[i0] - t0, 3) == 0
    assert grid[i1] <= t1
