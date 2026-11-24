import numpy as np

from preprocessing.bi3_derived import INPUT_HZ, SCORED_STEP
from preprocessing.extract_trips import POST_S, trip_span

GRID = 0.02 + np.arange(1001) / INPUT_HZ  # a 100 s recording's grid, starting at its first sample (0.02 s)


def test_start_and_end_snap_to_the_nearest_grid_sample():
    """A trip start between grid samples moves to the nearer one, by at most half a grid step."""
    first, start, end, last = trip_span(12.35, 19.09, GRID, history_s=3.0)
    assert np.isclose(GRID[start], 12.32) and np.isclose(GRID[end], 19.12)


def test_history_and_padding_are_whole_scored_steps_from_the_start():
    """Every scored sample (every SCORED_STEP-th from the first) must fall on the start: 7 steps of history (2.8 s of the 3 s), and
    padding up to POST_S after the end."""
    first, start, end, last = trip_span(12.35, 19.09, GRID, history_s=3.0)
    assert start - first == 7 * SCORED_STEP
    assert (last - start) % SCORED_STEP == 0
    assert GRID[last] <= GRID[end] + POST_S + 1e-9 < GRID[last] + SCORED_STEP / INPUT_HZ


def test_history_and_padding_stop_at_the_recording_edges():
    """Near the recording's start and end the span is cut there, still in whole scored steps from the start."""
    first, start, end, last = trip_span(1.3, 99.0, GRID, history_s=3.0)
    assert first >= 0 and (start - first) % SCORED_STEP == 0 and first < SCORED_STEP
    assert last <= len(GRID) - 1 < last + SCORED_STEP and (last - start) % SCORED_STEP == 0
    assert trip_span(GRID[0], 10.0, GRID, history_s=3.0)[:2] == [0, 0]  # a robot's first trip starts at the recording start
