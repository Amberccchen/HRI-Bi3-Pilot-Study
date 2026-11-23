import numpy as np
import pytest

from preprocessing import extract_trips
from preprocessing.extract_trips import INPUT_BIN_S, INPUT_HZ, binned_track, drop_repeated
from tests.test_human_trips import OVERLAPPING_AND_SAME_STATION_VISITS, fake_recording, visit


def track(time, x, theta=None):
    x = np.asarray(x, float)
    return np.asarray(time, float), np.c_[x, np.zeros_like(x), np.zeros_like(x) if theta is None else theta]


def test_sample_is_the_mean_of_its_centred_bin():
    """A bin anchored at its start, or closed at its end, would pick up the wrong raw samples."""
    time, xyt = track([0.94, 0.96, 1.0, 1.04, 1.05, 1.06], [10.0, 1.0, 2.0, 3.0, 20.0, 30.0])
    values, empty = binned_track(time, xyt, np.array([1.0]))
    # bin [0.95, 1.05): 0.96, 1.0 and 1.04; 0.94 is before it, 1.05 and 1.06 are at or after its end
    assert values[0, 0] == pytest.approx(2.0) and not empty[0]


def test_repeated_samples_are_dropped_before_averaging():
    """Mocap holds the last position between updates; counting the held copies would weight a position by how long it was held."""
    time, xyt = track([0.96, 0.98, 1.0, 1.02, 1.04], [0.0, 0.0, 0.0, 0.0, 1.0])
    t, p = drop_repeated(time, xyt)
    values, _ = binned_track(t, p, np.array([1.0]))
    assert values[0, 0] == pytest.approx(0.5)


def test_empty_bin_takes_the_last_sample_before_its_end():
    """A tracking gap: the empty bin keeps the latest known position instead of looking ahead to the next update."""
    time, xyt = track([0.0, 0.86, 1.3], [0.0, 1.0, 5.0])
    values, empty = binned_track(time, xyt, np.array([0.0, 0.9, 1.0, 1.3]))
    assert empty.tolist() == [False, False, True, False]
    np.testing.assert_allclose(values[:, 0], [0.0, 1.0, 1.0, 5.0])


def test_grid_time_before_any_raw_sample_is_refused():
    time, xyt = track([1.0, 1.1], [0.0, 1.0])
    with pytest.raises(ValueError):
        binned_track(time, xyt, np.array([0.5]))


def test_heading_is_a_circular_mean():
    """Headings either side of +-pi average to pi, not to 0."""
    time, xyt = track([0.97, 0.99, 1.01, 1.03], [0.0, 1.0, 2.0, 3.0], theta=[np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05, -np.pi + 0.05])
    values, _ = binned_track(time, xyt, np.array([1.0]))
    assert abs(np.angle(np.exp(1j * (values[0, 2] - np.pi)))) < 1e-9


def test_no_sample_uses_a_raw_sample_half_a_bin_or_more_after_its_time():
    """A step in the raw positions at time s: every sample before s - bin / 2 must not see it, whatever the grid phase."""
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0, 10, 500))
    step = time[250]
    t, p = track(time, (time >= step).astype(float))
    grid = np.arange(0.05, 9.9, 0.01)
    values, _ = binned_track(t, p, grid)
    assert np.all(values[grid + INPUT_BIN_S / 2 <= step, 0] == 0)
    assert np.all(values[grid - INPUT_BIN_S / 2 >= step, 0] == 1)


# Trip 0 starts at 1.3 s, so its history is cut at the recording start; trip 1 ends with its padding cut at the recording end (30 s).
EDGE_VISITS = [visit(0, 0, "0", 0.2, 0.8, 1.3), visit(1, 2, "2", 6.0, 7.03, 8.0), visit(2, 5, "5", 16.0, 17.0, 18.0)]


def test_series_spans_the_track_grid_and_every_fourth_sample_is_a_track_time(monkeypatch):
    """At the recording's edges as well, the series starts and ends with the 2.5 Hz track, shares its times and its start, and its
    end index is the last sample not after the trip end."""
    fake_recording(monkeypatch, EDGE_VISITS)
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "5")}
    trips = extract_trips.human_trips("laas", "laas99", "cv", human_xy, hz=2.5, history_s=3.0)
    assert [t["start_index"] for t in trips] == [3, 7] and trips[1]["time"][-1] < trips[1]["duration_s"] + extract_trips.POST_S
    factor = int(round(INPUT_HZ / 2.5))
    for trip in trips:
        series = trip["input_tracks_10hz"]
        assert len(series["time"]) == factor * (len(trip["time"]) - 1) + 1 and series["time"][::factor] == trip["time"]
        np.testing.assert_allclose(np.diff(series["time"]), 1 / INPUT_HZ, atol=1.5e-3)  # times are stored to 1 ms
        assert series["start_index"] == factor * trip["start_index"] and series["time"][series["start_index"]] == 0
        duration = trip["t_end"] - trip["t_start"]
        assert series["time"][series["end_index"]] <= duration + 1e-6 < series["time"][series["end_index"]] + 1 / INPUT_HZ


def test_person_trip_series_matches_tracks_at_shared_times(monkeypatch):
    """On a straight walk sampled at 10 Hz, the series at the track times equals the track away from the recording's ends, in the room frame."""
    fake_recording(monkeypatch, OVERLAPPING_AND_SAME_STATION_VISITS)
    human_xy = {label: [float(label), 1.0] for label in ("0", "2", "3", "5")}
    for trip in extract_trips.human_trips("laas", "laas99", "cv", human_xy, hz=2.5, history_s=3.0):
        series = trip["input_tracks_10hz"]
        for agent, rows in trip["tracks"].items():
            dense = np.asarray(series["tracks"][agent])[::4]
            middle = slice(2, -2)  # the 0.5 s average is cut at the recording's ends
            np.testing.assert_allclose(dense[middle, :2], np.asarray(rows)[middle, :2], atol=2e-3)
