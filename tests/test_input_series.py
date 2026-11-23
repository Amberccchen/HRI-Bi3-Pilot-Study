import json

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from metrics.bi3_metrics import bi3_metrics
from preprocessing.bi3_derived import INPUT_HZ, RECORDINGS_FILE, SCORED_STEP, load_trips, trip_arrays
from preprocessing.extract_trips import GAP_CONTEXT_S, INPUT_BIN_S, binned_track, drop_repeated, recording_arrays
from tests.test_human_trips import OVERLAPPING_AND_SAME_STATION_VISITS, person_trips, visit


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


def test_heading_is_a_circular_mean():
    """Headings either side of +-pi average to pi, not to 0."""
    time, xyt = track([0.97, 0.99, 1.01, 1.03], [0.0, 1.0, 2.0, 3.0], theta=[np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05, -np.pi + 0.05])
    values, _ = binned_track(time, xyt, np.array([1.0]))
    assert abs(np.angle(np.exp(1j * (values[0, 2] - np.pi)))) < 1e-9


def gap_track():
    """Raw samples at 50 Hz along a curve, with no samples between 2.0 and 3.0 s (bins 2.1 to 2.9 are empty), heading turning through +-pi."""
    time = np.r_[np.arange(0.0, 2.0001, 0.02), np.arange(3.0, 5.0001, 0.02)]
    theta = np.angle(np.exp(1j * (np.pi - 1.0 + 0.5 * time)))
    return time, np.c_[time, np.sin(time), theta]


def test_gap_bins_take_pchip_through_the_samples_either_side():
    """Empty bins get the PCHIP interpolant through the raw samples from GAP_CONTEXT_S before the gap to GAP_CONTEXT_S after it, the
    heading unwrapped first, so a heading through +-pi does not swing round the circle."""
    time, xyt = gap_track()
    grid = np.arange(1.0, 4.0001, 0.1)
    values, empty = binned_track(time, xyt, grid)
    gap = (grid > 2.05) & (grid < 2.95)
    assert empty.tolist() == gap.tolist()
    context = (time >= 2.0 - GAP_CONTEXT_S - 1e-9) & (time <= 3.0 + GAP_CONTEXT_S + 1e-9)
    expected = PchipInterpolator(time[context], np.c_[xyt[context, :2], np.unwrap(xyt[context, 2])])(grid[gap])
    np.testing.assert_allclose(values[gap, :2], expected[:, :2], atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * values[gap, 2]), np.exp(1j * expected[:, 2]), atol=1e-12)
    np.testing.assert_allclose(np.exp(1j * values[gap, 2]), np.exp(1j * (np.pi - 1.0 + 0.5 * grid[gap])), atol=1e-3)  # a smooth turn


def test_gap_bins_use_no_raw_sample_more_than_the_context_after_the_gap():
    """Changing the raw samples later than GAP_CONTEXT_S after the gap's end changes no gap sample."""
    time, xyt = gap_track()
    grid = np.arange(1.0, 4.0001, 0.1)
    values, empty = binned_track(time, xyt, grid)
    changed = xyt.copy()
    changed[time > 3.0 + GAP_CONTEXT_S + 1e-9] += 7.0
    np.testing.assert_array_equal(binned_track(time, changed, grid)[0][empty], values[empty])


def test_bins_beyond_the_raw_samples_hold_the_nearest_sample():
    """At a recording edge there is no sample on one side of the gap: the bins hold the first or last sample."""
    time, xyt = track([1.0, 1.02, 1.5, 1.52], [1.0, 2.0, 3.0, 4.0])
    values, empty = binned_track(time, xyt, np.array([0.8, 1.0, 1.7, 1.8]))
    assert empty.tolist() == [True, False, True, True]
    np.testing.assert_allclose(values[:, 0], [1.0, 1.5, 4.0, 4.0])


def test_no_sample_outside_a_gap_uses_a_raw_sample_half_a_bin_or_more_after_its_time():
    """A step in the raw positions at time s: every non-empty bin ending at or before s must not see it, whatever the grid phase."""
    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0, 10, 2000))
    step = time[1000]
    t, p = track(time, (time >= step).astype(float))
    grid = np.arange(0.05, 9.9, 0.01)
    values, empty = binned_track(t, p, grid)
    assert np.all(values[(grid + INPUT_BIN_S / 2 <= step) & ~empty, 0] == 0)
    assert np.all(values[(grid - INPUT_BIN_S / 2 >= step) & ~empty, 0] == 1)


# Trip 0 starts at 1.33 s (snapped to 1.3), so its history is cut at the recording start; trip 1 ends with its padding cut at the recording end (30 s).
EDGE_VISITS = [visit(0, 0, "0", 0.2, 0.8, 1.33), visit(1, 2, "2", 6.0, 7.03, 8.0), visit(2, 5, "5", 16.0, 17.0, 18.0)]
EDGE_XY = {label: [float(label), 1.0] for label in ("0", "2", "5")}


def test_trips_are_slices_of_the_recording_series_and_scored_samples_every_fourth(monkeypatch):
    """The assembled trip's 10 Hz series is the recording's arrays from its first to its last index, its 2.5 Hz tracks and labels every
    SCORED_STEP-th of them from the first, with the trip start a scored sample at time 0, at the recording's edges as well."""
    trips, series = person_trips(monkeypatch, EDGE_VISITS, EDGE_XY)
    assert [t["start_index"] for t in trips] == [3, 7] and trips[0]["t_start"] == 1.3 and trips[0]["t_end"] == 7.0
    for trip in trips:
        first, start, end, last = trip["recording_indices"]
        dense = trip["input_tracks_10hz"]
        assert (0 <= len(series["robot"]["xyt"]) - 1 - last < SCORED_STEP) if trip["trip"] == 1 else first < SCORED_STEP
        for agent, fields in series.items():
            np.testing.assert_array_equal(dense["tracks"][agent], fields["xyt"][first: last + 1])
            np.testing.assert_array_equal(dense["empty_bins"][agent], np.flatnonzero(fields["empty"][first: last + 1]))
            np.testing.assert_array_equal(trip["tracks"][agent], dense["tracks"][agent][::SCORED_STEP])
            assert trip["labels"][agent]["state"].tolist() == fields["state"][first: last + 1: SCORED_STEP].tolist()
        np.testing.assert_array_equal(trip["time"], dense["time"][::SCORED_STEP])
        np.testing.assert_allclose(np.diff(dense["time"]), 1 / INPUT_HZ, atol=1.5e-3)  # times are rounded to 1 ms
        assert dense["start_index"] == start - first == SCORED_STEP * trip["start_index"] and trip["time"][trip["start_index"]] == 0
        assert dense["end_index"] == end - first and trip["end_index"] == (end - first) // SCORED_STEP
        assert dense["time"][dense["end_index"]] == pytest.approx(trip["duration_s"])


def test_real_metrics_are_computed_on_the_scored_tracks(monkeypatch):
    trips, _ = person_trips(monkeypatch, OVERLAPPING_AND_SAME_STATION_VISITS, {label: [float(label), 1.0] for label in ("0", "2", "3", "5")})
    for trip in trips:
        path = slice(trip["start_index"], trip["end_index"] + 1)
        others = np.stack([np.asarray(track, float)[path, :2] for agent, track in trip["tracks"].items() if agent != trip["agent"]], axis=1)
        metrics = bi3_metrics(trip["time"][path], np.asarray(trip["tracks"][trip["agent"]], float)[path, :2], others)
        assert trip["real_metrics"]["path_efficiency"] == pytest.approx(metrics["path_efficiency"], abs=1e-4)


def test_load_trips_assembles_the_trips_from_the_files(monkeypatch, tmp_path):
    """Written as main writes them (trips.json metadata and recordings.npz), the trips load back equal to the ones assembled in memory."""
    trips, series = person_trips(monkeypatch, EDGE_VISITS, EDGE_XY)
    grid = np.arange(len(series["robot"]["xyt"])) / INPUT_HZ
    np.savez_compressed(tmp_path / RECORDINGS_FILE, **recording_arrays(trips[0]["recording"], grid, series))
    keys = [k for k in trips[0] if k not in trip_arrays(trips[0], series)]
    (tmp_path / "trips.json").write_text(json.dumps([{k: t[k] for k in keys} for t in trips]))
    loaded = load_trips(tmp_path)
    assert [t["id"] for t in loaded] == [t["id"] for t in trips]
    for got, want in zip(loaded, trips):
        assert list(got["tracks"]) == list(want["tracks"])
        np.testing.assert_array_equal(got["tracks"]["human0"], want["tracks"]["human0"])
        np.testing.assert_array_equal(got["input_tracks_10hz"]["tracks"]["robot"], want["input_tracks_10hz"]["tracks"]["robot"])
        assert got["labels"]["human1"]["goal"].tolist() == want["labels"]["human1"]["goal"].tolist()
        assert (got["start_index"], got["end_index"], got["real_metrics"]) == (want["start_index"], want["end_index"], want["real_metrics"])
