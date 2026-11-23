"""The real positions the model reads come from the trip's 10 Hz series, and the other agents' velocity is a central difference on it."""

import numpy as np
import pytest

from tests.test_human_prompt import SERIES_WIGGLE_M, SITE, person_trip
from traj_gen.harness.episodes import Episode
from traj_gen.harness.presets import PRESETS
from traj_gen.observation.scene import PROMPT_HISTORY_STEPS, Scene


def scene_at(trip: dict, i: int) -> Scene:
    return Scene(trip, SITE, i, np.asarray(trip["tracks"][trip["agent"]], float)[: i + 1, :2])


def test_other_agents_positions_and_history_come_from_the_series_not_the_scored_tracks():
    """Moving the scored track must not move what the model reads, and moving the series at the trip samples must."""
    trip = person_trip()
    i = trip["start_index"]
    trip["tracks"]["human1"] = (np.asarray(trip["tracks"]["human1"]) + 5.0).tolist()
    series = trip["input_tracks_10hz"]["tracks"]["human1"]
    series[::4, :2] += np.c_[np.arange(len(series[::4])), np.zeros(len(series[::4]))]  # trip sample k moves k metres along x
    p = next(o for o in scene_at(trip, i).others if o["id"] == "human1")
    line_x = np.linspace(1.2, 1.6, 12)
    assert p["xy"][0] == pytest.approx(line_x[i] + i)
    np.testing.assert_allclose(p["history"][:, 0], line_x[i - PROMPT_HISTORY_STEPS: i] + np.arange(i - PROMPT_HISTORY_STEPS, i))


def test_velocity_is_the_central_difference_over_one_series_sample_each_side():
    """The test series is a straight line with +w one sample after each trip sample and -w one sample before it, so the central
    difference is the line's velocity plus 2w / 0.2 s on both coordinates. A backward difference over the trip samples would give the
    line's velocity alone, and one over the series a different offset."""
    trip = person_trip()
    p = next(o for o in scene_at(trip, trip["start_index"]).others if o["id"] == "human1")
    line_v = np.array([0.4, -0.7]) / (11 * 0.4)  # human1 walks (1.2, 0.5) -> (1.6, -0.2) over 11 steps of 0.4 s
    np.testing.assert_allclose(p["velocity"], line_v + 2 * SERIES_WIGGLE_M / 0.2)


def test_velocity_at_the_ends_of_the_series_is_one_sided():
    """At the first and last series samples there is no sample on one side: the difference to the one neighbour, never a wrap-around."""
    trip = person_trip()
    line_v = np.array([0.4, -0.7]) / (11 * 0.4)
    first = next(o for o in scene_at(trip, 0).others if o["id"] == "human1")
    np.testing.assert_allclose(first["velocity"], line_v + SERIES_WIGGLE_M / 0.1)  # sample 1 has +w, sample 0 none
    last = next(o for o in scene_at(trip, len(trip["time"]) - 1).others if o["id"] == "human1")
    np.testing.assert_allclose(last["velocity"], line_v + SERIES_WIGGLE_M / 0.1)  # the sample before the last has -w


def test_planned_persons_history_before_set_off_comes_from_the_series(tmp_path):
    trip = person_trip()
    trip["tracks"]["human0"] = (np.asarray(trip["tracks"]["human0"]) + 5.0).tolist()
    episode = Episode(tmp_path, PRESETS["pilot_v2"].harness("human").experiment(), trip, SITE, "participant", "think", 0)
    np.testing.assert_allclose(episode.track, trip["input_tracks_10hz"]["tracks"]["human0"][::4][: trip["start_index"] + 1, :2])
