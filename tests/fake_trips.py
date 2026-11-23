"""A fake person trip and the same trip with the robot planned, with their 10 Hz series, shared by the harness tests."""

import numpy as np

from preprocessing.bi3_derived import INPUT_HZ, SCORED_STEP

SITE = {"room_m": 3.5, "footprint_m": [0.33, 0.34],
        "stations": {"2": {"members": ["2"], "xy": [0.1, -1.5], "human_xy": [0.15, -1.66]},
                     "3": {"members": ["3"], "xy": [0.2, 1.4], "human_xy": [0.1, 1.65]},
                     "5": {"members": ["5"], "xy": [1.6, -0.4], "human_xy": [1.7, -0.3]}}}


SERIES_WIGGLE_M = 0.02  # offset of the series between trip samples, so that a velocity read from the series differs from one read from the trip samples


def with_input_series(trip: dict) -> dict:
    """The trip with its 10 Hz series (input_tracks_10hz) as load_trips gives it: each track interpolated linearly between the trip
    samples, which the series holds exactly at every SCORED_STEP-th sample, plus +SERIES_WIGGLE_M on both coordinates one sample after a
    trip sample and -SERIES_WIGGLE_M one sample before it."""
    time = np.asarray(trip["time"], float)
    dense_time = np.round(np.arange(len(time) * SCORED_STEP - SCORED_STEP + 1) / INPUT_HZ + time[0], 3)
    wiggle = SERIES_WIGGLE_M * np.sin(np.pi / 2 * np.arange(len(dense_time)))  # 0 at the trip samples, +, 0, - between them
    tracks = {}
    for agent, track in trip["tracks"].items():
        track = np.asarray(track, float)
        dense = np.stack([np.interp(dense_time, time, track[:, k]) for k in range(3)], axis=1)
        dense[:, :2] += wiggle[:, None]
        tracks[agent] = dense
    start = trip["start_index"] * SCORED_STEP
    return {**trip, "input_tracks_10hz": {"time": dense_time, "start_index": start, "end_index": len(dense_time) - 1, "tracks": tracks,
                                          "empty_bins": {agent: np.array([], int) for agent in tracks}}}


def person_trip() -> dict:
    n = 12
    line = lambda a, b: np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n), np.zeros(n)].tolist()
    labels = lambda goal, after: {"state": ["transit"] * n, "goal": [goal] * n, "next_goal": [after] * n}
    return with_input_series({"id": "um_um99_cv_human0_01", "site": "um", "agent": "human0", "start_index": 8, "start_goal": "2", "goal": "3", "next_goal": "2",
            "goal_xy": SITE["stations"]["3"]["human_xy"], "time": (np.arange(n) * 0.4 - 3.2).tolist(),
            "tracks": {"robot": line([-1.0, 0.0], [-0.5, 0.2]), "human0": line([0.1, -1.4], [0.1, -0.2]), "human1": line([1.2, 0.5], [1.6, -0.2])},
            "labels": {"robot": labels("2", "5"), "human0": labels("3", "2"), "human1": labels("5", "3")}})


def robot_trip() -> dict:
    """The person trip with the robot as the planned agent: the robot drives up the middle, human0 stands at the centre, in its way."""
    trip = person_trip()
    n = len(trip["time"])
    trip = {**trip, "id": "um_um99_cv_robot_01", "agent": "robot", "goal": "3", "goal_xy": SITE["stations"]["3"]["xy"]}
    trip["tracks"] = {**trip["tracks"], "robot": np.c_[np.zeros(n), np.linspace(-1.0, 0.0, n), np.zeros(n)].tolist(),
                      "human0": np.c_[np.zeros(n), np.zeros(n), np.zeros(n)].tolist()}
    return with_input_series(trip)
