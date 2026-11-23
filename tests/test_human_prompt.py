import numpy as np

from preprocessing.bi3_derived import INPUT_HZ, SCORED_STEP

from traj_gen.harness.presets import Harness
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import Scene
from traj_gen.outputs.positions import PositionPlanner

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


def test_person_prompt_has_no_robot_template_and_gives_the_others_goals_and_sizes():
    """A robot template leaking into a person's prompt, the others' goals missing or swapped, or the planned person's own
    radius given as the robot's, would render without an error."""
    trip = person_trip()
    planner = PositionPlanner(Harness("single_shot", "positions", "m", "text", "participant").experiment(), trip, "participant", "nothink")
    builder = PromptBuilder("participant", planner, (), 20, False)
    scene = Scene(trip, SITE, trip["start_index"], np.asarray(trip["tracks"]["human0"])[:, :2])
    system, turn = builder.system(scene), builder.first(scene).text
    text = system + "\n" + turn
    for robot_phrase in ("motion planner", "Robot goal", "the robot's path", "Robot: at"):
        assert robot_phrase not in text
    assert "You are P1" in system and "station 3: (0.10, 1.65)" in system  # where people stand, not the robot's stopping point
    assert "Your goal: station 3 at (0.10, 1.65)" in turn and "Your goal after that: station 2." in turn
    other_person, robot = (next(line for line in turn.splitlines() if line.startswith(name)) for name in ("P2", "The robot: at"))
    assert "Goal: station 5. Next goal: station 3." in other_person
    assert "Goal: station 2." in robot and "Next goal" not in robot  # the robot's goal only
    assert "The robot is a disk of radius 0.24 m and the other participant takes up about a disk of radius 0.25 m." in turn  # UM robot, 0.237 m
    assert "[-0.50, 0.40], [-0.12, 0.52], [0.26, 0.64]" in turn  # the walking-pace example
