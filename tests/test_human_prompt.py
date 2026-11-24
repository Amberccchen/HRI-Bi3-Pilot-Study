import numpy as np

from tests.fake_trips import SITE, person_trip
from traj_gen.harness.presets import Harness
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import Scene, track_motion
from traj_gen.outputs.positions import PositionPlanner


def test_person_prompt_has_no_robot_template_and_gives_the_others_goals_and_sizes():
    """A robot template leaking into a person's prompt, the others' goals missing or swapped, or the planned person's own
    radius given as the robot's, would render without an error."""
    trip = person_trip()
    planner = PositionPlanner(Harness("single_shot", "positions", "m", "text", "participant").experiment(), trip, "participant", "nothink")
    builder = PromptBuilder("participant", planner, (), 20, False)
    track = np.asarray(trip["tracks"]["human0"], float)[: trip["start_index"] + 1, :2]
    scene = Scene(trip, SITE, trip["start_index"], track, track_motion(track))
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
