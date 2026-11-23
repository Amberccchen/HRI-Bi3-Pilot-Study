import numpy as np
import pytest

from tests.test_human_prompt import SITE, person_trip
from traj_gen.agents import comfort_distance, contact_distance
from traj_gen.client import QwenClient
from traj_gen.episodes import Episode
from traj_gen.experiments import EXPERIMENTS
from traj_gen.prompting import PromptBuilder
from traj_gen.scene import Scene


def robot_trip() -> dict:
    """The person trip with the robot as the planned agent: the robot drives up the middle, human0 stands at the centre, in its way."""
    trip = person_trip()
    n = len(trip["time"])
    trip = {**trip, "id": "um_um99_cv_robot_01", "agent": "robot", "goal": "3", "goal_xy": SITE["stations"]["3"]["xy"]}
    trip["tracks"] = {**trip["tracks"], "robot": np.c_[np.zeros(n), np.linspace(-1.0, 0.0, n), np.zeros(n)].tolist(),
                      "human0": np.c_[np.zeros(n), np.zeros(n), np.zeros(n)].tolist()}
    return trip


def prompts(exp_id: str, trip: dict, framing: str) -> str:
    exp = EXPERIMENTS[exp_id]
    planner = exp.planner(exp, trip, framing, "think")
    builder = PromptBuilder(framing, planner, exp.images, 20, exp.predict)
    scene = Scene(trip, SITE, trip["start_index"], np.asarray(trip["tracks"][trip["agent"]])[: trip["start_index"] + 1, :2])
    return builder.system(scene) + "\n" + builder.first(scene).text


def test_harness_framings_go_only_with_the_harness_mode():
    """A harness task (which says waiting is a step of distance 0) with a position reply, or the harness reply with a plain task, would render
    without an error and mix two setups."""
    trip = person_trip()
    for exp_id, framing in (("rh_px_bev_conv", "participant_harness"), ("rh_harness_bev_conv", "participant")):
        exp = EXPERIMENTS[exp_id]
        with pytest.raises(ValueError, match="harness framings go only with the harness mode"):
            PromptBuilder(framing, exp.planner(exp, trip, framing, "think"), exp.images, 20, exp.predict)
    assert EXPERIMENTS["rh_harness_bev_conv"].accepts("participant_harness") and not EXPERIMENTS["rh_harness_bev_conv"].accepts("participant")
    assert EXPERIMENTS["rh_px_bev_conv"].accepts("participant") and not EXPERIMENTS["rh_px_bev_conv"].accepts("participant_harness")


def test_person_harness_gives_only_the_top_speed_and_a_person_scale_example():
    """The person harness must not state the largest step (people asked for exactly it), while the robot harness keeps its stated 24 px."""
    person = prompts("rh_harness_bev_conv", person_trip(), "participant_harness")
    assert "You can walk at most 300 px/s." in person and "px in one step" not in person and "maximum" not in person
    assert '{"steps": [[0, 80], [-15, 80], [0, 0]]}' in person and "at most 115 degrees" in person
    robot = prompts("rh_harness_bev_conv", robot_trip(), "harness")
    assert "So it moves at most 24 px in one step." in robot and '{"steps": [[0, 20], [-15, 22], [0, 0]]}' in robot and "at most 35 degrees" in robot


def test_future_trajectories_only_in_the_future_experiments():
    trip = person_trip()
    assert "will actually be" not in prompts("rh_harness_bev_conv", trip, "participant_harness")
    future = prompts("rh_harness_bev_conv_future", trip, "participant_harness")
    assert "Where the other participant and the robot will actually be" in future and "Robot: t+0.4 s" in future and "P2: t+0.4 s" in future


def test_harness_steps_are_cut_to_the_turn_and_speed_limits():
    """Turns beyond the largest turn and distances beyond one step's travel (or below 0) must be cut, as the prompt says."""
    trip = person_trip()
    exp = EXPERIMENTS["rh_harness_bev_conv"]
    planner = exp.planner(exp, trip, "participant_harness", "think")
    scene = Scene(trip, SITE, trip["start_index"], np.asarray(trip["tracks"]["human0"])[: trip["start_index"] + 1, :2])
    heading = planner.heading(scene)
    start = planner.units.pos(scene.track[-1])
    uv = planner.units.pos(planner.positions(np.array([[170.0, 500.0], [0.0, -30.0]]), scene)[0])
    step_px = planner.units.length(planner.step_m)
    first = uv[0] - start
    assert np.linalg.norm(first) == pytest.approx(step_px)  # 500 px cut to one step (120 px at 1.5 m/s)
    assert np.degrees(np.arctan2(first[1], first[0])) == pytest.approx(heading + 115.0)  # 170 deg cut to 115
    assert np.allclose(uv[1], uv[0])  # a negative distance waits


def test_plan_check_counts_contact_against_the_recorded_future_and_needs_a_robot_trip(tmp_path):
    """The plan check must score the plan against where the people really are, with the scored contact and personal-space distances."""
    exp = EXPERIMENTS["rh_harness_bev_conv_future_check"]
    trip = robot_trip()
    episode = Episode(tmp_path, exp, trip, SITE, "harness", "think", 0)
    scene = episode.scene()
    straight = np.array([[0.0, 24.0]] * 7)  # keep driving along the current heading, into the person standing at y = 0 m
    counts, text = episode.conflicts(straight, scene)
    assert counts[0] > 0 and "in contact" in text
    positions = episode.planner.positions(straight, scene)[0]
    i = len(episode.track)
    near = np.linalg.norm(positions[: len(trip["time"]) - i] - np.asarray(trip["tracks"]["human0"])[i:, :2][: len(positions)], axis=1)
    assert counts == (int((near < contact_distance(SITE)).sum()), int(((near >= contact_distance(SITE)) & (near < comfort_distance(SITE))).sum()))
    with pytest.raises(ValueError, match="needs a robot trip"):
        Episode(tmp_path, exp, person_trip(), SITE, "participant_harness", "think", 0)


def test_harness_thinking_budget_is_recorded():
    assert QwenClient.describe(True, EXPERIMENTS["rh_harness_bev_conv"].planner.think_max_tokens)["max_tokens"] == 32768
    assert QwenClient.describe(True, EXPERIMENTS["rh_px_bev_conv"].planner.think_max_tokens)["max_tokens"] == 16384


def test_plan_check_and_harness_text_use_the_scored_distances():
    """Contact is scored with the robot's inscribed disk (half the footprint's shorter side) plus the person's body radius, and personal
    space with the person's 0.25 m disk. A plan check or task text on the drawn (enclosing) disk would give the wrong threshold."""
    assert contact_distance(SITE) == pytest.approx(0.33 / 2 + 0.15)
    assert comfort_distance(SITE) == pytest.approx(0.33 / 2 + 0.25)
    robot = prompts("rh_harness_bev_conv", robot_trip(), "harness")
    assert "less than 63 px apart" in robot and "at least 83 px apart" in robot  # 0.315 m and 0.415 m at 200 px/m
