import numpy as np
import pytest

from tests.test_human_prompt import SITE, person_trip, with_input_series
from traj_gen.agents import PERSON_BODY_RADIUS_M, PERSONAL_SPACE_RADIUS_M, comfort_distance, contact_distance, pair_distances
from traj_gen.harness.clients import QwenClient
from traj_gen.harness.episodes import Episode
from traj_gen.harness.presets import Harness
from traj_gen.harness.experiments import Experiment
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import Scene


def robot_trip() -> dict:
    """The person trip with the robot as the planned agent: the robot drives up the middle, human0 stands at the centre, in its way."""
    trip = person_trip()
    n = len(trip["time"])
    trip = {**trip, "id": "um_um99_cv_robot_01", "agent": "robot", "goal": "3", "goal_xy": SITE["stations"]["3"]["xy"]}
    trip["tracks"] = {**trip["tracks"], "robot": np.c_[np.zeros(n), np.linspace(-1.0, 0.0, n), np.zeros(n)].tolist(),
                      "human0": np.c_[np.zeros(n), np.zeros(n), np.zeros(n)].tolist()}
    return with_input_series(trip)


# The experiments of the tests: positions in pixels and steer in one conversation on the map, steer with the recorded futures, and with the plan check.
SETTINGS = {"positions": {"output": "positions"}, "steer": {"output": "steer"}, "steer_future": {"output": "steer", "prediction": "true_future"},
            "steer_check": {"output": "steer", "prediction": "true_future", "check": 2}}


def experiment(name: str, framing: str) -> Experiment:
    return Harness(**{"episode": "rolling_conv", "units": "px", "images": "bev", "framing": framing, **SETTINGS[name]}).experiment()


def prompts(name: str, trip: dict, framing: str) -> str:
    exp = experiment(name, framing)
    planner = exp.planner(exp, trip, framing, "think")
    builder = PromptBuilder(framing, planner, exp.images, 20, exp.predict)
    scene = Scene(trip, SITE, trip["start_index"], np.asarray(trip["tracks"][trip["agent"]])[: trip["start_index"] + 1, :2])
    return builder.system(scene) + "\n" + builder.first(scene).text


def test_scored_distance_framings_go_only_with_the_steer_modes():
    """A scored-distance task (which says waiting is a step of distance 0) with a position reply, or the steer reply with a plain task, would render
    without an error and mix two setups."""
    trip = person_trip()
    for name, framing, other in (("positions", "scored_distances", "participant"), ("steer", "participant", "scored_distances")):
        exp = experiment(name, other)
        with pytest.raises(ValueError, match="scored-distance framings go only with the steer and steer_vel modes"):
            PromptBuilder(framing, exp.planner(exp, trip, framing, "think"), exp.images, 20, exp.predict)


def test_person_steer_prompt_gives_only_the_top_speed_and_a_person_scale_example():
    """The person steer prompt must not state the largest step (people asked for exactly it), while the robot one keeps its stated 13 px (0.12 m at 112 px/m)."""
    person = prompts("steer", person_trip(), "scored_distances")
    assert "You can walk at most 168 px/s." in person and "px in one step" not in person and "maximum" not in person
    assert '{"steps": [[0, 45], [-15, 45], [0, 0]]}' in person and "at most 131 degrees" in person
    robot = prompts("steer", robot_trip(), "robot_scored_distances")
    assert "So it moves at most 13 px in one step." in robot and '{"steps": [[0, 11], [-15, 12], [0, 0]]}' in robot and "at most 39 degrees" in robot


def test_future_trajectories_only_in_the_future_experiments():
    trip = person_trip()
    assert "will actually be" not in prompts("steer", trip, "scored_distances")
    future = prompts("steer_future", trip, "scored_distances")
    assert "Where the other participant and the robot will actually be" in future and "Robot: t+0.4 s" in future and "P2: t+0.4 s" in future


def test_steer_steps_are_cut_to_the_turn_and_speed_limits():
    """Turns beyond the largest turn and distances beyond one step's travel (or below 0) must be cut, as the prompt says."""
    trip = person_trip()
    exp = experiment("steer", "scored_distances")
    planner = exp.planner(exp, trip, "scored_distances", "think")
    scene = Scene(trip, SITE, trip["start_index"], np.asarray(trip["tracks"]["human0"])[: trip["start_index"] + 1, :2])
    heading = planner.heading(scene)
    start = planner.units.pos(scene.track[-1])
    uv = planner.units.pos(planner.positions(np.array([[170.0, 500.0], [0.0, -30.0]]), scene)[0])
    step_px = planner.units.length(planner.step_m)
    first = uv[0] - start
    assert np.linalg.norm(first) == pytest.approx(step_px)  # 500 px cut to one step (67 px at 1.5 m/s)
    assert np.degrees(np.arctan2(first[1], first[0])) == pytest.approx(heading + 131.0)  # 170 deg cut to 131
    assert np.allclose(uv[1], uv[0])  # a negative distance waits


def test_plan_check_counts_contact_against_the_recorded_future(tmp_path):
    """The plan check must score the plan against where the people really are, with the scored contact and personal-space distances."""
    exp = experiment("steer_check", "robot_scored_distances")
    trip = robot_trip()
    episode = Episode(tmp_path, exp, trip, SITE, "robot_scored_distances", "think", 0)
    scene = episode.scene()
    straight = np.array([[0.0, 24.0]] * 7)  # keep driving along the current heading, into the person standing at y = 0 m
    counts, text = episode.conflicts(straight, scene)
    assert counts[0] > 0 and "in contact" in text
    positions = episode.planner.positions(straight, scene)[0]
    i = len(episode.track)
    near = np.linalg.norm(positions[: len(trip["time"]) - i] - np.asarray(trip["tracks"]["human0"])[i:, :2][: len(positions)], axis=1)
    assert counts == (int((near < contact_distance(SITE)).sum()), int(((near >= contact_distance(SITE)) & (near < comfort_distance(SITE))).sum()))


def person_check(tmp_path, robot: list[float], other: list[float]) -> tuple[tuple[int, int], str | None]:
    """Plan check of a person who waits in place for 3 steps, while the robot and the other participant stand at the given x offsets (m)
    from the person at each of those steps."""
    trip = person_trip()
    i = trip["start_index"] + 1
    here = np.asarray(trip["tracks"]["human0"])[trip["start_index"]]
    tracks = {a: np.asarray(trip["tracks"][a], float).copy() for a in ("robot", "human1")}
    for agent, offsets in (("robot", robot), ("human1", other)):
        tracks[agent][i: i + 3] = here + np.c_[offsets, np.zeros(3), np.zeros(3)]
    trip["tracks"] = {**trip["tracks"], **{a: t.tolist() for a, t in tracks.items()}}
    episode = Episode(tmp_path, experiment("steer_check", "scored_distances"), trip, SITE, "scored_distances", "think", 0)
    return episode.conflicts(np.zeros((3, 2)), episode.scene())


def test_person_plan_check_uses_the_thresholds_of_each_pair(tmp_path):
    """A planned person touches the robot below the robot-person contact distance (0.315 m here), and the other participant below 0.30 m,
    with that participant's personal space below 0.40 m. The robot has no personal space, so 0.39 m from it is nothing, though it is
    inside the robot's own personal-space distance to a person (0.415 m) and the person-person one. Using the robot's distances for the
    person pair, or giving the robot a personal space, would change these counts."""
    assert pair_distances("human0", "robot", SITE) == pytest.approx((contact_distance(SITE), contact_distance(SITE)))
    assert pair_distances("human0", "human1", SITE) == pytest.approx((2 * PERSON_BODY_RADIUS_M, PERSON_BODY_RADIUS_M + PERSONAL_SPACE_RADIUS_M["human"]))
    far = [3.0] * 3
    counts, text = person_check(tmp_path, [0.0, 3.0, 3.0], far)
    assert counts == (1, 0) and "from the robot at" in text and "so you are in contact" in text
    counts, text = person_check(tmp_path, far, [0.35, 0.35, 3.0])
    assert counts == (0, 2) and "from the other participant (P2) at" in text and "so you are inside personal space" in text
    assert person_check(tmp_path, [0.39] * 3, far) == ((0, 0), None)
    counts, text = person_check(tmp_path, [3.0, 0.39, 3.0], [3.0, 0.29, 0.35])
    assert counts == (1, 1) and "at t + 0.8 s" in text and "from the other participant (P2)" in text


def test_person_plan_check_reports_the_worst_agent_at_the_first_conflict(tmp_path):
    """At one step the robot is in contact at 0.31 m and the other participant only inside personal space at 0.305 m: contact comes first,
    though the participant is nearer. With both in contact, the nearer one is reported."""
    counts, text = person_check(tmp_path, [0.31, 3.0, 3.0], [0.305, 0.25, 3.0])
    assert counts == (2, 0) and "at t + 0.4 s" in text and "35 px from the robot" in text
    counts, text = person_check(tmp_path, [0.2, 3.0, 3.0], [0.1, 3.0, 3.0])
    assert "11 px from the other participant (P2)" in text


def test_person_plan_check_text_gives_the_person_thresholds(tmp_path):
    """The person's feedback must speak to the person and give the distances of the person task prompt, not the robot's personal space."""
    _, text = person_check(tmp_path, [0.0, 3.0, 3.0], [3.0] * 3)
    assert text.startswith("Check of your plan against where the other participant and the robot will actually be: at t + 0.4 s you would be at")
    assert ("contact with the robot below 35 px, contact with the other participant below 34 px, the other participant's personal space below 45 px,"
            " the robot has none") in text  # 0.315, 0.30 and 0.40 m at 112 px/m
    assert "the robot would be" not in text and "46 px" not in text


def test_steer_thinking_budget_is_recorded():
    assert QwenClient.describe(True, experiment("steer", "scored_distances").planner.think_max_tokens)["max_tokens"] == 32768
    assert QwenClient.describe(True, experiment("positions", "participant").planner.think_max_tokens)["max_tokens"] == 16384


def test_plan_check_and_scored_distance_framings_use_the_scored_distances():
    """Contact is scored with the robot's inscribed disk (half the footprint's shorter side) plus the person's body radius, and personal
    space with the person's 0.25 m disk. A plan check or task text on the drawn (enclosing) disk would give the wrong threshold."""
    assert contact_distance(SITE) == pytest.approx(0.33 / 2 + 0.15)
    assert comfort_distance(SITE) == pytest.approx(0.33 / 2 + 0.25)
    robot = prompts("steer", robot_trip(), "robot_scored_distances")
    assert "less than 35 px apart" in robot and "at least 46 px apart" in robot  # 0.315 m and 0.415 m at 112 px/m
