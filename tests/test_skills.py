"""The skills output mode (traj_gen/outputs/skills.py): wording and sides, the sided pedestrian force against the package's, passes in
synthetic scenes judged from geometry only, the 10 Hz reading of the agent being passed, the walker's state across decisions, reply
validation, the option list, and the skill labels of the GIFs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from tests.fake_trips import SITE, person_trip
from traj_gen.analysis.animate import load_episode
from traj_gen.harness.episodes import save_trajectory
from traj_gen.harness.presets import PRESETS
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import Scene, track_motion
from traj_gen.outputs.skills import FORCE_CONFIG, Config, SidedSocialForce, SocialForce, crossing, pass_substeps, side_is_free, turning_sign, walk_substeps

HALF_SIDE = 2.0  # the 3.5 m room: walls 2.0 m from the centre
N = 120  # samples of a synthetic 10 Hz series: 12 s


def series(start, velocity, n: int = N) -> np.ndarray:
    """(n, 2) positions at 10 Hz of an agent moving at a constant velocity from start."""
    return np.asarray(start, float) + 0.1 * np.arange(n)[:, None] * np.asarray(velocity, float)


def cross(a: np.ndarray, b: np.ndarray) -> float:
    """The z component of a x b: positive when b points to the left of a."""
    return float(a[0] * b[1] - a[1] * b[0])


def pair_force(force, walker_xy, walker_v, other_xy, other_v) -> np.ndarray:
    """The force a pedestrian force with the parameters of FORCE_CONFIG gives the walker (row 0) in a two-agent scene."""
    state = np.array([[*walker_xy, *walker_v], [*other_xy, *other_v]], float)
    force.peds = SimpleNamespace(pos=lambda: state[:, :2], vel=lambda: state[:, 2:4], size=lambda: 2)
    force.config, force.factor = Config(FORCE_CONFIG.config["sided_social_force"]), FORCE_CONFIG.config["sided_social_force"]["factor"]
    return force._get_force()[0]


# Wording and sides


def test_crossing_pair_needs_the_speed_and_the_angle_range():
    d = np.array([0.0, 1.0])
    at = lambda speed, deg: speed * np.array([np.sin(np.radians(deg)), np.cos(np.radians(deg))])  # deg from d
    assert not crossing(d, at(0.14, 90)) and crossing(d, at(0.16, 90))
    assert not crossing(d, at(1.0, 44)) and crossing(d, at(1.0, 46))
    assert crossing(d, at(1.0, 134)) and not crossing(d, at(1.0, 136))
    assert crossing(d, at(1.0, -90))  # from either side


def test_turning_sign_of_each_side_word():
    d = np.array([1.0, 0.0])  # the walker heads +x
    right_to_left = np.array([0.0, 0.8])  # the agent crosses from the walker's right to its left
    assert turning_sign("left", d, right_to_left, 0) == 1 and turning_sign("right", d, right_to_left, 0) == -1
    # In front of an agent moving to the walker's left: the walker ends on the agent's left side of travel, i.e. with the agent on its right.
    assert turning_sign("in front", d, right_to_left, 0) == -1 and turning_sign("behind", d, right_to_left, 0) == 1
    assert turning_sign("in front", d, -right_to_left, 0) == 1 and turning_sign("behind", d, -right_to_left, 0) == -1
    assert turning_sign("in front", d, np.array([0.0, 0.1]), -1) == -1 and turning_sign("behind", d, np.array([0.0, 0.1]), 1) == 1  # too slow: kept


# The force


def test_sided_force_equals_the_package_force_with_the_package_side():
    """Imported through the skills module, which restores the logging the package import changes."""
    for walker_v, other_xy, other_v, side in [((0.0, 1.0), (-0.05, 3.0), (0.0, -1.0), 1),   # head-on, agent slightly left: step right
                                              ((0.0, 1.0), (0.05, 3.0), (0.0, -1.0), -1),
                                              ((0.0, 1.0), (1.5, 1.5), (-0.8, 0.0), -1)]:  # crossing from the right
        expected = pair_force(SocialForce(), (0, 0), walker_v, other_xy, other_v)
        assert np.linalg.norm(expected) > 0.1
        np.testing.assert_allclose(pair_force(SidedSocialForce(side, 0.0), (0, 0), walker_v, other_xy, other_v), expected, rtol=1e-12, atol=1e-12)


def test_sided_force_turns_with_the_scene():
    """A pair whose t and e lie on either side of the -x axis: the package's unwrapped theta loses the force for the walker heading +x."""
    sided = SidedSocialForce(1, 0.0)
    walker_v, other_xy, other_v = np.array([1.0, 0.0]), np.array([3.0, -0.05]), np.array([-1.0, -0.1])
    base = pair_force(sided, (0, 0), walker_v, other_xy, other_v)
    assert np.linalg.norm(base) > 0.5
    for deg in (90, 180, 270):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        R = np.array([[c, -s], [s, c]])
        np.testing.assert_allclose(pair_force(sided, (0, 0), R @ walker_v, R @ other_xy, R @ other_v), R @ base, atol=1e-9)


# Passes in synthetic scenes, judged from geometry

WALKER, GOAL = np.array([0.0, -1.5]), np.array([0.0, 1.6])
D = np.array([0.0, 1.0])


def run_pass(other: np.ndarray, side: str, speed: float = 1.0, delta: float = 0.3, steps: int = 50):
    return pass_substeps(WALKER, speed * D, GOAL, speed, other, 0, side, 0, delta, HALF_SIDE, steps)[:4]


@pytest.mark.parametrize("case, other, speed", [("head-on", series((0.0, 1.5), (0.0, -1.0)), 1.0),
                                                ("standing", series((0.0, 0.0), (0.0, 0.0)), 1.0),
                                                ("overtaking", series((0.0, -0.8), (0.0, 0.4)), 1.2)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_pass_keeps_the_agent_on_the_chosen_side(case, other, speed, side):
    positions, _, _, distances = run_pass(other, side, speed)
    k = int(np.flatnonzero(positions[:, 1] >= other[1:len(positions) + 1, 1])[0])  # the walker draws level with the agent
    beside = other[k + 1] - positions[k]
    assert np.sign(cross(D, beside)) == (1 if side == "left" else -1), case
    assert distances.min() > 0.30  # no contact on the way


@pytest.mark.parametrize("side", ["in front", "behind"])
def test_a_crossing_pass_crosses_on_the_chosen_side(side):
    other = series((-1.5, 0.0), (0.8, 0.0))  # crosses the walker's line x = 0 at t = 1.9 s, from its left to its right
    positions, _, _, _ = run_pass(other, side)
    k = int(np.flatnonzero(positions[:, 1] >= other[1:len(positions) + 1, 1])[0])  # the walker reaches the agent's line of travel
    walker_first = other[k + 1, 0] < positions[k, 0]  # the agent has not yet reached the walker's x
    assert walker_first == (side == "in front")


def test_a_larger_delta_keeps_a_larger_distance():
    other = series((0.0, 1.5), (0.0, -1.0))
    closest = [run_pass(other, "left", delta=delta)[3].min() for delta in (0.0, 0.3, 0.6)]
    assert closest[0] < closest[1] < closest[2]


def test_the_agent_is_read_from_the_series_at_each_substep():
    """Far for samples 0-2, then right ahead: sub-steps 0 and 1 (position at j, j + 1, velocity from j - 1 .. j + 2) are unchanged, the
    rest react."""
    far = series((1.9, -1.9), (0.0, 0.0), 10)
    jumping = far.copy()
    jumping[3:] = [0.0, -1.0]
    a = pass_substeps(WALKER, D, GOAL, 1.0, far, 0, "left", 0, 0.3, HALF_SIDE, 4)[0]
    b = pass_substeps(WALKER, D, GOAL, 1.0, jumping, 0, "left", 0, 0.3, HALF_SIDE, 4)[0]
    np.testing.assert_array_equal(a[:2], b[:2])
    assert np.abs(a[2:] - b[2:]).min() > 1e-4


def test_walk_goes_straight_at_its_speed_and_stops_at_the_goal():
    positions, v = walk_substeps(np.array([0.0, 0.0]), np.array([0.3, 0.4]), 1.0, 8)  # 0.5 m away
    np.testing.assert_allclose(positions[:4], np.outer([0.1, 0.2, 0.3, 0.4], [0.6, 0.8]), atol=1e-12)
    np.testing.assert_allclose(positions[-1], [0.3, 0.4], atol=1e-12)


# Wall check


def test_a_side_toward_a_close_wall_is_not_offered():
    d, still = np.array([1.0, 0.0]), np.zeros(2)
    p2 = np.array([0.0, 1.7])  # 0.3 m from the wall at y = 2.0
    assert side_is_free("left", d, p2, still, 0.30, HALF_SIDE) and not side_is_free("right", d, p2, still, 0.30, HALF_SIDE)
    robot = np.array([1.5, 0.0])  # 0.5 m from the wall at x = 2.0. The walker heads +y, so keeping the robot on its left passes between it and the wall
    for contact, free in ((0.32, True), (0.48, False)):  # UM and LAAS robot contact distances: 0.5 >= 0.47, 0.5 < 0.63
        assert side_is_free("left", np.array([0.0, 1.0]), robot, still, contact, HALF_SIDE) == free
        assert side_is_free("right", np.array([0.0, 1.0]), robot, still, contact, HALF_SIDE)
    moving = np.array([0.0, 1.0])  # in front of an agent heading for the wall
    assert not side_is_free("in front", d, p2, moving, 0.30, HALF_SIDE) and side_is_free("behind", d, p2, moving, 0.30, HALF_SIDE)


# The planner on the fake trip

def skill_setup(units: str = "m"):
    harness = PRESETS["harness_v2"].harness("human", {"output": "skills", "units": units})
    exp, trip = harness.experiment(), person_trip()
    track, start = np.asarray(trip["tracks"][trip["agent"]], float)[:, :2], trip["start_index"]
    planner = exp.planner(exp, trip, harness.framing, "think")
    return planner, trip, track, start


def scene_at(trip, track, i):
    return Scene(trip, SITE, i, track[: i + 1], track_motion(track[: i + 1]))


def decide(planner, scene, reply: dict):
    planner.closing_text(scene)
    value = planner.parse(json.dumps(reply), 1)
    assert value is not None, reply
    return planner.positions(value, scene)


def test_walk_wait_and_continue():
    planner, trip, track, start = skill_setup()
    scene = scene_at(trip, track, start)
    planner.closing_text(scene)
    assert planner.parse('{"skill": "continue"}', 1) is None  # nothing under way yet
    xy, goal = scene.track[-1], scene.goal_xy
    walked, raw = decide(planner, scene, {"skill": "walk", "speed": 1.2})
    np.testing.assert_allclose(walked[0], xy + 0.48 * (goal - xy) / np.linalg.norm(goal - xy), atol=1e-12)
    assert raw.tolist() == [1.2]
    track = np.vstack([track[: start + 1], walked])
    later = Scene(trip, SITE, start + 1, track, planner.motion(None, scene, track))
    assert later.speed == pytest.approx(1.2)
    kept, _ = decide(planner, later, {"skill": "continue"})
    np.testing.assert_allclose(np.linalg.norm(kept[0] - walked[0]), 0.48, atol=1e-12)
    assert planner.record["continued"] and not planner.record["switched"] and planner.record["speed_m_s"] == 1.2
    waited, raw = decide(planner, later, {"skill": "wait"})
    np.testing.assert_array_equal(waited[0], later.track[-1])
    assert raw.tolist() == [0.0] and planner.motion(None, later, track)[1] == 0.0 and planner.record["switched"]


def test_a_pass_starts_from_the_velocity_the_last_skill_left():
    planner, trip, track, start = skill_setup()
    for other in ("robot", "human1"):  # both others far away and still
        trip["input_tracks_10hz"]["tracks"][other][:, :2] = [-1.9, -1.9] if other == "robot" else [1.9, -1.9]
    scene = scene_at(trip, track, start)
    walked, _ = decide(planner, scene, {"skill": "walk", "speed": 1.0})
    track = np.vstack([track[: start + 1], walked])
    later = Scene(trip, SITE, start + 1, track, planner.motion(None, scene, track))
    side = planner.offered["agents"]["robot"][0]
    passed, _ = decide(planner, later, {"skill": "pass", "agent": "robot", "side": side, "speed": 1.0})
    assert np.linalg.norm(passed[0] - walked[0]) == pytest.approx(0.40, abs=0.01)  # from rest it would be about 0.16 m


def test_delta_is_the_passed_agents_extra_body_radius():
    """The fake trip is at UM: robot body radius 0.165 m (half the shorter footprint side), person 0.15 m."""
    planner, trip, track, start = skill_setup()
    scene = scene_at(trip, track, start)
    decide(planner, scene, {"skill": "pass", "agent": "person", "side": "behind", "speed": 1.0})
    assert planner.record["delta_m"] == pytest.approx(0.0)
    decide(planner, scene, {"skill": "pass", "agent": "robot", "side": "behind", "speed": 1.0})
    assert planner.record["delta_m"] == pytest.approx(0.015)
    assert planner.manifest_fields()["decisions"] == 2 and len(planner.passes) == 2 and planner.passes[-1]["closest_m"] > 0


@pytest.mark.parametrize("reply", ["no json", '["walk"]', '{"skill": "run", "speed": 1}', '{"skill": "walk"}', '{"skill": "walk", "speed": 0}',
                                   '{"skill": "walk", "speed": "fast"}', '{"skill": "walk", "speed": true}',
                                   '{"skill": "pass", "agent": "robot", "side": "left", "speed": 1}',
                                   '{"skill": "pass", "agent": "dog", "side": "behind", "speed": 1}',
                                   '{"skill": "pass", "agent": "robot", "speed": 1}',
                                   '{"skill": "pass", "agent": "robot", "side": "behind", "speed": NaN}',
                                   '{"skill": "pass", "agent": ["robot"], "side": "behind", "speed": 1}',
                                   '{"skill": "pass", "agent": "robot", "side": ["behind"], "speed": 1}',
                                   '{"skill": "pass", "agent": {"a": 1}, "side": {"b": 2}, "speed": 1}'])
def test_invalid_replies(reply):
    planner, trip, track, start = skill_setup()
    planner.closing_text(scene_at(trip, track, start))  # the fake trip offers in front / behind for both others
    assert planner.parse(reply, 1) is None


def test_valid_replies_in_pixels():
    planner, trip, track, start = skill_setup("px")
    scene = scene_at(trip, track, start)
    planner.closing_text(scene)
    value = planner.parse('Sure: {"skill": "pass", "agent": "robot", "side": "behind", "speed": 112, "clearance": 67}', 1)
    assert value == {"skill": "pass", "agent": "robot", "side": "behind", "speed": 112}
    planner.positions(value, scene)
    assert planner.record["speed_m_s"] == pytest.approx(1.0)
    assert planner.parse('{"skill": "wait", "speed": 1}', 1) == {"skill": "wait"}


def test_options_leave_out_blocked_sides_and_offer_continue_after_a_choice():
    planner, trip, track, start = skill_setup()
    trip["input_tracks_10hz"]["tracks"]["human1"][:, :2] = [1.8, -0.5]  # standing by the right wall, the walker heads +y: no room on its right
    scene = scene_at(trip, track, start)
    text = planner.closing_text(scene)
    assert planner.offered == {"continue": False, "agents": {"robot": ["in front", "behind"], "person": ["right"]}}
    assert '"continue"' not in text and '"right" to keep P2 on your right' in text and "on your left" not in text
    decide(planner, scene, {"skill": "pass", "agent": "person", "side": "right", "speed": 1.0})
    assert planner.closing_text(scene).splitlines()[1] == '- "continue": keep passing P2 with "side": "right", at 1.00 m/s'


def test_the_option_list_closes_both_turns_after_the_true_future():
    harness = PRESETS["harness_v2"].harness("human", {"output": "skills", "prediction": "true_future"})
    exp, trip = harness.experiment(), person_trip()
    track, start = np.asarray(trip["tracks"][trip["agent"]], float)[:, :2], trip["start_index"]
    planner = exp.planner(exp, trip, harness.framing, "think")
    builder = PromptBuilder(harness.framing, planner, exp.images, 1, exp.predict)
    for text in (builder.first(scene_at(trip, track, start)).text, builder.followup(scene_at(trip, track, start + 1), 0.4).text):
        assert text.index("will actually be") < text.index("Your options now:")
        assert text.splitlines()[-1].startswith('- "pass"')


@pytest.mark.parametrize("overrides", [{"episode": "rolling"}, {"execute_s": 0.8}, {"prediction": "true_future", "check": 2}])
def test_skills_run_only_in_the_conversation_every_step(overrides):
    with pytest.raises(ValueError):
        PRESETS["harness_v2"].harness("human", {"output": "skills", **overrides})


def test_gif_labels_each_frame_with_the_skill_that_moved_the_walker_there(tmp_path):
    trip = {**person_trip(), "end_index": 11}  # the real path of the time view needs the trip's end
    start, t = trip["start_index"], np.asarray(trip["time"], float)
    save_trajectory(tmp_path, t[start: start + 3], {a: np.asarray(trip["tracks"][a], float)[start: start + 3, :2] for a in trip["tracks"]})
    (tmp_path / "manifest.json").write_text(json.dumps({"trip": trip["id"], "agent": trip["agent"]}))
    plans = [{"time_s": float(t[start]), "positions_m": [], "skill": "walk"},
             {"time_s": float(t[start + 1]), "positions_m": [], "skill": "pass", "agent": "person", "side": "behind"}]
    (tmp_path / "plans.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plans))
    assert load_episode(tmp_path, {trip["id"]: trip}, {trip["site"]: SITE})["skills"] == [None, "walk", "pass P2 behind"]
    (tmp_path / "plans.jsonl").write_text("".join(json.dumps({"time_s": p["time_s"], "positions_m": [], "reply": [[0, 0.1]]}) + "\n" for p in plans))
    assert load_episode(tmp_path, {trip["id"]: trip}, {trip["site"]: SITE})["skills"] is None


def test_a_pass_started_inside_the_enlarged_robot_stays_finite():
    """The LAAS robot (delta 0.184 m) standing 0.25 m ahead of a walker that walked into it, and the degenerate |D| = 0 case: the walker
    moving straight away from it at 1 / lambda = 0.5 m/s, inside the enlarged robot."""
    robot = series((0.0, 0.25), (0.0, 0.0), 10)
    positions, v, _, distances, nonfinite = pass_substeps(np.zeros(2), np.array([0.0, 1.0]), np.array([0.0, 1.6]), 1.0, robot, 0, "left", 0, 0.184,
                                                          HALF_SIDE, 6)
    assert np.isfinite(positions).all() and np.isfinite(v).all() and nonfinite == 0 and distances[-1] > 0.25
    positions, v, _, _, nonfinite = pass_substeps(np.zeros(2), np.array([0.0, -0.5]), np.array([0.0, -1.6]), 0.5, series((0.0, 0.1), (0.0, 0.0), 10),
                                                  0, "left", 0, 0.184, HALF_SIDE, 1)
    assert np.isfinite(positions).all() and np.isfinite(v).all() and nonfinite == 1
