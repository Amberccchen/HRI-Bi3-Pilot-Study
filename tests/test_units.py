"""The units setting: every number the model reads or writes goes through it, and replies come back to the same metres in either units."""

import numpy as np
import pytest

from tests.test_harness import robot_trip
from tests.test_human_prompt import SITE, person_trip
from traj_gen.harness.presets import Harness
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import DT, Scene
from traj_gen.outputs.planner import UNITS

PX, M = UNITS["px"], UNITS["m"]


def setup(output: str, units: str, framing: str, trip: dict, episode: str = "rolling_conv"):
    harness = Harness(episode, output, units, "bev", framing)
    exp = harness.experiment()
    planner = exp.planner(exp, trip, framing, "think")
    track = np.asarray(trip["tracks"][trip["agent"]], float)[: trip["start_index"] + 1, :2]
    return planner, Scene(trip, SITE, trip["start_index"], track)


def test_pixel_conversions_go_both_ways_and_flip_the_vertical_axis():
    """The pixel frame has v down: a position, a vector, a length and a speed must come back to the same metres, and +y must be -v."""
    xy = np.array([[0.3, -1.2], [-1.75, 1.75]])
    np.testing.assert_allclose(PX.pos_to_metres(PX.pos(xy)), xy, atol=1e-12)
    v = np.array([0.5, 0.25])
    np.testing.assert_allclose(PX.vec(v), [0.5 * PX.transform.scale, -0.25 * PX.transform.scale])
    np.testing.assert_allclose(PX.vec_to_metres(PX.vec(v)), v, atol=1e-12)
    assert PX.length_to_metres(PX.length(1.5)) == pytest.approx(1.5)
    np.testing.assert_allclose(PX.pos([0.0, 1.0]) - PX.pos([0.0, 0.0]), [0.0, -PX.transform.scale])
    assert M.pos_text([0.3, -1.2]) == "(0.30, -1.20)" and M.len_text(1.5) == "1.50"


@pytest.mark.parametrize("kind", ["human", "robot"])
def test_a_position_reply_gives_the_same_path_in_either_units(kind):
    """The same plan written in pixels and in metres must execute to the same positions and raw speeds."""
    trip, framing = (person_trip(), "participant") if kind == "human" else (robot_trip(), "neutral")
    plan_m = np.array([[0.05, -0.5], [0.1, -0.3], [0.2, -0.1]])
    planner_px, scene = setup("positions", "px", framing, trip)
    planner_m, _ = setup("positions", "m", framing, trip)
    a, raw_a = planner_px.positions(PX.pos(plan_m), scene)
    b, raw_b = planner_m.positions(plan_m, scene)
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(raw_a, raw_b, atol=1e-9)


def test_a_steer_reply_gives_the_same_path_in_either_units():
    """Distances scale with the units. A positive turn goes from the first axis toward the second, which is clockwise on the map in pixels
    (v down) and anticlockwise in metres (y up), so the same path has the opposite turns in the two units."""
    trip = person_trip()
    planner_px, scene = setup("steer", "px", "harness", trip)
    planner_m, _ = setup("steer", "m", "harness", trip)
    steps_m = np.array([[20.0, 0.3], [-40.0, 0.5], [10.0, 0.0], [5.0, 0.9]])  # the last one beyond a step at the speed cap (0.6 m)
    steps_px = np.c_[-steps_m[:, 0], PX.length(1.0) * steps_m[:, 1]]
    a, raw_a = planner_px.positions(steps_px, scene)
    b, raw_b = planner_m.positions(steps_m, scene)
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(raw_a, raw_b, atol=1e-9)
    np.testing.assert_allclose(raw_b, steps_m[:, 1] / DT)
    assert planner_px.heading(scene) == pytest.approx(-planner_m.heading(scene))


def test_a_velocity_reply_gives_the_same_path_in_either_units():
    trip = robot_trip()
    velocities_m = np.array([[0.2, 0.1], [0.4, 0.0]])  # the second above the robot's cap, scaled down to it
    planner_px, scene = setup("velseq", "px", "neutral", trip)
    planner_m, _ = setup("velseq", "m", "neutral", trip)
    a, raw_a = planner_px.positions(PX.vec(velocities_m), scene)
    b, raw_b = planner_m.positions(velocities_m, scene)
    np.testing.assert_allclose(a, b, atol=1e-9)
    np.testing.assert_allclose(raw_a, raw_b, atol=1e-9)


@pytest.mark.parametrize("output, framing", [("positions", "participant"), ("steer", "harness")])
def test_metre_prompts_name_metres_and_the_metre_axes_only(output, framing):
    """Every number and axis word must follow the units: a pixel number or a u/v axis left in a metre prompt would render without an error."""
    planner, scene = setup(output, "m", framing, person_trip())
    builder = PromptBuilder(framing, planner, ("bev",), 20, False)
    text = builder.system(scene) + "\n" + builder.first(scene).text + "\n" + builder.followup(scene, 2.0).text
    assert " px" not in text and "pixel" not in text and "+u" not in text and "+v" not in text
    assert "You can walk at most 1.50 m/s." in text and "Your goal: station 3 at (0.10, 1.65)" in text
    if output == "positions":
        assert "Plan your path as points in metres (x, y): one position of yours every 0.4 s" in text
        assert '{"points": [[-0.50, 0.40], [-0.12, 0.52], [0.26, 0.64]]}' in text
    else:
        assert "positive turns anticlockwise on the map, from +x toward +y; negative turns clockwise" in text
        assert "degrees (0 = +x, right; 90 = +y, up)" in text and '{"steps": [[0, 0.40], [-15, 0.40], [0, 0.00]]}' in text


def test_pixel_prompts_keep_the_pixel_wording():
    planner, scene = setup("steer", "px", "harness", person_trip())
    text = PromptBuilder("harness", planner, ("bev",), 20, False).first(scene).text
    assert "positive turns clockwise on the map, from +u toward +v; negative turns anticlockwise" in text
    assert "degrees (0 = +u, right; 90 = +v, down)" in text and '{"steps": [[0, 45], [-15, 45], [0, 0]]}' in text  # 0.4 m at 112 px/m
