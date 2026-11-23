"""The steer_vel output: a turn rate and a speed held for each step, so the agent follows circular arcs (the unicycle model)."""

import numpy as np
import pytest

from tests.fake_trips import SITE, person_trip, robot_trip
from traj_gen.agents import SPEED_CAP_M_S
from traj_gen.harness.presets import Harness
from traj_gen.observation.prompting import PromptBuilder
from traj_gen.observation.scene import DT, Scene
from traj_gen.outputs.steer import SteerPlanner


def setup(units: str, trip: dict, framing: str):
    exp = Harness("rolling_conv", "steer_vel", units, "bev", framing).experiment()
    planner = exp.planner(exp, trip, framing, "think")
    track = np.asarray(trip["tracks"][trip["agent"]], float)[: trip["start_index"] + 1, :2]
    return planner, Scene(trip, SITE, trip["start_index"], track)


def euler(start: np.ndarray, heading_deg: float, steps: np.ndarray, substeps: int = 20000) -> np.ndarray:
    """The unicycle integrated in small time steps (an independent implementation): (n, 2) positions at the step ends, in the reply's frame."""
    p, theta, out = np.array(start, float), np.radians(heading_deg), []
    h = DT / substeps
    for rate, speed in steps:
        for _ in range(substeps):  # midpoint rule on the heading
            mid = theta + np.radians(rate) * h / 2
            p = p + speed * h * np.array([np.cos(mid), np.sin(mid)])
            theta += np.radians(rate) * h
        out.append(p.copy())
    return np.array(out)


@pytest.mark.parametrize("units", ["m", "px"])
def test_steps_follow_the_integrated_unicycle(units):
    """The closed-form arcs must match the unicycle integrated in small steps, with the heading carried from step to step."""
    planner, scene = setup(units, person_trip(), "scored_distances")
    scale = planner.units.length(1.0)
    steps = np.array([[90.0, 1.0 * scale], [0.0, 0.8 * scale], [-200.0, 1.2 * scale], [0.0, 0.0], [45.0, 0.5 * scale]])
    expected = euler(planner.units.pos(scene.track[-1]), planner.heading(scene), steps)
    np.testing.assert_allclose(planner.units.pos(planner.positions(steps, scene)[0]), expected, atol=1e-6 * scale)


def test_zero_turn_rate_walks_straight_along_the_heading():
    planner, scene = setup("m", person_trip(), "scored_distances")
    heading = np.radians(planner.heading(scene))
    positions, raw = planner.positions(np.array([[0.0, 1.0], [0.0, 1.0]]), scene)
    direction = np.array([np.cos(heading), np.sin(heading)])
    np.testing.assert_allclose(positions, scene.track[-1] + np.outer([0.4, 0.8], direction), atol=1e-12)
    np.testing.assert_allclose(raw, [1.0, 1.0])


def test_a_held_turn_rate_stays_on_the_circle_of_radius_speed_over_turn_rate():
    """At 90 deg/s and 1 m/s the agent stays on a circle of radius 1 / (pi / 2) m through its start, tangent to its heading."""
    planner, scene = setup("m", person_trip(), "scored_distances")
    heading = np.radians(planner.heading(scene))
    positions, _ = planner.positions(np.array([[90.0, 1.0]] * 4), scene)
    radius = 1.0 / np.radians(90.0)
    centre = scene.track[-1] + radius * np.array([-np.sin(heading), np.cos(heading)])  # to the left: a positive turn is anticlockwise in metres
    np.testing.assert_allclose(np.linalg.norm(positions - centre, axis=1), radius, atol=1e-12)
    end_heading = heading + np.radians(4 * 90.0 * DT)  # the heading has turned 144 deg
    np.testing.assert_allclose(positions[-1], centre + radius * np.array([np.sin(end_heading), -np.cos(end_heading)]), atol=1e-12)


@pytest.mark.parametrize("kind", ["human", "robot"])
def test_turn_rate_and_speed_are_cut_to_the_caps(kind):
    """A turn rate beyond the steer turn cap per step divided by the step, or a speed beyond the cap (or below 0), must be cut, while
    the raw speed is kept for the speed-violation metric."""
    trip, framing = (person_trip(), "scored_distances") if kind == "human" else (robot_trip(), "robot_scored_distances")
    planner, scene = setup("m", trip, framing)
    cap_rate, cap_speed = SteerPlanner.MAX_TURN_DEG[kind] / DT, SPEED_CAP_M_S[kind]
    cut, raw = planner.positions(np.array([[5000.0, 9.0], [-5000.0, -1.0]]), scene)
    capped, _ = planner.positions(np.array([[cap_rate, cap_speed], [-cap_rate, 0.0]]), scene)
    np.testing.assert_allclose(cut, capped, atol=1e-12)
    np.testing.assert_allclose(raw, [9.0, 1.0])


def test_prompt_states_the_turn_rate_cap_and_the_speed_unit():
    planner, scene = setup("px", person_trip(), "scored_distances")
    text = PromptBuilder("scored_distances", planner, ("bev",), 20, False).first(scene).text
    assert f"A turn rate is at most {SteerPlanner.MAX_TURN_DEG['human'] / DT:g} degrees per second" in text and "in px/s" in text
    assert '{"steps": [[0, 112], [-40, 112], [0, 0]]}' in text  # 1 m/s at 112 px per metre
    robot, robot_scene = setup("m", robot_trip(), "robot_scored_distances")
    text = PromptBuilder("robot_scored_distances", robot, ("bev",), 20, False).first(robot_scene).text
    assert f"A turn rate is at most {SteerPlanner.MAX_TURN_DEG['robot'] / DT:g} degrees per second" in text and "in m/s, from 0 to the maximum speed" in text
