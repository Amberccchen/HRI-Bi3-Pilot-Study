"""The skills output mode: every 0.4 s the model chooses a skill, and code turns the choice into motion at the 10 Hz of the input series.

- walk: a straight line to the goal at the chosen speed, ignoring the other agents.
- pass: toward the goal at the chosen speed under social force from one named agent, on the chosen side (SidedSocialForce), with
  the walls repelling. It lasts until another skill is chosen.
- wait: stop at once.
- continue: keep the current skill with its agent, side and speed.

The pass uses PySocialForce's goal force (DesiredForce), wall force (ObstacleForce) and helpers as they are, and its pedestrian force
(the Moussaid et al. 2009 model) with the turning side set by the skill. Inside a 0.4 s decision the agent being passed is read from the
trip's 10 Hz input series at every 0.1 s sub-step (position, and velocity as the observation defines it), never from its future in the
text. The walker's 10 Hz velocity carries over from one decision to the next.

Sides. "left" / "right" keep the agent on the walker's left or right, judged along d, the walker's direction to its goal. "in front" /
"behind" cross ahead of or behind a moving agent's direction of travel. They are offered when the agent moves at CROSSING_MIN_SPEED_M_S
or more and its direction is within CROSSING_ANGLES_DEG of d, and the others get "left" / "right". A side is offered only when the walls
leave room to pass there (side_is_free).
"""

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from preprocessing.bi3_derived import INPUT_HZ, SCORED_STEP
from traj_gen.agents import PERSON_BODY_RADIUS_M, SPEED_CAP_M_S, agent_kind, body_radius, pair_distances, room_half_side
from traj_gen.observation import load_prompt
from traj_gen.observation.scene import DT, Scene, person_name, series_velocity
from traj_gen.outputs.planner import QueryPlanner

# Importing pysocialforce sets the root logger to DEBUG, adds a stderr handler and opens file.log in the working directory
# (pysocialforce/utils/logging.py). Both are undone right after the import.
_root_logger = logging.getLogger()
_root_level, _log_file_existed = _root_logger.level, Path("file.log").exists()
from pysocialforce.forces import DesiredForce, ObstacleForce, SocialForce  # noqa: E402
from pysocialforce.scene import EnvState, PedState  # noqa: E402
from pysocialforce.utils import Config, stateutils  # noqa: E402
from pysocialforce.utils import logging as pysocialforce_logging  # noqa: E402

for _handler in (pysocialforce_logging.c_handler, pysocialforce_logging.f_handler):
    _root_logger.removeHandler(_handler)
    _handler.close()
_root_logger.setLevel(_root_level)
if not _log_file_existed and Path(pysocialforce_logging.f_handler.baseFilename).stat().st_size == 0:
    Path(pysocialforce_logging.f_handler.baseFilename).unlink()


SUB_DT = 1 / INPUT_HZ  # s per sub-step of a skill. SCORED_STEP sub-steps make one 0.4 s decision
CROSSING_MIN_SPEED_M_S = 0.15     # an agent at least this fast can be crossed in front of or behind
CROSSING_ANGLES_DEG = (45.0, 135.0)  # ... when its direction of travel is within this range of the walker's direction to its goal
PASS_CAP_MULTIPLIER = 1.3  # the walker's speed in a pass is capped at this times the chosen speed (the package's max_speed_multiplier)
# Force parameters. Goal force and pedestrian force: the package's DefaultConfig (the configuration its Simulator runs). Wall force: the
# fallbacks of ObstacleForce._get_force, since DefaultConfig's (factor 10, threshold 3.0 m) push harder than the goal force at every
# station's standing point. Keyed by the forces' snake-case class names, as Force.init looks them up.
FORCE_CONFIG = Config({"desired_force": {"factor": 1.0, "relaxation_time": 0.5, "goal_threshold": 0.2},
                       "sided_social_force": {"factor": 5.1, "lambda_importance": 2.0, "gamma": 0.35, "n": 2, "n_prime": 3},
                       "obstacle_force": {"factor": 1.0, "sigma": 0.2, "threshold": 0.2}})
WALL_POINTS_PER_M = 10  # wall sampling of the package's EnvState (its default resolution)
SIDE_ROOM_M = PERSON_BODY_RADIUS_M  # a side is free when the wall is at least the contact distance plus this beyond the agent's centre
SIDES = {"crossing": ("in front", "behind"), "beside": ("left", "right")}
REPLY_AGENTS = {"robot": "robot", "human": "person"}  # an other agent's kind -> its name in the reply's agent field


class SidedSocialForce(SocialForce):
    """The pedestrian force of PySocialForce 1.1.2 (forces.SocialForce, MIT licence, github.com/yuxiang-gao/PySocialForce), the model of
    Moussaid, Helbing, Garnier, Johansson, Combe and Theraulaz 2009, "Experimental study of the behavioural mechanisms underlying
    self-organization in human crowds", Proc. R. Soc. B 276:2755. Per pair, with the interaction vector D = lambda (v_other - v_self)
    + e (e: unit vector from the other to self), t = D / |D|, theta the angle from e to t and B = gamma |D|:
        f = exp(-d / B - (n' B theta)^2) t + s exp(-d / B - (n B theta)^2) left_normal(t)
    Three edits to the package's computation:
    - the turning side s is the given side (+1 keeps the other on self's left, -1 on its right) instead of -sign(theta).
    - d is the centre distance minus delta, at least 0, which enlarges the other by delta and multiplies both terms by exp(delta / B),
      bounded inside the enlarged other by the force at d = 0.
    - theta is wrapped to (-pi, pi]. The package takes the difference of two arctan2 angles unwrapped, so a pair whose t and e lie on
      either side of the -x axis gets a theta near 2 pi and no force."""

    def __init__(self, side: int, delta: float):
        super().__init__()
        self.side, self.delta = side, delta

    def _get_force(self):
        lambda_importance = self.config("lambda_importance", 2.0)
        gamma = self.config("gamma", 0.35)
        n = self.config("n", 2)
        n_prime = self.config("n_prime", 3)

        pos_diff = stateutils.each_diff(self.peds.pos())  # self - other
        diff_direction, diff_length = stateutils.normalize(pos_diff)
        vel_diff = -1.0 * stateutils.each_diff(self.peds.vel())  # other - self
        interaction_vec = lambda_importance * vel_diff + diff_direction
        interaction_direction, interaction_length = stateutils.normalize(interaction_vec)
        theta = stateutils.vector_angles(interaction_direction) - stateutils.vector_angles(diff_direction)
        theta = np.angle(np.exp(1j * theta))  # wrapped to (-pi, pi]
        B = gamma * interaction_length
        distance = np.maximum(diff_length - self.delta, 0.0)

        force_velocity_amount = np.exp(-1.0 * distance / B - np.square(n_prime * B * theta))
        force_angle_amount = self.side * np.exp(-1.0 * distance / B - np.square(n * B * theta))
        force_velocity = force_velocity_amount.reshape(-1, 1) * interaction_direction
        force_angle = force_angle_amount.reshape(-1, 1) * stateutils.left_normal(interaction_direction)
        force = force_velocity + force_angle
        return np.sum(force.reshape((self.peds.size(), -1, 2)), axis=1) * self.factor


def crossing(d: np.ndarray, v: np.ndarray) -> bool:
    """Whether an agent of velocity v gets the "in front" / "behind" pair for a walker heading along the unit vector d."""
    speed = np.linalg.norm(v)
    if speed < CROSSING_MIN_SPEED_M_S:
        return False
    angle = np.degrees(np.arccos(np.clip(np.dot(d, v) / speed, -1.0, 1.0)))
    return CROSSING_ANGLES_DEG[0] <= angle <= CROSSING_ANGLES_DEG[1]


def turning_sign(side: str, xy: np.ndarray, v: np.ndarray, goal: np.ndarray, other_xy: np.ndarray, other_v: np.ndarray, previous: int) -> int:
    """The turning side s of SidedSocialForce for a side word, from t, the interaction direction the force uses (D = lambda (other_v - v)
    + unit(xy - other_xy)), and d, the walker's direction to its goal. With cross(a, b) = a_x b_y - a_y b_x:
    - "left": s = -sign(t . d), "right": s = +sign(t . d), so the push s left_normal(t) moves the walker away from that side of d;
    - "in front": s = +sign(cross(t, other_v)), "behind": s = -sign(cross(t, other_v)), so the push's component along the agent's
      velocity is positive in front and negative behind.
    The previous sign is kept when the quantity is exactly 0, or for in front / behind when the agent is slower than
    CROSSING_MIN_SPEED_M_S. Without a previous sign (0) the base sign applies: +1 left, -1 right, and for in front / behind
    -sign(cross(d, other_v)) and +sign(cross(d, other_v)), which the rule above gives for t = -d."""
    d = unit(goal - xy)
    t = unit(FORCE_CONFIG.config["sided_social_force"]["lambda_importance"] * (other_v - v) + unit(xy - other_xy))
    if side in ("left", "right"):
        s = int(np.sign(np.dot(t, d))) * (-1 if side == "left" else 1)
        base = 1 if side == "left" else -1
    elif np.linalg.norm(other_v) < CROSSING_MIN_SPEED_M_S:
        s = base = 0
    else:
        s = int(np.sign(t[0] * other_v[1] - t[1] * other_v[0])) * (1 if side == "in front" else -1)
        base = int(np.sign(d[0] * other_v[1] - d[1] * other_v[0])) * (-1 if side == "in front" else 1)
    return s or previous or base


def side_is_free(side: str, d: np.ndarray, other_xy: np.ndarray, other_v: np.ndarray, contact: float, half_side: float) -> bool:
    """Whether the walls leave room to pass an agent on a side: the distance from the agent's centre to the wall square (half side
    half_side) along the side's direction is at least contact + SIDE_ROOM_M. The direction: the agent's direction of travel (in front),
    against it (behind), the right normal of d (left: the walker passes on the agent's right as seen along d) or its left normal (right)."""
    n = {"in front": other_v, "behind": -other_v, "left": np.array([d[1], -d[0]]), "right": np.array([-d[1], d[0]])}[side]
    n = n / np.linalg.norm(n)
    with np.errstate(divide="ignore", invalid="ignore"):  # a zero component never reaches that pair of walls
        reach = np.where(n != 0, (half_side * np.sign(n) - other_xy) / n, np.inf)
    return float(reach.min()) >= contact + SIDE_ROOM_M


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else np.zeros(2)


def pass_substeps(xy: np.ndarray, v: np.ndarray, goal: np.ndarray, speed: float, other: np.ndarray, j: int, side: str, sign: int, delta: float,
                  half_side: float, n: int) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    """n sub-steps of SUB_DT of a pass from sample j of the passed agent's (N, 2) 10 Hz series, inside walls of half side half_side. Sub-step
    k reads the agent at sample j + k (position, and series_velocity), re-reads the turning sign (turning_sign, from sign), takes the
    walker's acceleration from the package's forces on a two-agent state (row 0 the walker, row 1 the agent): the goal force toward goal
    at the desired speed, the sided pedestrian force from the agent and the wall force. It integrates the walker as the package's
    PedState.step does: v <- v + dt a capped at min(PASS_CAP_MULTIPLIER speed, the person speed cap), then xy <- xy + dt v. A sub-step
    whose pedestrian force is not finite (|D| = 0 inside the enlarged agent, where B = 0) uses the goal and wall forces only. Returns the
    (n, 2) walker positions, its velocity and the turning sign after the last sub-step, the (n,) centre distances to the agent at
    samples j + 1 .. j + n, and the number of sub-steps without the pedestrian force."""
    h = half_side
    wall_points = EnvState([(-h, h, -h, -h), (-h, h, h, h), (-h, -h, -h, h), (h, h, -h, h)], WALL_POINTS_PER_M).obstacles
    cap = np.array([min(PASS_CAP_MULTIPLIER * speed, SPEED_CAP_M_S["human"])])
    positions, distances, nonfinite = [], [], 0
    for k in range(n):
        other_xy, other_v = other[j + k], series_velocity(other, j + k)
        sign = turning_sign(side, xy, v, goal, other_xy, other_v, sign)
        peds = PedState(np.array([[*xy, *v, *goal], [*other_xy, *other_v, *other_xy]]), None, Config({"agent_radius": PERSON_BODY_RADIUS_M}))
        peds.max_speeds = np.array([speed, 0.0])  # DesiredForce reads the desired speed from max_speeds
        goal_and_walls, pedestrian = (DesiredForce(), ObstacleForce()), SidedSocialForce(sign, delta)
        for force in (*goal_and_walls, pedestrian):
            force.init(SimpleNamespace(peds=peds, get_obstacles=lambda: wall_points), FORCE_CONFIG)
        a = sum(force.get_force() for force in goal_and_walls)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            a_pedestrian = pedestrian.get_force()[0]
        if np.isfinite(a_pedestrian).all():
            a = a + a_pedestrian
        else:
            nonfinite += 1
        v = PedState.capped_velocity((v + SUB_DT * a)[None], cap)[0]
        xy = xy + SUB_DT * v
        positions.append(xy)
        distances.append(np.linalg.norm(xy - other[j + k + 1]))
    return np.array(positions), v, sign, np.array(distances), nonfinite


def walk_substeps(xy: np.ndarray, goal: np.ndarray, speed: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n sub-steps of SUB_DT of a straight walk toward goal at speed, stopping at the goal: the (n, 2) positions and the final velocity."""
    positions, v = [], np.zeros(2)
    for _ in range(n):
        step = goal - xy
        v = speed * unit(step) if np.linalg.norm(step) > 0 else np.zeros(2)
        xy = xy + unit(step) * min(speed * SUB_DT, np.linalg.norm(step))
        positions.append(xy)
    return np.array(positions), v


class SkillPlanner(QueryPlanner):
    """The skills mode for a planned person: a reply is one skill choice (parse), executed for one 0.4 s step at 10 Hz (positions). The
    planner is made per episode and holds the skill under way, the walker's 10 Hz velocity and heading, the options of the current
    turn (closing_text), the record of the last decision (plan_record) and the passes so far (manifest_fields). Speeds of a reply are in
    the units, and a speed above the person speed cap is cut to it. The pass enlarges the agent being passed by delta, its body radius
    minus a person's, since the pedestrian force treats both as points."""

    name = "skills"
    template = system_template = followup_template = reply_template = "skills"
    key = "skill"
    scored_distance_framings = True
    scored_sizes = True

    def __init__(self, exp, trip, framing, thinking):
        super().__init__(exp, trip, framing, thinking)
        self.others = {REPLY_AGENTS[agent_kind(a)]: a for a in trip["tracks"] if a != trip["agent"]}  # reply agent name -> agent id
        self.names = {agent: "the robot" if agent == "robot" else person_name(i) for agent, i in self.others.items()}  # reply agent name -> its name in the prompts
        self.current = None    # the skill under way: skill, agent, side, speed_m_s, requested_m_s, delta_m
        self.sign = 0          # the turning sign of the pass under way
        self.velocity = None   # the walker's velocity (m/s) at the end of the last sub-step, None before the first decision
        self.heading = None    # its last heading while moving (radians in the room frame)
        self.offered = None    # the options of the current turn: {"continue": bool, "agents": {reply agent: [sides]}}
        self.record, self.passes, self.decisions, self.switches = None, [], 0, 0

    def max_steps(self, remaining: int) -> int:
        return 1

    def reply_fields(self) -> dict:
        return {}

    @property
    def prefill(self) -> str:
        return '{"skill": "'

    def closing_text(self, scene: Scene) -> str:
        """The option list of this turn, which parse checks replies against: continue (once a skill is under way), walk, wait, and per
        other agent its wording pair minus the sides the walls block."""
        u, d = self.units, unit(scene.goal_xy - scene.track[-1])
        half_side = room_half_side(scene.site, "human")
        self.offered = {"continue": self.current is not None, "agents": {}}
        lines = []
        if self.current is not None:
            c = self.current
            fields = {"unit": u.name, "speed": u.len_text(c["speed_m_s"]) if c["speed_m_s"] is not None else None}
            if c["skill"] == "pass":
                fields |= {"agent": c["agent"], "name": self.names[c["agent"]], "side": c["side"]}
            lines.append(load_prompt("skill_options", f"continue_{c['skill']}").format(**fields))
        lines += [load_prompt("skill_options", "walk"), load_prompt("skill_options", "wait")]
        for p in scene.others:
            agent = REPLY_AGENTS[p["kind"]]
            name = self.names[agent]
            contact = pair_distances(self.trip["agent"], p["id"], scene.site)[0]
            sides = [s for s in SIDES["crossing" if crossing(d, p["velocity"]) else "beside"]
                     if side_is_free(s, d, p["xy"], p["velocity"], contact, half_side)]
            if not sides:
                continue
            self.offered["agents"][agent] = sides
            phrases = [load_prompt("skill_options", "side_" + s.replace(" ", "_")).format(name=name) for s in sides]
            lines.append(load_prompt("skill_options", f"pass_{len(sides)}").format(agent=agent, name=name, side1=phrases[0], side2=phrases[-1]))
        return load_prompt("skill_options", "list").format(lines="\n".join(lines))

    def parse(self, text: str, max_steps: int) -> dict | None:
        """The reply's fields if they are a valid choice among the options of the current turn (closing_text), else None. Invalid: no JSON
        object, an unknown skill, continue when not offered, walk or pass without a finite speed above 0, pass without an offered agent,
        and a side offered for it (both strings). Fields a skill does not use are dropped."""
        def number(x) -> bool:
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                return False
            try:
                return math.isfinite(float(x))
            except OverflowError:  # an int beyond the float range
                return False
        try:
            reply = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except ValueError:  # no braces or bad JSON
            return None
        skill = reply.get("skill") if isinstance(reply, dict) else None
        if skill == "wait" or (skill == "continue" and self.offered["continue"]):
            return {"skill": skill}
        if skill not in ("walk", "pass") or not number(reply.get("speed")) or reply["speed"] <= 0:
            return None
        if skill == "walk":
            return {"skill": "walk", "speed": reply["speed"]}
        agent, side = reply.get("agent"), reply.get("side")
        if not isinstance(agent, str) or not isinstance(side, str) or side not in self.offered["agents"].get(agent, ()):
            return None
        return {"skill": "pass", "agent": agent, "side": side, "speed": reply["speed"]}

    def positions(self, value, scene):
        """The walker's position after one 0.4 s step of the chosen skill, and the requested speed (m/s, 0 for wait). Advances the
        planner's state, so it is called once per executed decision."""
        if self.velocity is None:  # the first decision starts from the motion the state gives
            self.velocity, self.heading = scene.speed * np.array([np.cos(scene.heading), np.sin(scene.heading)]), scene.heading
        previous = self.current
        if value["skill"] != "continue":
            self.current = self.choice(value, scene)
            self.sign = 0
        c, xy, goal = self.current, scene.track[-1], scene.goal_xy
        j = SCORED_STEP * (len(scene.track) - 1)  # the series sample at the scene's time
        nonfinite = 0  # pass sub-steps without the pedestrian force
        if c["skill"] == "walk":
            substeps, self.velocity = walk_substeps(xy, goal, c["speed_m_s"], SCORED_STEP)
        elif c["skill"] == "wait":
            substeps, self.velocity = np.repeat(xy[None], SCORED_STEP, axis=0), np.zeros(2)
        else:
            other = np.asarray(self.trip["input_tracks_10hz"]["tracks"][self.others[c["agent"]]], float)[:, :2]
            substeps, self.velocity, self.sign, distances, nonfinite = pass_substeps(xy, self.velocity, goal, c["speed_m_s"], other, j, c["side"],
                                                                                     self.sign, c["delta_m"], room_half_side(scene.site, "human"), SCORED_STEP)
            if value["skill"] == "pass":
                self.passes.append({"start_s": scene.time_s, "agent": c["agent"], "side": c["side"], "delta_m": c["delta_m"], "closest_m": np.inf})
            self.passes[-1] |= {"end_s": round(scene.time_s + DT, 3), "closest_m": min(self.passes[-1]["closest_m"], float(distances.min()))}
        if np.linalg.norm(self.velocity) > 0:
            self.heading = float(np.arctan2(self.velocity[1], self.velocity[0]))
        switched = previous is not None and any(previous.get(k) != c.get(k) for k in ("skill", "agent", "side"))
        self.decisions += 1
        self.switches += switched
        self.record = {"options": self.offered, **c, "continued": value["skill"] == "continue", "switched": switched,
                       "turning_side": {1: "left", -1: "right"}.get(self.sign) if c["skill"] == "pass" else None,
                       "substeps_m": substeps.tolist(), "nonfinite_substeps": nonfinite}
        return substeps[-1:], np.array([c["requested_m_s"]])

    def choice(self, value: dict, scene: Scene) -> dict:
        """A new skill choice in metres: the speed cut to the person speed cap, and for a pass the passed agent's delta."""
        c = {"skill": value["skill"], "agent": None, "side": None, "speed_m_s": None, "requested_m_s": 0.0, "delta_m": None}
        if value["skill"] == "wait":
            return c
        requested = float(self.units.length_to_metres(value["speed"]))
        c |= {"speed_m_s": min(requested, self.speed_cap), "requested_m_s": requested}
        if value["skill"] == "pass":
            c |= {"agent": value["agent"], "side": value["side"],
                  "delta_m": body_radius(self.others[value["agent"]], scene.site) - body_radius(self.trip["agent"], scene.site)}
        return c

    def motion(self, value, scene, track):
        return self.heading, float(np.linalg.norm(self.velocity))

    def plan_record(self, value) -> dict:
        return {"reply": value, **self.record}

    def manifest_fields(self) -> dict:
        return {"decisions": self.decisions, "switches": self.switches, "passes": self.passes}
