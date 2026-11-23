"""What the planner knows at one time index of a trip, and how it is put into words.

A Scene is built for array index i of a trip (trip["time"][i] seconds after the trip start)
from the planned agent's track so far (trip["agent"]) and the other agents' tracks. The planned
agent's track is an argument, not read from the trip: in rolling episodes it is the real pre-start
history followed by what the generated plan actually executed. Leakage rule: only samples at indices <= i are ever
read, so nothing after the query time can reach the text or the map.

Coordinates are written in a Metres or Pixels object: metres in the room frame (x right, y up), or pixels of
the BEV image (u right, v down), so the pixel output mode differs from the metre mode only
by the coordinate system. Text comes from the templates in prompts/ (str.format fields).
"""

from functools import cache
from pathlib import Path

import numpy as np

from preprocessing.bi3_derived import TRIP_HISTORY_S, TRIP_HZ, label_members
from traj_gen.agents import STATION_POSITION, agent_kind, body_radius, prompt_radius


PROMPTS = Path(__file__).parent / "prompts"
DT = 1 / TRIP_HZ     # s between samples
PROMPT_HISTORY_STEPS = int(TRIP_HISTORY_S * TRIP_HZ)    # the most whole steps that fit in the history the trips carry
PROMPT_HISTORY_S = round(PROMPT_HISTORY_STEPS * DT, 1)  # 2.8, as written in the prompts
# Horizons of the constant-velocity prediction (prediction experiments): 2 s is the rolling execution window
# (the robot moves on this plan before it can re-plan), 1 s halfway, so a person's position is given for the
# middle and the end of the stretch the robot is committed to.
PREDICT_S = (1.0, 2.0)
VELOCITY_STEPS = 2   # velocity is the displacement over 0.8 s. The tracks are already smoothed over 0.5 s, but one step is noisy


@cache
def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text().strip()


class Metres:
    """Metres in the room frame: x right, y up."""

    name = "m"
    decimals = 2  # centimetres

    def pos(self, xy) -> np.ndarray:
        return np.asarray(xy, float)

    def vec(self, v) -> np.ndarray:
        return np.asarray(v, float)

    def length(self, m: float) -> float:
        return float(m)

    def num(self, v: float) -> str:
        return f"{v:.{self.decimals}f}"

    def pair(self, a: np.ndarray) -> str:
        return f"({self.num(a[0])}, {self.num(a[1])})"

    def pos_text(self, xy) -> str:
        return self.pair(self.pos(xy))

    def vec_text(self, v) -> str:
        return self.pair(self.vec(v))

    def len_text(self, m: float) -> str:
        return self.num(self.length(m))

    def bounds(self, room_m: float) -> dict:
        """The room's extent along the two coordinates, for the frame description."""
        corners = self.pos([[-room_m / 2, -room_m / 2], [room_m / 2, room_m / 2]])
        return {"c0_min": self.num(corners[:, 0].min()), "c0_max": self.num(corners[:, 0].max()),
                "c1_min": self.num(corners[:, 1].min()), "c1_max": self.num(corners[:, 1].max())}


class Pixels(Metres):
    """Pixels of the BEV image (u right, v down, origin top-left). transform is a bev.BevTransform."""

    name = "px"
    decimals = 0

    def __init__(self, transform):
        self.transform = transform

    def pos(self, xy) -> np.ndarray:
        return self.transform.to_pixel(xy)

    def vec(self, v) -> np.ndarray:
        return self.transform.to_pixel(v) - self.transform.to_pixel([0.0, 0.0])

    def length(self, m: float) -> float:
        return float(m) * self.transform.scale


def person_name(agent_id: str) -> str:
    """P<k+1> for agent "human<k>", the person's name in the prompts and maps."""
    return f"P{int(agent_id.removeprefix('human')) + 1}"


def station_text(label: str, stations: dict) -> str:
    members = label_members(label, stations)
    if not members:
        return "no further goal"
    if len(members) > 1:
        return "station " + " or ".join(members) + " (shared table)"
    return f"station {label}"


class Scene:
    """State at index i of a trip. track is the (>= i + 1, 2) track of the planned agent (trip["agent"]) aligned with the trip arrays.

    others holds one dict per other agent, in the trip's tracks order: id, kind, position, history, velocity, speed,
    constant-velocity prediction, whether it stands at a station, its goal labels, and its distance from the planned agent.
    """

    def __init__(self, trip: dict, site: dict, i: int, track: np.ndarray):
        self.site = site
        self.agent = trip["agent"]
        self.time_s = trip["time"][i]
        self.track = np.asarray(track, float)[: i + 1, :2]
        self.goal, self.next_goal = trip["goal"], trip["next_goal"]
        self.goal_xy = np.asarray(trip["goal_xy"], float)
        self.others = []
        for other, other_track in trip["tracks"].items():
            if other == self.agent:
                continue
            xy = np.asarray(other_track, float)[: i + 1, :2]
            k = min(VELOCITY_STEPS, i)
            v = (xy[-1] - xy[-1 - k]) / (k * DT)
            half = site["room_m"] / 2  # agents stay in the room, so the prediction does too
            labels = trip["labels"][other]
            self.others.append({
                "id": other, "kind": agent_kind(other), "xy": xy[-1], "history": xy[-1 - PROMPT_HISTORY_STEPS: -1], "velocity": v,
                "speed": float(np.linalg.norm(v)), "predicted": np.clip(xy[-1] + np.outer(PREDICT_S, v), -half, half), "standing": labels["state"][i] == "at_station",
                "goal": labels["goal"][i], "next_goal": labels["next_goal"][i],
                "distance": float(np.linalg.norm(xy[-1] - self.track[-1]))})
        self.history = self.track[-1 - PROMPT_HISTORY_STEPS: -1]

    def stations_text(self, units: Metres) -> str:
        """Station positions, then one sentence per shared table: the robot's stopping points for a planned robot, where people stand for a planned person."""
        stations, key = self.site["stations"], STATION_POSITION[agent_kind(self.agent)]
        lines = [f"station {k}: {units.pos_text(s[key])}" for k, s in stations.items() if len(s["members"]) == 1]
        lines += [load_prompt("shared_table").format(a=a, b=b, pos=units.pos_text(s[key]))
                  for k, s in stations.items() if len(s["members"]) == 2 for a, b in [s["members"]]]
        return "\n".join(lines)

    def other_text(self, p: dict, units: Metres, predict: bool, scored_sizes: bool = False) -> str:
        """One other agent's line: position, motion, goal and next goal (for the robot, its goal only, as a station name).
        scored_sizes gives the robot's scoring radius instead of its drawn (enclosing) radius."""
        human_planned = agent_kind(self.agent) == "human"
        fields = {"pos": units.pos_text(p["xy"]), "velocity": units.vec_text(p["velocity"]), "speed": units.len_text(p["speed"]),
                  "distance": units.len_text(p["distance"]), "unit": units.name, "history": ", ".join(units.pos_text(q) for q in p["history"]),
                  "hist_s": PROMPT_HISTORY_S, "dt": DT}
        if p["kind"] == "robot":
            name = "the robot"
            text = load_prompt("robot_other_human").format(radius=units.len_text((body_radius if scored_sizes else prompt_radius)(p["id"], self.site)),
                                                           goal=station_text(p["goal"], self.site["stations"]), **fields)
        else:
            name = person_name(p["id"])
            activity = f"standing at {station_text(p['goal'], self.site['stations'])}" if p["standing"] else "walking"
            text = load_prompt("person_human" if human_planned else "person").format(
                name=name, activity=activity, goal=station_text(p["goal"], self.site["stations"]),
                next_goal=station_text(p["next_goal"], self.site["stations"]), **fields)
        if predict:
            (h1, h2), (p1, p2) = PREDICT_S, p["predicted"]
            text += " " + load_prompt("person_prediction").format(name=name, h1=f"{h1:g}", pos1=units.pos_text(p1), h2=f"{h2:g}", pos2=units.pos_text(p2))
        return text

    def text(self, units: Metres, predict: bool, scored_sizes: bool = False) -> str:
        """The per-turn state block of the planned agent, with the other agents. predict adds each other agent's constant-velocity positions."""
        fields = {"time_s": f"{self.time_s:.1f}", "unit": units.name, "dt": DT, "hist_s": PROMPT_HISTORY_S,
                  "goal": station_text(self.goal, self.site["stations"]), "goal_pos": units.pos_text(self.goal_xy),
                  "goal_dist": units.len_text(np.linalg.norm(self.goal_xy - self.track[-1]))}
        own = {"pos": units.pos_text(self.track[-1]), "history": ", ".join(units.pos_text(q) for q in self.history)}
        if agent_kind(self.agent) == "robot":
            return load_prompt("scene").format(robot_pos=own["pos"], robot_history=own["history"], **fields,
                                               people="\n".join(self.other_text(p, units, predict) for p in self.others if p["kind"] == "human"))
        return load_prompt("scene_human").format(
            name=person_name(self.agent), **own, **fields, next_goal=station_text(self.next_goal, self.site["stations"]),
            others="\n".join(self.other_text(p, units, predict, scored_sizes) for p in self.others))
