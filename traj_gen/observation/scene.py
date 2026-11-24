"""What the planner knows at one time index of a trip, and how it is put into words.

A Scene is built for array index i of a trip (trip["time"][i] seconds after the trip start)
from the planned agent's track so far (trip["agent"]) and the other agents' tracks. The planned
agent's track is an argument, not read from the trip: in rolling episodes it is the real pre-start
history (from the 10 Hz series) followed by what the generated plan actually executed. The other agents' positions are
read from the trip's 10 Hz series (trip["input_tracks_10hz"]), whose every SCORED_STEP-th sample is a trip sample, so
series sample SCORED_STEP * i is at the scene's time. Leakage rule: positions are read only up to the scene's time; the
velocity is a central difference over the series samples one before and one after it (0.1 s each side).

Coordinates are written in a Metres or Pixels object: metres in the room frame (x right, y up), or pixels of
the BEV image (u right, v down), so the pixel output mode differs from the metre mode only
by the coordinate system. Text comes from the templates in prompts/ (str.format fields).
"""

import numpy as np

from preprocessing.bi3_derived import INPUT_HZ, SCORED_STEP, TRIP_HISTORY_S, TRIP_HZ, label_members
from traj_gen.agents import STATION_POSITION, agent_kind, body_radius, prompt_radius
from traj_gen.observation import PLANNED_AGENT_WORDS, load_prompt
from traj_gen.observation.units import Metres


DT = 1 / TRIP_HZ     # s between samples
PROMPT_HISTORY_STEPS = int(TRIP_HISTORY_S * TRIP_HZ)    # the most whole steps that fit in the history the trips carry
PROMPT_HISTORY_S = round(PROMPT_HISTORY_STEPS * DT, 1)  # 2.8, as written in the prompts
# Horizons of the constant-velocity prediction (prediction experiments): 2 s is the rolling execution window
# (the robot moves on this plan before it can re-plan), 1 s halfway, so a person's position is given for the
# middle and the end of the stretch the robot is committed to.
PREDICT_S = (1.0, 2.0)


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
        j = SCORED_STEP * i  # the series sample at the scene's time
        for other, series in trip["input_tracks_10hz"]["tracks"].items():
            if other == self.agent:
                continue
            dense = np.asarray(series, float)[:, :2]
            xy = dense[: j + 1: SCORED_STEP]
            before, after = max(j - 1, 0), min(j + 1, len(dense) - 1)  # one sample either side, one-sided at the ends of the series
            v = (dense[after] - dense[before]) * INPUT_HZ / (after - before)
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
        lines += [load_prompt("system", "shared_table").format(a=a, b=b, pos=units.pos_text(s[key]))
                  for k, s in stations.items() if len(s["members"]) == 2 for a, b in [s["members"]]]
        return "\n".join(lines)

    def other_text(self, p: dict, units: Metres, predict: bool, scored_sizes: bool, first_turn: bool) -> str:
        """One other agent's line: position, motion, goal and next goal (for the robot, its goal only, as a station name).
        The robot's size is given in the first turn only: its scoring radius with scored_sizes, else its drawn (enclosing) radius."""
        fields = {"pos": units.pos_text(p["xy"]), "velocity": units.vec_text(p["velocity"]), "speed": units.len_text(p["speed"]),
                  "distance": units.len_text(p["distance"]), "unit": units.name, "history": ", ".join(units.pos_text(q) for q in p["history"]),
                  "hist_s": PROMPT_HISTORY_S, "dt": DT}
        if p["kind"] == "robot":
            name = "the robot"
            radius = units.len_text((body_radius if scored_sizes else prompt_radius)(p["id"], self.site))
            size = ", " + load_prompt("state", "robot_size").format(radius=radius, unit=units.name) if first_turn else ""
            text = load_prompt("state", "other_robot").format(size=size, goal=station_text(p["goal"], self.site["stations"]), **fields)
        else:
            name = person_name(p["id"])
            activity = f"standing at {station_text(p['goal'], self.site['stations'])}" if p["standing"] else "walking"
            text = load_prompt("state", "other_person").format(
                name=name, activity=activity, **PLANNED_AGENT_WORDS[agent_kind(self.agent)], goal=station_text(p["goal"], self.site["stations"]),
                next_goal=station_text(p["next_goal"], self.site["stations"]), **fields)
        if predict:
            (h1, h2), (p1, p2) = PREDICT_S, p["predicted"]
            text += " " + load_prompt("state", "other_constvel_future").format(name=name, h1=f"{h1:g}", pos1=units.pos_text(p1), h2=f"{h2:g}", pos2=units.pos_text(p2))
        return text

    def text(self, units: Metres, predict: bool, scored_sizes: bool, first_turn: bool) -> str:
        """The per-turn state block of the planned agent, with the other agents. predict adds each other agent's constant-velocity positions.
        The note on what the others' goals mean and the robot's size are given in the first turn of a conversation only."""
        kind = agent_kind(self.agent)
        fields = {"time_s": f"{self.time_s:.1f}", "unit": units.name, "dt": DT, "hist_s": PROMPT_HISTORY_S,
                  "others_note": f" ({load_prompt('state', 'others_note', kind)})" if first_turn else "",
                  "goal": station_text(self.goal, self.site["stations"]), "goal_pos": units.pos_text(self.goal_xy),
                  "goal_dist": units.len_text(np.linalg.norm(self.goal_xy - self.track[-1]))}
        own = {"pos": units.pos_text(self.track[-1]), "history": ", ".join(units.pos_text(q) for q in self.history)}
        if kind == "robot":
            return load_prompt("state", "scene", "robot").format(robot_pos=own["pos"], robot_history=own["history"], **fields,
                                               people="\n".join(self.other_text(p, units, predict, scored_sizes, first_turn) for p in self.others if p["kind"] == "human"))
        return load_prompt("state", "scene", "human").format(
            name=person_name(self.agent), **own, **fields, next_goal=station_text(self.next_goal, self.site["stations"]),
            others="\n".join(self.other_text(p, units, predict, scored_sizes, first_turn) for p in self.others))
