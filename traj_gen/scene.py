"""What the planner knows at one time index of a trip, and how it is put into words.

A Scene is built for array index i of a trip (trip["time"][i] seconds after the trip start)
from the robot's track so far and the people's tracks. The robot track is an argument, not
read from the trip: in rolling episodes it is the real pre-start history followed by what
the generated plan actually executed. Leakage rule: only samples at indices <= i are ever
read, so nothing after the query time can reach the text or the map.

Coordinates are written in a Units: metres in the room frame (x right, y up), or pixels of
the BEV image (u right, v down), so the pixel output mode differs from the metre mode only
by the coordinate system. Text comes from the templates in prompts/ (str.format fields).
"""

from pathlib import Path

import numpy as np


PROMPTS = Path(__file__).parent / "prompts"
DT = 0.4             # s between samples, from extract_trips.HZ = 2.5
HISTORY_STEPS = 7    # 2.8 s: the most whole steps that fit in the 3 s of history the trips carry
HISTORY_S = round(HISTORY_STEPS * DT, 1)  # 2.8, as written in the prompts
VELOCITY_STEPS = 2   # velocity = displacement over 0.8 s; the tracks are already smoothed over 0.5 s, one step is noisy


def load_prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text().strip()


class Units:
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


class Pixels(Units):
    """Pixels of the BEV image (u right, v down, origin top-left); transform is a bev.BevTransform."""

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


def station_text(label: str) -> str:
    if label == "":
        return "no further goal"
    if "/" in label:
        return "station " + " or ".join(label.split("/")) + " (shared table)"
    return f"station {label}"


class Scene:
    """State at index i of a trip; robot is the (>= i + 1, 2) robot track aligned with the trip arrays."""

    def __init__(self, trip: dict, site: dict, i: int, robot: np.ndarray):
        self.site = site
        self.time_s = trip["time"][i]
        self.robot = np.asarray(robot, float)[: i + 1, :2]
        self.goal = trip["robot_goal"]
        self.goal_xy = np.asarray(trip["robot_goal_xy"], float)
        self.people = []
        for h, track in enumerate(trip["humans"]):
            xy = np.asarray(track, float)[: i + 1, :2]
            k = min(VELOCITY_STEPS, i)
            v = (xy[-1] - xy[-1 - k]) / (k * DT)
            self.people.append({
                "name": f"P{h + 1}", "xy": xy[-1], "history": xy[-1 - HISTORY_STEPS: -1], "velocity": v,
                "speed": float(np.linalg.norm(v)), "standing": trip["human_state"][h][i] == "at_station",
                "goal": trip["human_goal"][h][i], "next_goal": trip["human_next_goal"][h][i],
                "distance": float(np.linalg.norm(xy[-1] - self.robot[-1]))})
        self.robot_history = self.robot[-1 - HISTORY_STEPS: -1]

    def stations_text(self, units: Units) -> str:
        """Station positions, then one sentence per shared table."""
        stations = self.site["stations"]
        lines = [f"station {k}: {units.pos_text(s['xy'])}" for k, s in stations.items() if "/" not in k]
        lines += [load_prompt("shared_table").format(a=a, b=b, pos=units.pos_text(s["xy"]))
                  for k, s in stations.items() if "/" in k for a, b in [s["members"]]]
        return "\n".join(lines)

    def person_text(self, p: dict, units: Units) -> str:
        activity = f"standing at {station_text(p['goal'])}" if p["standing"] else "walking"
        return load_prompt("person").format(
            name=p["name"], pos=units.pos_text(p["xy"]), activity=activity, goal=station_text(p["goal"]),
            next_goal=station_text(p["next_goal"]), velocity=units.vec_text(p["velocity"]),
            speed=units.len_text(p["speed"]), distance=units.len_text(p["distance"]), unit=units.name,
            history=", ".join(units.pos_text(q) for q in p["history"]), hist_s=HISTORY_S, dt=DT)

    def text(self, units: Units) -> str:
        """The per-turn state block."""
        return load_prompt("scene").format(
            time_s=f"{self.time_s:.1f}", robot_pos=units.pos_text(self.robot[-1]), unit=units.name, dt=DT,
            hist_s=HISTORY_S, robot_history=", ".join(units.pos_text(q) for q in self.robot_history),
            goal=station_text(self.goal), goal_pos=units.pos_text(self.goal_xy),
            goal_dist=units.len_text(np.linalg.norm(self.goal_xy - self.robot[-1])),
            people="\n".join(self.person_text(p, units) for p in self.people))
