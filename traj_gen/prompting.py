"""Assemble the system prompt and user turns for one framing, output mode and set of images.

A PromptBuilder is made once per condition (framing x output mode x images x unit system) and
then asked for the system prompt and the first user turn of a Scene, or for the short
follow-up turn of a conversation. What differs between output modes lives in a subclass
(template, example, how the example is written in the chosen units); everything else is shared.
Image kinds in a Turn are "bev" (render_bev of the turn's scene) and "cam" (camera_frame of the trip),
in the order they are described in the text.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "metrics"))
from bev import SIZE_PX  # noqa: E402
from scene import DT, HISTORY_S, Scene, Units, load_prompt  # noqa: E402
from traj_gen_metrics import GOAL_TOL, HUMAN_RADIUS, SPEED_CAP  # noqa: E402


FRAMINGS = ("neutral", "social", "social_body", "cautious")  # social_body: social + the centre distance the bodies need; cautious: social, stronger (waiting/detours are fine)


@dataclass
class Turn:
    text: str
    images: list[str]


class PromptBuilder:
    """Waypoint plan in the units' coordinates; the base for the other output modes."""

    template = "output_waypoints"
    key = "waypoints"  # key of the JSON object the model replies with: {key: [[a, b], ...]}
    # Example numbers in metres (converted to the units in use): a short slow path, ~0.1 m per step (0.25 m/s)
    # in an arbitrary place so it does not hint at where the robot should go.
    example = [(-0.50, 0.40), (-0.40, 0.45), (-0.30, 0.50)]

    def __init__(self, framing: str, units: Units, images: tuple[str, ...], max_steps: int):
        assert framing in FRAMINGS and set(images) <= {"bev", "cam"}
        self.framing, self.units, self.images, self.max_steps = framing, units, images, max_steps

    def to_units(self, value) -> np.ndarray:
        return self.units.pos(value)

    def example_text(self) -> str:
        return ", ".join("[" + ", ".join(self.units.num(c) for c in self.to_units(p)) + "]" for p in self.example)

    def output_text(self) -> str:
        output = load_prompt(self.template).format(unit=self.units.name, dt=DT, dt2=2 * DT, max_steps=self.max_steps)
        return output + "\n" + load_prompt("reply").format(key=self.key, example=self.example_text())

    def system(self, scene: Scene) -> str:
        u, site = self.units, scene.site
        frame = load_prompt(f"frame_{u.name}").format(image_px=SIZE_PX, **u.bounds(site["room_m"]))
        return load_prompt("system").format(
            room_side=u.len_text(site["room_m"]), unit=u.name, frame=frame, stations=scene.stations_text(u),
            robot_radius=u.len_text(site["robot_radius_m"]), speed_cap=u.len_text(SPEED_CAP), dt=DT,
            goal_tol=u.len_text(GOAL_TOL))

    def captions(self, scene: Scene, kinds: tuple[str, ...]) -> str:
        """The "Attached images" block for the given kinds (empty without images)."""
        if not kinds:
            return ""
        fields = {"time_s": f"{scene.time_s:.1f}", "grid": load_prompt(f"grid_{self.units.name}"), "hist_s": HISTORY_S,
                  "goal_tol": self.units.len_text(GOAL_TOL), "unit": self.units.name}
        items = "\n".join(load_prompt(f"caption_{k}").format(n=n, **fields) for n, k in enumerate(kinds, 1))
        return load_prompt("images").format(items=items)

    def first(self, scene: Scene) -> Turn:
        """First user turn: task, images, state, output format. With the camera, scene is the trip start (t = 0)."""
        u, r = self.units, scene.site["robot_radius_m"]
        task = load_prompt(f"task_{self.framing}").format(robot_radius=u.len_text(r), person_radius=u.len_text(HUMAN_RADIUS), min_centre=u.len_text(r + HUMAN_RADIUS), unit=u.name)
        parts = [task, self.captions(scene, self.images), scene.text(self.units), self.output_text()]
        return Turn("\n\n".join(p for p in parts if p), list(self.images))

    @property
    def prefill(self) -> str:
        """The start of a reply, to continue from when the thinking is cut off."""
        return f'{{"{self.key}": [['

    def parse(self, text: str) -> np.ndarray | None:
        """The (k, 2) array under self.key of the JSON object in the reply, or None if the reply is not a valid plan."""
        try:
            plan = np.atleast_2d(np.array(json.loads(text[text.index("{"): text.rindex("}") + 1])[self.key], float))  # atleast_2d: the model sometimes drops the inner brackets of a single pair
        except (ValueError, KeyError, TypeError):  # no braces, bad JSON, missing key, or entries that are not numbers
            return None
        ok = plan.ndim == 2 and plan.shape[1] == 2 and 1 <= len(plan) <= self.max_steps and np.isfinite(plan).all()
        return plan if ok else None

    def followup(self, scene: Scene, executed_s: float) -> Turn:
        """Later turn of a conversation: the new state after executing the previous plan (the camera photo is not repeated)."""
        kinds = tuple(k for k in self.images if k == "bev")
        text = load_prompt("followup").format(executed_s=f"{executed_s:g}", images=self.captions(scene, kinds), scene=scene.text(self.units))
        return Turn(text.replace("\n\n\n", "\n\n"), list(kinds))


class PixelPrompt(PromptBuilder):
    """Points on the BEV image, in pixels (units must be scene.Pixels)."""

    template = "output_pixels"
    key = "points"


class VelocityPrompt(PromptBuilder):
    """One velocity for the next step."""

    template = "output_velocity"
    key = "velocities"
    example = [(0.20, 0.10)]  # m/s, below the cap

    def to_units(self, value) -> np.ndarray:
        return self.units.vec(value)


class VelocitySeqPrompt(VelocityPrompt):
    """A velocity per step."""

    template = "output_velocity_seq"
    example = [(0.20, 0.10), (0.20, 0.15), (0.15, 0.20)]
