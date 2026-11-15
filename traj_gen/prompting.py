"""Assemble the system prompt and user turns for one framing, output mode and set of images.

A PromptBuilder is made once per condition (framing x output mode x images) and then asked for
the system prompt and the first user turn of a Scene, or for the short follow-up turn of a
conversation. The output mode is a planners.QueryPlanner: it supplies the template, reply key,
example, units and how the example is written in those units. Everything else is shared.
Image kinds in a Turn are "bev" (render_bev of the turn's scene) and "cam" (camera_frame of the trip),
in the order they are described in the text.
"""

import json
from dataclasses import dataclass

import numpy as np

from traj_gen.agents import ARRIVAL_RADIUS_M, SPEED_CAP_M_S, agent_kind, prompt_radius
from traj_gen.bev import SIZE_PX
from traj_gen.scene import DT, PREDICT_S, PROMPT_HISTORY_S, Scene, load_prompt


FRAMINGS = ("neutral", "social", "social_body", "cautious")  # social_body: social + the centre distance the bodies need; cautious: social, stronger (waiting/detours are fine)


@dataclass
class Turn:
    text: str
    images: list[str]


class PromptBuilder:
    """Prompts for one framing, set of images and output mode. mode is a QueryPlanner (template, key, example, units, to_units)."""

    def __init__(self, framing: str, mode, images: tuple[str, ...], max_steps: int, predict: bool):
        self.framing, self.mode, self.images, self.max_steps, self.predict = framing, mode, images, max_steps, predict

    def example_text(self) -> str:
        return ", ".join("[" + ", ".join(self.mode.units.num(c) for c in self.mode.to_units(p)) + "]" for p in self.mode.example)

    def output_text(self) -> str:
        output = load_prompt(self.mode.template).format(unit=self.mode.units.name, dt=DT, dt2=2 * DT, max_steps=self.max_steps)
        return output + "\n" + load_prompt("reply").format(key=self.mode.key, example=self.example_text())

    def system(self, scene: Scene) -> str:
        u, site, kind = self.mode.units, scene.site, agent_kind(scene.agent)
        frame = load_prompt(f"frame_{u.name}").format(image_px=SIZE_PX, **u.bounds(site["room_m"]))
        return load_prompt("system").format(
            room_side=u.len_text(site["room_m"]), unit=u.name, frame=frame, stations=scene.stations_text(u),
            robot_radius=u.len_text(prompt_radius(scene.agent, site)), speed_cap=u.len_text(SPEED_CAP_M_S[kind]), dt=DT,
            goal_tol=u.len_text(ARRIVAL_RADIUS_M[kind]))

    def captions(self, scene: Scene, kinds: tuple[str, ...]) -> str:
        """The "Attached images" block for the given kinds (empty without images)."""
        if not kinds:
            return ""
        fields = {"time_s": f"{scene.time_s:.1f}", "grid": load_prompt(f"grid_{self.mode.units.name}"), "hist_s": PROMPT_HISTORY_S,
                  "goal_tol": self.mode.units.len_text(ARRIVAL_RADIUS_M[agent_kind(scene.agent)]), "unit": self.mode.units.name}
        items = []
        for n, k in enumerate(kinds, 1):
            caption = load_prompt(f"caption_{k}").format(n=n, **fields)
            if k == "bev" and self.predict:
                caption += " " + load_prompt("caption_bev_prediction").format(h1=f"{PREDICT_S[0]:g}", h2=f"{PREDICT_S[1]:g}")
            items.append(caption)
        return load_prompt("images").format(items="\n".join(items))

    def first(self, scene: Scene) -> Turn:
        """First user turn: task, images, state, output format. With the camera, scene is the trip start (t = 0)."""
        u, r, person_radius = self.mode.units, prompt_radius(scene.agent, scene.site), prompt_radius("human0", scene.site)
        task = load_prompt(f"task_{self.framing}").format(robot_radius=u.len_text(r), person_radius=u.len_text(person_radius), min_centre=u.len_text(r + person_radius), unit=u.name)
        parts = [task, self.captions(scene, self.images), scene.text(self.mode.units, self.predict), self.output_text()]
        return Turn("\n\n".join(p for p in parts if p), list(self.images))

    @property
    def prefill(self) -> str:
        """The start of a reply, to continue from when the thinking is cut off."""
        return f'{{"{self.mode.key}": [['

    def parse(self, text: str) -> np.ndarray | None:
        """The (k, 2) array under self.mode.key of the JSON object in the reply, or None if the reply is not a valid plan."""
        try:
            plan = np.atleast_2d(np.array(json.loads(text[text.index("{"): text.rindex("}") + 1])[self.mode.key], float))  # atleast_2d: the model sometimes drops the inner brackets of a single pair
        except (ValueError, KeyError, TypeError):  # no braces, bad JSON, missing key, or entries that are not numbers
            return None
        ok = plan.ndim == 2 and plan.shape[1] == 2 and 1 <= len(plan) <= self.max_steps and np.isfinite(plan).all()
        return plan if ok else None

    def followup(self, scene: Scene, executed_s: float) -> Turn:
        """Later turn of a conversation: the new state after executing the previous plan (the camera photo is not repeated)."""
        kinds = tuple(k for k in self.images if k == "bev")
        text = load_prompt("followup").format(executed_s=f"{executed_s:g}", images=self.captions(scene, kinds), scene=scene.text(self.mode.units, self.predict))
        return Turn(text.replace("\n\n\n", "\n\n"), list(kinds))
