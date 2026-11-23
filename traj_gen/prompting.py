"""Assemble the system prompt and user turns for one framing, output mode and set of images.

A PromptBuilder is made once per condition (framing x output mode x images) and then asked for
the system prompt and the first user turn of a Scene, or for the short follow-up turn of a
conversation. The output mode is a planners.QueryPlanner: it supplies the template, reply key,
example, units and how the example is written in those units. Everything else is shared.
The planned agent's kind picks the templates: a planned person reads the "_human" version of the
system prompt, output format, map captions and follow-up (TEMPLATE_SUFFIX), and its own framings.
The output mode also names its system and follow-up templates, whether the text gives the scored body sizes, and any extra
text after the state; the harness mode (planners.HarnessPlanner) uses these, and the harness framings go only with it.
Image kinds in a Turn are "bev" (render_bev of the turn's scene) and "cam" (camera_frame of the trip),
in the order they are described in the text.
"""

import json
from dataclasses import dataclass

import numpy as np

from traj_gen.agents import (ARRIVAL_RADIUS_M, PERSON_BODY_RADIUS_M, PERSON_PROMPT_RADIUS_M, PERSONAL_SPACE_RADIUS_M, ROOM_MARGIN_M, SPEED_CAP_M_S, body_radius,
                             comfort_distance, contact_distance, prompt_radius)
from traj_gen.bev import SIZE_PX
from traj_gen.scene import DT, PREDICT_S, PROMPT_HISTORY_S, Scene, load_prompt, person_name


# Prompt framing (task_<framing>.txt) -> the planned-agent kind it is written for. social_body is social plus the centre distance
# the bodies need. cautious is a stronger social prompt (waiting and detours are fine). participant: walk as a participant of the study would, given the robot's and the other person's sizes.
# harness (robot) and participant_harness (person): the scored contact and personal-space distances, and that waiting is allowed; only with the harness mode.
FRAMINGS = {"neutral": "robot", "social": "robot", "social_body": "robot", "cautious": "robot", "harness": "robot", "participant": "human", "participant_harness": "human"}
HARNESS_FRAMINGS = {"harness", "participant_harness"}
TEMPLATE_SUFFIX = {"robot": "", "human": "_human"}  # of the templates that differ by the planned agent's kind


@dataclass
class Turn:
    text: str
    images: list[str]


class PromptBuilder:
    """Prompts for one framing, set of images and output mode. mode is a QueryPlanner (template, key, example, units, to_units,
    and kind, the planned agent's kind, which framing must be written for)."""

    def __init__(self, framing: str, mode, images: tuple[str, ...], max_steps: int, predict: bool):
        self.framing, self.mode, self.images, self.max_steps, self.predict, self.kind = framing, mode, images, max_steps, predict, mode.kind
        if FRAMINGS[framing] != self.kind:
            raise ValueError(f"framing {framing!r} is written for a planned {FRAMINGS[framing]}, not a {self.kind}")
        if (framing in HARNESS_FRAMINGS) != mode.harness:
            raise ValueError(f"framing {framing!r} does not go with the {mode.name} output mode: the harness framings go only with the harness mode")
        self.suffix = TEMPLATE_SUFFIX[self.kind]

    def output_text(self) -> str:
        output = load_prompt(self.mode.template + self.suffix).format(unit=self.mode.units.name, dt=DT, dt2=2 * DT, max_steps=self.max_steps, **self.mode.output_fields())
        return output + "\n" + load_prompt("reply").format(key=self.mode.key, example=self.mode.example_text())

    def system(self, scene: Scene) -> str:
        u, site = self.mode.units, scene.site
        frame = load_prompt(f"frame_{u.name}").format(image_px=SIZE_PX, **u.bounds(site["room_m"]))
        fields = {"room_side": u.len_text(site["room_m"]), "unit": u.name, "frame": frame, "stations": scene.stations_text(u),
                  "speed_cap": u.len_text(SPEED_CAP_M_S[self.kind]), "dt": DT, "goal_tol": u.len_text(ARRIVAL_RADIUS_M[self.kind])}
        radius = (body_radius if self.mode.scored_sizes else prompt_radius)(scene.agent, site)
        template = load_prompt(self.mode.system_template + self.suffix)
        if self.kind == "robot":
            return template.format(robot_radius=u.len_text(radius), step=u.len_text(SPEED_CAP_M_S[self.kind] * DT), **fields)
        return template.format(name=person_name(scene.agent), person_radius=u.len_text(radius), margin=u.len_text(ROOM_MARGIN_M[self.kind]), **fields)

    def captions(self, scene: Scene, kinds: tuple[str, ...]) -> str:
        """The "Attached images" block for the given kinds (empty without images)."""
        if not kinds:
            return ""
        fields = {"time_s": f"{scene.time_s:.1f}", "grid": load_prompt(f"grid_{self.mode.units.name}"), "hist_s": PROMPT_HISTORY_S,
                  "goal_tol": self.mode.units.len_text(ARRIVAL_RADIUS_M[self.kind]), "unit": self.mode.units.name}
        items = []
        for n, k in enumerate(kinds, 1):
            caption = load_prompt(f"caption_{k}" + (self.suffix if k == "bev" else "")).format(n=n, **fields)
            if k == "bev" and self.predict:
                caption += " " + load_prompt("caption_bev_prediction" + self.suffix).format(h1=f"{PREDICT_S[0]:g}", h2=f"{PREDICT_S[1]:g}")
            items.append(caption)
        return load_prompt("images").format(items="\n".join(items))

    def first(self, scene: Scene) -> Turn:
        """First user turn: task, images, state, output format. With the camera, scene is the trip start (t = 0)."""
        u, r = self.mode.units, prompt_radius("robot", scene.site)
        site = scene.site
        task = load_prompt(f"task_{self.framing}").format(robot_radius=u.len_text(r), person_radius=u.len_text(PERSON_PROMPT_RADIUS_M), min_centre=u.len_text(r + PERSON_PROMPT_RADIUS_M), unit=u.name,
                                                       contact=u.len_text(contact_distance(site)), comfort=u.len_text(comfort_distance(site)),
                                                       contact_person=u.len_text(2 * PERSON_BODY_RADIUS_M), comfort_person=u.len_text(PERSON_BODY_RADIUS_M + PERSONAL_SPACE_RADIUS_M["human"]))
        parts = [task, self.captions(scene, self.images), self.state(scene, "\n\n"), self.output_text()]
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
        text = load_prompt(self.mode.followup_template + self.suffix).format(executed_s=f"{executed_s:g}", images=self.captions(scene, kinds), scene=self.state(scene, "\n"))
        return Turn(text.replace("\n\n\n", "\n\n"), list(kinds))

    def state(self, scene: Scene, separator: str) -> str:
        """The per-turn state block, then the output mode's extra text (future trajectories, the current heading), if any."""
        return separator.join(t for t in (scene.text(self.mode.units, self.predict, self.mode.scored_sizes), self.mode.extra_text(scene)) if t)
