"""Harnesses: everything that sets the model's prompts and loop, except the model, backend, effort, thinking, samples, concurrency and trips.

A harness is eight settings:
  episode     single_shot (one query at the trip start), rolling (re-plan every execute_s, each time in a fresh chat) or rolling_conv
              (the same in one growing conversation, harness.episodes.ConversationEpisode)
  output      the output mode: positions (one point per step in the world frame, outputs.positions.PositionPlanner), steer (a turn and a
              distance per step from the current heading, with the harness prompts, outputs.steer.SteerPlanner), steer_vel (a turn rate and
              a speed held for each step, a circular arc, with steer's prompts otherwise, outputs.steer.SteerVelocityPlanner), vel1 (one
              velocity step) or velseq (a velocity sequence). vel1 and velseq (robot only) are deprecated: kept for the runs that used them, not for new runs
  units       px (pixels of the map: u right, v down) or m (metres of the room frame: x right, y up), for every number the model reads or
              writes (outputs.planner.UNITS). px needs a map (images bev or bev_cam)
  images      text (no image), bev (the top-down map) or bev_cam (the map and the camera view)
  framing     the task prompt, one of observation.prompting.FRAMINGS. Its kind (robot or human) is the planned agent's
  prediction  none, true_future (the other agents' recorded future positions in the text each turn) or constvel_future (each person's
              constant-velocity position at t + 1 s and t + 2 s, in the text and on the map). constvel_future is deprecated: it is not
              confirmed to be complete
  check       plan-check revision rounds, 0 = off (each plan is tested against the recorded futures, the first conflict is reported back).
              Only with prediction=true_future
  execute_s   executed motion between queries, rolling episodes only (DEFAULT_EXECUTE_S: 2.0 s, 0.4 s for vel1). None for single shot

Pairing rules: the harness framings (observation.prompting.HARNESS_FRAMINGS) go only with steer and steer_vel, and those only with them; vel1 and velseq have
no person templates, so they go only with a robot framing; px units need a map.

The experiment id (the run folder level above the framing) is derived from the settings (Harness.id), in this order:
  ss / rh (single_shot / rolling and rolling_conv), output, units, images, conv (rolling_conv), true_future / constvel_future, then check
  (2 rounds) or check<N> (another count), then replan<execute_s>s when execute_s is not the output's default (for example
rh_positions_px_bev_conv, rh_steer_px_bev_conv_true_future_replan0.8s).
Framing is not in the id: it is the next folder level, <run>/<id>/<framing>/. Each harness writes its settings to
<run>/<id>/<framing>/harness.json (HARNESS_FILE) when it first runs there.

PRESETS are named harnesses for either planned agent: each gives the framing per planned-agent kind. The run command takes preset names,
picks the framing of the planned agent (a person, or the robot with --robot) and applies --overrides to each of them.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from traj_gen.harness.episodes import ConversationEpisode, RollingEpisode, SingleShotEpisode
from traj_gen.harness.experiments import EXPERIMENTS, LEGACY_IDS, Experiment
from traj_gen.observation.prompting import FRAMINGS, HARNESS_FRAMINGS
from traj_gen.observation.scene import DT
from traj_gen.outputs.planner import UNITS
from traj_gen.outputs.positions import PositionPlanner
from traj_gen.outputs.steer import SteerPlanner, SteerVelocityPlanner
from traj_gen.outputs.velocity import StepVelocityPlanner, VelocityPlanner

HARNESS_FILE = "harness.json"
EPISODES = {"single_shot": SingleShotEpisode, "rolling": RollingEpisode, "rolling_conv": ConversationEpisode}
OUTPUTS = {"positions": PositionPlanner, "steer": SteerPlanner, "steer_vel": SteerVelocityPlanner, "vel1": StepVelocityPlanner, "velseq": VelocityPlanner}
IMAGES = {"text": (), "bev": ("bev",), "bev_cam": ("bev", "cam")}
PREDICTIONS = ("none", "true_future", "constvel_future")  # constvel_future is deprecated
DEFAULT_EXECUTE_S = {"positions": 2.0, "steer": 2.0, "steer_vel": 2.0, "vel1": 0.4, "velseq": 2.0}  # 2.0 s: 5 steps, about 7 queries per trip; vel1 re-plans every step
ROBOT_ONLY_OUTPUTS = ("vel1", "velseq")  # deprecated. No person output templates: only prompts/output/robot/ has them


@dataclass(frozen=True)
class Harness:
    episode: str
    output: str
    units: str
    images: str
    framing: str
    prediction: str = "none"
    check: int = 0
    execute_s: float | None = None  # None: the output's default for a rolling episode

    def __post_init__(self):
        for name, value, allowed in (("episode", self.episode, EPISODES), ("output", self.output, OUTPUTS), ("units", self.units, UNITS), ("images", self.images, IMAGES),
                                     ("framing", self.framing, FRAMINGS), ("prediction", self.prediction, PREDICTIONS)):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {', '.join(allowed)}")
        if (self.framing in HARNESS_FRAMINGS) != OUTPUTS[self.output].harness:
            with_harness = ", ".join(k for k, v in OUTPUTS.items() if v.harness)
            raise ValueError(f"framing {self.framing!r} does not go with output {self.output!r}: the harness framings ({', '.join(sorted(HARNESS_FRAMINGS))}) go only with {with_harness}, and those only with them")
        if self.output in ROBOT_ONLY_OUTPUTS and self.kind != "robot":
            raise ValueError(f"output {self.output!r} has no person templates, so it does not go with the person framing {self.framing!r}")
        if self.units == "px" and "bev" not in IMAGES[self.images]:
            raise ValueError(f"units px are pixels of the map, so they need a map: images={self.images!r} has none")
        if isinstance(self.check, bool) or not isinstance(self.check, int) or self.check < 0:
            raise ValueError(f"check must be a whole number of revision rounds, 0 or more, got {self.check!r}")
        if self.check and self.prediction != "true_future":
            raise ValueError(f"check={self.check} needs prediction=true_future: the plan check tests each plan against the recorded futures")
        if self.episode == "single_shot":
            if self.execute_s is not None:
                raise ValueError(f"execute_s is only for rolling episodes, got {self.execute_s} with episode=single_shot")
            return
        execute_s = DEFAULT_EXECUTE_S[self.output] if self.execute_s is None else float(self.execute_s)
        steps = round(execute_s / DT)
        if steps < 1 or abs(steps * DT - execute_s) >= 1e-9:
            raise ValueError(f"execute_s={execute_s:g} is not a positive whole number of {DT} s steps")
        object.__setattr__(self, "execute_s", execute_s)

    @property
    def kind(self) -> str:
        """The planned agent's kind (robot or human), from the framing."""
        return FRAMINGS[self.framing]

    @property
    def id(self) -> str:
        return self._id(self.prediction)

    def _id(self, prediction: str) -> str:
        parts = ["ss" if self.episode == "single_shot" else "rh", self.output, self.units, self.images]
        if self.episode == "rolling_conv":
            parts.append("conv")
        if prediction != "none":
            parts.append(prediction)
        if self.check:
            parts.append("check" if self.check == 2 else f"check{self.check}")
        if self.execute_s is not None and self.execute_s != DEFAULT_EXECUTE_S[self.output]:
            parts.append(f"replan{self.execute_s:g}s")
        return "_".join(parts)

    def experiment(self) -> Experiment:
        """The Experiment the episodes and planners run (the framing is passed to them separately)."""
        return Experiment(self.id, OUTPUTS[self.output], self.units, IMAGES[self.images], EPISODES[self.episode], self.execute_s,
                          base=self._id("none") if self.prediction == "constvel_future" else None,
                          true_future=self.prediction == "true_future", check_rounds=self.check)

    def settings(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Preset:
    """A named harness for either planned agent: the settings other than the framing, and the framing per planned-agent kind."""
    episode: str
    output: str
    units: str
    images: str
    framings: tuple[tuple[str, str], ...]  # (kind, framing) pairs

    def harness(self, kind: str, overrides: dict | None = None) -> Harness:
        """The harness for a planned agent of this kind, with the overrides applied (a framing override replaces the kind's framing)."""
        framing = dict(self.framings)[kind]
        return Harness(**{"episode": self.episode, "output": self.output, "units": self.units, "images": self.images, "framing": framing,
                          **(overrides or {})})


# The true_future and plan-check levels are overrides of these: --overrides prediction=true_future, and prediction=true_future check=2.
PRESETS = {
    # The simple setup: map positions in one conversation. Person: the person pilot, run 20261115_pilot_52d7b27 (rh_px_bev_conv/participant).
    # Robot: the robot report's simple harness, run 20261116_robot_px_f379f0c (rh_px_bev_conv/neutral).
    "pilot": Preset("rolling_conv", "positions", "px", "bev", (("human", "participant"), ("robot", "neutral"))),
    # The harness: steer steps with the scored distances. Person: run 20261123_human_harness_rate_claude_sonnet_b2aaeb7 (rh_heading_rate_bev_conv/participant_space before the port).
    # Robot: rh_steer_bev_conv/robot_harness, whose robot test runs (2026-11-16, 2026-11-17) predate the port.
    "harness_v1": Preset("rolling_conv", "steer", "px", "bev", (("human", "harness"), ("robot", "robot_harness"))),
}
DEFAULT_PRESET = "pilot"


def parse_overrides(items: list[str]) -> dict:
    """KEY=VALUE strings as harness settings, with check as an integer and execute_s as a number. Raises ValueError on an unknown key, a
    repeated key or a value of the wrong type (the allowed values are checked when the harness is made)."""
    names = [f.name for f in fields(Harness)]
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise ValueError(f"override {item!r} is not KEY=VALUE")
        if key not in names:
            raise ValueError(f"override key {key!r} is not a harness setting: {', '.join(names)}")
        if key in out:
            raise ValueError(f"override key {key!r} is given twice")
        try:
            out[key] = int(value) if key == "check" else float(value) if key == "execute_s" else value
        except ValueError:
            raise ValueError(f"override {key}={value!r}: {key} takes {'a whole number' if key == 'check' else 'a number of seconds'}") from None
    return out


def settings_of(exp: Experiment, framing: str) -> dict:
    """The harness settings of a registered model experiment and a framing (the inverse of Harness.experiment)."""
    def key(table: dict, value) -> str:
        return next(k for k, v in table.items() if v == value)
    prediction = "true_future" if exp.true_future else "constvel_future" if exp.predict else "none"
    return {"episode": key(EPISODES, exp.episode), "output": key(OUTPUTS, exp.planner), "units": exp.units, "images": key(IMAGES, exp.images), "framing": framing,
            "prediction": prediction, "check": exp.check_rounds, "execute_s": exp.execute_s}


def write_harness_file(run: Path, name: str, harness: Harness, overrides: dict, check_only: bool = False) -> None:
    """Write <run>/<id>/<framing>/harness.json, or check it if it exists: raises ValueError if the stored settings differ from the harness's.
    With check_only, nothing is written."""
    path = run / harness.id / harness.framing / HARNESS_FILE
    if path.exists():
        stored = json.loads(path.read_text())["settings"]
        if stored != harness.settings():
            changed = {k: (stored.get(k), v) for k, v in harness.settings().items() if stored.get(k) != v}
            raise ValueError(f"{path} holds other settings than harness {name} (stored, now: {changed}): start a new run folder")
        return
    if check_only:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"id": harness.id, "preset": name, "overrides": overrides, "settings": harness.settings()}, indent=2) + "\n")


def run_experiments(run: Path) -> dict[str, Experiment]:
    """Experiment folder name -> Experiment, for every experiment folder of a run: from <id>/<framing>/harness.json where one exists,
    otherwise from the registry (harness.experiments.EXPERIMENTS, with the old folder names of harness.experiments.LEGACY_IDS)."""
    out = {folder: exp for folder, exp in EXPERIMENTS.items() if (run / folder).is_dir()}
    out |= {old: EXPERIMENTS[new] for old, new in LEGACY_IDS.items() if (run / old).is_dir()}
    for path in sorted(run.glob(f"*/*/{HARNESS_FILE}")):
        stored = json.loads(path.read_text())
        exp = Harness(**stored["settings"]).experiment()
        folder, framing = path.parent.parent.name, path.parent.name
        if exp.id != folder or stored["settings"]["framing"] != framing:
            raise ValueError(f"{path}: its settings give the folder {exp.id}/{stored['settings']['framing']}, not {folder}/{framing}")
        out[folder] = exp
    return out
