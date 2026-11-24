"""Harnesses: everything that sets the model's prompts and loop, except the model, backend, effort, thinking, samples, concurrency and trips.

A harness is eight settings:
  episode     single_shot (one query at the trip start), rolling (re-plan every execute_s, each time in a fresh chat) or rolling_conv
              (the same in one growing conversation, harness.episodes.ConversationEpisode)
  output      the output mode: positions (one point per step in the world frame, outputs.positions.PositionPlanner), steer (a turn and a
              distance per step from the current heading, with the steer prompts, outputs.steer.SteerPlanner), steer_vel (a turn rate and
              a speed held for each step, a circular arc, with steer's prompts otherwise, outputs.steer.SteerVelocityPlanner), vel1 (one
              velocity step) or velseq (a velocity sequence). vel1 and velseq (robot only) are deprecated: kept for the runs that used them, not for new runs
  units       px (pixels of the map: u right, v down) or m (metres of the room frame: x right, y up), for every number the model reads or
              writes (outputs.planner.UNITS). px needs a map (images bev or bev_cam)
  images      text (no image), bev (the top-down map) or bev_cam (the map and the camera view)
  framing     the task prompt, one of observation.prompting.FRAMINGS. Its kind (robot or human) is the planned agent's
  prediction  none, true_future (the other agents' recorded future positions in the text each turn, outputs.planner.TRUE_FUTURE_S ahead) or constvel_future (each person's
              constant-velocity position at t + 1 s and t + 2 s, in the text and on the map). constvel_future is deprecated: it is not
              confirmed to be complete
  check       plan-check revision rounds, 0 = off (each plan is tested against the recorded futures, the first conflict is reported back).
              Only with prediction=true_future
  execute_s   executed motion between queries, rolling episodes only (DEFAULT_EXECUTE_S: 0.8 s, 0.4 s for vel1, 2.0 s for velseq). None for single shot

Pairing rules: the scored-distance framings (observation.prompting.SCORED_DISTANCE_FRAMINGS) go only with steer and steer_vel, and those only with them; vel1 and velseq have
no person templates, so they go only with a robot framing; px units need a map.

The experiment id (the run folder level above the framing) is derived from the settings (Harness.id), in this order:
  ss / rh (single_shot / rolling and rolling_conv), output, units, images, conv (rolling_conv), true_future / constvel_future, then check
  (2 rounds) or check<N> (another count), then replan<execute_s>s when execute_s is not the output's default (for example
rh_positions_px_bev_conv, rh_steer_px_bev_conv_true_future_replan1.2s).
Framing is not in the id: it is the next folder level, <run>/<id>/<framing>/. Each harness writes its settings to
<run>/<id>/<framing>/harness.json (HARNESS_FILE) when it first runs there.

PRESETS are named harnesses for either planned agent: each gives the framing per planned-agent kind. The run command takes preset names,
picks the framing of the planned agent (a person, or the robot with --robot) and applies --overrides to each of them.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from traj_gen.harness.episodes import ConversationEpisode, RollingEpisode, SingleShotEpisode
from traj_gen.harness.experiments import BASELINES, Experiment
from traj_gen.observation.prompting import FRAMINGS, SCORED_DISTANCE_FRAMINGS
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
DEFAULT_EXECUTE_S = {"positions": 0.8, "steer": 0.8, "steer_vel": 0.8, "vel1": 0.4, "velseq": 2.0}  # 0.8 s: 2 steps, about 5 decisions in a mean person traversal; vel1 re-plans every step
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
        if (self.framing in SCORED_DISTANCE_FRAMINGS) != OUTPUTS[self.output].scored_distance_framings:
            steer_outputs = ", ".join(k for k, v in OUTPUTS.items() if v.scored_distance_framings)
            raise ValueError(f"framing {self.framing!r} does not go with output {self.output!r}: the scored-distance framings ({', '.join(sorted(SCORED_DISTANCE_FRAMINGS))}) go only with {steer_outputs}, and those only with them")
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
    # The simple setup: map positions in one conversation, with the participant task for a person and the neutral task for the robot.
    "pilot_v2": Preset("rolling_conv", "positions", "px", "bev", (("human", "participant"), ("robot", "neutral"))),
    # Steer steps, with the scored distances in the task.
    "harness_v2": Preset("rolling_conv", "steer", "px", "bev", (("human", "scored_distances"), ("robot", "robot_scored_distances"))),
}
DEFAULT_PRESET = "pilot_v2"


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
    """Experiment folder name -> Experiment, for every experiment folder of a run: from <id>/<framing>/harness.json, and the baseline
    folders (harness.experiments.BASELINES). Raises ValueError when a harness.json's settings give another folder than the one it is in."""
    out = {name: exp for name, exp in BASELINES.items() if (run / name).is_dir()}
    for path in sorted(run.glob(f"*/*/{HARNESS_FILE}")):
        stored = json.loads(path.read_text())
        exp = Harness(**stored["settings"]).experiment()
        folder, framing = path.parent.parent.name, path.parent.name
        if exp.id != folder or stored["settings"]["framing"] != framing:
            raise ValueError(f"{path}: its settings give the folder {exp.id}/{stored['settings']['framing']}, not {folder}/{framing}")
        out[folder] = exp
    return out
