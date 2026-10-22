"""Episodes: one planner on one trip, executed against the replayed people, and everything saved.

The robot track is aligned with the trip arrays: the real history up to start_index, then one
executed position per time step (so the people's replay at the same index is the people at that
time). An episode ends at the first executed step within GOAL_TOL of the goal, or at the end of
the human replay (the last index of the trip arrays).

- SingleShotEpisode: one query at the start; the whole (retimed) plan is executed.
- RollingEpisode: re-query every execute_s of executed motion, each time in a fresh chat.
- ConversationEpisode: the same in one growing chat.
- BaselineEpisode: single shot without a model (straight, replay), one folder per trip.

Status: ok, invalid (single shot: unusable reply), failed (rolling: unusable reply after one
retry), error (exception; traceback saved). The files written per episode are listed in PLAN.md
(Saved results); manifest.json comes last, so a folder with a manifest is a finished episode.
"""

import itertools
import json
import subprocess
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np

from bev import render_result
from bi3_metrics import bi3_metrics
from client import ANSWER_MAX_TOKENS, INSTRUCT_SAMPLING, MODEL, THINK_MAX_TOKENS, THINKING_EFFORT, THINKING_SAMPLING, Chat, Sample
from prompting import Turn
from scene import DT, Scene, load_prompt
from traj_gen_metrics import GOAL_TOL, other_stations, traj_gen_metrics

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ["traj_gen", "metrics", "preprocessing"]


@lru_cache(maxsize=None)
def code_version() -> dict:
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", *CODE_DIRS))}


class Episode:
    """Shared machinery; subclasses decide when to query and how much of a plan to execute."""

    uses_model = True

    def __init__(self, run: Path, exp, trip: dict, site: dict, framing: str | None, thinking: str | None, sample: int):
        self.exp, self.trip, self.site, self.framing, self.thinking, self.sample = exp, trip, site, framing, thinking, sample
        self.folder = self.folder_for(run, exp.id, trip["id"], framing, thinking, sample)
        self.planner = exp.planner(exp, trip, framing, thinking)
        self.start = trip["start_index"]
        self.goal = np.asarray(trip["robot_goal_xy"], float)
        self.track = np.asarray(trip["robot"], float)[: self.start + 1, :2]
        self.status, self.error = "ok", None
        self.plans, self.records, self.raw, self.n_queries = [], [], [], 0
        self.inputs: dict[str, bytes] = {}  # file name under inputs/ -> PNG bytes sent
        self.refs: dict[bytes, str] = {}    # PNG bytes -> "inputs/<name>", to point at images in the logged prompts

    @classmethod
    def folder_for(cls, run: Path, exp_id: str, trip_id: str, framing: str, thinking: str, sample: int) -> Path:
        return run / exp_id / framing / thinking / trip_id / f"s{sample}"

    @classmethod
    def conditions(cls, framings, thinkings, samples: int):
        return itertools.product(framings, thinkings, range(samples))

    def done(self) -> bool:
        """Finished earlier (an episode that ended in an error is run again)."""
        manifest = self.folder / "manifest.json"
        return manifest.exists() and json.loads(manifest.read_text())["status"] != "error"

    def scene(self) -> Scene:
        return Scene(self.trip, self.site, len(self.track) - 1, self.track)

    def open_turn(self, scene: Scene) -> tuple[Chat, Turn]:
        """The chat to ask and the user turn to add to it."""
        chat = self.planner.chat(scene)
        return chat, chat.client.builder.first(scene)

    def query(self, scene: Scene, retries: int) -> np.ndarray | None:
        """Ask the model for a plan (in the output units); an invalid reply is sampled again up to `retries` times."""
        chat, turn = self.open_turn(scene)
        images = self.planner.images(scene, turn.images)
        self.n_queries += 1
        for kind, png in images.items():
            name = "cam.png" if kind == "cam" else f"q{self.n_queries - 1}_{kind}.png"
            self.inputs[name], self.refs[png] = png, f"inputs/{name}"
        for attempt in range(retries + 1):
            sample = chat.ask(turn, images)[0] if attempt == 0 else chat.resample(1)[0]
            self.log(scene, chat, attempt, sample)
            if sample.value is not None:
                chat.accept(sample)
                break
        return sample.value

    def log(self, scene: Scene, chat: Chat, attempt: int, s: Sample) -> None:
        messages = [{"role": m["role"], "content": [p[1] if p[0] == "text" else {"image": self.refs[p[2]]} for p in m["parts"]]} for m in chat.messages]
        self.records.append({
            "query": self.n_queries - 1, "attempt": attempt, "time_s": scene.time_s, "messages": messages, "reply": s.content,
            "reasoning": s.reasoning, "finish": s.finish, "valid": s.value is not None, "forced": load_prompt("think_cutoff") in s.reasoning,
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens, "latency_s": s.latency_s})

    def plan(self, value: np.ndarray | None, scene: Scene, steps: int | None) -> np.ndarray:
        """Positions for a reply, with the plan recorded; steps = how many plan steps will be executed (None = all), for the raw speeds."""
        positions, raw = self.planner.positions(value, scene)
        self.plans.append({"query": self.n_queries - 1 if self.n_queries else None, "time_s": scene.time_s, "units": self.exp.units.name if self.exp.units else None,
                           "reply": None if value is None else value.tolist(), "positions_m": positions.tolist(),
                           "raw_speed": None if raw is None else raw.tolist()})
        if raw is not None:
            self.raw.append(raw[:steps])
        return positions

    def execute(self, positions: np.ndarray) -> bool:
        """Append the positions to the track, up to the goal or the end of the replay; True when the episode is over."""
        take = positions[: len(self.trip["time"]) - len(self.track)]
        hit = np.flatnonzero(np.linalg.norm(take - self.goal, axis=1) <= GOAL_TOL)
        if hit.size:
            take = take[: hit[0] + 1]
        self.track = np.vstack([self.track, take])
        return bool(hit.size) or len(self.track) == len(self.trip["time"])

    def run(self) -> None:
        raise NotImplementedError

    def go(self) -> None:
        """Run and save; an exception becomes status error instead of stopping the whole run."""
        t0 = time.perf_counter()
        try:
            self.run()
        except Exception:
            self.status, self.error = "error", traceback.format_exc()
        self.save(time.perf_counter() - t0)

    def metrics(self) -> dict | None:
        n = len(self.track) - self.start
        if n < 3:  # velocity and acceleration need a few samples (nothing executed after an invalid reply)
            return None
        sl = slice(self.start, self.start + n)
        humans = np.stack([np.asarray(h, float)[sl, :2] for h in self.trip["humans"]], axis=1)
        t, robot = np.asarray(self.trip["time"], float)[sl], self.track[self.start:]
        raw = np.concatenate(self.raw) if self.raw else None
        return {**bi3_metrics(t, robot, humans), **traj_gen_metrics(
            t, robot, humans, self.goal, self.site["robot_radius_m"], self.site["room_m"], other_stations(self.site, self.trip["robot_goal"]), raw)}

    def manifest(self, wall_s: float) -> dict:
        t = self.trip
        model = {"name": MODEL, "thinking": self.thinking == "think", "reasoning_effort": THINKING_EFFORT, "sampling": THINKING_SAMPLING if self.thinking == "think" else INSTRUCT_SAMPLING,
                 "max_tokens": THINK_MAX_TOKENS if self.thinking == "think" else ANSWER_MAX_TOKENS} if self.uses_model else None
        return {
            "trip": t["id"], "site": t["site"], "controller": t["controller"], "experiment": self.exp.id, "framing": self.framing,
            "thinking": self.thinking, "sample": self.sample, "images": list(self.exp.images), "output_mode": self.exp.prompt.key if self.exp.prompt else None,
            "horizon": type(self).__name__, "execute_s": self.exp.execute_s, "model": model, "status": self.status, "error": self.error,
            "start_index": self.start, "steps_executed": len(self.track) - self.start - 1, "metrics": self.metrics(),
            "queries": self.n_queries, "samples": len(self.records), "retries": sum(r["attempt"] > 0 for r in self.records),
            "invalid_replies": sum(not r["valid"] for r in self.records), "forced_answers": sum(r["forced"] for r in self.records),
            "prompt_tokens": sum(r["prompt_tokens"] for r in self.records), "completion_tokens": sum(r["completion_tokens"] for r in self.records),
            "query_latency_s": sum(r["latency_s"] for r in self.records), "wall_s": wall_s, "code": code_version()}

    def save(self, wall_s: float) -> None:
        f, t, a = self.folder, self.trip, self.start
        (f / "inputs").mkdir(parents=True, exist_ok=True)
        for name, png in self.inputs.items():
            (f / "inputs" / name).write_bytes(png)
        n = len(self.track) - a
        times = np.asarray(t["time"], float)[a: a + n]
        humans = [np.asarray(h, float)[a: a + n, :2] for h in t["humans"]]
        header = "t,x,y" + "".join(f",h{h + 1}_x,h{h + 1}_y" for h in range(len(humans)))
        np.savetxt(f / "trajectory.csv", np.column_stack([times, self.track[a:], *humans]), delimiter=",", header=header, comments="", fmt="%.4f")
        for name, rows in (("plans.jsonl", self.plans), ("queries.jsonl", self.records)):
            (f / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        title = f"{self.exp.id} {self.framing or ''} {self.thinking or ''} {t['id']} {self.status}"
        (f / "result.png").write_bytes(render_result(self.site, self.goal, self.track[a:], np.asarray(t["robot"], float)[a: t["end_index"] + 1, :2], humans, title))
        manifest = self.manifest(wall_s)
        manifest["files"] = sorted(str(p.relative_to(f)) for p in f.rglob("*") if p.is_file()) + ["manifest.json"]
        (f / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class SingleShotEpisode(Episode):
    def run(self) -> None:
        scene = self.scene()
        value = self.query(scene, retries=0)
        if value is None:
            self.status = "invalid"
            return
        self.execute(self.plan(value, scene, None))


class BaselineEpisode(SingleShotEpisode):
    """No model: the planner's positions are executed as they are; one folder per trip."""

    uses_model = False

    @classmethod
    def folder_for(cls, run, exp_id, trip_id, framing, thinking, sample) -> Path:
        return run / exp_id / trip_id

    @classmethod
    def conditions(cls, framings, thinkings, samples: int):
        return [(None, None, 0)]

    def run(self) -> None:
        self.execute(self.plan(None, self.scene(), None))


class RollingEpisode(Episode):
    def run(self) -> None:
        steps = round(self.exp.execute_s / DT)
        while True:
            scene = self.scene()
            value = self.query(scene, retries=1)
            if value is None:
                self.status = "failed"
                return
            positions = self.plan(value, scene, steps)[:steps]
            positions = np.vstack([positions, np.repeat(positions[-1:], steps - len(positions), axis=0)])  # a plan shorter than the window: hold the last position
            if self.execute(positions):
                return


class ConversationEpisode(RollingEpisode):
    """One chat for the whole episode: the first turn is the full prompt, later turns the follow-up (no camera photo)."""

    chat = None

    def open_turn(self, scene: Scene):
        if self.chat is None:
            self.chat, turn = super().open_turn(scene)
            return self.chat, turn
        return self.chat, self.chat.client.builder.followup(scene, self.exp.execute_s)
