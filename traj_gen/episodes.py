"""Episodes: one planner on one trip, executed against the replayed people, and everything saved.

The planned agent (trip["agent"]) has a track aligned with the trip arrays: its real history up to
start_index, then one executed position per time step (so the other agents' replay at the same index
is the others at that time). An episode ends at the first executed step within the agent's arrival
radius of the goal, or at the end of the replay (the last index of the trip arrays).

- SingleShotEpisode: one query at the start, and the whole (retimed) plan is executed.
- RollingEpisode: re-query every execute_s of executed motion, each time in a fresh chat.
- ConversationEpisode: the same in one growing chat.
- BaselineEpisode: single shot without a model (straight, replay), one folder per trip.

Status: ok, invalid_reply (an unusable reply: single shot at once, rolling after one retry),
error (server or transport error, with the traceback in the manifest). Each episode folder holds
inputs/ (the images sent), trajectory.csv, plans.jsonl, queries.jsonl, result.png and manifest.json.
manifest.json is written last, so a folder with a manifest is a finished episode.
"""

import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

from metrics.bi3_metrics import bi3_metrics
from metrics.traj_gen_metrics import move_end_index, traj_gen_metrics
from preprocessing.bi3_derived import provenance
from traj_gen.prompting import HARD_CAP_FRAMINGS, comfort_distance, contact_distance
from traj_gen.agents import ARRIVAL_RADIUS_M, PERSONAL_SPACE_RADIUS_M, ROOM_MARGIN_M, SPEED_CAP_M_S, agent_kind, body_radii, body_radius
from traj_gen.bev import render_result
from traj_gen.client import TRANSPORT_ERRORS, Chat, Sample, describe_model
from traj_gen.prompting import Turn
from traj_gen.scene import DT, Scene, load_prompt, person_name

# Layout of a run folder: the trip list, the configuration, then <experiment>/... episode folders, and the outputs of analyze and report.
TEST_TRIPS = "test_trips.json"
RUN_CONFIG = "run.json"
ANALYSIS_DIR = "analysis"
REPORT_DIR = "report"


def real_path(trip: dict) -> np.ndarray:
    """(n, 2) positions of the trip's planned agent, as recorded, from the trip's start to its end, in the room frame."""
    return np.asarray(trip["tracks"][trip["agent"]], float)[trip["start_index"]: trip["end_index"] + 1, :2]


def save_trajectory(folder: Path, times: np.ndarray, positions: dict[str, np.ndarray]) -> None:
    """Write <folder>/trajectory.csv from times (T,) and {agent id: (T, 2) positions}, columns t, <id>_x, <id>_y in the dict's order."""
    header = "t" + "".join(f",{agent}_x,{agent}_y" for agent in positions)
    np.savetxt(folder / "trajectory.csv", np.column_stack([times, *positions.values()]), delimiter=",", header=header, comments="", fmt="%.4f")


def load_trajectory(folder: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read <folder>/trajectory.csv: times (T,) and {agent id: (T, 2) positions} in column order."""
    path = folder / "trajectory.csv"
    columns = path.read_text().split("\n", 1)[0].split(",")[1:]
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data[:, 0], {columns[k].removesuffix("_x"): data[:, 1 + k: 3 + k] for k in range(0, len(columns), 2)}


HARD_CAP_RETRIES = 3  # a plan over the hard speed cap is sampled again up to this many times


class Episode:
    """Shared machinery. Subclasses decide when to query and how much of a plan to execute."""

    def __init__(self, run: Path, exp, trip: dict, site: dict, framing: str | None, thinking: str | None, sample: int):
        self.exp, self.trip, self.site, self.framing, self.thinking, self.sample = exp, trip, site, framing, thinking, sample
        self.folder = self.folder_for(run, exp.id, trip["id"], framing, thinking, sample)
        self.planner = exp.planner(exp, trip, framing, thinking)
        self.agent, self.start = trip["agent"], trip["start_index"]
        self.goal = np.asarray(trip["goal_xy"], float)
        self.arrival_radius = ARRIVAL_RADIUS_M[agent_kind(self.agent)]
        self.track = np.asarray(trip["tracks"][self.agent], float)[: self.start + 1, :2]
        self.others = [a for a in trip["tracks"] if a != self.agent]
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
        """Ask the model for a plan (in the output units). An invalid reply is sampled again up to `retries` times."""
        if self.framing in HARD_CAP_FRAMINGS:
            retries = max(retries, HARD_CAP_RETRIES)
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
        if sample.value is None or not getattr(self.exp, "check_rounds", 0):
            return sample.value
        return self.revise(scene, chat, sample.value, retries)

    def conflicts(self, value: np.ndarray, scene: Scene) -> tuple[tuple[int, int], str | None]:
        """(contact steps, personal-space steps) of a plan against the people's logged future, and the feedback text on its first conflict (None if none)."""
        positions, _ = self.planner.positions(value, scene)
        i, u = len(self.track), self.planner.units
        people = [o for o in self.others if agent_kind(o) == "human"]
        n = min(len(positions), len(self.trip["time"]) - i)
        if n <= 0:
            return (0, 0), None
        future = {o: np.asarray(self.trip["tracks"][o], float)[i: i + n, :2] for o in people}
        dist = np.stack([np.linalg.norm(positions[:n] - future[o], axis=1) for o in people], axis=1)  # (n, people)
        contact, comfort = contact_distance(self.site), comfort_distance(self.site)
        near = dist.min(axis=1)
        counts = (int((near < contact).sum()), int(((near >= contact) & (near < comfort)).sum()))
        bad = np.flatnonzero(near < comfort)
        if not bad.size:
            return counts, None
        k = bad[0]
        j = int(dist[k].argmin())
        text = load_prompt("plan_check").format(
            t=f"{DT * (k + 1):.1f}", robot=u.pos_text(positions[k]), dist=u.len_text(near[k]), person=person_name(people[j]), person_at=u.pos_text(future[people[j]][k]),
            kind="in contact" if near[k] < contact else "inside personal space", contact=u.len_text(contact), comfort=u.len_text(comfort),
            n_contact=counts[0], n_space=counts[1], unit=u.name)
        return counts, text

    def revise(self, scene: Scene, chat: Chat, value: np.ndarray, retries: int) -> np.ndarray:
        """Report the plan's first conflict with the logged future and ask for a revision, up to check_rounds times; the plan with the fewest contact
        steps (then personal-space steps) wins, the latest on ties."""
        best, best_counts = value, None
        for _ in range(self.exp.check_rounds + 1):
            counts, text = self.conflicts(value, scene)
            if best_counts is None or counts <= best_counts:
                best, best_counts = value, counts
            if text is None or _ == self.exp.check_rounds:
                break
            turn = Turn(text, [])
            for attempt in range(retries + 1):
                sample = chat.ask(turn, {})[0] if attempt == 0 else chat.resample(1)[0]
                self.log(scene, chat, attempt, sample)
                if sample.value is not None:
                    break
            if sample.value is None:  # keep the conversation well formed: the last valid plan stands
                chat.messages.pop()
                break
            chat.accept(sample)
            value = sample.value
        return best

    def log(self, scene: Scene, chat: Chat, attempt: int, s: Sample) -> None:
        messages = [{"role": m["role"], "content": [p[1] if p[0] == "text" else {"image": self.refs[p[2]]} for p in m["parts"]]} for m in chat.messages]
        self.records.append({
            "query": self.n_queries - 1, "attempt": attempt, "time_s": scene.time_s, "messages": messages, "reply": s.content,
            "reasoning": s.reasoning, "finish": s.finish, "valid": s.value is not None, "forced": load_prompt("think_cutoff") in s.reasoning,
            "prompt_tokens": s.prompt_tokens, "completion_tokens": s.completion_tokens, "latency_s": s.latency_s})

    def plan(self, value: np.ndarray | None, scene: Scene, steps: int | None) -> np.ndarray:
        """Positions for a reply, with the plan recorded. steps is how many plan steps will be executed (None for all), for the raw speeds."""
        positions, raw = self.planner.positions(value, scene)
        self.plans.append({"query": self.n_queries - 1 if self.n_queries else None, "time_s": scene.time_s, "units": self.planner.units.name if self.planner.units else None,
                           "reply": None if value is None else value.tolist(), "positions_m": positions.tolist(),
                           "raw_speed": None if raw is None else raw.tolist()})
        if raw is not None:
            self.raw.append(raw[:steps])
        return positions

    def execute(self, positions: np.ndarray) -> bool:
        """Append the positions to the track, up to the goal or the end of the replay. Returns True when the episode is over."""
        take = positions[: len(self.trip["time"]) - len(self.track)]
        hit = np.flatnonzero(np.linalg.norm(take - self.goal, axis=1) <= self.arrival_radius)
        if hit.size:
            take = take[: hit[0] + 1]
        self.track = np.vstack([self.track, take])
        return bool(hit.size) or len(self.track) == len(self.trip["time"])

    def run(self) -> None:
        raise NotImplementedError

    def go(self) -> None:
        """Run and save. A server or transport error becomes status error instead of stopping the whole run."""
        t0 = time.perf_counter()
        try:
            self.run()
        except TRANSPORT_ERRORS:
            self.status, self.error = "error", traceback.format_exc()
        self.save(time.perf_counter() - t0)

    def metrics(self) -> dict | None:
        n = len(self.track) - self.start
        if n < 3:  # velocity and acceleration need a few samples (nothing executed after an invalid reply)
            return None
        sl = slice(self.start, self.start + n)
        others = np.stack([np.asarray(self.trip["tracks"][o], float)[sl, :2] for o in self.others], axis=1)
        t, agent = np.asarray(self.trip["time"], float)[sl], self.track[self.start:]
        raw = np.concatenate(self.raw) if self.raw else None
        kind = agent_kind(self.agent)
        with_space = [k for k, o in enumerate(self.others) if agent_kind(o) in PERSONAL_SPACE_RADIUS_M]
        return {**bi3_metrics(t, agent, others), **traj_gen_metrics(
            t, agent, others, self.goal, body_radius(self.agent, self.site), body_radii(self.others, self.site),
            others[:, with_space], [PERSONAL_SPACE_RADIUS_M[agent_kind(self.others[k])] for k in with_space], self.arrival_radius,
            SPEED_CAP_M_S[kind], self.site["room_m"] / 2 + ROOM_MARGIN_M[kind], raw)}

    def manifest(self, wall_s: float) -> dict:
        t = self.trip
        model = describe_model(self.thinking == "think") if self.exp.uses_model else None
        return {
            "trip": t["id"], "site": t["site"], "controller": t["controller"], "agent": self.agent, "experiment": self.exp.id, "framing": self.framing,
            "thinking": self.thinking, "sample": self.sample, "images": list(self.exp.images), "output_mode": self.planner.name,
            "horizon": type(self).__name__, "execute_s": self.exp.execute_s, "model": model, "status": self.status, "error": self.error,
            "start_index": self.start, "steps_executed": len(self.track) - self.start - 1, "metrics": self.metrics(),
            "move_end_index": move_end_index(self.track[self.start:], self.goal, self.arrival_radius, self.site, t["goal"]),  # from start_index
            "queries": self.n_queries, "samples": len(self.records), "retries": sum(r["attempt"] > 0 for r in self.records),
            "invalid_replies": sum(not r["valid"] for r in self.records), "forced_answers": sum(r["forced"] for r in self.records),
            "prompt_tokens": sum(r["prompt_tokens"] for r in self.records), "completion_tokens": sum(r["completion_tokens"] for r in self.records),
            "query_latency_s": sum(r["latency_s"] for r in self.records), "wall_s": wall_s, "code": provenance()}

    def save(self, wall_s: float) -> None:
        f, t, a = self.folder, self.trip, self.start
        (f / "inputs").mkdir(parents=True, exist_ok=True)
        for name, png in self.inputs.items():
            (f / "inputs" / name).write_bytes(png)
        n = len(self.track) - a
        times = np.asarray(t["time"], float)[a: a + n]
        others = {o: np.asarray(t["tracks"][o], float)[a: a + n, :2] for o in self.others}
        save_trajectory(f, times, {agent: self.track[a:] if agent == self.agent else others[agent] for agent in t["tracks"]})
        for name, rows in (("plans.jsonl", self.plans), ("queries.jsonl", self.records)):
            (f / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        title = f"{self.exp.id} {self.framing or ''} {self.thinking or ''} {t['id']} {self.status}"
        (f / "result.png").write_bytes(render_result(self.site, self.agent, self.goal, self.track[a:], real_path(t), others, title))
        manifest = self.manifest(wall_s)
        manifest["files"] = sorted(str(p.relative_to(f)) for p in f.rglob("*") if p.is_file()) + ["manifest.json"]
        (f / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


class SingleShotEpisode(Episode):
    def run(self) -> None:
        scene = self.scene()
        value = self.query(scene, retries=0)
        if value is None:
            self.status = "invalid_reply"
            return
        self.execute(self.plan(value, scene, None))


class BaselineEpisode(SingleShotEpisode):
    """No model: the planner's positions are executed as they are, in one folder per trip."""

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
                self.status = "invalid_reply"
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
