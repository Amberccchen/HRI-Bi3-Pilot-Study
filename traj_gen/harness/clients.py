"""Chat clients for the generating model: messages from prompt turns and images, samples, parsed plans.

Two backends answer the same chats: Qwen through the vLLM OpenAI API (the default) and Claude through the
Anthropic Messages API. select_backend() picks one for the whole process. Each client turns a chat's
messages into its API's request and a reply into a Sample, so nothing above this file knows the backend.

A Chat holds one conversation (system prompt, then user turns and the model's replies). A single
query is a Chat with one ask(). A rolling conversation keeps asking and accept()s the reply it
wants to continue from, with its reasoning, so the model reads its earlier thinking in later turns. Per sample the client returns the
parsed plan (None when the reply is not a valid plan), the raw reply, the reasoning text, the parts the reply adds to the conversation,
the finish reason, token counts and latency.

What the reply must look like (JSON key, example, validity checks, the start of a reply) comes from the PromptBuilder of the
output mode, so the client has no per-mode branches.
"""

import base64
import http.client
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

try:
    import anthropic
except ImportError:  # only the Claude backend needs it
    anthropic = None

from traj_gen.observation import load_prompt
from traj_gen.observation.prompting import PromptBuilder, Turn


BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen3.8-27B-FP8"
MAX_IMAGES = 10  # the server's --limit-mm-per-prompt (hri-sim scripts/serve.sh)
THINKING_EFFORT = "medium"  # the template's default "xhigh" does not finish within 24000 tokens (8 of 8 runs), and "low" is not shorter than medium
SERVER_MAX_LEN = 131072  # hri-sim scripts/serve.sh MAX_LEN, which must be started at this value. A forced answer must fit prompt + thinking + answer into it
ANSWER_MAX_TOKENS = 1280  # longest allowed answer is 75 pairs (outputs.planner.MAX_PLAN_STEPS) at about 15 tokens per pair = 1125. Also the cap for a forced answer
TIMEOUT_S = 3600  # a thinking reply of the full budget takes minutes on a busy server
# Sampling from the model card. Thinking mode and instruct (non-thinking) mode differ.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


# Claude. No sampling parameters are sent: Sonnet 5.5 and Opus 5.5 reject non-default values and every model rejects them from 2026-10-21.
CLAUDE_MODEL = "claude-sonnet-5-5"
CLAUDE_THINKING_EFFORT = "high"  # default effort with thinking on, chosen by the user for the first Sonnet tests (the Qwen runs use medium). --effort overrides it
CLAUDE_THINK_MAX_TOKENS = 32000  # thinking and answer together (streamed, so no HTTP timeout). Medium effort mostly needs a few thousand
CLAUDE_ANSWER_MAX_TOKENS = 4096  # without thinking, the answer only (at most 75 pairs, about 1100 tokens)
CLAUDE_MAX_IMAGES = 100          # the API's limit per request, so no map is ever replaced by a note
# Errors that end one episode (status error) and are tried again on resume. Other API errors (bad request, authentication, not found) are setup mistakes and stop the run.
TRANSPORT_ERRORS = (OSError, http.client.HTTPException) + ((anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError) if anthropic else ())


@dataclass
class Sample:
    value: np.ndarray | None  # the plan as an (k, 2) array in the output units, None for an invalid reply
    content: str              # the reply text the plan was read from
    reasoning: str            # the model's thinking ("" without thinking)
    reply_parts: list         # the reply as accept() adds it to the conversation: ("reasoning", str) and ("text", str) for Qwen, ("claude_blocks", content) for Claude
    finish: str               # vLLM finish reason of the (last) call: "stop" or "length"
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


class QwenClient:
    """Settings shared by all chats of one condition: output mode (via the builder) and thinking on/off. Sends chats to the vLLM server."""

    max_images = MAX_IMAGES

    def __init__(self, builder: PromptBuilder, thinking: bool):
        self.builder, self.thinking = builder, thinking
        self.sampling = THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING

    def chat(self, system: str) -> "Chat":
        return Chat(self, system)

    @staticmethod
    def describe(thinking: bool, think_max_tokens: int) -> dict:
        """What the manifest records about the model of a condition."""
        return {"backend": "qwen", "name": MODEL, "thinking": thinking, "reasoning_effort": THINKING_EFFORT,
                "sampling": THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING, "max_tokens": think_max_tokens if thinking else ANSWER_MAX_TOKENS}

    @staticmethod
    def _api_messages(messages: list[dict]) -> list[dict]:
        """The chat's messages in the OpenAI format, with an earlier reply's reasoning in the assistant message's reasoning field (vLLM hands
        it to the chat template as reasoning_content)."""
        def part(p):
            if p[0] == "text":
                return {"type": "text", "text": p[1]}
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p[2]).decode()}}
        out = []
        for m in messages:
            message = {"role": m["role"], "content": [part(p) for p in m["parts"] if p[0] != "reasoning"]}
            reasoning = [p[1] for p in m["parts"] if p[0] == "reasoning"]
            out.append(message | {"reasoning": reasoning[0]} if reasoning else message)
        return out

    def _call(self, messages: list[dict], extra: list[dict] | None = None, **kwargs) -> tuple[dict, float]:
        """One chat completion for the conversation so far (+ extra trailing messages): the choice with its usage, and the latency."""
        t0 = time.perf_counter()
        payload = {"model": MODEL, "messages": self._api_messages(messages) + (extra or []),
                   "chat_template_kwargs": {"enable_thinking": self.thinking, "preserve_thinking": True,  # the template keeps the think block of every earlier reply
                                    **({"reasoning_effort": THINKING_EFFORT} if self.thinking else {})},
                   **self.sampling, **kwargs}
        r = post("/v1/chat/completions", payload)
        return {**r["choices"][0], "usage": r["usage"]}, time.perf_counter() - t0

    def sample(self, messages: list[dict]) -> Sample:
        b = self.builder
        choice, latency = self._call(messages, max_tokens=b.mode.think_max_tokens if self.thinking else ANSWER_MAX_TOKENS)
        message, usage = choice["message"], choice["usage"]
        content, reasoning = message.get("content") or "", message.get("reasoning") or ""
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        if self.thinking and choice["finish_reason"] == "length" and b.parse(content) is None:
            # The budget ran out in the thinking (or in the answer after it): close the thinking and force the answer,
            # continuing from the reasoning so far with the start of the JSON already written.
            room = SERVER_MAX_LEN - usage["prompt_tokens"] - ANSWER_MAX_TOKENS - 512  # 512: the cutoff note, the prefill and template tokens
            if usage["completion_tokens"] > room:  # late in a long conversation the full thinking leaves no room for the answer: keep its start
                reasoning = reasoning[: int(len(reasoning) * max(room, 0) / usage["completion_tokens"])]
            reasoning += "\n\n" + load_prompt("backend", "think_cutoff")
            choice, dt = self._call(messages, [{"role": "assistant", "reasoning": reasoning, "content": b.prefill}], max_tokens=ANSWER_MAX_TOKENS,
                                    continue_final_message=True, add_generation_prompt=False)
            message, usage = choice["message"], choice["usage"]
            # The qwen3 reasoning parser finds no </think> in this continuation and files it all under "reasoning".
            content = b.prefill + (message.get("content") or message.get("reasoning") or "")
            latency += dt
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
        return Sample(b.parse(content), content, reasoning, [("reasoning", reasoning), ("text", content)], choice["finish_reason"], prompt_tokens,
                      completion_tokens, latency)


class ClaudeClient:
    """The same for Claude through the Messages API (credentials from ANTHROPIC_API_KEY).

    Thinking on is adaptive thinking at the selected effort (default CLAUDE_THINKING_EFFORT), with the thinking summary kept as the reasoning text. Thinking off is the model's
    own way of switching it off. A reply cut off by the token limit is an invalid reply: there is no forced
    answer, since Claude takes no assistant prefill. The model is the one given to select_backend().
    An accepted reply enters the conversation as its whole content, thinking blocks included and unchanged, so that later turns read the
    earlier thinking. The API reads a thinking block only while every message before it is unchanged, which holds as the conversation is
    only ever appended to (no map is replaced below CLAUDE_MAX_IMAGES).
    """

    max_images = CLAUDE_MAX_IMAGES
    model = CLAUDE_MODEL
    effort = CLAUDE_THINKING_EFFORT  # with thinking on
    api = None  # one anthropic.Anthropic for all threads, made on first use

    def __init__(self, builder: PromptBuilder, thinking: bool):
        self.builder, self.thinking = builder, thinking

    def chat(self, system: str) -> "Chat":
        return Chat(self, system + "\n\n" + load_prompt("backend", "claude_answer_only"))  # Sonnet without thinking otherwise reasons aloud in the reply, in several JSON objects

    @classmethod
    def request_options(cls, thinking: bool) -> dict:
        """The thinking and effort fields of a request. Models that cannot switch thinking off (Opus 5.5, Fable) get adaptive thinking at low effort."""
        if thinking:
            return {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": cls.effort}}
        if cls.model.startswith("claude-sonnet-5-5"):
            return {"thinking": {"type": "between_tools"}}  # Sonnet 5.5 rejects {"type": "disabled"}
        if cls.model.startswith("claude-haiku-5-5"):
            return {"thinking": {"type": "disabled"}, "output_config": {"effort": "high"}}
        return {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}

    @classmethod
    def describe(cls, thinking: bool, think_max_tokens: int) -> dict:  # think_max_tokens is Qwen's budget: Claude keeps CLAUDE_THINK_MAX_TOKENS
        return {"backend": "claude", "name": cls.model, "thinking": thinking, **cls.request_options(thinking), "sampling": None, "system_suffix": load_prompt("backend", "claude_answer_only"),
                "max_tokens": CLAUDE_THINK_MAX_TOKENS if thinking else CLAUDE_ANSWER_MAX_TOKENS}

    @staticmethod
    def _api_messages(messages: list[dict]) -> list[dict]:
        def parts(p):
            if p[0] == "claude_blocks":
                return list(p[1])
            if p[0] == "text":
                return [{"type": "text", "text": p[1]}]
            return [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(p[2]).decode()}}]
        return [{"role": m["role"], "content": [block for p in m["parts"] for block in parts(p)]} for m in messages if m["role"] != "system"]

    def sample(self, messages: list[dict]) -> Sample:
        cls = type(self)
        if cls.api is None:
            cls.api = anthropic.Anthropic(max_retries=5)
        system = "".join(p[1] for m in messages if m["role"] == "system" for p in m["parts"])
        t0 = time.perf_counter()
        with cls.api.messages.stream(model=cls.model, max_tokens=CLAUDE_THINK_MAX_TOKENS if self.thinking else CLAUDE_ANSWER_MAX_TOKENS,
                                     system=system, messages=self._api_messages(messages), cache_control={"type": "ephemeral"},  # the rolling conversation resends its whole history
                                     **self.request_options(self.thinking)) as stream:
            r = stream.get_final_message()
        latency = time.perf_counter() - t0
        content = "".join(b.text for b in r.content if b.type == "text")
        reasoning = "\n".join(b.thinking for b in r.content if b.type == "thinking")
        usage = r.usage
        prompt_tokens = usage.input_tokens + (usage.cache_read_input_tokens or 0) + (usage.cache_creation_input_tokens or 0)
        finish = {"end_turn": "stop", "max_tokens": "length"}.get(r.stop_reason, r.stop_reason)  # vLLM's names; a refusal stays "refusal"
        return Sample(self.builder.parse(content), content, reasoning, [("claude_blocks", r.content)], finish, prompt_tokens, usage.output_tokens, latency)


BACKENDS = {"qwen": QwenClient, "claude": ClaudeClient}
_backend = {"name": "qwen"}


def select_backend(name: str, model: str | None = None, effort: str | None = None) -> None:
    """Use this backend for every chat of the process; model and effort (thinking on) are for Claude only."""
    if name not in BACKENDS:
        raise ValueError(f"backend {name!r} is not one of {list(BACKENDS)}")
    if (model or effort) and name != "claude":
        raise ValueError("--model and --effort are only for the claude backend")
    if name == "claude":
        if anthropic is None:
            raise ValueError("the claude backend needs the anthropic package")
        ClaudeClient.model = model or CLAUDE_MODEL
        ClaudeClient.effort = effort or CLAUDE_THINKING_EFFORT
    _backend["name"] = name


def make_client(builder: PromptBuilder, thinking: bool):
    return BACKENDS[_backend["name"]](builder, thinking)


def describe_model(thinking: bool, think_max_tokens: int) -> dict:
    return BACKENDS[_backend["name"]].describe(thinking, think_max_tokens)


def backend_label() -> dict:
    """Backend and model of the process, to record in the run folder."""
    if _backend["name"] == "claude":
        return {"backend": "claude", "model": ClaudeClient.model, "effort": ClaudeClient.effort}
    return {"backend": "qwen", "model": MODEL}


class Chat:
    """One conversation. messages hold parts ("text", str) or ("image", kind, png_bytes), so old maps can be dropped, and the replies' parts
    (Sample.reply_parts)."""

    def __init__(self, client, system: str):
        self.client = client
        self.messages = [{"role": "system", "parts": [("text", system)]}]

    def ask(self, turn: Turn, images: dict[str, bytes], n: int = 1) -> list[Sample]:
        """Add the user turn (images: kind -> PNG bytes, those named in turn.images) and sample n replies."""
        assert self.messages[-1]["role"] != "user", "accept() a reply before asking again"
        parts = [("image", k, images[k]) for k in turn.images] + [("text", turn.text)]
        self.messages.append({"role": "user", "parts": parts})
        self._cap_images()
        return self.resample(n)

    def accept(self, sample: Sample) -> None:
        """Continue the conversation from this reply, its reasoning included."""
        self.messages.append({"role": "assistant", "parts": sample.reply_parts})

    def resample(self, n: int = 1) -> list[Sample]:
        """n new samples for the last user turn: the retry for an invalid sample."""
        with ThreadPoolExecutor(n) as pool:  # separate requests (not n=) so every sample has its own usage and latency
            return list(pool.map(lambda _: self.client.sample(self.messages), range(n)))

    def _cap_images(self) -> None:
        """Replace the oldest maps by a text note until the prompt fits the backend's image limit (the camera frame stays)."""
        images = [(m["parts"], i) for m in self.messages for i, part in enumerate(m["parts"]) if part[0] == "image"]
        maps = [(parts, i) for parts, i in images if parts[i][1] == "bev"]
        for parts, i in maps[: max(0, len(images) - self.client.max_images)]:
            parts[i] = ("text", load_prompt("backend", "map_omitted"))
