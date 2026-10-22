"""Qwen chat client over the vLLM OpenAI API: messages from prompt turns and images, samples, parsed plans.

A Chat holds one conversation (system prompt, then user turns and the model's replies). A single
query is a Chat with one ask(); a rolling conversation keeps asking and accept()s the reply it
wants to continue from. Per sample the client returns the parsed plan (None when the reply is not
a valid plan), the raw reply, the reasoning text, the finish reason, token counts and latency.

What the reply must look like (JSON key, example, validity checks, the start of a reply) comes from the PromptBuilder of the
output mode, so the client has no per-mode branches. Reply format choices and their measurements: TRACKER.md, Phase 3.
"""

import base64
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from prompting import PromptBuilder, Turn
from scene import load_prompt


BASE = "http://localhost:8000"
MODEL = "Qwen/Qwen3.8-27B-FP8"
MAX_IMAGES = 10  # the server's --limit-mm-per-prompt (hri-sim scripts/serve.sh)
THINKING_EFFORT = "medium"  # the template's default "xhigh" never finished within 24000 tokens (8 of 8 runs); "low" was not shorter than medium
THINK_MAX_TOKENS = 16384  # thinking + answer in one call: medium effort needs 1-6k tokens, text-only waypoints ~8k (8 of 12 were forced at 8192); beyond this the answer is forced. Server runs with MAX_LEN=65536 so long conversations still fit
ANSWER_MAX_TOKENS = 1024  # longest answer is 40 pairs, about 600 tokens; also the cap for a forced answer
TIMEOUT_S = 3600  # a thinking reply of the full budget takes minutes on a busy server
# Sampling from the model card (PLAN Decision 8); thinking mode and instruct (non-thinking) mode differ.
THINKING_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
INSTRUCT_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}


@dataclass
class Sample:
    value: np.ndarray | None  # the plan as an (k, 2) array in the output units; None = invalid reply
    content: str              # the reply text the plan was read from
    reasoning: str            # the model's thinking ("" without thinking)
    finish: str               # vLLM finish reason of the (last) call: "stop" or "length"
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


def post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


class QwenClient:
    """Settings shared by all chats of one condition: output mode (via the builder) and thinking on/off."""

    def __init__(self, builder: PromptBuilder, thinking: bool):
        self.builder, self.thinking = builder, thinking
        self.sampling = THINKING_SAMPLING if thinking else INSTRUCT_SAMPLING

    def chat(self, system: str) -> "Chat":
        return Chat(self, system)


class Chat:
    """One conversation. messages hold parts ("text", str) or ("image", kind, png_bytes) so old maps can be dropped."""

    def __init__(self, client: QwenClient, system: str):
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
        """Continue the conversation from this reply (its final text only, not its thinking)."""
        self.messages.append({"role": "assistant", "parts": [("text", sample.content)]})

    def resample(self, n: int = 1) -> list[Sample]:
        """n new samples for the last user turn: the retry for an invalid sample."""
        with ThreadPoolExecutor(n) as pool:  # separate requests (not n=) so every sample has its own usage and latency
            return list(pool.map(lambda _: self._sample(), range(n)))

    def _cap_images(self) -> None:
        """Replace the oldest maps by a text note until the prompt fits the server's image limit (the camera frame stays)."""
        images = [(m["parts"], i) for m in self.messages for i, part in enumerate(m["parts"]) if part[0] == "image"]
        maps = [(parts, i) for parts, i in images if parts[i][1] == "bev"]
        for parts, i in maps[: max(0, len(images) - MAX_IMAGES)]:
            parts[i] = ("text", load_prompt("map_omitted"))

    def _api_messages(self) -> list[dict]:
        def part(p):
            if p[0] == "text":
                return {"type": "text", "text": p[1]}
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(p[2]).decode()}}
        return [{"role": m["role"], "content": [part(p) for p in m["parts"]]} for m in self.messages]

    def _call(self, extra: list[dict] | None = None, **kwargs) -> tuple[dict, float]:
        """One chat completion for the conversation so far (+ extra trailing messages); returns choice with usage, and latency."""
        c = self.client
        t0 = time.perf_counter()
        payload = {"model": MODEL, "messages": self._api_messages() + (extra or []),
                   "chat_template_kwargs": {"enable_thinking": c.thinking, "preserve_thinking": False,  # earlier replies enter the history without a (empty) think block
                                    **({"reasoning_effort": THINKING_EFFORT} if c.thinking else {})},
                   **c.sampling, **kwargs}
        r = post("/v1/chat/completions", payload)
        return {**r["choices"][0], "usage": r["usage"]}, time.perf_counter() - t0

    def _sample(self) -> Sample:
        b = self.client.builder
        choice, latency = self._call(max_tokens=THINK_MAX_TOKENS if self.client.thinking else ANSWER_MAX_TOKENS)
        message, usage = choice["message"], choice["usage"]
        content, reasoning = message.get("content") or "", message.get("reasoning") or ""
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        if self.client.thinking and choice["finish_reason"] == "length" and b.parse(content) is None:
            # The budget ran out in the thinking (or in the answer after it): close the thinking and force the answer,
            # continuing from the reasoning so far with the start of the JSON already written.
            reasoning += "\n\n" + load_prompt("think_cutoff")
            choice, dt = self._call([{"role": "assistant", "reasoning": reasoning, "content": b.prefill}], max_tokens=ANSWER_MAX_TOKENS,
                                    continue_final_message=True, add_generation_prompt=False)
            message, usage = choice["message"], choice["usage"]
            # The qwen3 reasoning parser finds no </think> in this continuation and files it all under "reasoning".
            content = b.prefill + (message.get("content") or message.get("reasoning") or "")
            latency += dt
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
        return Sample(b.parse(content), content, reasoning, choice["finish_reason"], prompt_tokens, completion_tokens, latency)
