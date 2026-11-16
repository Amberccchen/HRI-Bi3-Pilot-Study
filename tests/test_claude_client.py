"""The Claude backend with a fake Messages API: the request it builds and the Sample it returns, without a network or a key."""

from types import SimpleNamespace as NS

import pytest

from traj_gen import client
from traj_gen.client import Chat, ClaudeClient
from traj_gen.prompting import Turn


class FakeBuilder:
    prefill = ""

    def parse(self, text):
        return text if text.startswith("{") else None


class FakeStream:
    def __init__(self, store, kwargs, message):
        store.append(kwargs)
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def fake_api(store, message):
    return NS(messages=NS(stream=lambda **kwargs: FakeStream(store, kwargs, message)))


def reply(text, stop="end_turn", thinking=""):
    content = ([NS(type="thinking", thinking=thinking)] if thinking else []) + [NS(type="text", text=text)]
    return NS(content=content, stop_reason=stop, usage=NS(input_tokens=10, cache_read_input_tokens=5, cache_creation_input_tokens=None, output_tokens=7))


@pytest.fixture
def requests(monkeypatch):
    store = []
    monkeypatch.setattr(ClaudeClient, "api", fake_api(store, reply('{"waypoints": [[0, 0]]}', thinking="plan")))
    monkeypatch.setattr(ClaudeClient, "model", "claude-sonnet-5-5")
    return store


def ask(thinking):
    chat = Chat(ClaudeClient(FakeBuilder(), thinking), "system text")
    return chat.ask(Turn("go", ["bev"]), {"bev": b"png"})[0]


def test_request_and_sample_with_thinking(requests):
    s = ask(True)
    r = requests[0]
    assert r["system"] == "system text" and r["model"] == "claude-sonnet-5-5"
    assert r["messages"] == [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}}, {"type": "text", "text": "go"}]}]
    assert r["thinking"] == {"type": "adaptive", "display": "summarized"} and r["output_config"] == {"effort": "medium"}
    assert not {"temperature", "top_p", "top_k"} & r.keys()
    assert (s.content, s.reasoning, s.finish, s.prompt_tokens, s.completion_tokens) == ('{"waypoints": [[0, 0]]}', "plan", "stop", 15, 7)
    assert s.value is not None


def test_thinking_off_per_model(requests, monkeypatch):
    ask(False)
    assert requests[0]["thinking"] == {"type": "between_tools"} and "output_config" not in requests[0]
    monkeypatch.setattr(ClaudeClient, "model", "claude-opus-5-5")
    ask(False)
    assert requests[1]["thinking"] == {"type": "adaptive"} and requests[1]["output_config"] == {"effort": "low"}


def test_finish_reasons(monkeypatch):
    for stop, finish in (("max_tokens", "length"), ("refusal", "refusal")):
        monkeypatch.setattr(ClaudeClient, "api", fake_api([], reply("no json", stop=stop)))
        s = ask(True)
        assert s.finish == finish and s.value is None


def test_qwen_request_unchanged(monkeypatch):
    sent = []
    monkeypatch.setattr(client, "post", lambda path, payload: sent.append(payload) or {"choices": [{"message": {"content": "{}", "reasoning": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    chat = Chat(client.QwenClient(FakeBuilder(), True), "system text")
    s = chat.ask(Turn("go", []), {})[0]
    assert sent[0]["messages"][0] == {"role": "system", "content": [{"type": "text", "text": "system text"}]}
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "medium"}
    assert sent[0]["temperature"] == 1.0 and sent[0]["max_tokens"] == client.THINK_MAX_TOKENS and s.reasoning == "r"


def test_select_backend(monkeypatch):
    monkeypatch.setattr(client, "_backend", {"name": "qwen"})
    with pytest.raises(ValueError):
        client.select_backend("qwen", "claude-opus-5-5")
    client.select_backend("claude", "claude-opus-5-5")
    assert client.backend_label() == {"backend": "claude", "model": "claude-opus-5-5"}
    assert client.describe_model(True)["backend"] == "claude"
    monkeypatch.setattr(ClaudeClient, "model", client.CLAUDE_MODEL)
