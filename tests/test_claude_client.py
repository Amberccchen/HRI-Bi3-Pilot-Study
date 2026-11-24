"""The Claude backend with a fake Messages API: the request it builds and the Sample it returns, without a network or a key."""

from types import SimpleNamespace as NS

import pytest

from traj_gen.harness import clients
from traj_gen.harness.clients import Chat, ClaudeClient
from traj_gen.observation.prompting import Turn


class FakeMode:
    think_max_tokens = 16384
    prefill = ""

    def parse(self, text, max_steps):
        return text if text.startswith("{") else None


class FakeBuilder:
    mode = FakeMode()
    max_steps = 75


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
    chat = ClaudeClient(FakeBuilder(), thinking).chat("system text")
    return chat.ask(Turn("go", ["bev"]), {"bev": b"png"})[0]


def test_request_and_sample_with_thinking(requests):
    s = ask(True)
    r = requests[0]
    assert r["system"].startswith("system text\n\n") and "only with your final answer" in r["system"] and r["model"] == "claude-sonnet-5-5"
    assert r["messages"] == [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"}}, {"type": "text", "text": "go"}]}]
    assert r["thinking"] == {"type": "adaptive", "display": "summarized"} and r["output_config"] == {"effort": "high"}
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


def test_qwen_request(monkeypatch):
    sent = []
    monkeypatch.setattr(clients, "post", lambda path, payload: sent.append(payload) or {"choices": [{"message": {"content": "{}", "reasoning": "r"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    chat = Chat(clients.QwenClient(FakeBuilder(), True), "system text")
    s = chat.ask(Turn("go", []), {})[0]
    assert sent[0]["messages"][0] == {"role": "system", "content": [{"type": "text", "text": "system text"}]}
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "medium"}
    assert sent[0]["temperature"] == 1.0 and sent[0]["max_tokens"] == FakeMode.think_max_tokens and s.reasoning == "r"


def test_select_backend(monkeypatch):
    monkeypatch.setattr(clients, "_backend", {"name": "qwen"})
    with pytest.raises(ValueError):
        clients.select_backend("qwen", "claude-opus-5-5")
    clients.select_backend("claude", "claude-opus-5-5", "medium")
    assert clients.backend_label() == {"backend": "claude", "model": "claude-opus-5-5", "effort": "medium"}
    assert clients.describe_model(True, 16384)["output_config"] == {"effort": "medium"}
    assert clients.describe_model(True, 16384)["backend"] == "claude"
    monkeypatch.setattr(ClaudeClient, "model", clients.CLAUDE_MODEL)
    monkeypatch.setattr(ClaudeClient, "effort", clients.CLAUDE_THINKING_EFFORT)


def test_run_folder_takes_one_model(tmp_path):
    from traj_gen.run import check_run_model
    label = {"backend": "claude", "model": "claude-sonnet-5-5", "effort": "high"}
    check_run_model(tmp_path, {}, label)  # a fresh folder takes any model
    check_run_model(tmp_path, label, label)
    with pytest.raises(ValueError):
        check_run_model(tmp_path, {"backend": "qwen", "model": "Qwen/Qwen3.8-27B-FP8"}, label)
    episode = tmp_path / "e" / "f" / "nothink" / "t" / "s0"
    episode.mkdir(parents=True)
    (episode / "manifest.json").write_text('{"model": {"name": "Qwen/Qwen3.8-27B-FP8", "thinking": false}}')  # an old Qwen episode, which has no backend field
    with pytest.raises(ValueError):
        check_run_model(tmp_path, {}, label)
    (episode / "manifest.json").write_text('{"model": {"name": "claude-sonnet-5-5", "thinking": true, "output_config": {"effort": "medium"}}}')
    with pytest.raises(ValueError):  # the same model at another effort
        check_run_model(tmp_path, {}, label)
    (episode / "manifest.json").write_text('{"model": {"name": "claude-sonnet-5-5", "thinking": true, "output_config": {"effort": "high"}}}')
    check_run_model(tmp_path, {}, label)


def two_turns(client):
    """Ask, accept the reply, ask again: the conversation the second request carries."""
    chat = Chat(client, "system text")
    chat.accept(chat.ask(Turn("first", []), {})[0])
    chat.ask(Turn("second", []), {})
    return chat


def test_claude_history_after_two_turns_holds_the_first_replys_blocks_unchanged(monkeypatch):
    """The second request must carry the first reply's whole content, its thinking block (with the signature the API checks) before its text,
    as the same objects: a history of the reply text alone would drop the earlier thinking silently."""
    store, first = [], reply('{"waypoints": [[0, 0]]}', thinking="plan one")
    first.content[0].signature = "sig"
    monkeypatch.setattr(ClaudeClient, "api", fake_api(store, first))
    two_turns(ClaudeClient(FakeBuilder(), True))
    history = store[1]["messages"]
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert history[1]["content"] == first.content and history[1]["content"][0].thinking == "plan one" and history[1]["content"][0].signature == "sig"
    assert history[2]["content"] == [{"type": "text", "text": "second"}]


def test_qwen_history_after_two_turns_holds_the_first_replys_reasoning(monkeypatch):
    """The second request must give the first reply's reasoning in its assistant message, and ask the template to keep it."""
    sent = []
    answers = iter([("{}", "plan one"), ("{}", "plan two")])
    def post(path, payload):
        sent.append(payload)
        content, reasoning = next(answers)
        return {"choices": [{"message": {"content": content, "reasoning": reasoning}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}
    monkeypatch.setattr(clients, "post", post)
    two_turns(clients.QwenClient(FakeBuilder(), True))
    assert sent[1]["messages"][2] == {"role": "assistant", "content": [{"type": "text", "text": "{}"}], "reasoning": "plan one"}
    assert sent[1]["chat_template_kwargs"]["preserve_thinking"] is True
