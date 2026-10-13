"""AnthropicClient request/retry logic, verified against a fake SDK client (no network)."""

from types import SimpleNamespace

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

from oncorag.generation.generator import AnthropicClient, LLMGenerator  # noqa: E402
from oncorag.generation.prompts import SYSTEM_PROMPT  # noqa: E402
from oncorag.schemas import Chunk, RetrievedChunk  # noqa: E402


class FakeMessages:
    def __init__(self, behaviours):
        self.behaviours, self.calls = list(behaviours), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self.behaviours.pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=action)])


def _client(behaviours, **kw):
    msgs = FakeMessages(behaviours)
    return AnthropicClient("key", "claude-test", client=SimpleNamespace(messages=msgs), **kw), msgs


def test_request_shape_and_text_extraction():
    c, msgs = _client(["  Answer [1].  "], max_tokens=123, temperature=0.0)
    assert c.complete("SYS", "USER") == "Answer [1]."
    call = msgs.calls[0]
    assert call["model"] == "claude-test" and call["max_tokens"] == 123
    assert call["system"] == "SYS" and call["messages"] == [{"role": "user", "content": "USER"}]
    assert call["temperature"] == 0.0


def test_temperature_is_omitted_when_unset():
    c, msgs = _client(["ok"], temperature=None)
    c.complete("s", "u")
    assert "temperature" not in msgs.calls[0]


def test_non_text_blocks_are_ignored():
    msgs = SimpleNamespace(create=lambda **kw: SimpleNamespace(content=[
        SimpleNamespace(type="thinking", text="ignore me"), SimpleNamespace(type="text", text="real")]))
    c = AnthropicClient("k", "m", client=SimpleNamespace(messages=msgs))
    assert c.complete("s", "u") == "real"


def test_connection_errors_are_retried():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    c, msgs = _client([err, "recovered"])
    assert c.complete("s", "u") == "recovered" and len(msgs.calls) == 2


def test_non_retryable_errors_propagate_immediately():
    c, msgs = _client([ValueError("bad request shape")])
    with pytest.raises(ValueError):
        c.complete("s", "u")
    assert len(msgs.calls) == 1


def test_missing_api_key_gives_actionable_error():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient(None, "m")


def test_generator_sends_the_grounding_system_prompt():
    c, msgs = _client(["Fact [1]."])
    ctx = [RetrievedChunk(chunk=Chunk(chunk_id="1:0", pmid="1", title="T", text="Body text."), score=1.0)]
    assert LLMGenerator(c).generate("Q?", ctx) == "Fact [1]."
    assert msgs.calls[0]["system"] == SYSTEM_PROMPT
    assert "Question: Q?" in msgs.calls[0]["messages"][0]["content"]
