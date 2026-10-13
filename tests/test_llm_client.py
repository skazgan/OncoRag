"""AnthropicClient request/retry logic, verified against a fake SDK client (no network)."""

from types import SimpleNamespace

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

from oncorag.generation.generator import AnthropicClient, LLMError, LLMGenerator  # noqa: E402
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


def test_default_request_is_accepted_by_the_real_sdk_signature():
    """Regression: the fake client above accepts anything, which hid that the real SDK's
    Messages.create() rejects `temperature`. Bind our default kwargs to the genuine signature."""
    import inspect

    from oncorag.config import Settings

    s = Settings(_env_file=None)
    assert s.llm_temperature is None
    c, msgs = _client(["ok"], max_tokens=s.llm_max_tokens, temperature=s.llm_temperature)
    c.complete("sys", "user")
    sig = inspect.signature(anthropic.resources.messages.Messages.create)
    sig.bind(None, **msgs.calls[0])  # raises TypeError on any argument the SDK doesn't accept


class _Block:
    def __init__(self, type, text=None):
        self.type, self.text = type, text


def _msg(blocks, stop_reason, output_tokens=700):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=SimpleNamespace(output_tokens=output_tokens))


class _Msgs:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def test_thinking_only_response_raises_an_actionable_error_and_is_not_retried():
    """Regression: with a small max_tokens the model spent everything on thinking, returned no
    text block, and the CLI printed an empty answer panel with 'support rate 0% clean'."""
    msgs = _Msgs(_msg([_Block("thinking")], "max_tokens"))
    c = AnthropicClient("k", "m", max_tokens=700, client=SimpleNamespace(messages=msgs))
    with pytest.raises(LLMError) as info:
        c.complete("s", "u")
    text = str(info.value)
    assert "stop_reason=max_tokens" in text and "thinking" in text and "700" in text
    assert "ONCORAG_LLM_MAX_TOKENS" in text and "ONCORAG_LLM_EFFORT" in text
    assert len(msgs.calls) == 1  # a deterministic failure must not be retried


def test_refusal_and_blank_text_are_reported_distinctly():
    refusal = AnthropicClient("k", "m", client=SimpleNamespace(messages=_Msgs(_msg([], "refusal"))))
    with pytest.raises(LLMError, match="declined"):
        refusal.complete("s", "u")
    blank = AnthropicClient("k", "m", client=SimpleNamespace(messages=_Msgs(_msg([_Block("text", "  \n")], "end_turn"))))
    with pytest.raises(LLMError, match="returned no text"):
        blank.complete("s", "u")


def test_effort_is_sent_as_output_config_only_when_set_and_sdk_accepts_it():
    import inspect

    c, msgs = _client(["ok"], effort="low")
    c.complete("s", "u")
    assert msgs.calls[0]["output_config"] == {"effort": "low"}
    sig = inspect.signature(anthropic.resources.messages.Messages.create)
    sig.bind(None, **msgs.calls[0])

    c2, msgs2 = _client(["ok"])
    c2.complete("s", "u")
    assert "output_config" not in msgs2.calls[0]


def test_generator_rejects_blank_client_output():
    class Blank:
        def complete(self, system, user):
            return "   "

    ctx = [RetrievedChunk(chunk=Chunk(chunk_id="1:0", pmid="1", title="T", text="Body."), score=1.0)]
    with pytest.raises(LLMError, match="empty answer"):
        LLMGenerator(Blank()).generate("Q?", ctx)


def test_new_defaults_leave_headroom_for_thinking():
    from oncorag.config import Settings

    s = Settings(_env_file=None)
    assert s.llm_max_tokens >= 4096 and s.llm_effort is None
