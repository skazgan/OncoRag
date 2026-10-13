"""Answer generators.

* ``LLMGenerator`` wraps any chat-completion client behind a tiny ``complete(system, user)``
  interface (``AnthropicClient`` ships; swapping providers means writing ~15 lines).
* ``ExtractiveGenerator`` needs no LLM: it assembles an answer from the most query-relevant
  sentences in the retrieved evidence, each carrying its citation marker. It is deterministic, so
  it powers offline demos, CI and regression tests of the retrieval + verification stack.
"""

from __future__ import annotations

from typing import Protocol

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from oncorag.config import Settings
from oncorag.generation.prompts import ABSTAIN_TOKEN, SYSTEM_PROMPT, build_user_prompt
from oncorag.schemas import RetrievedChunk
from oncorag.text import content_terms, split_sentences


class Generator(Protocol):
    name: str

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str: ...


class ChatClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_tokens: int = 700,
        temperature: float | None = 0.0,
        *,
        client: object | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Run: pip install -e '.[llm]'") from exc
        if client is None and not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it, put it in .env, or use "
                "ONCORAG_LLM_BACKEND=extractive."
            )
        self._anthropic = anthropic
        # `client` is injectable so the request/retry logic is unit-testable without network.
        self._client = client or anthropic.Anthropic(api_key=api_key)
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def complete(self, system: str, user: str) -> str:
        @retry(
            retry=retry_if_exception_type(
                (
                    self._anthropic.RateLimitError,
                    self._anthropic.APIConnectionError,
                    self._anthropic.InternalServerError,
                )
            ),
            stop=stop_after_attempt(4),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            reraise=True,
        )
        def _call() -> str:
            kwargs: dict[str, object] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            msg = self._client.messages.create(**kwargs)  # type: ignore[arg-type]
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

        return _call().strip()


class LLMGenerator:
    def __init__(self, client: ChatClient, name: str = "llm") -> None:
        self.client = client
        self.name = name

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        return self.client.complete(SYSTEM_PROMPT, build_user_prompt(question, contexts))


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_coverage: float = 0.25) -> None:
        self.max_sentences = max_sentences
        self.min_coverage = min_coverage

    def generate(self, question: str, contexts: list[RetrievedChunk]) -> str:
        q_terms = content_terms(question)
        if not q_terms or not contexts:
            return f"{ABSTAIN_TOKEN} No usable evidence was retrieved."

        scored: list[tuple[float, int, int, str]] = []  # (coverage, source, position, sentence)
        for src, rc in enumerate(contexts, start=1):
            for pos, sent in enumerate(split_sentences(rc.chunk.text)):
                coverage = len(q_terms & content_terms(sent)) / len(q_terms)
                scored.append((coverage, src, pos, sent))

        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        picked: list[tuple[float, int, int, str]] = []
        per_source: dict[int, int] = {}
        for item in scored:
            if item[0] < self.min_coverage:
                break
            if per_source.get(item[1], 0) >= 2:
                continue
            picked.append(item)
            per_source[item[1]] = per_source.get(item[1], 0) + 1
            if len(picked) == self.max_sentences:
                break

        if not picked:
            return f"{ABSTAIN_TOKEN} The retrieved sources do not address this question."
        picked.sort(key=lambda t: (t[1], t[2]))  # reading order: by source, then position
        return " ".join(f"{sent.rstrip().rstrip('.')} [{src}]." for _, src, _, sent in picked)


def get_generator(settings: Settings) -> Generator:
    if settings.llm_backend == "anthropic":
        client = AnthropicClient(
            settings.anthropic_api_key,
            settings.llm_model,
            settings.llm_max_tokens,
            settings.llm_temperature,
        )
        return LLMGenerator(client, name=f"anthropic:{settings.llm_model}")
    return ExtractiveGenerator()
