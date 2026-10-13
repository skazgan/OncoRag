"""Second-stage rerankers.

First-stage retrieval optimises recall over thousands of chunks cheaply (bi-encoder + BM25).
A reranker re-scores only the ~25 survivors with a more expensive model that sees the query and
the passage *together*, which is where precision@k is won.

* ``CrossEncoderReranker``: real neural reranker (needs the ``ml`` extra).
* ``LexicalReranker``: deterministic offline heuristic (term coverage, title match, phrase
  match, entity overlap). Scores are in [0, 1] and are meaningful as an evidence-strength signal,
  which is what the abstention gate uses.
"""

from __future__ import annotations

import math
from typing import Protocol

from oncorag.config import Settings
from oncorag.ingestion.entities import EntityExtractor, get_extractor
from oncorag.schemas import RetrievedChunk
from oncorag.text import content_terms, tokenize


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


def _finalise(candidates: list[RetrievedChunk], scores: list[float], top_k: int) -> list[RetrievedChunk]:
    for cand, sc in zip(candidates, scores, strict=True):
        cand.rerank_score = sc
        cand.score = sc
    return sorted(candidates, key=lambda c: (-c.score, c.chunk.chunk_id))[:top_k]


class NoopReranker:
    name = "none"

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return candidates[:top_k]


class LexicalReranker:
    name = "lexical"

    def __init__(self, extractor: EntityExtractor | None = None) -> None:
        self.extractor = extractor or get_extractor()

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        q_terms = content_terms(query)
        q_tokens = tokenize(query)
        q_bigrams = {f"{a} {b}" for a, b in zip(q_tokens, q_tokens[1:], strict=False)}
        q_entities = self.extractor.extract(query).as_set()

        scores: list[float] = []
        for cand in candidates:
            c = cand.chunk
            body_tokens = tokenize(c.text)
            body_terms = set(body_tokens)
            title_terms = content_terms(c.title)
            body_bigrams = {f"{a} {b}" for a, b in zip(body_tokens, body_tokens[1:], strict=False)}

            if not q_terms:
                scores.append(0.0)
                continue
            coverage = len(q_terms & (body_terms | title_terms)) / len(q_terms)
            title_hit = len(q_terms & title_terms) / len(q_terms)
            phrase = len(q_bigrams & body_bigrams) / len(q_bigrams) if q_bigrams else 0.0
            entity = (
                len(q_entities & {*c.genes, *c.cancer_types}) / len(q_entities) if q_entities else 0.0
            )
            raw = 0.55 * coverage + 0.15 * title_hit + 0.15 * phrase + 0.15 * entity
            scores.append(min(1.0, raw))
        return _finalise(candidates, scores, top_k)


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError("Run: pip install -e '.[ml]'") from exc
        self._model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, f"{c.chunk.title}. {c.chunk.text}") for c in candidates]
        logits = self._model.predict(pairs, show_progress_bar=False)
        scores = [1.0 / (1.0 + math.exp(-float(x))) for x in logits]  # sigmoid -> (0, 1)
        return _finalise(candidates, scores, top_k)


def get_reranker(settings: Settings) -> Reranker:
    if settings.reranker_backend == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    if settings.reranker_backend == "lexical":
        return LexicalReranker()
    return NoopReranker()
