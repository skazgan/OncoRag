"""RAGPipeline: the single entry point used by the CLI, the API and the evaluation harness.

    question
       |-> retrieve (dense + BM25 -> RRF -> soft entity boost)      ~ recall
       |-> rerank top candidates                                    ~ precision
       |-> diversify (cap chunks per article) and take top_k
       |-> abstention gate (weak evidence => don't call the LLM)
       |-> generate with numbered sources
       |-> verify citations / groundedness
       '-> Answer (text, citations, contexts, verification, per-stage latency)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from oncorag.config import Settings
from oncorag.embeddings import Embedder, get_embedder
from oncorag.generation.citations import build_citations, is_abstention, verify_answer
from oncorag.generation.generator import Generator, get_generator
from oncorag.generation.prompts import ABSTAIN_TOKEN
from oncorag.index.manager import IndexManager
from oncorag.retrieval.hybrid import HybridRetriever, RetrievalMode
from oncorag.retrieval.rerank import NoopReranker, Reranker, get_reranker
from oncorag.schemas import Answer, RetrievedChunk, SearchFilters


@dataclass
class RetrievalOptions:
    mode: RetrievalMode = RetrievalMode.HYBRID
    rerank: bool = True
    entity_boost: float | None = None  # None -> Settings.entity_boost
    top_k: int | None = None  # contexts passed on; None -> Settings.top_k_final


def diversify(candidates: list[RetrievedChunk], top_k: int, max_per_article: int) -> list[RetrievedChunk]:
    """Cap chunks per article so one verbose paper cannot crowd out independent evidence."""
    out: list[RetrievedChunk] = []
    per_pmid: dict[str, int] = {}
    for cand in candidates:
        if per_pmid.get(cand.chunk.pmid, 0) >= max_per_article:
            continue
        per_pmid[cand.chunk.pmid] = per_pmid.get(cand.chunk.pmid, 0) + 1
        out.append(cand)
        if len(out) == top_k:
            break
    return out


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        index: IndexManager,
        retriever: HybridRetriever,
        reranker: Reranker,
        generator: Generator,
    ) -> None:
        self.settings = settings
        self.index = index
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        persist: bool = True,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> RAGPipeline:
        settings = settings or Settings()
        embedder = embedder or get_embedder(settings)
        index = IndexManager(settings, embedder, persist=persist)
        retriever = HybridRetriever(index.chunks, index.store, index.bm25, embedder, settings)
        return cls(
            settings,
            index,
            retriever,
            reranker or get_reranker(settings),
            generator or get_generator(settings),
        )

    # -- ingestion keeps the retriever's BM25 handle fresh -------------
    def add_articles(self, articles) -> dict[str, int]:  # type: ignore[no-untyped-def]
        stats = self.index.add_articles(articles)
        self.retriever.bm25 = self.index.bm25
        self.retriever.chunks = self.index.chunks
        return stats

    # -- retrieval only (used by /search and by evaluation) ------------
    def retrieve(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        options: RetrievalOptions | None = None,
    ) -> list[RetrievedChunk]:
        opts = options or RetrievalOptions()
        s = self.settings
        top_k = opts.top_k or s.top_k_final
        candidates = self.retriever.retrieve(
            question,
            mode=opts.mode,
            filters=filters,
            top_k=s.top_k_candidates,
            entity_boost=opts.entity_boost,
        )
        reranker = self.reranker if opts.rerank else NoopReranker()
        reranked = reranker.rerank(question, candidates, top_k=len(candidates))
        return diversify(reranked, top_k, s.max_chunks_per_article)

    # -- full RAG ------------------------------------------------------
    def ask(
        self,
        question: str,
        *,
        filters: SearchFilters | None = None,
        options: RetrievalOptions | None = None,
    ) -> Answer:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        contexts = self.retrieve(question, filters=filters, options=options)
        timings["retrieve_rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        if not contexts:
            return Answer(
                question=question,
                answer=f"{ABSTAIN_TOKEN} No relevant sources were found in the index.",
                abstained=True,
                abstain_reason="no_results",
                timings_ms=timings,
            )

        threshold = self.settings.abstain_below
        best = contexts[0].rerank_score
        if threshold is not None and best is not None and best < threshold:
            return Answer(
                question=question,
                answer=(
                    f"{ABSTAIN_TOKEN} The indexed literature does not appear to address this "
                    "question (best evidence score "
                    f"{best:.2f} is below the {threshold:.2f} threshold)."
                ),
                abstained=True,
                abstain_reason="low_evidence_score",
                contexts=contexts,
                timings_ms=timings,
            )

        t1 = time.perf_counter()
        text = self.generator.generate(question, contexts)
        timings["generate_ms"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        abstained = is_abstention(text)
        verification = None if abstained else verify_answer(text, contexts)
        timings["verify_ms"] = (time.perf_counter() - t2) * 1000
        timings["total_ms"] = sum(timings.values())

        return Answer(
            question=question,
            answer=text,
            abstained=abstained,
            abstain_reason="model_abstained" if abstained else None,
            citations=[] if abstained else build_citations(text, contexts),
            contexts=contexts,
            verification=verification,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
        )

    def close(self) -> None:
        self.index.close()
