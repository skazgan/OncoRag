"""Hybrid retrieval: dense (Qdrant) + sparse (BM25) fused with Reciprocal Rank Fusion.

RRF is used instead of a weighted sum of raw scores because cosine similarities and BM25 scores
live on incomparable scales; RRF only uses ranks, needs no per-corpus calibration and is the
robust default in practice.

    RRF(d) = sum over rankers r of  w_r / (k + rank_r(d))

An optional, *soft* entity boost multiplies the fused score of chunks that mention the genes /
cancer types detected in the query. It is soft (a re-weighting, not a filter) so a query that
mentions "p53" does not lose relevant chunks that only say "TP53-mutant".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from oncorag.config import Settings
from oncorag.embeddings import Embedder
from oncorag.index.lexical import BM25Index
from oncorag.index.vector_store import QdrantStore
from oncorag.ingestion.entities import EntityExtractor, get_extractor
from oncorag.schemas import Chunk, RetrievedChunk, SearchFilters


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match rankings")
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
    return fused


class HybridRetriever:
    def __init__(
        self,
        chunks: Mapping[str, Chunk],
        store: QdrantStore,
        bm25: BM25Index,
        embedder: Embedder,
        settings: Settings,
        extractor: EntityExtractor | None = None,
    ) -> None:
        self.chunks = chunks
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings
        self.extractor = extractor or get_extractor()

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        filters: SearchFilters | None = None,
        top_k: int | None = None,
        entity_boost: float | None = None,
    ) -> list[RetrievedChunk]:
        mode = RetrievalMode(mode)
        s = self.settings
        top_k = top_k or s.top_k_candidates
        boost = s.entity_boost if entity_boost is None else entity_boost

        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []
        if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            dense_hits = self.store.search(
                self.embedder.embed_query(query), s.top_k_dense, filters
            )
        if mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID):
            sparse_hits = self.bm25.search(query, s.top_k_sparse, filters)

        dense_rank = {cid: r for r, (cid, _) in enumerate(dense_hits, start=1)}
        sparse_rank = {cid: r for r, (cid, _) in enumerate(sparse_hits, start=1)}

        if mode is RetrievalMode.DENSE:
            scores = {cid: sc for cid, sc in dense_hits}
            fused: dict[str, float] | None = None
        elif mode is RetrievalMode.SPARSE:
            scores = {cid: sc for cid, sc in sparse_hits}
            fused = None
        else:
            fused = reciprocal_rank_fusion(
                [[c for c, _ in dense_hits], [c for c, _ in sparse_hits]], k=s.rrf_k
            )
            scores = dict(fused)

        if boost > 0 and scores:
            q_entities = self.extractor.extract(query).as_set()
            if q_entities:
                for cid in list(scores):
                    chunk = self.chunks[cid]
                    overlap = len(q_entities & {*chunk.genes, *chunk.cancer_types}) / len(q_entities)
                    scores[cid] *= 1.0 + boost * overlap

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [
            RetrievedChunk(
                chunk=self.chunks[cid],
                score=score,
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                fused_score=fused.get(cid) if fused else None,
            )
            for cid, score in ranked
            if cid in self.chunks
        ]
