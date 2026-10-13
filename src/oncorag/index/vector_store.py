"""Qdrant vector store (embedded local mode by default, server mode via ONCORAG_QDRANT_URL)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from oncorag.schemas import Chunk, SearchFilters

_NAMESPACE = uuid.UUID("6f1d3c52-8a0e-4d8c-9a55-1c3b7a1f0e11")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-ingesting a chunk overwrites instead of duplicating it."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def to_qdrant_filter(filters: SearchFilters | None) -> models.Filter | None:
    if filters is None or filters.is_empty:
        return None
    must: list[models.Condition] = []
    if filters.genes:
        must.append(models.FieldCondition(key="genes", match=models.MatchAny(any=filters.genes)))
    if filters.cancer_types:
        must.append(
            models.FieldCondition(key="cancer_types", match=models.MatchAny(any=filters.cancer_types))
        )
    if filters.year_min or filters.year_max:
        must.append(
            models.FieldCondition(
                key="year", range=models.Range(gte=filters.year_min, lte=filters.year_max)
            )
        )
    return models.Filter(must=must)


class QdrantStore:
    def __init__(
        self,
        dim: int,
        *,
        path: str | Path | None = None,
        url: str | None = None,
        collection: str = "oncorag_chunks",
    ) -> None:
        self.dim = dim
        self.collection = collection
        self._is_server = bool(url)
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
        else:
            self.client = QdrantClient(":memory:")
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )
        if not self._is_server:
            return  # payload indexes only exist on a Qdrant server; local mode ignores them
        for field, schema in (
            ("genes", models.PayloadSchemaType.KEYWORD),
            ("cancer_types", models.PayloadSchemaType.KEYWORD),
            ("pmid", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
        ):
            self.client.create_payload_index(self.collection, field, field_schema=schema)

    def upsert(self, chunks: Sequence[Chunk], vectors: np.ndarray, batch_size: int = 256) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for i in range(0, len(chunks), batch_size):
            batch = [
                models.PointStruct(
                    id=point_id(c.chunk_id),
                    vector=vectors[i + j].tolist(),
                    payload={
                        "chunk_id": c.chunk_id,
                        "pmid": c.pmid,
                        "genes": c.genes,
                        "cancer_types": c.cancer_types,
                        "year": c.year,
                    },
                )
                for j, c in enumerate(chunks[i : i + batch_size])
            ]
            self.client.upsert(self.collection, points=batch, wait=True)

    def delete_pmids(self, pmids: Sequence[str]) -> None:
        if not pmids:
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="pmid", match=models.MatchAny(any=list(pmids)))]
                )
            ),
            wait=True,
        )

    def search(
        self, vector: np.ndarray, limit: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        res = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=to_qdrant_filter(filters),
            with_payload=["chunk_id"],
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points if p.payload]

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def close(self) -> None:
        self.client.close()
