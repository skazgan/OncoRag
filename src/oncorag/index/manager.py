"""IndexManager: keeps chunks (source of truth), the Qdrant collection and the BM25 index consistent.

On disk (``index_dir``):
    chunks.jsonl   one chunk per line - the source of truth, human-inspectable and diffable
    manifest.json  embedder name / dimension / counts, used to reject mismatched embedders
    qdrant/        embedded Qdrant storage (omitted when a Qdrant server is used)

Re-ingesting an article replaces *all* of its previous chunks (stale chunks are deleted first),
so the index never serves text from an older version of the same PMID.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from oncorag.config import Settings
from oncorag.embeddings import Embedder
from oncorag.index.lexical import BM25Index
from oncorag.index.vector_store import QdrantStore
from oncorag.ingestion.chunking import chunk_article
from oncorag.ingestion.entities import EntityExtractor, get_extractor
from oncorag.schemas import Article, Chunk

log = logging.getLogger(__name__)


class IndexMismatchError(RuntimeError):
    pass


class IndexManager:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        *,
        persist: bool = True,
        extractor: EntityExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.persist = persist
        self.extractor = extractor or get_extractor()
        self.dir = Path(settings.index_dir)
        self.chunks: dict[str, Chunk] = {}

        if persist:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._check_manifest()
            self._load_chunks()
            store_path = None if settings.qdrant_url else self.dir / "qdrant"
            self.store = QdrantStore(embedder.dim, path=store_path, url=settings.qdrant_url)
        else:
            self.store = QdrantStore(embedder.dim)
        self.bm25 = BM25Index(list(self.chunks.values()))

    # -- persistence ---------------------------------------------------
    @property
    def _chunks_path(self) -> Path:
        return self.dir / "chunks.jsonl"

    @property
    def _manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def _check_manifest(self) -> None:
        if not self._manifest_path.exists():
            return
        manifest = json.loads(self._manifest_path.read_text())
        if manifest.get("embedder") != self.embedder.name or manifest.get("dim") != self.embedder.dim:
            raise IndexMismatchError(
                f"Index was built with embedder '{manifest.get('embedder')}' "
                f"(dim {manifest.get('dim')}) but the current embedder is "
                f"'{self.embedder.name}' (dim {self.embedder.dim}). Re-ingest with "
                "`oncorag ingest --rebuild` or switch ONCORAG_EMBEDDING_* back."
            )

    def _load_chunks(self) -> None:
        if not self._chunks_path.exists():
            return
        with open(self._chunks_path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    chunk = Chunk.model_validate_json(line)
                    self.chunks[chunk.chunk_id] = chunk

    def _save(self) -> None:
        if not self.persist:
            return
        tmp = self._chunks_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for chunk in self.chunks.values():
                fh.write(chunk.model_dump_json() + "\n")
        tmp.replace(self._chunks_path)  # atomic swap: a crash never leaves a half-written file
        self._manifest_path.write_text(
            json.dumps(
                {
                    "embedder": self.embedder.name,
                    "dim": self.embedder.dim,
                    "n_chunks": len(self.chunks),
                    "n_articles": len({c.pmid for c in self.chunks.values()}),
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )

    # -- ingestion -----------------------------------------------------
    def build_chunks(self, article: Article) -> list[Chunk]:
        chunks = chunk_article(
            article,
            max_words=self.settings.chunk_max_words,
            overlap_sentences=self.settings.chunk_overlap_sentences,
        )
        for chunk in chunks:
            ents = self.extractor.extract(f"{chunk.title}. {chunk.text}")
            chunk.genes, chunk.cancer_types, chunk.variants = (
                ents.genes,
                ents.cancer_types,
                ents.variants,
            )
        return chunks

    def add_articles(self, articles: Iterable[Article], *, batch_articles: int = 64) -> dict[str, int]:
        stats = {"articles": 0, "chunks": 0, "replaced_articles": 0, "skipped_no_abstract": 0}
        batch: list[Article] = []

        def flush() -> None:
            if not batch:
                return
            pmids = [a.pmid for a in batch]
            existing = {c.pmid for c in self.chunks.values()}
            stale = {p for p in pmids if p in existing}
            if stale:
                self.store.delete_pmids(sorted(stale))
                self.chunks = {k: v for k, v in self.chunks.items() if v.pmid not in stale}
                stats["replaced_articles"] += len(stale)

            new_chunks: list[Chunk] = []
            for art in batch:
                art_chunks = self.build_chunks(art)
                if not art_chunks:
                    stats["skipped_no_abstract"] += 1
                    continue
                new_chunks.extend(art_chunks)
                stats["articles"] += 1
            if new_chunks:
                vectors = self.embedder.embed_documents([c.embed_text for c in new_chunks])
                self.store.upsert(new_chunks, vectors)
                for c in new_chunks:
                    self.chunks[c.chunk_id] = c
                stats["chunks"] += len(new_chunks)
            batch.clear()

        for article in articles:
            batch.append(article)
            if len(batch) >= batch_articles:
                flush()
        flush()

        self.bm25 = BM25Index(list(self.chunks.values()))
        self._save()
        return stats

    # -- introspection -------------------------------------------------
    def stats(self) -> dict[str, object]:
        chunks = list(self.chunks.values())
        years = [c.year for c in chunks if c.year]
        gene_counts: dict[str, int] = {}
        for c in chunks:
            for g in c.genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1
        return {
            "embedder": self.embedder.name,
            "n_chunks": len(chunks),
            "n_articles": len({c.pmid for c in chunks}),
            "year_range": [min(years), max(years)] if years else None,
            "top_genes": sorted(gene_counts.items(), key=lambda kv: -kv[1])[:10],
        }

    def close(self) -> None:
        self.store.close()
