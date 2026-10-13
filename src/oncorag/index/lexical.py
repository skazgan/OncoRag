"""BM25 implemented from scratch over an inverted index (numpy postings).

Why hand-rolled instead of ``rank_bm25``? Scoring only touches the postings of the query terms
(O(sum of posting lengths)) instead of looping over every document in Python, it supports
metadata masks, and it is ~60 lines that are easy to explain and unit-test.

BM25 matters in this domain: exact tokens such as ``KRAS``, ``G12C`` or ``T790M`` are precisely
where dense embeddings are weakest and lexical match is strongest.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from oncorag.schemas import Chunk, SearchFilters
from oncorag.text import tokenize


def chunk_matches(chunk: Chunk, filters: SearchFilters | None) -> bool:
    if filters is None or filters.is_empty:
        return True
    if filters.genes and not set(filters.genes) & set(chunk.genes):
        return False
    if filters.cancer_types and not set(filters.cancer_types) & set(chunk.cancer_types):
        return False
    if filters.year_min and (chunk.year is None or chunk.year < filters.year_min):
        return False
    return not (filters.year_max and (chunk.year is None or chunk.year > filters.year_max))


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks = list(chunks)
        self.ids = [c.chunk_id for c in self.chunks]
        n = len(self.chunks)

        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len = np.zeros(n, dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            toks = tokenize(chunk.embed_text)
            doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((i, tf))

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if n else 0.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for term, plist in postings.items():
            idx = np.fromiter((d for d, _ in plist), dtype=np.int64, count=len(plist))
            tfs = np.fromiter((t for _, t in plist), dtype=np.float32, count=len(plist))
            self.postings[term] = (idx, tfs)
            df = len(plist)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def __len__(self) -> int:
        return len(self.chunks)

    def search(
        self, query: str, limit: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idx, tf = posting
            scores[idx] += self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + norm[idx])

        if filters is not None and not filters.is_empty:
            mask = np.fromiter(
                (chunk_matches(c, filters) for c in self.chunks), dtype=bool, count=len(self.chunks)
            )
            scores = np.where(mask, scores, 0.0)

        top = np.argpartition(-scores, min(limit, len(scores) - 1))[:limit]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0.0]
