"""Embedding backends behind a single interface.

* ``HashingEmbedder``: deterministic signed feature hashing of word uni/bi-grams. No model
  download, no randomness, runs anywhere. It is a *stand-in* so tests, CI and the offline demo
  work; semantically it is closer to bag-of-words than to a neural model, so real quality
  numbers must be produced with ``SentenceTransformerEmbedder``.
* ``SentenceTransformerEmbedder``: real dense embeddings (default BAAI/bge-small-en-v1.5).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from oncorag.config import Settings
from oncorag.text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@lru_cache(maxsize=500_000)
def _hash64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _embed_one(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False))
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, tf in feats.items():
            h = _hash64(feat)
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign * (1.0 + math.log(tf))
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


def embedding_dimension(model: object) -> int:
    """Output size of a SentenceTransformer. The method was renamed in sentence-transformers 5.x
    (the old name now emits a FutureWarning on every run), so prefer the new one when present."""
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension  # type: ignore[attr-defined]
    return int(fn())


class SentenceTransformerEmbedder:
    # bge models are trained with an instruction prefix on the *query* side only.
    _BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -e '.[ml]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = embedding_dimension(self._model)
        self._batch = batch_size
        self._prefix = self._BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                batch_size=self._batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._prefix + text])[0]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashingEmbedder(settings.embedding_dim)
