"""Central configuration. Every knob is an environment variable (prefix ``ONCORAG_``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ONCORAG_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Storage
    index_dir: Path = Path("index_store")
    qdrant_url: str | None = None  # None -> embedded local mode inside index_dir

    # Embeddings
    embedding_backend: Literal["hashing", "sentence-transformers"] = "hashing"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384  # only used by the hashing backend

    # Reranking
    reranker_backend: Literal["none", "lexical", "cross-encoder"] = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation
    llm_backend: Literal["extractive", "anthropic"] = "extractive"
    llm_model: str = "claude-sonnet-5-5"
    llm_max_tokens: int = 700
    llm_temperature: float | None = 0.0  # set to null/empty to omit the parameter
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "ONCORAG_ANTHROPIC_API_KEY"),
    )

    # PubMed / NCBI E-utilities
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None

    # Chunking
    chunk_max_words: int = 170
    chunk_overlap_sentences: int = 1

    # Retrieval depths
    top_k_dense: int = 40
    top_k_sparse: int = 40
    top_k_candidates: int = 25  # passed to the reranker
    top_k_final: int = 5  # contexts given to the generator
    rrf_k: int = 60
    entity_boost: float = 0.5
    max_chunks_per_article: int = 2

    # Abstention: if the best reranked score is below this, skip generation entirely.
    # None disables the gate. Scores are only comparable within one reranker backend, so
    # re-run `oncorag calibrate` when you change the reranker or corpus (0.45 was picked on the
    # bundled demo eval set with the lexical reranker; it is in-sample and not a universal value).
    abstain_below: float | None = 0.45
