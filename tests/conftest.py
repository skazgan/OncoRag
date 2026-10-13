from __future__ import annotations

from pathlib import Path

import pytest

from oncorag.config import Settings
from oncorag.ingestion import load_articles_jsonl
from oncorag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
DEMO_CORPUS = ROOT / "data" / "sample" / "demo_corpus.jsonl"
DEMO_EVAL = ROOT / "data" / "eval" / "demo_eval.jsonl"
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path) -> Settings:
    # _env_file=None: tests must never depend on a developer's local .env
    return Settings(_env_file=None, index_dir=tmp_path / "idx")


@pytest.fixture(scope="session")
def demo_articles():
    return load_articles_jsonl(DEMO_CORPUS)


@pytest.fixture
def pipeline(demo_articles) -> RAGPipeline:
    """In-memory pipeline over the synthetic demo corpus (hashing embedder, lexical reranker,
    extractive generator): fully deterministic and offline."""
    pipe = RAGPipeline.from_settings(Settings(_env_file=None), persist=False)
    pipe.add_articles(demo_articles)
    yield pipe
    pipe.close()
