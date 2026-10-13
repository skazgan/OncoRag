import json

import pytest

from oncorag.config import Settings
from oncorag.embeddings import HashingEmbedder
from oncorag.generation.generator import ExtractiveGenerator, LLMGenerator
from oncorag.index.manager import IndexManager, IndexMismatchError
from oncorag.ingestion import load_articles_jsonl
from oncorag.pipeline import RAGPipeline
from oncorag.schemas import Article

from .conftest import DEMO_CORPUS


class FakeClient:
    """Stands in for an LLM: returns canned text and records what it was sent."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


def _llm_pipeline(demo_articles, reply):
    client = FakeClient(reply)
    pipe = RAGPipeline.from_settings(Settings(_env_file=None), persist=False,
                                     generator=LLMGenerator(client, name="fake"))
    pipe.add_articles(demo_articles)
    return pipe, client


# ---- ask() behaviour -------------------------------------------------------
def test_answerable_question_gets_cited_verified_answer(pipeline):
    ans = pipeline.ask("Why does BRAF V600E colorectal cancer not respond to BRAF inhibitor monotherapy?")
    assert not ans.abstained
    assert ans.citations and ans.citations[0].pmid == "SYN006" and ans.citations[0].url is None
    assert ans.verification and ans.verification.invalid_markers == []
    assert ans.contexts[0].chunk.pmid == "SYN006"
    assert {"retrieve_rerank_ms", "generate_ms", "verify_ms", "total_ms"} <= set(ans.timings_ms)


def test_off_topic_question_abstains_without_calling_the_generator(demo_articles):
    pipe, client = _llm_pipeline(demo_articles, "should never be used [1].")
    ans = pipe.ask("Which football team won the 2022 World Cup?")
    assert ans.abstained and ans.abstain_reason in {"low_evidence_score", "no_results"}
    assert client.calls == []  # the gate saved an LLM call


def test_abstention_gate_can_be_disabled(demo_articles):
    s = Settings(_env_file=None, abstain_below=None)
    pipe = RAGPipeline.from_settings(s, persist=False, generator=LLMGenerator(FakeClient("INSUFFICIENT_EVIDENCE x")))
    pipe.add_articles(demo_articles)
    ans = pipe.ask("Which football team won the 2022 World Cup?")
    assert ans.abstain_reason == "model_abstained"  # generator was reached this time


def test_llm_receives_numbered_sources_and_injection_guard(demo_articles):
    pipe, client = _llm_pipeline(demo_articles, "Sotorasib and adagrasib inhibit KRAS G12C [1].")
    pipe.ask("Which drugs inhibit KRAS G12C in lung cancer?")
    system, user = client.calls[0]
    assert "<context>" in user and "[1] (PMID SYN001" in user
    assert "untrusted" in system and "INSUFFICIENT_EVIDENCE" in system


def test_hallucinated_citation_from_the_model_is_caught(demo_articles):
    pipe, _ = _llm_pipeline(demo_articles, "Sotorasib inhibits KRAS G12C in lung cancer [9].")
    ans = pipe.ask("Which drugs inhibit KRAS G12C in lung cancer?")
    assert ans.verification.invalid_markers == [9] and not ans.verification.is_clean
    assert ans.citations == []  # phantom markers never become citations


def test_empty_index_abstains():
    pipe = RAGPipeline.from_settings(Settings(_env_file=None), persist=False)
    ans = pipe.ask("anything about KRAS?")
    assert ans.abstained and ans.abstain_reason == "no_results"


def test_extractive_generator_abstains_when_nothing_matches(pipeline):
    gen = ExtractiveGenerator()
    ctx = pipeline.retrieve("KRAS G12C")
    assert gen.generate("zebra quantum bicycle", ctx).startswith("INSUFFICIENT_EVIDENCE")


# ---- IndexManager ------------------------------------------------------------
def test_chunks_are_annotated_with_entities(pipeline):
    c = pipeline.index.chunks["SYN004:0"]
    assert {"EGFR", "MET", "KRAS", "BRAF", "ERBB2"} <= set(c.genes)
    assert "non-small cell lung cancer" in c.cancer_types and "C797S" in c.variants


def test_persistence_round_trip(settings, demo_articles):
    pipe = RAGPipeline.from_settings(settings)
    stats = pipe.add_articles(demo_articles)
    assert stats["articles"] == 28
    pipe.close()

    reopened = RAGPipeline.from_settings(settings)
    assert len(reopened.index.chunks) == stats["chunks"]
    assert reopened.index.store.count() == stats["chunks"]
    ans = reopened.ask("Which ABL1 mutation causes resistance to imatinib in CML?")
    assert ans.contexts[0].chunk.pmid == "SYN027"
    manifest = json.loads((settings.index_dir / "manifest.json").read_text())
    assert manifest["embedder"] == "hashing-384" and manifest["n_articles"] == 28
    reopened.close()


def test_reingesting_an_article_replaces_its_old_chunks(demo_articles):
    pipe = RAGPipeline.from_settings(Settings(_env_file=None), persist=False)
    pipe.add_articles(demo_articles)
    before = len(pipe.index.chunks)

    revised = Article(pmid="SYN001", title="Revised", abstract_sections=[("", "Completely different text about zebrafish models of melanoma. " * 3)])
    stats = pipe.add_articles([revised])
    assert stats["replaced_articles"] == 1
    assert len(pipe.index.chunks) == before  # same number of chunks, not before+1
    assert pipe.index.store.count() == before
    assert pipe.index.chunks["SYN001:0"].title == "Revised"
    assert pipe.retrieve("zebrafish melanoma")[0].chunk.pmid == "SYN001"
    assert all("sotorasib" not in c.text.lower() for c in pipe.index.chunks.values() if c.pmid == "SYN001")


def test_embedder_mismatch_is_rejected(settings, demo_articles):
    pipe = RAGPipeline.from_settings(settings)
    pipe.add_articles(demo_articles[:3])
    pipe.close()
    with pytest.raises(IndexMismatchError, match="hashing-384"):
        IndexManager(settings, HashingEmbedder(64))


def test_articles_without_abstract_are_counted_not_crashed(pipeline):
    stats = pipeline.add_articles([Article(pmid="X1", title="No abstract")])
    assert stats["skipped_no_abstract"] == 1 and stats["articles"] == 0


def test_load_articles_jsonl_reports_bad_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"pmid": "1", "title": "ok", "abstract": "text"}\n{"title": "no pmid"}\n')
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_articles_jsonl(bad)
    assert len(load_articles_jsonl(DEMO_CORPUS)) == 28


def test_shrinking_article_leaves_no_orphaned_vectors():
    """Deterministic IDs make same-size re-ingests overwrite in place, but when a revised article
    produces FEWER chunks the old trailing chunks must be deleted explicitly."""
    pipe = RAGPipeline.from_settings(Settings(_env_file=None, chunk_max_words=40), persist=False)
    long_text = " ".join(f"Sentence {i} discusses KRAS biology in considerable detail here." for i in range(20))
    pipe.add_articles([Article(pmid="L1", title="Long", abstract_sections=[("", long_text)])])
    n_before = sum(c.pmid == "L1" for c in pipe.index.chunks.values())
    assert n_before >= 3 and pipe.index.store.count() == n_before

    pipe.add_articles([Article(pmid="L1", title="Short", abstract_sections=[("", "One short revised sentence about BRAF only.")])])
    assert [c.chunk_id for c in pipe.index.chunks.values()] == ["L1:0"]
    assert pipe.index.store.count() == 1  # orphans "L1:1", "L1:2", ... are gone from the vector store
    hits = pipe.index.store.search(pipe.index.embedder.embed_query("KRAS biology"), limit=10)
    assert all(cid == "L1:0" for cid, _ in hits)
