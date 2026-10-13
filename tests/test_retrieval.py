import numpy as np
import pytest

from oncorag.config import Settings
from oncorag.embeddings import HashingEmbedder
from oncorag.index.lexical import BM25Index
from oncorag.pipeline import RetrievalOptions
from oncorag.retrieval.hybrid import RetrievalMode, reciprocal_rank_fusion
from oncorag.schemas import Chunk, RetrievedChunk, SearchFilters


def _chunk(cid, text, genes=(), cancers=(), year=2020, title="t"):
    return Chunk(chunk_id=cid, pmid=cid.split(":")[0], title=title, text=text, year=year,
                 genes=list(genes), cancer_types=list(cancers))


# ---- RRF ---------------------------------------------------------------
def test_rrf_scores_and_ordering():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_rewards_agreement_over_a_single_high_rank():
    fused = reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]])
    assert max(fused, key=fused.get) == "shared"


def test_rrf_weights_and_validation():
    fused = reciprocal_rank_fusion([["a"], ["b"]], weights=[2.0, 1.0])
    assert fused["a"] > fused["b"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


# ---- BM25 --------------------------------------------------------------
def test_bm25_ranks_exact_variant_token_first():
    idx = BM25Index([
        _chunk("1:0", "EGFR T790M resistance mutation after first line therapy"),
        _chunk("2:0", "EGFR mutation positive lung cancer first line therapy outcomes"),
        _chunk("3:0", "unrelated text about immunotherapy biomarkers"),
    ])
    hits = idx.search("T790M", limit=3)
    assert hits[0][0] == "1:0" and len(hits) == 1  # zero-score docs are never returned


def test_bm25_idf_downweights_common_terms():
    idx = BM25Index([_chunk(f"{i}:0", "cancer therapy study") for i in range(10)]
                    + [_chunk("rare:0", "cancer therapy study sotorasib")])
    assert idx.search("cancer sotorasib", limit=1)[0][0] == "rare:0"


def test_bm25_filters_and_empty_cases():
    idx = BM25Index([
        _chunk("1:0", "kras inhibitor response", genes=["KRAS"], year=2018),
        _chunk("2:0", "kras inhibitor response", genes=["KRAS"], year=2023),
        _chunk("3:0", "kras inhibitor response", genes=["BRAF"], year=2023),
    ])
    hits = idx.search("kras inhibitor", 5, SearchFilters(genes=["KRAS"], year_min=2020))
    assert [h[0] for h in hits] == ["2:0"]
    assert BM25Index([]).search("anything", 5) == []
    assert idx.search("zzzz", 5) == []


# ---- Embedder ----------------------------------------------------------
def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = e.embed_query("KRAS G12C inhibitor"), e.embed_query("KRAS G12C inhibitor")
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
    docs = e.embed_documents(["KRAS G12C inhibitor in lung cancer", "radiotherapy fractionation schedule"])
    assert float(a @ docs[0]) > float(a @ docs[1])


# ---- Full hybrid retriever on the demo corpus -----------------------------
def test_hybrid_retrieves_the_right_paper(pipeline):
    res = pipeline.retrieve("Which drugs inhibit KRAS G12C in lung cancer?")
    assert res[0].chunk.pmid == "SYN001"
    assert res[0].dense_rank is not None or res[0].sparse_rank is not None


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_every_mode_runs(pipeline, mode):
    res = pipeline.retrieve("osimertinib resistance C797S", options=RetrievalOptions(mode=mode, rerank=False))
    assert res and res[0].chunk.pmid == "SYN004"


def test_sparse_mode_has_only_sparse_ranks_and_dense_only_dense(pipeline):
    sparse = pipeline.retrieve("PARP inhibitor", options=RetrievalOptions(mode=RetrievalMode.SPARSE, rerank=False))
    dense = pipeline.retrieve("PARP inhibitor", options=RetrievalOptions(mode=RetrievalMode.DENSE, rerank=False))
    assert all(r.dense_rank is None and r.sparse_rank for r in sparse)
    assert all(r.sparse_rank is None and r.dense_rank for r in dense)


def test_metadata_filters_are_applied_in_both_retrievers(pipeline):
    flt = SearchFilters(genes=["BRCA1"])
    for mode in (RetrievalMode.DENSE, RetrievalMode.SPARSE):
        res = pipeline.retrieve("resistance mechanisms", filters=flt,
                                options=RetrievalOptions(mode=mode, rerank=False, top_k=10))
        assert res and all("BRCA1" in r.chunk.genes for r in res)
    res = pipeline.retrieve("treatment", filters=SearchFilters(year_min=2023, year_max=2023),
                            options=RetrievalOptions(top_k=20))
    assert res and all(r.chunk.year == 2023 for r in res)


def test_entity_boost_promotes_chunks_that_match_query_entities(pipeline):
    plain = pipeline.retriever.retrieve("resistance in lung cancer", entity_boost=0.0, top_k=25)
    boosted = pipeline.retriever.retrieve("resistance in lung cancer", entity_boost=3.0, top_k=25)

    def lung_share(rs, n=5):
        return sum("lung cancer" in r.chunk.cancer_types for r in rs[:n]) / n

    assert lung_share(boosted) >= lung_share(plain)
    assert boosted[0].score >= plain[0].score  # multiplicative boost never lowers a score


def test_diversification_caps_chunks_per_article():
    from oncorag.pipeline import diversify

    mk = lambda cid: RetrievedChunk(chunk=_chunk(cid, "x"), score=1.0)  # noqa: E731
    out = diversify([mk("a:0"), mk("a:1"), mk("a:2"), mk("b:0")], top_k=4, max_per_article=2)
    assert [r.chunk.chunk_id for r in out] == ["a:0", "a:1", "b:0"]


def test_reranker_scores_are_bounded_and_sorted(pipeline):
    res = pipeline.retrieve("Why are MSI-high tumors sensitive to checkpoint inhibitors?")
    scores = [r.rerank_score for r in res]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)
    assert res[0].chunk.pmid == "SYN012"


def test_settings_do_not_leak_env_file(tmp_path):
    assert Settings(_env_file=None).embedding_backend == "hashing"
