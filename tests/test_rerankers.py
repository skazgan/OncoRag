from oncorag.config import Settings
from oncorag.retrieval.rerank import LexicalReranker, NoopReranker, get_reranker
from oncorag.schemas import Chunk, RetrievedChunk


def _rc(cid, title, text, genes=()):
    return RetrievedChunk(chunk=Chunk(chunk_id=cid, pmid=cid, title=title, text=text, genes=list(genes)), score=0.0)


def test_noop_preserves_order_and_truncates():
    cands = [_rc("a", "t", "x"), _rc("b", "t", "y"), _rc("c", "t", "z")]
    assert [c.chunk.chunk_id for c in NoopReranker().rerank("q", cands, 2)] == ["a", "b"]


def test_lexical_reranker_prefers_full_term_coverage_and_handles_edges():
    rr = LexicalReranker()
    cands = [_rc("partial", "Other", "sotorasib trial design"),
             _rc("full", "Sotorasib in KRAS G12C", "sotorasib inhibits KRAS G12C", genes=["KRAS"])]
    out = rr.rerank("sotorasib KRAS G12C", cands, 2)
    assert out[0].chunk.chunk_id == "full" and out[0].rerank_score > out[1].rerank_score
    assert rr.rerank("anything", [], 3) == []
    assert rr.rerank("the of and", [_rc("a", "t", "text")], 1)[0].rerank_score == 0.0  # all stopwords


def test_factory_selects_backend():
    assert get_reranker(Settings(_env_file=None, reranker_backend="none")).name == "none"
    assert get_reranker(Settings(_env_file=None, reranker_backend="lexical")).name == "lexical"
