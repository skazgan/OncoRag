from oncorag.generation.citations import (
    build_citations,
    extract_markers,
    is_abstention,
    normalize_marker_placement,
    strip_markers,
    verify_answer,
)
from oncorag.schemas import Chunk, RetrievedChunk


def _ctx(*texts):
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"P{i}:0", pmid=f"P{i}", title=f"Title {i}", text=t, year=2022, journal="J"),
            score=1.0,
        )
        for i, t in enumerate(texts, start=1)
    ]


CTX = _ctx(
    "Sotorasib and adagrasib inhibit KRAS G12C in non-small cell lung cancer. Responses were not durable.",
    "A threshold of 10 mutations per megabase was used for the pembrolizumab approval.",
)


def test_marker_parsing_variants():
    assert extract_markers("claim [1]. other [2][3] and [1, 4]") == [1, 2, 3, 4]
    assert extract_markers("no markers (see Table 1)") == []
    assert strip_markers("claim [1][2] here") == "claim here"


def test_marker_normalisation_reattaches_trailing_markers():
    assert normalize_marker_placement("First claim. [1] Second claim. [2]") == "First claim [1]. Second claim [2]."
    assert normalize_marker_placement("Already fine [1]. Next [2].") == "Already fine [1]. Next [2]."


def test_clean_answer_passes():
    ans = "Sotorasib and adagrasib inhibit KRAS G12C in non-small cell lung cancer [1]."
    rep = verify_answer(ans, CTX)
    assert rep.is_clean and rep.support_rate == 1.0 and rep.cited_markers == [1]


def test_phantom_citation_is_flagged():
    rep = verify_answer("Sotorasib inhibits KRAS G12C in lung cancer [7].", CTX)
    assert rep.invalid_markers == [7]
    assert not rep.is_clean
    assert rep.uncited_sentences  # its only marker was invalid, so the sentence counts as uncited


def test_uncited_sentence_is_flagged():
    rep = verify_answer("Sotorasib inhibits KRAS G12C [1]. Resistance always develops within weeks.", CTX)
    assert rep.uncited_sentences == ["Resistance always develops within weeks."]


def test_fabricated_number_is_flagged_even_if_words_match():
    rep = verify_answer("A threshold of 20 mutations per megabase was used for the pembrolizumab approval [2].", CTX)
    s = rep.sentence_support[0]
    assert s.unsupported_numbers == ["20"] and not s.supported
    ok = verify_answer("A threshold of 10 mutations per megabase was used for the pembrolizumab approval [2].", CTX)
    assert ok.sentence_support[0].supported


def test_off_topic_claim_is_weakly_supported():
    rep = verify_answer("Aspirin reduces colorectal polyp formation in elderly smokers [1].", CTX)
    assert not rep.sentence_support[0].supported


def test_trailing_marker_style_is_handled_end_to_end():
    rep = verify_answer("Sotorasib and adagrasib inhibit KRAS G12C in non-small cell lung cancer. [1]", CTX)
    assert rep.is_clean


def test_abstention_detection_and_no_verification():
    assert is_abstention("INSUFFICIENT_EVIDENCE The sources do not cover this.")
    assert is_abstention("  insufficient_evidence: nothing found")
    assert not is_abstention("Sufficient evidence exists [1].")
    assert verify_answer("INSUFFICIENT_EVIDENCE nope", CTX).sentence_support == []


def test_build_citations_only_returns_cited_valid_sources():
    cites = build_citations("A [2]. B [1]. C [9].", CTX)
    assert [c.marker for c in cites] == [2, 1]
    assert cites[0].url is None  # non-numeric (synthetic) IDs never produce a dead PubMed link


def test_real_pmid_gets_pubmed_url():
    ctx = [RetrievedChunk(chunk=Chunk(chunk_id="123:0", pmid="123", title="T", text="x"), score=1.0)]
    assert build_citations("Claim [1].", ctx)[0].url == "https://pubmed.ncbi.nlm.nih.gov/123/"
