from oncorag.ingestion.chunking import chunk_article
from oncorag.schemas import Article
from oncorag.text import split_sentences, tokenize


def test_sentence_split_keeps_abbreviations_together():
    text = "Smith et al. reported a response (e.g. in KRAS G12C tumors). Median PFS was 6.3 months."
    sents = split_sentences(text)
    assert len(sents) == 2
    assert sents[0].startswith("Smith et al. reported")


def test_sentence_split_does_not_break_on_decimals_or_pvalues():
    sents = split_sentences("The HR was 0.62 (95% CI, 0.5-0.8; P<0.05). Toxicity was manageable.")
    assert len(sents) == 2


def test_tokenize_emits_hyphen_parts_and_keeps_gene_symbols_intact():
    toks = tokenize("PD-L1 expression in KRAS and NRAS mutants")
    assert "pd-l1" in toks and "pd" in toks
    # 4-letter symbols ending in "s" must not be stemmed
    assert "kras" in toks and "nras" in toks
    assert "express" not in toks  # light stemming only strips plural "s"


def test_chunking_preserves_section_labels_and_ids():
    art = Article(
        pmid="1",
        title="T",
        abstract_sections=[("BACKGROUND", "Alpha beta gamma. Delta epsilon."), ("RESULTS", "Zeta eta theta.")],
    )
    chunks = chunk_article(art, max_words=200)
    assert [c.chunk_id for c in chunks] == ["1:0"]
    assert chunks[0].section == "Background"
    assert "Zeta eta theta." in chunks[0].text


def test_chunking_splits_long_abstracts_with_sentence_overlap():
    sentences = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    art = Article(pmid="2", title="T", abstract_sections=[("", " ".join(sentences))])
    chunks = chunk_article(art, max_words=40, overlap_sentences=1, min_words=5)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        last_of_a = split_sentences(a.text)[-1]
        assert last_of_a in b.text  # one-sentence overlap
    for c in chunks[:-1]:
        assert len(c.text.split()) <= 40


def test_chunking_merges_tiny_tail_instead_of_emitting_fragment():
    long_part = " ".join(f"Sentence {i} is made of eight whole words." for i in range(10))
    art = Article(pmid="3", title="T", abstract_sections=[("", long_part + " Tail.")])
    chunks = chunk_article(art, max_words=40, overlap_sentences=0, min_words=10)
    assert chunks[-1].text.endswith("Tail.")
    assert all(len(c.text.split()) >= 10 for c in chunks)


def test_chunking_empty_abstract_returns_nothing():
    assert chunk_article(Article(pmid="4", title="T", abstract_sections=[])) == []
