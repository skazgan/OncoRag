"""Sentence-window chunking of abstracts.

Design choices (see docs/design-decisions.md):
* Chunk on *sentence* boundaries, never mid-sentence, so every chunk is quotable evidence.
* Keep structured-abstract labels (BACKGROUND / METHODS / RESULTS / CONCLUSIONS) as metadata.
  A result sentence retrieved without knowing it came from a *preclinical* paper is dangerous.
* Overlap by whole sentences so a claim and its qualifier are rarely separated.
* Most abstracts are 150-300 words, so many articles yield only 1-2 chunks; the title is
  prepended at embedding time (``Chunk.embed_text``) rather than duplicated into ``text``.
"""

from __future__ import annotations

from oncorag.schemas import Article, Chunk
from oncorag.text import split_sentences


def _words(sentence: str) -> int:
    return len(sentence.split())


def chunk_article(
    article: Article,
    *,
    max_words: int = 170,
    overlap_sentences: int = 1,
    min_words: int = 30,
) -> list[Chunk]:
    sentences: list[tuple[str, str]] = [
        (label, sent)
        for label, text in article.abstract_sections
        for sent in split_sentences(text)
    ]
    if not sentences:
        return []

    groups: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    new_in_current = 0
    word_count = 0

    for item in sentences:
        w = _words(item[1])
        if current and word_count + w > max_words and new_in_current > 0:
            groups.append(current)
            current = current[-overlap_sentences:] if overlap_sentences else []
            word_count = sum(_words(s) for _, s in current)
            new_in_current = 0
        current.append(item)
        word_count += w
        new_in_current += 1

    if new_in_current > 0:
        # A short tail is merged into the previous chunk instead of becoming a useless fragment.
        tail_words = sum(_words(s) for _, s in current[-new_in_current:])
        if groups and tail_words < min_words:
            groups[-1] = groups[-1] + current[-new_in_current:]
        else:
            groups.append(current)

    chunks: list[Chunk] = []
    for position, group in enumerate(groups):
        chunks.append(
            Chunk(
                chunk_id=f"{article.pmid}:{position}",
                pmid=article.pmid,
                title=article.title,
                text=" ".join(s for _, s in group),
                section=group[0][0].title() if group[0][0] else "",
                journal=article.journal,
                year=article.year,
                position=position,
            )
        )
    return chunks
