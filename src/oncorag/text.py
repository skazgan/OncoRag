"""Tokenisation and sentence splitting shared by chunking, BM25, reranking and verification."""

from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how if in
    into is it its may might of on or our s such than that the their then there these they this
    those to was we were what when where which who whom why will with within without would you
    your also both each other some more most many much very than via per versus vs among between
    during after before over under using used use""".split()  # noqa: SIM905
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@lru_cache(maxsize=200_000)
def _stem(token: str) -> str:
    """Very light plural stripping. Aggressive stemming hurts gene symbols and variants."""
    if any(ch.isdigit() for ch in token):
        return token  # variants / gene names such as g12c, pd-l1, ros1
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    # len > 4 keeps 4-letter gene symbols that end in "s" intact (KRAS, NRAS, HRAS)
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lowercase tokens. Hyphenated terms are emitted whole *and* split (``pd-l1`` -> ``pd-l1``,
    ``pd``, ``l1``) so both spellings match."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        parts = raw.split("-") if "-" in raw else []
        for tok in [raw, *parts]:
            if not tok:
                continue
            if not keep_stopwords and tok in STOPWORDS:
                continue
            out.append(_stem(tok))
    return out


def content_terms(text: str) -> set[str]:
    return set(tokenize(text))


_ABBREV_END = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|vs|fig|figs|approx|dr|cf|ref|refs|eq|spp|sp|vol|pp)\.)$",
    re.IGNORECASE,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter tuned for biomedical abstracts (abbreviations, p-values,
    gene symbols). Avoids a heavyweight NLP dependency."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = _SENT_SPLIT.split(text)
    sentences: list[str] = []
    for piece in pieces:
        if sentences and _ABBREV_END.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if s.strip()]
