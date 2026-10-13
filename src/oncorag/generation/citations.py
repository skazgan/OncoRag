"""Citation parsing and post-hoc verification of generated answers.

Generation is never trusted blindly. After the model answers we check, per sentence:

1. every cited marker ``[n]`` refers to a context that was actually provided (no phantom sources);
2. every factual sentence carries at least one citation;
3. the sentence is lexically supported by the chunks it cites (content-term coverage), and every
   number it states also appears in those chunks (numbers are where hallucinations hurt most).

This is a cheap *lexical* groundedness check, not entailment: it catches fabricated sources,
unsupported numbers and off-topic sentences, but a fluent paraphrase that flips a conclusion can
pass. Production systems add an NLI model or LLM judge on top (see docs/design-decisions.md);
the check is intentionally a pluggable function.
"""

from __future__ import annotations

import re

from oncorag.generation.prompts import ABSTAIN_TOKEN
from oncorag.schemas import Citation, RetrievedChunk, SentenceSupport, VerificationReport
from oncorag.text import content_terms, split_sentences

_MARKER_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_NUMBER = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)?%?")
SUPPORT_THRESHOLD = 0.5


_TRAILING_MARKERS = re.compile(
    r"([.!?])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)(?=\S|$)"
)


def normalize_marker_placement(text: str) -> str:
    """Move markers written after terminal punctuation ("claim. [1] Next") to before it
    ("claim [1]. Next") so each marker stays attached to the sentence it supports."""

    def _swap(m: re.Match[str]) -> str:
        markers = "".join(re.findall(r"\[[^\]]+\]", m.group(2)))
        return f" {markers}{m.group(1)} "

    return re.sub(r"\s+", " ", _TRAILING_MARKERS.sub(_swap, text)).strip()


def extract_markers(text: str) -> list[int]:
    seen: list[int] = []
    for grp in _MARKER_GROUP.findall(text):
        for num in re.split(r"[,;]", grp):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


def strip_markers(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKER_GROUP.sub("", text)).strip()


def is_abstention(answer: str) -> bool:
    return answer.strip().upper().startswith(ABSTAIN_TOKEN)


def build_citations(answer: str, contexts: list[RetrievedChunk]) -> list[Citation]:
    """Only sources actually cited in the answer, in marker order."""
    out: list[Citation] = []
    for marker in extract_markers(answer):
        if 1 <= marker <= len(contexts):
            c = contexts[marker - 1].chunk
            out.append(
                Citation(
                    marker=marker,
                    pmid=c.pmid,
                    title=c.title,
                    journal=c.journal,
                    year=c.year,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/" if c.pmid.isdigit() else None,
                )
            )
    return out


def verify_answer(answer: str, contexts: list[RetrievedChunk]) -> VerificationReport:
    report = VerificationReport()
    if is_abstention(answer):
        return report

    n_ctx = len(contexts)
    answer = normalize_marker_placement(answer)
    all_markers = extract_markers(answer)
    report.cited_markers = [m for m in all_markers if 1 <= m <= n_ctx]
    report.invalid_markers = [m for m in all_markers if not 1 <= m <= n_ctx]

    for sentence in split_sentences(answer):
        markers = extract_markers(sentence)
        claim = strip_markers(sentence)
        terms = content_terms(claim)
        if len(terms) < 3:  # fragments such as "See below." are not factual claims
            continue
        valid = [m for m in markers if 1 <= m <= n_ctx]
        if not valid:
            report.uncited_sentences.append(claim)
            continue

        evidence_text = " ".join(f"{contexts[m - 1].chunk.title}. {contexts[m - 1].chunk.text}" for m in valid)
        evidence_terms = content_terms(evidence_text)
        support = len(terms & evidence_terms) / len(terms)

        evidence_numbers = set(_NUMBER.findall(evidence_text))
        missing = sorted(n for n in set(_NUMBER.findall(claim)) if n not in evidence_numbers)

        report.sentence_support.append(
            SentenceSupport(
                sentence=claim,
                markers=valid,
                support=round(support, 3),
                unsupported_numbers=missing,
                supported=support >= SUPPORT_THRESHOLD and not missing,
            )
        )
    return report
