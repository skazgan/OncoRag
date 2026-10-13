"""Bootstrap an evaluation set from *your own indexed corpus* with an LLM.

For a real PubMed corpus nobody has hand-labelled relevance judgements. A standard, defensible
approach is synthetic question generation: sample chunks, ask an LLM for a question that can only
be answered from that chunk, and treat the source paper as the relevant document. It is biased
toward lexical overlap with the source text (so absolute numbers are optimistic), which is why
the README reports it as a *relative* benchmark for ablations, and why a small hand-written gold
set should sit next to it.
"""

from __future__ import annotations

import json
import random
import re

from oncorag.evaluation.dataset import EvalExample
from oncorag.generation.generator import ChatClient
from oncorag.schemas import Chunk

_SYSTEM = (
    "You write evaluation questions for a biomedical retrieval system. Given one passage from a "
    "paper abstract, write ONE specific, self-contained question that this passage answers and "
    "that a researcher might plausibly ask. Do not mention 'the passage', 'the study' or 'the "
    "authors'; include the gene, variant, cancer type or drug if relevant. Return JSON only: "
    '{"question": "...", "keywords": ["2-4 words/terms that a correct answer must contain"]}'
)


def _parse(raw: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data.get("question"), str) else None


def generate_eval_set(
    chunks: list[Chunk], client: ChatClient, *, n: int = 50, seed: int = 7, min_words: int = 40
) -> list[EvalExample]:
    rng = random.Random(seed)
    eligible = [c for c in chunks if len(c.text.split()) >= min_words]
    # one question per article keeps the set from over-weighting prolific papers
    by_pmid: dict[str, list[Chunk]] = {}
    for c in eligible:
        by_pmid.setdefault(c.pmid, []).append(c)
    pmids = sorted(by_pmid)
    rng.shuffle(pmids)

    examples: list[EvalExample] = []
    for pmid in pmids:
        if len(examples) >= n:
            break
        chunk = rng.choice(by_pmid[pmid])
        data = _parse(client.complete(_SYSTEM, f"Title: {chunk.title}\nPassage: {chunk.text}"))
        if not data:
            continue
        kws = data.get("keywords") or []
        examples.append(
            EvalExample(
                id=f"syn{len(examples) + 1:03d}",
                question=str(data["question"]).strip(),
                relevant_pmids=[pmid],
                answerable=True,
                expected_keywords=[str(k) for k in kws][:4] if isinstance(kws, list) else [],
            )
        )
    return examples
