"""Evaluation dataset format (JSONL, one example per line).

    {"id": "q01", "question": "...", "relevant_pmids": ["123", "456"],
     "answerable": true, "expected_keywords": ["sotorasib", "G12C"]}

``answerable: false`` marks questions the corpus cannot answer; the system is *expected to abstain*.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    id: str
    question: str
    relevant_pmids: list[str] = Field(default_factory=list)
    answerable: bool = True
    expected_keywords: list[str] = Field(default_factory=list)


def load_eval_set(path: str | Path) -> list[EvalExample]:
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                examples.append(EvalExample.model_validate_json(line))
    return examples


def save_eval_set(examples: list[EvalExample], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")
