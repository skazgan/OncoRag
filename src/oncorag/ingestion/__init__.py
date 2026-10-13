from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from oncorag.schemas import Article


def load_articles_jsonl(path: str | Path) -> list[Article]:
    """Load articles from JSONL. Accepts either ``abstract`` (plain string) or
    ``abstract_sections`` ([[label, text], ...])."""
    articles: list[Article] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if "abstract" in raw and "abstract_sections" not in raw:
                    raw["abstract_sections"] = [["", raw.pop("abstract")]]
                articles.append(Article.model_validate(raw))
            except Exception as exc:  # surface the exact line, don't fail silently
                raise ValueError(f"{path}:{line_no}: invalid article record: {exc}") from exc
    return articles


def save_articles_jsonl(articles: Iterable[Article], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for art in articles:
            fh.write(art.model_dump_json() + "\n")
            n += 1
    return n
