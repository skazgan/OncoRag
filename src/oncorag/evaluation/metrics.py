"""Information-retrieval and answer-quality metrics (binary relevance, article-level).

Chunks are mapped to their PMID and de-duplicated in rank order before scoring: the question
"did we surface the right *paper*" is what a researcher cares about, and scoring chunks would
reward returning five chunks of the same paper.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def dedupe(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return sum(1 for d in top if d in relevant) / k if k else 0.0


def hit_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 13
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Small eval sets make point estimates misleading;
    reporting an interval keeps ablation claims honest."""
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=n)) for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot) - 1]
