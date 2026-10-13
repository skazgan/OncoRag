"""Evaluation runner: retrieval ablations and end-to-end answer quality.

Retrieval ablation (answerable questions only) compares configurations on the same queries:

    dense only | BM25 only | hybrid (RRF) | hybrid + rerank | hybrid + rerank + entity boost

so every component has to justify its complexity with numbers.

Answer-level metrics (all questions):
    citation_validity   share of answers with no phantom citation markers
    groundedness        mean share of sentences that pass the lexical support check
    keyword_coverage    share of expected keywords present in the answer (answerable only)
    abstain_precision   of the unanswerable questions, share the system correctly refused
    answer_rate         of the answerable questions, share the system attempted
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from oncorag.evaluation.dataset import EvalExample
from oncorag.evaluation.metrics import (
    bootstrap_ci,
    dedupe,
    hit_at_k,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from oncorag.pipeline import RAGPipeline, RetrievalOptions
from oncorag.retrieval.hybrid import RetrievalMode


@dataclass(frozen=True)
class AblationConfig:
    name: str
    mode: RetrievalMode
    rerank: bool
    entity_boost: float


def default_ablations(entity_boost: float = 0.5) -> list[AblationConfig]:
    return [
        AblationConfig("Dense only", RetrievalMode.DENSE, False, 0.0),
        AblationConfig("BM25 only", RetrievalMode.SPARSE, False, 0.0),
        AblationConfig("Hybrid (RRF)", RetrievalMode.HYBRID, False, 0.0),
        AblationConfig("Hybrid + rerank", RetrievalMode.HYBRID, True, 0.0),
        AblationConfig("Hybrid + rerank + entity boost", RetrievalMode.HYBRID, True, entity_boost),
    ]


def evaluate_retrieval(
    pipeline: RAGPipeline, examples: list[EvalExample], cfg: AblationConfig, k: int = 5
) -> dict[str, object]:
    answerable = [e for e in examples if e.answerable and e.relevant_pmids]
    per_q: dict[str, list[float]] = {"recall": [], "precision": [], "hit": [], "mrr": [], "ndcg": []}
    latencies: list[float] = []
    for ex in answerable:
        t0 = time.perf_counter()
        results = pipeline.retrieve(
            ex.question,
            options=RetrievalOptions(
                mode=cfg.mode, rerank=cfg.rerank, entity_boost=cfg.entity_boost, top_k=k * 2
            ),
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = dedupe(r.chunk.pmid for r in results)
        rel = set(ex.relevant_pmids)
        per_q["recall"].append(recall_at_k(ranked, rel, k))
        per_q["precision"].append(precision_at_k(ranked, rel, k))
        per_q["hit"].append(hit_at_k(ranked, rel, k))
        per_q["mrr"].append(reciprocal_rank(ranked, rel))
        per_q["ndcg"].append(ndcg_at_k(ranked, rel, k))

    return {
        "config": cfg.name,
        "n": len(answerable),
        f"recall@{k}": mean(per_q["recall"]),
        f"hit@{k}": mean(per_q["hit"]),
        f"precision@{k}": mean(per_q["precision"]),
        "mrr": mean(per_q["mrr"]),
        f"ndcg@{k}": mean(per_q["ndcg"]),
        "mrr_ci95": bootstrap_ci(per_q["mrr"]),
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
    }


def evaluate_answers(pipeline: RAGPipeline, examples: list[EvalExample]) -> dict[str, object]:
    validity: list[float] = []
    grounded: list[float] = []
    keyword_cov: list[float] = []
    answered_answerable: list[float] = []
    abstained_unanswerable: list[float] = []
    failures: list[dict[str, object]] = []

    for ex in examples:
        ans = pipeline.ask(ex.question)
        if not ex.answerable:
            abstained_unanswerable.append(1.0 if ans.abstained else 0.0)
            if not ans.abstained:
                failures.append({"id": ex.id, "type": "failed_to_abstain", "answer": ans.answer})
            continue

        answered_answerable.append(0.0 if ans.abstained else 1.0)
        if ans.abstained:
            failures.append({"id": ex.id, "type": "unnecessary_abstention", "reason": ans.abstain_reason})
            continue
        v = ans.verification
        validity.append(1.0 if v and not v.invalid_markers else 0.0)
        grounded.append(v.support_rate if v else 0.0)
        if ex.expected_keywords:
            text = ans.answer.lower()
            keyword_cov.append(
                sum(kw.lower() in text for kw in ex.expected_keywords) / len(ex.expected_keywords)
            )

    return {
        "n_answerable": sum(e.answerable for e in examples),
        "n_unanswerable": sum(not e.answerable for e in examples),
        "answer_rate": mean(answered_answerable),
        "citation_validity": mean(validity),
        "groundedness": mean(grounded),
        "keyword_coverage": mean(keyword_cov),
        "abstain_precision": mean(abstained_unanswerable) if abstained_unanswerable else None,
        "failures": failures,
    }


def calibrate_abstention(
    pipeline: RAGPipeline, examples: list[EvalExample], thresholds: list[float] | None = None
) -> dict[str, object]:
    """Sweep the abstention threshold on a labelled set.

    Reranker scores are only comparable within one reranker backend, so the threshold must be
    re-calibrated whenever the reranker or corpus changes (a cross-encoder's sigmoid scores live
    on a different scale from the lexical heuristic's). Returns the sweep plus a recommendation.
    """
    if not any(e.answerable for e in examples) or all(e.answerable for e in examples):
        raise ValueError(
            "Calibration needs BOTH answerable and unanswerable questions "
            '(add off-topic questions with "answerable": false to your eval set).'
        )
    scored: list[tuple[float, bool]] = []
    for ex in examples:
        results = pipeline.retrieve(ex.question)
        best = results[0].rerank_score if results and results[0].rerank_score is not None else 0.0
        scored.append((best, ex.answerable))

    n_ans = max(1, sum(a for _, a in scored))
    n_un = max(1, sum(not a for _, a in scored))
    grid = thresholds or [round(i * 0.05, 2) for i in range(0, 20)]
    rows = []
    for t in grid:
        kept = sum(1 for s, a in scored if a and s >= t) / n_ans
        refused = sum(1 for s, a in scored if not a and s < t) / n_un
        rows.append(
            {"threshold": t, "answerable_kept": kept, "unanswerable_refused": refused,
             "balanced_accuracy": (kept + refused) / 2}
        )
    best_bal = max(r["balanced_accuracy"] for r in rows)
    tied = [r["threshold"] for r in rows if r["balanced_accuracy"] == best_bal]
    return {
        "n_answerable": sum(a for _, a in scored),
        "n_unanswerable": sum(not a for _, a in scored),
        "sweep": rows,
        "recommended_threshold": tied[len(tied) // 2],
        "best_balanced_accuracy": best_bal,
    }


def run_evaluation(
    pipeline: RAGPipeline,
    examples: list[EvalExample],
    *,
    k: int = 5,
    ablations: list[AblationConfig] | None = None,
) -> dict[str, object]:
    ablations = ablations or default_ablations(pipeline.settings.entity_boost)
    return {
        "k": k,
        "embedder": pipeline.index.embedder.name,
        "reranker": pipeline.reranker.name,
        "generator": pipeline.generator.name,
        "corpus": pipeline.index.stats(),
        "retrieval": [evaluate_retrieval(pipeline, examples, cfg, k) for cfg in ablations],
        "answers": evaluate_answers(pipeline, examples),
    }


def to_markdown(report: dict[str, object]) -> str:
    k = report["k"]
    rows = report["retrieval"]
    corpus = report["corpus"]
    lines = [
        f"**Corpus:** {corpus['n_articles']} articles / {corpus['n_chunks']} chunks  "  # type: ignore[index]
        f"**Embedder:** `{report['embedder']}`  **Reranker:** `{report['reranker']}`  "
        f"**Generator:** `{report['generator']}`",
        "",
        f"| Configuration | Recall@{k} | Hit@{k} | Precision@{k} | MRR (95% CI) | nDCG@{k} | p50 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:  # type: ignore[union-attr]
        lo, hi = r["mrr_ci95"]
        lines.append(
            f"| {r['config']} | {r[f'recall@{k}']:.3f} | {r[f'hit@{k}']:.3f} | "
            f"{r[f'precision@{k}']:.3f} | {r['mrr']:.3f} ({lo:.2f}-{hi:.2f}) | "
            f"{r[f'ndcg@{k}']:.3f} | {r['latency_p50_ms']:.1f} |"
        )
    a = report["answers"]
    abstain = a["abstain_precision"]  # type: ignore[index]
    lines += [
        "",
        "| Answer-level metric | Value |",
        "|---|---|",
        f"| Answer rate (answerable) | {a['answer_rate']:.2f} |",  # type: ignore[index]
        f"| Citation validity | {a['citation_validity']:.2f} |",  # type: ignore[index]
        f"| Groundedness (lexical) | {a['groundedness']:.2f} |",  # type: ignore[index]
        f"| Keyword coverage | {a['keyword_coverage']:.2f} |",  # type: ignore[index]
        f"| Abstention on unanswerable | {'n/a' if abstain is None else f'{abstain:.2f}'} |",
    ]
    return "\n".join(lines)
