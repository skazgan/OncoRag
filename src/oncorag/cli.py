"""`oncorag` command line interface."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from oncorag.config import Settings
from oncorag.ingestion import load_articles_jsonl
from oncorag.ingestion.entities import get_extractor
from oncorag.ingestion.pubmed import DEFAULT_QUERY, PubMedClient, PubMedError
from oncorag.pipeline import RAGPipeline, RetrievalOptions
from oncorag.retrieval.hybrid import RetrievalMode
from oncorag.schemas import SearchFilters

app = typer.Typer(help="OncoRAG: evidence-grounded RAG over cancer-genomics literature.", no_args_is_help=True)
console = Console()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CORPUS = REPO_ROOT / "data" / "sample" / "demo_corpus.jsonl"
DEMO_EVAL = REPO_ROOT / "data" / "eval" / "demo_eval.jsonl"


def _settings(index_dir: Path | None) -> Settings:
    s = Settings()
    if index_dir is not None:
        s.index_dir = index_dir
    return s


def _filters(genes: list[str] | None, cancers: list[str] | None, year_min: int | None) -> SearchFilters:
    ex = get_extractor()
    g = [ex.normalize_gene(x) or x for x in genes or []]
    c = [ex.normalize_cancer(x) or x.lower() for x in cancers or []]
    return SearchFilters(genes=g, cancer_types=c, year_min=year_min)


IndexDirOpt = Annotated[Path | None, typer.Option("--index-dir", help="Override ONCORAG_INDEX_DIR.")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


@app.command()
def ingest(
    query: Annotated[str, typer.Option(help="PubMed query (default: cancer genomics / targeted therapy).")] = DEFAULT_QUERY,
    max_results: Annotated[int, typer.Option(help="Maximum PMIDs to fetch.")] = 500,
    min_year: Annotated[int | None, typer.Option(help="Only articles published from this year.")] = None,
    from_file: Annotated[Path | None, typer.Option(help="Ingest a JSONL file instead of querying PubMed.")] = None,
    rebuild: Annotated[bool, typer.Option(help="Delete the existing index first.")] = False,
    index_dir: IndexDirOpt = None,
) -> None:
    """Fetch articles (PubMed or JSONL), chunk, embed and index them."""
    settings = _settings(index_dir)

    if from_file:
        articles = load_articles_jsonl(from_file)
        console.print(f"Loaded {len(articles)} articles from {from_file}")
    else:
        try:
            client = PubMedClient(settings.ncbi_email, settings.ncbi_api_key)
            with console.status("Querying PubMed..."):
                articles = client.search_and_fetch(query, max_results=max_results, min_year=min_year)
        except PubMedError as exc:
            console.print(f"[red]PubMed error:[/red] {exc}")
            raise typer.Exit(1) from None
        console.print(f"Fetched {len(articles)} articles with abstracts")

    # Only discard the old index once new data is safely in hand; a failed download must not
    # destroy a working index.
    if rebuild and settings.index_dir.exists():
        shutil.rmtree(settings.index_dir)

    pipeline = RAGPipeline.from_settings(settings)
    with console.status("Chunking, embedding and indexing..."):
        stats = pipeline.add_articles(articles)
    pipeline.close()
    console.print(f"[green]Indexed[/green] {stats}")


@app.command()
def demo(index_dir: IndexDirOpt = None, rebuild: bool = True) -> None:
    """Index the bundled SYNTHETIC demo corpus (offline; for trying the system, not for science)."""
    if not DEMO_CORPUS.exists():
        raise typer.BadParameter(f"Demo corpus not found at {DEMO_CORPUS}")
    ingest(from_file=DEMO_CORPUS, rebuild=rebuild, index_dir=index_dir)  # type: ignore[call-arg]
    console.print('Now try:  oncorag ask "Which drugs target KRAS G12C?"')


def _print_answer(ans) -> None:  # type: ignore[no-untyped-def]
    style = "yellow" if ans.abstained else "green"
    console.print(Panel(ans.answer, title="Answer" + (" (abstained)" if ans.abstained else ""), border_style=style))
    if ans.citations:
        table = Table(title="Sources", show_lines=False)
        table.add_column("#", justify="right")
        table.add_column("PMID")
        table.add_column("Title")
        table.add_column("Year", justify="right")
        for c in ans.citations:
            table.add_row(str(c.marker), c.pmid, c.title, str(c.year or ""))
        console.print(table)
    v = ans.verification
    if v:
        flags = []
        if v.invalid_markers:
            flags.append(f"invalid markers {v.invalid_markers}")
        if v.uncited_sentences:
            flags.append(f"{len(v.uncited_sentences)} uncited sentence(s)")
        weak = [s for s in v.sentence_support if not s.supported]
        if weak:
            flags.append(f"{len(weak)} weakly supported sentence(s)")
        console.print(
            f"Verification: support rate {v.support_rate:.0%}"
            + (f"  [red]{'; '.join(flags)}[/red]" if flags else "  [green]clean[/green]")
        )
    console.print(f"[dim]{ans.timings_ms}[/dim]")


@app.command()
def ask(
    question: str,
    gene: Annotated[list[str] | None, typer.Option("--gene", "-g", help="Filter by gene (repeatable).")] = None,
    cancer: Annotated[list[str] | None, typer.Option("--cancer", "-c", help="Filter by cancer type.")] = None,
    year_min: Annotated[int | None, typer.Option(help="Minimum publication year.")] = None,
    mode: Annotated[RetrievalMode, typer.Option(help="dense | sparse | hybrid")] = RetrievalMode.HYBRID,
    no_rerank: bool = False,
    top_k: int = 5,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    index_dir: IndexDirOpt = None,
) -> None:
    """Ask a question; get a cited, verified answer."""
    pipeline = RAGPipeline.from_settings(_settings(index_dir))
    if not pipeline.index.chunks:
        raise typer.BadParameter("Index is empty. Run `oncorag demo` or `oncorag ingest` first.")
    ans = pipeline.ask(
        question,
        filters=_filters(gene, cancer, year_min),
        options=RetrievalOptions(mode=mode, rerank=not no_rerank, top_k=top_k),
    )
    pipeline.close()
    if as_json:
        console.print_json(ans.model_dump_json())
    else:
        _print_answer(ans)


@app.command()
def search(
    question: str,
    mode: Annotated[RetrievalMode, typer.Option()] = RetrievalMode.HYBRID,
    no_rerank: bool = False,
    top_k: int = 5,
    index_dir: IndexDirOpt = None,
) -> None:
    """Retrieval only (no generation): see exactly what the retriever returns and why."""
    pipeline = RAGPipeline.from_settings(_settings(index_dir))
    results = pipeline.retrieve(
        question, options=RetrievalOptions(mode=mode, rerank=not no_rerank, top_k=top_k)
    )
    pipeline.close()
    table = Table(title=f"{mode.value} retrieval")
    for col in ("rank", "PMID", "score", "dense", "bm25", "title"):
        table.add_column(col)
    for i, r in enumerate(results, start=1):
        table.add_row(
            str(i), r.chunk.pmid, f"{r.score:.3f}", str(r.dense_rank or "-"), str(r.sparse_rank or "-"), r.chunk.title[:70]
        )
    console.print(table)


@app.command()
def stats(index_dir: IndexDirOpt = None) -> None:
    """Show index statistics."""
    pipeline = RAGPipeline.from_settings(_settings(index_dir))
    console.print_json(json.dumps(pipeline.index.stats()))
    pipeline.close()


@app.command("eval-generate")
def eval_generate(
    out: Annotated[Path, typer.Option(help="Where to write the JSONL eval set.")] = Path("data/eval/synthetic_eval.jsonl"),
    n: int = 50,
    index_dir: IndexDirOpt = None,
) -> None:
    """Create a synthetic eval set from the indexed corpus using the configured LLM."""
    from oncorag.evaluation.dataset import save_eval_set
    from oncorag.evaluation.synthetic_qa import generate_eval_set
    from oncorag.generation.generator import AnthropicClient

    settings = _settings(index_dir)
    try:
        client = AnthropicClient(settings.anthropic_api_key, settings.llm_model, 400, settings.llm_temperature)
    except RuntimeError as exc:  # missing key / missing extra: show the fix, not a traceback
        raise typer.BadParameter(str(exc)) from exc
    pipeline = RAGPipeline.from_settings(settings)
    examples = generate_eval_set(list(pipeline.index.chunks.values()), client, n=n)
    pipeline.close()
    save_eval_set(examples, out)
    console.print(f"[green]Wrote {len(examples)} questions to {out}[/green]")


@app.command("eval")
def evaluate(
    dataset: Annotated[Path, typer.Option(help="JSONL eval set.")] = DEMO_EVAL,
    k: int = 5,
    out_dir: Annotated[Path, typer.Option(help="Where to write the report.")] = Path("reports"),
    index_dir: IndexDirOpt = None,
) -> None:
    """Run the retrieval ablation and answer-quality evaluation."""
    from oncorag.evaluation.dataset import load_eval_set
    from oncorag.evaluation.runner import run_evaluation, to_markdown

    pipeline = RAGPipeline.from_settings(_settings(index_dir))
    report = run_evaluation(pipeline, load_eval_set(dataset), k=k)
    pipeline.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps(report, indent=2, default=str))
    md = to_markdown(report)
    (out_dir / f"eval-{stamp}.md").write_text(md)
    console.print(md)
    failures = report["answers"]["failures"]  # type: ignore[index]
    if failures:
        console.print("\n[yellow]Failure cases:[/yellow]")
        for f in failures:
            console.print(f"  {f}")


@app.command()
def calibrate(
    dataset: Annotated[Path, typer.Option(help="JSONL eval set with answerable and unanswerable questions.")] = DEMO_EVAL,
    index_dir: IndexDirOpt = None,
) -> None:
    """Sweep the abstention threshold and recommend ONCORAG_ABSTAIN_BELOW for this reranker/corpus."""
    from oncorag.evaluation.dataset import load_eval_set
    from oncorag.evaluation.runner import calibrate_abstention

    pipeline = RAGPipeline.from_settings(_settings(index_dir))
    if pipeline.reranker.name == "none":
        raise typer.BadParameter("Abstention needs reranker scores; set ONCORAG_RERANKER_BACKEND.")
    try:
        result = calibrate_abstention(pipeline, load_eval_set(dataset))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        pipeline.close()

    table = Table(title=f"Abstention sweep ({result['n_answerable']} answerable / {result['n_unanswerable']} unanswerable)")
    for col in ("threshold", "answerable kept", "unanswerable refused", "balanced acc."):
        table.add_column(col, justify="right")
    for r in result["sweep"]:  # type: ignore[union-attr]
        table.add_row(f"{r['threshold']:.2f}", f"{r['answerable_kept']:.2f}", f"{r['unanswerable_refused']:.2f}", f"{r['balanced_accuracy']:.2f}")
    console.print(table)
    console.print(
        f"Recommended: ONCORAG_ABSTAIN_BELOW={result['recommended_threshold']} "
        f"(balanced accuracy {result['best_balanced_accuracy']:.2f}, in-sample: validate on held-out questions)"
    )


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Start the FastAPI server (docs at /docs)."""
    import uvicorn

    uvicorn.run("oncorag.api:app_factory", host=host, port=port, factory=True, reload=reload)


if __name__ == "__main__":
    app()
