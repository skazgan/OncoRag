import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from oncorag.api import create_app
from oncorag.cli import app as cli_app
from oncorag.config import Settings
from oncorag.evaluation import metrics as m
from oncorag.evaluation.dataset import EvalExample, load_eval_set
from oncorag.evaluation.runner import calibrate_abstention, run_evaluation, to_markdown
from oncorag.evaluation.synthetic_qa import generate_eval_set
from oncorag.pipeline import RAGPipeline

from .conftest import DEMO_CORPUS, DEMO_EVAL


# ---- API ----------------------------------------------------------------------
@pytest.fixture
def client(pipeline):
    with TestClient(create_app(pipeline=pipeline)) as c:
        yield c


def test_health_and_stats(client):
    assert client.get("/health").json()["chunks"] == 28
    assert client.get("/stats").json()["n_articles"] == 28


def test_ask_endpoint_returns_structured_answer(client):
    r = client.post("/ask", json={"question": "Which drugs inhibit KRAS G12C in lung cancer?"})
    assert r.status_code == 200
    body = r.json()
    assert body["abstained"] is False and body["citations"][0]["pmid"] == "SYN001"
    assert body["verification"]["invalid_markers"] == []


def test_ask_supports_filters_and_modes(client):
    r = client.post("/ask", json={"question": "resistance mechanisms", "mode": "sparse", "rerank": False,
                                  "filters": {"genes": ["BRCA2"]}, "top_k": 3})
    assert r.status_code == 200
    assert all("BRCA2" in c["chunk"]["genes"] for c in r.json()["contexts"])


def test_search_endpoint_is_retrieval_only(client):
    r = client.post("/search", json={"question": "PARP inhibitors BRCA", "top_k": 2})
    assert r.status_code == 200 and len(r.json()) == 2 and "chunk" in r.json()[0]


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "valid question here", "top_k": 99},
                                     {"question": "valid question here", "mode": "magic"}])
def test_request_validation(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_empty_index_returns_503():
    empty = RAGPipeline.from_settings(Settings(_env_file=None), persist=False)
    with TestClient(create_app(pipeline=empty)) as c:
        assert c.post("/ask", json={"question": "anything at all"}).status_code == 503


# ---- Metrics -------------------------------------------------------------------
def test_ir_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], {"b", "d"}
    assert m.recall_at_k(ranked, rel, 2) == 0.5
    assert m.precision_at_k(ranked, rel, 4) == 0.5
    assert m.reciprocal_rank(ranked, rel) == 0.5
    assert m.hit_at_k(ranked, rel, 1) == 0.0
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3)
    assert m.ndcg_at_k(ranked, rel, 4) == pytest.approx((0.6309 + 0.4307) / (1 + 0.6309), abs=1e-3)
    assert m.dedupe(["a", "a", "b", "a"]) == ["a", "b"]


def test_percentile_and_bootstrap_ci():
    assert m.percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert m.percentile([], 0.9) == 0.0
    vals = [0.0, 1.0] * 20
    lo, hi = m.bootstrap_ci(vals)
    assert lo < 0.5 < hi and lo >= 0 and hi <= 1


# ---- Evaluation runner -----------------------------------------------------------
def test_full_evaluation_on_demo_set(pipeline):
    report = run_evaluation(pipeline, load_eval_set(DEMO_EVAL), k=5)
    by_name = {r["config"]: r for r in report["retrieval"]}
    assert len(by_name) == 5
    assert by_name["Hybrid + rerank"]["recall@5"] >= 0.9
    ans = report["answers"]
    assert ans["citation_validity"] == 1.0
    assert ans["abstain_precision"] is not None and ans["abstain_precision"] >= 0.75
    md = to_markdown(report)
    assert "| Hybrid + rerank |" in md and "Abstention on unanswerable" in md


def test_calibration_recommends_a_separating_threshold(pipeline):
    res = calibrate_abstention(pipeline, load_eval_set(DEMO_EVAL))
    assert 0.3 <= res["recommended_threshold"] <= 0.6
    assert res["best_balanced_accuracy"] >= 0.9


def test_synthetic_eval_generation_with_fake_llm(pipeline):
    class Fake:
        def complete(self, system, user):
            return 'Sure!\n{"question": "What does this passage say?", "keywords": ["kras", "egfr"]}'

    exs = generate_eval_set(list(pipeline.index.chunks.values()), Fake(), n=5, min_words=20)
    assert len(exs) == 5
    assert len({e.relevant_pmids[0] for e in exs}) == 5  # one question per article
    assert exs[0].expected_keywords == ["kras", "egfr"]

    class Broken:
        def complete(self, system, user):
            return "not json"

    with pytest.raises(RuntimeError, match="first 10 model replies"):
        generate_eval_set(list(pipeline.index.chunks.values()), Broken(), n=3)


def test_eval_example_roundtrip():
    ex = EvalExample(id="x", question="q", relevant_pmids=["1"], answerable=False)
    assert EvalExample.model_validate_json(ex.model_dump_json()) == ex


# ---- CLI -----------------------------------------------------------------------------
def test_cli_demo_ask_and_stats(tmp_path):
    runner = CliRunner()
    idx = str(tmp_path / "cli_idx")
    r = runner.invoke(cli_app, ["demo", "--index-dir", idx])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli_app, ["ask", "Which ABL1 mutation causes imatinib resistance in CML?", "--json", "--index-dir", idx])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["citations"][0]["pmid"] == "SYN027"
    r = runner.invoke(cli_app, ["ask", "KRAS", "--gene", "p53", "--index-dir", idx, "--json"])
    assert r.exit_code == 0  # alias "p53" is normalised to TP53 before filtering
    r = runner.invoke(cli_app, ["stats", "--index-dir", idx])
    assert "n_articles" in r.output


def test_cli_ask_on_empty_index_fails_cleanly(tmp_path):
    r = CliRunner().invoke(cli_app, ["ask", "question here", "--index-dir", str(tmp_path / "empty")])
    assert r.exit_code != 0 and "Index is empty" in r.output.replace("\n", " ")


def test_cli_ingest_from_file(tmp_path):
    r = CliRunner().invoke(cli_app, ["ingest", "--from-file", str(DEMO_CORPUS), "--index-dir", str(tmp_path / "i")])
    assert r.exit_code == 0 and "Indexed" in r.output


def test_cli_search_eval_and_calibrate(tmp_path):
    runner = CliRunner()
    idx = str(tmp_path / "idx")
    assert runner.invoke(cli_app, ["demo", "--index-dir", idx]).exit_code == 0

    r = runner.invoke(cli_app, ["search", "PARP inhibitor BRCA", "--index-dir", idx, "--top-k", "3"])
    assert r.exit_code == 0 and "SYN007" in r.output

    out = tmp_path / "reports"
    r = runner.invoke(cli_app, ["eval", "--index-dir", idx, "--out-dir", str(out)])
    assert r.exit_code == 0, r.output
    assert list(out.glob("eval-*.json")) and list(out.glob("eval-*.md"))
    report = json.loads(next(out.glob("eval-*.json")).read_text())
    assert report["retrieval"][0]["config"] == "Dense only"

    r = runner.invoke(cli_app, ["calibrate", "--index-dir", idx])
    assert r.exit_code == 0 and "Recommended: ONCORAG_ABSTAIN_BELOW=" in r.output.replace("\n", " ")


def test_calibration_requires_both_classes(pipeline):
    only_answerable = [e for e in load_eval_set(DEMO_EVAL) if e.answerable]
    with pytest.raises(ValueError, match="BOTH answerable and unanswerable"):
        calibrate_abstention(pipeline, only_answerable)


def test_cli_eval_generate_without_key_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ONCORAG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    r = CliRunner().invoke(cli_app, ["eval-generate", "--index-dir", str(tmp_path / "i")])
    assert r.exit_code != 0 and "ANTHROPIC_API_KEY" in r.output.replace("\n", " ")


def test_failed_pubmed_fetch_keeps_the_existing_index_and_leaks_no_key(tmp_path):
    """Regression: `ingest --rebuild` used to delete the index *before* downloading, and a
    misplaced Anthropic key in the NCBI slot ended up in the traceback."""
    runner = CliRunner()
    idx = tmp_path / "idx"
    assert runner.invoke(cli_app, ["demo", "--index-dir", str(idx)]).exit_code == 0
    before = (idx / "chunks.jsonl").read_text()

    leaked = "sk-ant-usr-TESTSECRET123"
    r = runner.invoke(cli_app, ["ingest", "--rebuild", "--index-dir", str(idx)],
                      env={"ONCORAG_NCBI_API_KEY": leaked})
    out = r.output.replace("\n", " ")
    assert r.exit_code == 1
    assert "PubMed error" in out and "LLM provider key" in out
    assert leaked not in r.output and "Traceback" not in r.output
    assert (idx / "chunks.jsonl").read_text() == before  # index survived the failed rebuild


def test_cli_ingest_uses_year_stratified_search_by_default(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import oncorag.cli as cli_mod
    from oncorag.ingestion import load_articles_jsonl

    captured = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def search_and_fetch(self, query, **kwargs):
            captured.update(kwargs)
            return load_articles_jsonl(DEMO_CORPUS)

    monkeypatch.setattr(cli_mod, "PubMedClient", FakeClient)
    runner = CliRunner()
    idx = str(tmp_path / "i")
    r = runner.invoke(cli_app, ["ingest", "--min-year", "2018", "--max-results", "50", "--index-dir", idx])
    assert r.exit_code == 0, r.output
    assert captured["by_year"] is True and captured["min_year"] == 2018
    assert captured["max_year"] == datetime.now(timezone.utc).year
    assert "By year:" in r.output and "2023: " in r.output  # transparency about the year spread

    r = runner.invoke(cli_app, ["ingest", "--min-year", "2018", "--newest-first", "--index-dir", idx])
    assert r.exit_code == 0 and captured["by_year"] is False

    r = runner.invoke(cli_app, ["ingest", "--index-dir", idx])  # no --min-year: nothing to stratify
    assert r.exit_code == 0 and captured["by_year"] is False and captured["max_year"] is None



# ---- empty / failed LLM output (regression: blank answer shown as "support rate 0% clean") ----------
class _BlankClient:
    def complete(self, system, user):
        return ""


def _failing_pipeline(demo_articles):
    from oncorag.generation.generator import LLMGenerator

    pipe = RAGPipeline.from_settings(Settings(_env_file=None), persist=False, generator=LLMGenerator(_BlankClient()))
    pipe.add_articles(demo_articles)
    return pipe


def test_empty_answer_is_never_reported_as_clean():
    from oncorag.generation.citations import verify_answer

    rep = verify_answer("", [])
    assert rep.sentence_support == [] and rep.is_clean is False


def test_api_returns_502_when_the_llm_gives_no_answer(demo_articles):
    with TestClient(create_app(pipeline=_failing_pipeline(demo_articles))) as c:
        r = c.post("/ask", json={"question": "Which drugs inhibit KRAS G12C in lung cancer?"})
    assert r.status_code == 502 and "LLM error" in r.json()["detail"]


def test_cli_ask_shows_a_clean_error_when_the_llm_gives_no_answer(demo_articles, monkeypatch):
    import oncorag.cli as cli_mod

    pipe = _failing_pipeline(demo_articles)
    monkeypatch.setattr(cli_mod.RAGPipeline, "from_settings", classmethod(lambda cls, *a, **k: pipe))
    r = CliRunner().invoke(cli_app, ["ask", "Which drugs inhibit KRAS G12C in lung cancer?"])
    assert r.exit_code == 1
    assert "LLM error" in r.output and "empty answer" in r.output.replace("\n", " ")
    assert "Traceback" not in r.output


def test_eval_records_generation_errors_and_keeps_going(demo_articles):
    from oncorag.evaluation.runner import evaluate_answers

    exs = [e for e in load_eval_set(DEMO_EVAL) if e.answerable][:3]
    res = evaluate_answers(_failing_pipeline(demo_articles), exs)
    assert [f["type"] for f in res["failures"]] == ["generation_error"] * 3
    assert res["answer_rate"] == 0.0


def test_eval_generate_reports_llm_failure_and_writes_nothing(tmp_path, monkeypatch):
    import oncorag.generation.generator as gen_mod
    from oncorag.generation.generator import LLMError

    class Boom:
        def __init__(self, *a, **k):
            pass

        def complete(self, system, user):
            raise LLMError("The 400-token limit was used up")

    monkeypatch.setattr(gen_mod, "AnthropicClient", Boom)
    runner = CliRunner()
    idx = str(tmp_path / "i")
    assert runner.invoke(cli_app, ["demo", "--index-dir", idx]).exit_code == 0
    out = tmp_path / "q.jsonl"
    r = runner.invoke(cli_app, ["eval-generate", "--index-dir", idx, "--out", str(out), "--n", "3"])
    assert r.exit_code == 1 and "Could not generate questions" in r.output and "Traceback" not in r.output
    assert not out.exists()
