.PHONY: install install-ml test lint format demo eval calibrate serve ingest clean

PY ?= python

install:            ## dev install (offline-capable defaults)
	$(PY) -m pip install -e ".[dev,llm]"

install-ml:         ## add neural embeddings + cross-encoder reranker
	$(PY) -m pip install -e ".[dev,llm,ml]"

test:               ## run the test suite with coverage
	$(PY) -m pytest --cov=oncorag --cov-report=term-missing:skip-covered

lint:
	ruff check src tests

format:
	ruff check --fix src tests

demo:               ## index the bundled SYNTHETIC corpus and ask a question (fully offline)
	oncorag demo
	oncorag ask "Which drugs inhibit KRAS G12C in lung cancer?"

eval:               ## retrieval ablation + answer-quality metrics -> reports/
	oncorag eval

calibrate:          ## pick the abstention threshold for the current reranker/corpus
	oncorag calibrate

serve:              ## API on http://127.0.0.1:8000  (docs at /docs)
	oncorag serve

ingest:             ## real PubMed ingestion (needs internet; set ONCORAG_NCBI_EMAIL)
	oncorag ingest --max-results 2000 --min-year 2018

clean:
	rm -rf index_store .pytest_cache .ruff_cache .coverage htmlcov dist build
