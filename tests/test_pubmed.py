import json

import httpx
import pytest

from oncorag.ingestion.pubmed import PubMedClient, parse_pubmed_xml

from .conftest import FIXTURES


def test_parse_fixture():
    arts = parse_pubmed_xml((FIXTURES / "pubmed_sample.xml").read_bytes())
    # the abstract-less editorial is dropped
    assert [a.pmid for a in arts] == ["90000001", "90000002"]

    a = arts[0]
    assert a.title == "Resistance to KRAS G12C inhibition in colorectal cancer."  # <i> flattened
    assert a.year == 2023
    assert a.journal == "Journal of Synthetic Oncology"
    assert a.doi == "10.0000/synthetic.0001"
    assert [label for label, _ in a.abstract_sections] == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]
    assert "IC 50" in a.abstract_sections[2][1] or "IC50" in a.abstract_sections[2][1]
    assert "P<0.01" in a.abstract_sections[2][1]  # XML entity decoded
    assert a.authors == ["Yilmaz A", "Demir C"]  # collective authors skipped
    assert "Colorectal Neoplasms" in a.mesh_terms


def test_parse_medline_date_and_unstructured_abstract():
    b = parse_pubmed_xml((FIXTURES / "pubmed_sample.xml").read_bytes())[1]
    assert b.year == 2019
    assert b.abstract_sections[0][0] == ""


def _mock_client(handler) -> PubMedClient:
    return PubMedClient(
        email="me@example.org",
        api_key="k",  # api key -> 10 req/s throttle keeps tests fast
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_search_paginates_and_sends_etiquette_params():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        start = int(request.url.params["retstart"])
        ids = [str(i) for i in range(start, start + int(request.url.params["retmax"]))]
        return httpx.Response(200, json={"esearchresult": {"idlist": ids}})

    pmids = _mock_client(handler).search("kras", max_results=1500)
    assert len(pmids) == 1500 and pmids[0] == "0" and pmids[-1] == "1499"
    assert len(seen) == 2  # 1000 + 500
    assert seen[0]["tool"] == "oncorag" and seen[0]["email"] == "me@example.org"
    assert seen[0]["api_key"] == "k"


def test_search_stops_when_results_run_out():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"esearchresult": {"idlist": ["1", "2", "3"]}})

    assert _mock_client(handler).search("rare", max_results=5000) == ["1", "2", "3"]


def test_fetch_batches_via_post_and_parses():
    xml = (FIXTURES / "pubmed_sample.xml").read_bytes()
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=xml)

    arts = list(_mock_client(handler).fetch([str(i) for i in range(450)], batch_size=200))
    assert len(calls) == 3  # 200 + 200 + 50
    assert all(c.method == "POST" for c in calls)
    assert len(arts) == 6  # 2 parseable articles per mocked batch


def test_transient_5xx_is_retried_then_succeeds():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=json.dumps({"esearchresult": {"idlist": ["7"]}}))

    assert _mock_client(handler).search("x", max_results=10) == ["7"]
    assert attempts["n"] == 2


def test_client_errors_are_not_retried():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400)

    with pytest.raises(httpx.HTTPStatusError):
        _mock_client(handler).search("x", max_results=10)
    assert attempts["n"] == 1
