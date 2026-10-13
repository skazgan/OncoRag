import json

import httpx
import pytest

from oncorag.ingestion.pubmed import PubMedClient, PubMedError, parse_pubmed_xml

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

    with pytest.raises(PubMedError, match="HTTP 400"):
        _mock_client(handler).search("x", max_results=10)
    assert attempts["n"] == 1


def test_errors_never_contain_the_url_or_credentials():
    """Regression: the raw httpx error embeds the full request URL, including api_key."""
    secret = "NCBI-SECRET-0123456789abcdef"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400)

    client = PubMedClient(email="me@example.org", api_key=secret,
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(PubMedError) as info:
        client.search("x", max_results=10)
    text = str(info.value)
    assert secret not in text and "api_key" not in text and "eutils" not in text
    assert "ONCORAG_NCBI_API_KEY" in text  # actionable hint for 400/401/403 with a key set
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_network_failure_is_a_clean_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(PubMedError, match="Could not reach NCBI"):
        _mock_client(handler).search("x", max_results=10)


@pytest.mark.parametrize("bad", ["sk-ant-api03-abc", "sk-proj-abc", "sk-anything"])
def test_llm_style_key_in_the_ncbi_slot_is_rejected_before_any_request(bad):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    with pytest.raises(PubMedError, match="LLM provider key") as info:
        PubMedClient(api_key=bad, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert bad not in str(info.value) and calls == []


def test_empty_api_key_is_treated_as_unset():
    c = PubMedClient(api_key="")
    assert c.api_key is None and "api_key" not in c._params()


def test_search_by_year_spreads_quota_across_years_and_dedupes():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.params
        requests.append(dict(p))
        year = int(p["mindate"])
        ids = [f"{year}{i:03d}" for i in range(int(p["retmax"]))]
        if year == 2021:
            ids[0] = "2020000"  # a PMID that also matched the previous year's window
        return httpx.Response(200, json={"esearchresult": {"idlist": ids}})

    pmids = _mock_client(handler).search_by_year("kras", total=100, min_year=2018, max_year=2022)
    assert [r["mindate"] for r in requests] == ["2018", "2019", "2020", "2021", "2022"]
    assert all(r["mindate"] == r["maxdate"] for r in requests)  # one calendar year per request
    assert all(r["retmax"] == "20" for r in requests)  # 100 / 5 years
    assert all(r["datetype"] == "pdat" and "sort" not in r for r in requests)  # relevance order
    assert len(pmids) == len(set(pmids)) == 99  # exactly one duplicate removed
    assert {p[:4] for p in pmids} == {"2018", "2019", "2020", "2021", "2022"}


def test_search_by_year_quota_rounds_up_and_result_is_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        year = int(request.url.params["mindate"])
        return httpx.Response(200, json={"esearchresult": {"idlist": [f"{year}{i:03d}" for i in range(int(request.url.params["retmax"]))]}})

    pmids = _mock_client(handler).search_by_year("x", total=10, min_year=2020, max_year=2022)  # 4 per year
    assert len(pmids) == 10


def test_search_by_year_validates_range_and_search_and_fetch_requires_both_years():
    c = _mock_client(lambda r: httpx.Response(200, json={"esearchresult": {"idlist": []}}))
    with pytest.raises(ValueError):
        c.search_by_year("x", total=10, min_year=2022, max_year=2020)
    with pytest.raises(ValueError, match="both min_year and max_year"):
        c.search_and_fetch("x", by_year=True, min_year=2020)


def test_newest_first_search_still_sorts_by_date_and_honours_max_year():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, json={"esearchresult": {"idlist": ["1"]}})

    _mock_client(handler).search("x", max_results=5, min_year=2019, max_year=2021)
    assert seen["sort"] == "pub_date" and seen["mindate"] == "2019" and seen["maxdate"] == "2021"
