"""PubMed access through NCBI E-utilities (esearch + efetch) and a robust XML parser.

Etiquette / reliability notes:
* NCBI allows 3 requests/s without an API key and 10/s with one; we throttle accordingly.
* Large result sets are fetched in batches via efetch POST (long PMID lists overflow GET URLs).
* Transient failures (429 / 5xx / network) are retried with exponential backoff.
* ``parse_pubmed_xml`` is a pure function so it is unit-tested against a fixture without network.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from oncorag.schemas import Article

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _text(elem: ET.Element | None) -> str:
    """Full text of an element including inline markup (<i>, <sup>, <sub>)."""
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def _year(article_el: ET.Element) -> int | None:
    for path in (
        ".//Article/Journal/JournalIssue/PubDate/Year",
        ".//Article/ArticleDate/Year",
        ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year",
    ):
        el = article_el.find(path)
        if el is not None and (el.text or "").strip().isdigit():
            return int(el.text.strip())  # type: ignore[union-attr]
    medline = article_el.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    if medline is not None and medline.text:
        digits = "".join(c if c.isdigit() else " " for c in medline.text).split()
        for d in digits:
            if len(d) == 4:
                return int(d)
    return None


def parse_pubmed_xml(xml_text: str | bytes) -> list[Article]:
    """Parse an efetch ``PubmedArticleSet``. Records without an abstract are skipped because
    there is nothing to retrieve from them."""
    root = ET.fromstring(xml_text)
    articles: list[Article] = []
    for node in root.findall(".//PubmedArticle"):
        pmid = _text(node.find(".//MedlineCitation/PMID"))
        title = _text(node.find(".//Article/ArticleTitle"))
        if not pmid or not title:
            continue

        sections: list[tuple[str, str]] = []
        for ab in node.findall(".//Article/Abstract/AbstractText"):
            text = _text(ab)
            if text:
                sections.append(((ab.get("Label") or "").strip(), text))
        if not sections:
            log.debug("PMID %s has no abstract; skipping", pmid)
            continue

        authors = []
        for au in node.findall(".//Article/AuthorList/Author"):
            last, init = _text(au.find("LastName")), _text(au.find("Initials"))
            if last:
                authors.append(f"{last} {init}".strip())

        doi = None
        for aid in node.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = _text(aid)

        articles.append(
            Article(
                pmid=pmid,
                title=title,
                abstract_sections=sections,
                journal=_text(node.find(".//Article/Journal/Title")),
                year=_year(node),
                authors=authors,
                mesh_terms=[_text(m) for m in node.findall(".//MeshHeadingList/MeshHeading/DescriptorName")],
                doi=doi,
            )
        )
    return articles


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class PubMedError(RuntimeError):
    """An NCBI request failed. Messages are written to be safe to print and share: they never
    contain the request URL, because NCBI credentials travel in the query string."""


class PubMedClient:
    def __init__(
        self,
        email: str | None = None,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        api_key = api_key or None  # treat "" as unset
        if api_key and api_key.startswith("sk-"):
            # Fail before any network call: NCBI keys are ~36 hex characters, whereas "sk-..."
            # is the shape of an LLM-provider secret that must never be sent to NCBI.
            raise PubMedError(
                "ONCORAG_NCBI_API_KEY looks like an LLM provider key (it starts with 'sk-'), "
                "not an NCBI key. Leave ONCORAG_NCBI_API_KEY empty unless you have a key from "
                "your NCBI account settings, and put your Anthropic key on ANTHROPIC_API_KEY."
            )
        self.email = email
        self.api_key = api_key
        self._min_interval = 1 / (10 if api_key else 3)
        self._last_call = 0.0
        self._client = client or httpx.Client(timeout=60.0)

    def _params(self, **extra: object) -> dict[str, object]:
        params: dict[str, object] = {"tool": "oncorag", **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, method: str, endpoint: str, params: dict[str, object]) -> httpx.Response:
        """Retrying request. Failures surface as ``PubMedError`` with a redacted message
        (``from None`` also drops the chained httpx error, whose text contains the full URL)."""
        try:
            return self._request_with_retry(method, endpoint, params)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            hint = ""
            if status in (400, 401, 403) and self.api_key:
                hint = " If you set ONCORAG_NCBI_API_KEY, check that it is a valid NCBI key, or leave it empty."
            elif status == 429:
                hint = " NCBI rate limit hit; set ONCORAG_NCBI_API_KEY to raise it."
            raise PubMedError(f"NCBI {endpoint} returned HTTP {status}.{hint}") from None
        except httpx.TransportError as exc:
            raise PubMedError(
                f"Could not reach NCBI ({type(exc).__name__}). Check your internet connection."
            ) from None

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _request_with_retry(
        self, method: str, endpoint: str, params: dict[str, object]
    ) -> httpx.Response:
        self._throttle()
        url = f"{EUTILS}/{endpoint}"
        resp = (
            self._client.post(url, data=params)
            if method == "POST"
            else self._client.get(url, params=params)
        )
        resp.raise_for_status()
        return resp

    def search(self, query: str, *, max_results: int = 200, min_year: int | None = None) -> list[str]:
        """Return PMIDs for a PubMed query, newest first."""
        pmids: list[str] = []
        page = 1000
        for start in range(0, max_results, page):
            params = self._params(
                db="pubmed",
                term=query,
                retmode="json",
                retstart=start,
                retmax=min(page, max_results - start),
                sort="pub_date",
            )
            if min_year:
                params.update(datetype="pdat", mindate=str(min_year), maxdate="3000")
            data = self._request("GET", "esearch.fcgi", params).json()["esearchresult"]
            ids = data.get("idlist", [])
            pmids.extend(ids)
            if len(ids) < min(page, max_results - start):
                break
        return pmids[:max_results]

    def fetch(self, pmids: list[str], *, batch_size: int = 200) -> Iterator[Article]:
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = self._params(db="pubmed", id=",".join(batch), retmode="xml", rettype="abstract")
            resp = self._request("POST", "efetch.fcgi", params)
            yield from parse_pubmed_xml(resp.content)

    def search_and_fetch(
        self, query: str, *, max_results: int = 200, min_year: int | None = None
    ) -> list[Article]:
        pmids = self.search(query, max_results=max_results, min_year=min_year)
        log.info("esearch returned %d PMIDs", len(pmids))
        return list(self.fetch(pmids))


# A sensible default query for the project's domain: recent, abstract-bearing literature on
# actionable cancer genes, biomarkers and targeted / immuno-therapy.
DEFAULT_QUERY = (
    '(("neoplasms"[MeSH Terms] OR cancer[Title/Abstract]) AND '
    "(mutation[Title/Abstract] OR genomic[Title/Abstract] OR biomarker[Title/Abstract] OR "
    '"targeted therapy"[Title/Abstract] OR immunotherapy[Title/Abstract])) AND '
    "hasabstract AND english[Language]"
)
