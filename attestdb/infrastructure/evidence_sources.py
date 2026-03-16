"""Built-in evidence sources for the autodidact daemon.

Each source is a synchronous function: search_fn(query: str) -> str
that returns text evidence suitable for claim extraction.

Sources auto-register based on available API keys. Free sources
(PubMed, Semantic Scholar) always register. Paid sources register
when their API key is present in the environment.

Priority order (lower = tried first):
  0 — Perplexity Sonar (synthesized answers with citations, ~$1/1K queries)
  1 — PubMed (free, biomedical literature — authoritative for bio vertical)
  2 — Semantic Scholar (free, all academic fields)
  3 — Serper (Google results fallback, ~$1/1K queries)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.autodidact import AutodidactDaemon

logger = logging.getLogger(__name__)

# Request timeout in seconds
_TIMEOUT = 30


def register_default_sources(
    daemon: "AutodidactDaemon",
    env_path: str | None = None,
) -> list[str]:
    """Auto-register evidence sources based on available API keys.

    Reads keys from os.environ (which includes .env if loaded).
    Returns list of registered source names.

    Args:
        daemon: The AutodidactDaemon to register sources on.
        env_path: Optional path to .env file to load keys from.
    """
    if env_path:
        _load_env(env_path)

    registered = []

    # Perplexity Sonar — best value paid source (~$1/1K queries)
    # Returns synthesized answers with citations, ideal for claim extraction
    pplx_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if pplx_key:
        daemon.register_source(
            "perplexity",
            lambda q: search_perplexity(q, pplx_key),
            cost_per_call=0.001,
            priority=0,
        )
        registered.append("perplexity")

    # Free sources — always available
    daemon.register_source("pubmed", search_pubmed, priority=1)
    registered.append("pubmed")

    daemon.register_source("semantic_scholar", search_semantic_scholar, priority=2)
    registered.append("semantic_scholar")

    # Serper — Google fallback (raw snippets, needs extraction)
    serper_key = os.environ.get("SERPER_API_KEY", "")
    if serper_key:
        daemon.register_source(
            "serper",
            lambda q: search_serper(q, serper_key),
            cost_per_call=0.001,
            priority=3,
        )
        registered.append("serper")

    logger.info("Autodidact evidence sources: %s", registered)
    return registered


# ---------------------------------------------------------------------------
# Perplexity Sonar — OpenAI-compatible API with built-in search
# ---------------------------------------------------------------------------


def search_perplexity(query: str, api_key: str) -> str:
    """Search via Perplexity Sonar API.

    Returns a synthesized answer with inline citations. Uses the
    OpenAI-compatible chat completions endpoint. Sonar models do
    built-in web search and return grounded answers — ideal for
    feeding into claim extraction.

    Cost: ~$1 per 1,000 queries (sonar model).
    """
    data = json.dumps({
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Provide a detailed, factual "
                    "answer with specific claims, relationships, and evidence. "
                    "Include source references where possible."
                ),
            },
            {"role": "user", "content": query},
        ],
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        "https://api.perplexity.ai/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Perplexity search failed: %s", exc)
        return ""

    # Extract the assistant message
    choices = result.get("choices", [])
    if not choices:
        return ""

    content = choices[0].get("message", {}).get("content", "")

    # Append citations if present (Sonar returns these in the response)
    citations = result.get("citations", [])
    if citations:
        cite_lines = [f"[{i + 1}] {url}" for i, url in enumerate(citations)]
        content += "\n\nSources:\n" + "\n".join(cite_lines)

    return content


# ---------------------------------------------------------------------------
# Serper — Google search results
# ---------------------------------------------------------------------------


def search_serper(query: str, api_key: str) -> str:
    """Search via Serper (Google). Returns snippets as text."""
    data = json.dumps({"q": query, "num": 5}).encode()

    req = urllib.request.Request(
        "https://google.serper.dev/search",
        data=data,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            result = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Serper search failed: %s", exc)
        return ""

    parts = []

    # Answer box
    answer_box = result.get("answerBox", {})
    if answer_box.get("answer"):
        parts.append(answer_box["answer"])
    elif answer_box.get("snippet"):
        parts.append(answer_box["snippet"])

    # Organic results
    for item in result.get("organic", []):
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        if snippet:
            source_line = f" (source: {link})" if link else ""
            parts.append(f"{title}: {snippet}{source_line}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# PubMed — NCBI E-utilities (free, no API key required)
# ---------------------------------------------------------------------------


def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search PubMed and return titles + abstracts as text.

    Uses NCBI E-utilities: esearch -> efetch. No API key required
    (rate-limited to 3 req/sec without key, 10/sec with NCBI_API_KEY).
    """
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # Build search params
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "sort": "relevance",
    }
    ncbi_key = os.environ.get("NCBI_API_KEY", "")
    if ncbi_key:
        params["api_key"] = ncbi_key

    # Step 1: Search for PMIDs
    search_url = f"{base}/esearch.fcgi?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(search_url, timeout=_TIMEOUT) as resp:
            search_data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("PubMed search failed: %s", exc)
        return ""

    pmids = search_data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return ""

    # Step 2: Fetch abstracts for found PMIDs
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if ncbi_key:
        fetch_params["api_key"] = ncbi_key

    fetch_url = f"{base}/efetch.fcgi?{urllib.parse.urlencode(fetch_params)}"

    # Rate limit: small delay between search and fetch
    time.sleep(0.35)

    try:
        with urllib.request.urlopen(fetch_url, timeout=_TIMEOUT) as resp:
            xml_text = resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        logger.warning("PubMed fetch failed: %s", exc)
        return ""

    return _parse_pubmed_xml(xml_text)


def _parse_pubmed_xml(xml_text: str) -> str:
    """Parse PubMed XML and extract titles + abstracts."""
    parts = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""

    for article in root.findall(".//PubmedArticle"):
        # Title
        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None and title_el.text else ""

        # Abstract
        abstract_parts = []
        for abs_text in article.findall(".//AbstractText"):
            label = abs_text.get("Label", "")
            text = abs_text.text or ""
            if label and text:
                abstract_parts.append(f"{label}: {text}")
            elif text:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        # Journal + year
        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None and journal_el.text else ""
        year_el = article.find(".//PubDate/Year")
        year = year_el.text if year_el is not None and year_el.text else ""

        # PMID
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

        if title and abstract:
            source = f" ({journal}, {year}, PMID:{pmid})" if journal else ""
            parts.append(f"{title}{source}\n{abstract}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Semantic Scholar — Academic Graph API (free, no key required)
# ---------------------------------------------------------------------------


def search_semantic_scholar(query: str, max_results: int = 5) -> str:
    """Search Semantic Scholar and return titles + abstracts as text.

    Uses the S2 Academic Graph API. Free tier: 1 request/second.
    """
    params = {
        "query": query,
        "limit": str(max_results),
        "fields": "title,abstract,venue,year,citationCount,url",
    }

    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "AttestDB-Autodidact/0.1")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        logger.warning("Semantic Scholar search failed: %s", exc)
        return ""

    papers = data.get("data", [])
    if not papers:
        return ""

    parts = []
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        venue = paper.get("venue", "")
        year = paper.get("year", "")
        paper_url = paper.get("url", "")

        if title and abstract:
            source = f" ({venue}, {year})" if venue else ""
            source_url = f" {paper_url}" if paper_url else ""
            parts.append(f"{title}{source}{source_url}\n{abstract}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env(path: str) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = value
    except FileNotFoundError:
        pass
