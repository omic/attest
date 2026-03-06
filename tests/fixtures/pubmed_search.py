"""PubMed search function for researcher evaluation.

Uses NCBI E-utilities (esearch + efetch) to retrieve PubMed abstracts.
No API key required for low-volume usage (< 3 requests/sec).
"""

from __future__ import annotations

import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_RATE_LIMIT_DELAY = 0.4  # seconds between calls


def pubmed_search_fn(question: str) -> str:
    """Search PubMed for abstracts relevant to a research question.

    Args:
        question: Natural-language research question.

    Returns:
        Concatenated abstract text with PMID markers for provenance extraction.
        Format: "--- PMID:12345 ---\\nTitle\\nAbstract text\\n"
    """
    # Step 1: Search for PMIDs
    search_params = urllib.parse.urlencode({
        "db": "pubmed",
        "term": question,
        "retmax": "3",
        "retmode": "xml",
    })
    search_url = f"{ESEARCH_URL}?{search_params}"

    try:
        with urllib.request.urlopen(search_url, timeout=15) as resp:
            search_xml = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("PubMed esearch failed: %s", e)
        return ""

    # Parse PMIDs from esearch response
    try:
        root = ET.fromstring(search_xml)
        pmids = [id_elem.text for id_elem in root.findall(".//IdList/Id") if id_elem.text]
    except ET.ParseError as e:
        logger.warning("Failed to parse esearch XML: %s", e)
        return ""

    if not pmids:
        return ""

    time.sleep(NCBI_RATE_LIMIT_DELAY)

    # Step 2: Fetch abstracts
    fetch_params = urllib.parse.urlencode({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "xml",
        "retmode": "xml",
    })
    fetch_url = f"{EFETCH_URL}?{fetch_params}"

    try:
        with urllib.request.urlopen(fetch_url, timeout=15) as resp:
            fetch_xml = resp.read().decode("utf-8")
    except Exception as e:
        logger.warning("PubMed efetch failed: %s", e)
        return ""

    # Parse abstracts
    parts = []
    try:
        root = ET.fromstring(fetch_xml)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "unknown"

            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            abstract_parts = []
            for abs_text in article.findall(".//AbstractText"):
                if abs_text.text:
                    abstract_parts.append(abs_text.text)
            abstract = " ".join(abstract_parts)

            if title or abstract:
                parts.append(f"--- PMID:{pmid} ---\n{title}\n{abstract}")
    except ET.ParseError as e:
        logger.warning("Failed to parse efetch XML: %s", e)
        return ""

    return "\n\n".join(parts)
