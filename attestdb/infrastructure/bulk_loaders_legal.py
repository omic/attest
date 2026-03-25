"""Legal data source loaders: CourtListener, Harvard CAP, US Code, Federal Register."""

from __future__ import annotations

import csv
import json
import logging
import os
import time

from attestdb.core.confidence import tier1_confidence
from attestdb.core.hashing import compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import BatchResult
from attestdb.infrastructure.bulk_loader import (
    _ingest_append_direct_fn as _ingest_append_direct,
    _register_entities_fn as _register_entities,
    _safe_entity_id,
)

logger = logging.getLogger(__name__)

# CourtListener bulk data URL (requires API key for full access)
COURTLISTENER_CITATIONS_URL = "https://www.courtlistener.com/api/bulk-data/citations/all.csv.gz"

# Court ID → namespace mapping
COURT_NAMESPACE = {
    "scotus": "scotus",
    "ca1": "1st_circuit", "ca2": "2nd_circuit", "ca3": "3rd_circuit",
    "ca4": "4th_circuit", "ca5": "5th_circuit", "ca6": "6th_circuit",
    "ca7": "7th_circuit", "ca8": "8th_circuit", "ca9": "9th_circuit",
    "ca10": "10th_circuit", "ca11": "11th_circuit",
    "cadc": "dc_circuit", "cafc": "fed_circuit",
}

# Court confidence by level
COURT_CONFIDENCE = {
    "scotus": 0.95,
    "circuit": 0.85,
    "district": 0.70,
    "bankruptcy": 0.60,
    "state": 0.70,
}


def _court_confidence(court_id: str) -> float:
    """Map a court identifier to an authority confidence level."""
    court_lower = court_id.lower()
    if court_lower == "scotus":
        return COURT_CONFIDENCE["scotus"]
    if court_lower.startswith("ca") and court_lower in COURT_NAMESPACE:
        return COURT_CONFIDENCE["circuit"]
    if "dist" in court_lower or "d." in court_lower:
        return COURT_CONFIDENCE["district"]
    if "bankr" in court_lower:
        return COURT_CONFIDENCE["bankruptcy"]
    return COURT_CONFIDENCE.get("state", 0.70)


def _court_namespace(court_id: str) -> str:
    """Map a court identifier to a namespace."""
    return COURT_NAMESPACE.get(court_id.lower(), court_id.lower())


def load_courtlistener_citations(
    pipeline,
    path: str,
    max_cases: int = 0,
) -> BatchResult:
    """Load CourtListener citation graph from CSV.

    Expected CSV format (header row):
        citing_id, cited_id, citing_case_name, cited_case_name,
        citing_court, cited_court, citing_date, cited_date, depth

    Each row represents a case-to-case citation. Creates:
    - Case entities with court-appropriate confidence
    - Citation claims (citing_case → cites → cited_case)
    - Namespace assignment by court jurisdiction

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to the citations CSV file.
        max_cases: Maximum citations to load (0 = unlimited).

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_pairs: set[tuple[str, str]] = set()

    with open(path, "r", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_cases and i >= max_cases:
                break

            citing_id = row.get("citing_id", "").strip()
            cited_id = row.get("cited_id", "").strip()
            citing_name = row.get("citing_case_name", "").strip()
            cited_name = row.get("cited_case_name", "").strip()
            citing_court = row.get("citing_court", "").strip()
            cited_court = row.get("cited_court", "").strip()

            if not citing_id or not cited_id:
                continue

            # Skip self-citations
            if citing_id == cited_id:
                continue

            # Deduplicate
            pair = (citing_id, cited_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Build entity IDs
            citing_eid = normalize_entity_id(f"case_{citing_id}")
            cited_eid = normalize_entity_id(f"case_{cited_id}")
            citing_display = citing_name or f"Case {citing_id}"
            cited_display = cited_name or f"Case {cited_id}"

            # Register entities
            if citing_eid not in entities:
                entities[citing_eid] = ("entity", citing_display, "{}")
            if cited_eid not in entities:
                entities[cited_eid] = ("entity", cited_display, "{}")

            # Confidence from citing court
            confidence = _court_confidence(citing_court)

            # Source ID
            source_id = f"courtlistener:{citing_id}_cites_{cited_id}"
            source_type = "human_annotation"

            # Compute IDs
            pred_id = "associated_with"
            pred_type = "relates_to"
            claim_id = compute_claim_id(
                citing_eid, pred_id, cited_eid,
                source_id, source_type, timestamp,
            )
            content_id = compute_content_id(citing_eid, pred_id, cited_eid)

            claim_rows.append((
                citing_eid, cited_eid,
                claim_id, content_id, pred_id, pred_type,
                confidence,
                source_type, source_id,
                "", "[]", "", "", "", timestamp, "active",
            ))

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "CourtListener: %d citations, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_us_code_sections(
    pipeline,
    path: str,
    max_sections: int = 0,
) -> BatchResult:
    """Load US Code sections from JSON/NDJSON file.

    Expected format (one JSON object per line):
    {
        "title": 42,
        "section": "1983",
        "heading": "Civil action for deprivation of rights",
        "statute_at_large": "...",
        "amendment_refs": ["14th Amendment"],
        "url": "https://..."
    }

    This data can be extracted from govinfo.gov US Code XML bulk downloads
    or from Cornell LII structured data.

    Produces claims:
    - statute → implements → amendment (when amendment_refs present)
    - statute entity with title/section metadata

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to NDJSON file of US Code sections.
        max_sections: Maximum sections to load (0 = unlimited).

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claim_ids: set[str] = set()
    count = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_sections and count >= max_sections:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            title_num = rec.get("title", "")
            section = (rec.get("section") or "").strip()
            heading = (rec.get("heading") or "").strip()
            amendment_refs = rec.get("amendment_refs") or []

            if not title_num or not section:
                continue

            # Entity: statute
            statute_label = f"{title_num}_usc_{section}"
            statute_eid = normalize_entity_id(statute_label)
            display = heading or f"Title {title_num} § {section}"
            if statute_eid not in entities:
                ext = {"title": str(title_num), "section": section}
                entities[statute_eid] = ("statute", display, json.dumps(ext))

            source_type = "database_import"

            # Claims: statute → implements → amendment
            for amendment in amendment_refs:
                amendment = amendment.strip()
                if not amendment:
                    continue
                amend_eid = normalize_entity_id(amendment)
                if amend_eid not in entities:
                    entities[amend_eid] = ("amendment", amendment, "{}")

                pred_id = "implements"
                source_id = f"uscode:{title_num}:{section}:amend:{amendment}"
                confidence = 0.95  # enacted law

                claim_id = compute_claim_id(
                    statute_eid, pred_id, amend_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claim_ids:
                    seen_claim_ids.add(claim_id)
                    content_id = compute_content_id(statute_eid, pred_id, amend_eid)
                    claim_rows.append((
                        statute_eid, amend_eid,
                        claim_id, content_id, pred_id, "relates_to",
                        confidence,
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))

            count += 1

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "US Code: %d claims, %d entities from %d sections in %.1fs",
        len(claim_rows), len(entities), count, elapsed,
    )
    return result


def load_federal_register_legal(
    pipeline,
    path: str,
    max_rules: int = 0,
) -> BatchResult:
    """Load Federal Register legal/civil rights regulations from JSON/NDJSON.

    Same format as finreg's load_federal_register_rules but filtered to
    legal domain agencies (DOJ, EEOC, etc.) and producing legal entity types.

    Expected format (one JSON object per line):
    {
        "title": "rule title",
        "document_number": "2024-12345",
        "type": "Rule|Proposed Rule|Notice",
        "agencies": [{"name": "Department of Justice"}],
        "cfr_references": [{"title": 28, "part": 42}],
        "publication_date": "2024-01-15"
    }

    Produces claims: regulation → implements → statute (via CFR references)

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to NDJSON file of Federal Register documents.
        max_rules: Maximum rules to load (0 = unlimited).

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claim_ids: set[str] = set()
    count = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_rules and count >= max_rules:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = (rec.get("title") or "").strip()
            doc_number = (rec.get("document_number") or "").strip()
            cfr_refs = rec.get("cfr_references") or []

            if not title or not doc_number:
                continue

            # Entity: regulation
            reg_eid = normalize_entity_id(f"regulation_{doc_number}")
            if reg_eid not in entities:
                ext = {"document_number": doc_number}
                entities[reg_eid] = ("regulation", title[:100], json.dumps(ext))

            source_type = "database_import"
            confidence = 0.90  # federal regulation

            # Claims: regulation → implements → CFR part (as statute proxy)
            for cfr in cfr_refs:
                cfr_title = cfr.get("title", "")
                cfr_part = cfr.get("part", "")
                if not cfr_title or not cfr_part:
                    continue

                cfr_eid = normalize_entity_id(f"cfr_title_{cfr_title}_part_{cfr_part}")
                if cfr_eid not in entities:
                    entities[cfr_eid] = ("statute", f"CFR Title {cfr_title} Part {cfr_part}", "{}")

                pred_id = "implements"
                source_id = f"fedreg_legal:{doc_number}:cfr:{cfr_title}:{cfr_part}"
                claim_id = compute_claim_id(
                    reg_eid, pred_id, cfr_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claim_ids:
                    seen_claim_ids.add(claim_id)
                    content_id = compute_content_id(reg_eid, pred_id, cfr_eid)
                    claim_rows.append((
                        reg_eid, cfr_eid,
                        claim_id, content_id, pred_id, "relates_to",
                        confidence,
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))

            count += 1

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "Federal Register (legal): %d claims, %d entities from %d rules in %.1fs",
        len(claim_rows), len(entities), count, elapsed,
    )
    return result


def load_harvard_cap(
    pipeline,
    path: str,
    max_cases: int = 0,
) -> BatchResult:
    """Load case law from Harvard Caselaw Access Project metadata.

    Expected format: NDJSON where each line is a case from CasesMetadata.json
    with fields: id, name, name_abbreviation, decision_date, court,
    citations, cites_to, docket_number, jurisdiction.

    Download from: https://static.case.law/{reporter}/{volume}/CasesMetadata.json

    Produces claims:
    - case → cites → case (case-to-case citations via cites_to[].case_ids)
    - case → interprets → statute (case-to-statute citations via cites_to[].category)

    Confidence by court level: SCOTUS=0.95, Circuit=0.85, State supreme=0.80, other=0.70.

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to NDJSON file of CAP case metadata.
        max_cases: Maximum cases to process (0 = unlimited).

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claim_ids: set[str] = set()
    count = 0
    case_cite_count = 0
    statute_cite_count = 0

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_cases and count >= max_cases:
                break
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = case.get("id", "")
            name = (case.get("name_abbreviation") or case.get("name") or "").strip()
            court = case.get("court", {})
            court_name = court.get("name_abbreviation", court.get("name", ""))
            decision_date = (case.get("decision_date") or "").strip()
            citations = case.get("citations", [])
            cites_to = case.get("cites_to", [])

            if not case_id or not name:
                continue

            # Build case entity ID from official citation if available
            official_cite = ""
            for cit in citations:
                if cit.get("type") == "official":
                    official_cite = cit.get("cite", "")
                    break
            case_label = official_cite or f"cap_{case_id}"
            case_eid = normalize_entity_id(case_label)
            case_eid, case_ext = _safe_entity_id(case_eid)

            if case_eid not in entities:
                ext = dict(case_ext) if case_ext else {}
                ext["cap_id"] = str(case_id)
                if official_cite:
                    ext["citation"] = official_cite
                if decision_date:
                    ext["decision_date"] = decision_date
                entities[case_eid] = ("case", name[:100], json.dumps(ext))

            # Confidence by court type
            court_abbrev = court_name.lower()
            if "supreme court of the united states" in court_abbrev or court_abbrev == "u.s.":
                confidence = 0.95
            elif "circuit" in court_abbrev or court_abbrev.startswith("ca"):
                confidence = 0.85
            elif "supreme" in court_abbrev:
                confidence = 0.80  # state supreme courts
            else:
                confidence = 0.70

            source_type = "database_import"

            # Process citations
            for cite_ref in cites_to:
                category = cite_ref.get("category", "")
                cite_text = cite_ref.get("cite", "").strip()
                if not cite_text:
                    continue

                if "reporter" in category and cite_ref.get("case_ids"):
                    # Case-to-case citation
                    target_cite = cite_text
                    target_eid = normalize_entity_id(target_cite)
                    target_eid, target_ext = _safe_entity_id(target_eid)

                    if target_eid not in entities:
                        t_ext = dict(target_ext) if target_ext else {}
                        t_ext["citation"] = cite_text
                        if cite_ref.get("year"):
                            t_ext["year"] = str(cite_ref["year"])
                        entities[target_eid] = ("case", cite_text, json.dumps(t_ext))

                    pred_id = "cites"
                    source_id = f"cap:{case_id}:cites:{cite_text}"
                    claim_id = compute_claim_id(
                        case_eid, pred_id, target_eid,
                        source_id, source_type, timestamp,
                    )
                    if claim_id not in seen_claim_ids:
                        seen_claim_ids.add(claim_id)
                        content_id = compute_content_id(case_eid, pred_id, target_eid)
                        claim_rows.append((
                            case_eid, target_eid,
                            claim_id, content_id, pred_id, "relates_to",
                            confidence,
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        case_cite_count += 1

                elif "statute" in category:
                    # Case-to-statute citation
                    statute_eid = normalize_entity_id(cite_text)
                    statute_eid, statute_ext = _safe_entity_id(statute_eid)

                    if statute_eid not in entities:
                        s_ext = dict(statute_ext) if statute_ext else {}
                        s_ext["citation"] = cite_text
                        entities[statute_eid] = ("statute", cite_text, json.dumps(s_ext))

                    pred_id = "interprets"
                    source_id = f"cap:{case_id}:interprets:{cite_text}"
                    claim_id = compute_claim_id(
                        case_eid, pred_id, statute_eid,
                        source_id, source_type, timestamp,
                    )
                    if claim_id not in seen_claim_ids:
                        seen_claim_ids.add(claim_id)
                        content_id = compute_content_id(case_eid, pred_id, statute_eid)
                        claim_rows.append((
                            case_eid, statute_eid,
                            claim_id, content_id, pred_id, "relates_to",
                            confidence,
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp, "active",
                        ))
                        statute_cite_count += 1

            count += 1

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "Harvard CAP: %d claims (%d case-to-case, %d case-to-statute), "
        "%d entities from %d cases in %.1fs",
        len(claim_rows), case_cite_count, statute_cite_count,
        len(entities), count, elapsed,
    )
    return result
