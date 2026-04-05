"""Corporate governance data source loaders: SEC EDGAR.

Loads company tickers (exchange listings) and insider transactions
(Forms 3/4/5 — officer, director, and 10% owner relationships).

Data sources:
- SEC company tickers: https://www.sec.gov/files/company_tickers_exchange.json
- SEC insider transactions: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime, timezone

from attestdb.core.hashing import compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import BatchResult
from attestdb.infrastructure.bulk_loader import (
    _ingest_append_direct_fn as _ingest_append_direct,
    _safe_entity_id,
)

logger = logging.getLogger(__name__)

_NANO = 1_000_000_000

CORPGOV_CONFIDENCE = {
    "sec_filing": 0.95,        # SEC-mandated disclosure
    "exchange_listing": 0.95,  # Exchange-verified
    "insider_transaction": 0.90,  # Self-reported, SEC-filed
}


def _date_to_ns(date_str: str) -> int:
    """Convert YYYY-MM-DD to nanosecond timestamp."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        return int(dt.timestamp() * _NANO)
    except (ValueError, TypeError):
        return int(time.time() * _NANO)


def load_company_tickers(
    pipeline,
    path: str,
) -> BatchResult:
    """Load SEC company tickers with exchange listings.

    Expected format: JSON with "fields" and "data" arrays.
    Each row: [cik, name, ticker, exchange].
    """
    t0 = time.time()
    timestamp = int(t0 * _NANO)

    with open(path) as f:
        data = json.load(f)

    rows = data.get("data", [])
    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"
    confidence = CORPGOV_CONFIDENCE["exchange_listing"]

    # Build exchange entities and SIC industry entities
    exchanges_seen: set[str] = set()

    for row in rows:
        cik, name, ticker, exchange = row[0], row[1], row[2], row[3]
        if not name or not ticker:
            continue

        # Company entity
        company_eid = _safe_entity_id(
            normalize_entity_id(f"company_{cik}")
        )[0]
        ext = {"cik": cik, "ticker": ticker}
        if exchange:
            ext["exchange"] = exchange
        entities[company_eid] = ("company", name, json.dumps(ext))

        # Exchange entity
        if exchange:
            ex_eid = _safe_entity_id(
                normalize_entity_id(f"exchange_{exchange}")
            )[0]
            if exchange not in exchanges_seen:
                entities[ex_eid] = (
                    "exchange", exchange, json.dumps({"name": exchange})
                )
                exchanges_seen.add(exchange)

            # company → listed_on → exchange
            pred_id = "listed_on"
            pred_type = "relates_to"
            source_id = f"sec:ticker:{cik}"
            claim_id = compute_claim_id(
                company_eid, pred_id, ex_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    company_eid, pred_id, ex_eid
                )
                claim_rows.append((
                    company_eid, ex_eid,
                    claim_id, content_id, pred_id, pred_type,
                    confidence,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(
        pipeline, entities, claim_rows, timestamp
    )
    elapsed = time.time() - t0
    logger.info(
        "Company tickers: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_insider_transactions(
    pipeline,
    insider_dir: str,
    max_filings: int = 0,
) -> BatchResult:
    """Load SEC insider transactions (Forms 3/4/5).

    Reads SUBMISSION.tsv and REPORTINGOWNER.tsv from the extracted ZIP.
    Creates person→role→company claims with real filing dates as timestamps.
    """
    t0 = time.time()

    sub_path = os.path.join(insider_dir, "SUBMISSION.tsv")
    owner_path = os.path.join(insider_dir, "REPORTINGOWNER.tsv")

    if not os.path.exists(sub_path) or not os.path.exists(owner_path):
        logger.error("Missing SUBMISSION.tsv or REPORTINGOWNER.tsv in %s",
                      insider_dir)
        return BatchResult(ingested=0)

    # Read submissions: accession → (issuer_cik, issuer_name, filing_date)
    submissions: dict[str, tuple[str, str, str]] = {}
    with open(sub_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        count = 0
        for row in reader:
            accession = row.get("ACCESSION_NUMBER", "").strip()
            issuer_cik = row.get("ISSUERCIK", "").strip()
            issuer_name = row.get("ISSUERNAME", "").strip()
            filing_date = row.get("FILING_DATE", "").strip()
            ticker = row.get("ISSUERTRADINGSYMBOL", "").strip()

            if accession and issuer_cik:
                submissions[accession] = (
                    issuer_cik, issuer_name, filing_date, ticker
                )
            count += 1
            if max_filings > 0 and count >= max_filings:
                break

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"
    confidence = CORPGOV_CONFIDENCE["insider_transaction"]

    # Read reporting owners
    with open(owner_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accession = row.get("ACCESSION_NUMBER", "").strip()
            if accession not in submissions:
                continue

            issuer_cik, issuer_name, filing_date, ticker = submissions[accession]
            owner_cik = row.get("RPTOWNERCIK", "").strip()
            owner_name = row.get("RPTOWNERNAME", "").strip()
            relationship = row.get("RPTOWNER_RELATIONSHIP", "").strip()
            title = row.get("RPTOWNER_TITLE", "").strip()

            if not owner_name or not relationship:
                continue

            timestamp = _date_to_ns(filing_date)

            # Company entity
            company_eid = _safe_entity_id(
                normalize_entity_id(f"company_{issuer_cik}")
            )[0]
            comp_ext = {"cik": issuer_cik}
            if ticker:
                comp_ext["ticker"] = ticker
            entities[company_eid] = (
                "company", issuer_name or f"CIK {issuer_cik}",
                json.dumps(comp_ext),
            )

            # Person entity
            person_eid = _safe_entity_id(
                normalize_entity_id(f"person_{owner_cik or owner_name}")
            )[0]
            person_ext = {}
            if owner_cik:
                person_ext["cik"] = owner_cik
            if title:
                person_ext["title"] = title
            entities[person_eid] = (
                "person", owner_name, json.dumps(person_ext),
            )

            # Determine predicate from relationship field
            rel_lower = relationship.lower()
            if "director" in rel_lower:
                pred_id = "director_of"
            elif "officer" in rel_lower:
                pred_id = "officer_of"
            elif "10" in rel_lower or "ten" in rel_lower:
                pred_id = "major_shareholder_of"
            else:
                pred_id = "affiliated_with"
            pred_type = "relates_to"

            source_id = f"sec:form345:{accession}"
            claim_id = compute_claim_id(
                person_eid, pred_id, company_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    person_eid, pred_id, company_eid
                )
                claim_rows.append((
                    person_eid, company_eid,
                    claim_id, content_id, pred_id, pred_type,
                    confidence,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

    if not claim_rows:
        return BatchResult(ingested=0)

    latest_ts = max(r[14] for r in claim_rows)
    result = _ingest_append_direct(
        pipeline, entities, claim_rows, latest_ts
    )
    elapsed = time.time() - t0
    logger.info(
        "Insider transactions: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result
