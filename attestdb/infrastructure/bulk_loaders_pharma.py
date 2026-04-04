"""Pharma safety data source loaders: FAERS, FDA safety communications, drug withdrawals.

All loaders consume real public data from FDA and produce claims using
the standard bulk insertion pipeline.

Data sources:
- FAERS: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
- FDA safety: https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts
- DailyMed labels: https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm
"""

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
    _safe_entity_id,
)

logger = logging.getLogger(__name__)

# FAERS quarterly data download base URL
FAERS_BASE_URL = "https://fis.fda.gov/content/Exports/"

# Confidence by evidence type (pharmacovigilance hierarchy)
PHARMA_CONFIDENCE = {
    "spontaneous_report": 0.55,
    "signal_detection": 0.75,
    "post_marketing_study": 0.85,
    "regulatory_communication": 0.90,
    "label_revision": 0.92,
    "regulatory_action": 0.95,
}

# FAERS outcome codes → severity
FAERS_OUTCOME = {
    "DE": "death",
    "LT": "life_threatening",
    "HO": "hospitalization",
    "DS": "disability",
    "CA": "congenital_anomaly",
    "RI": "required_intervention",
    "OT": "other",
}


def load_faers(
    pipeline,
    drug_path: str,
    reaction_path: str,
    max_reports: int = 0,
) -> BatchResult:
    """Load FAERS drug-adverse event pairs from quarterly ASCII files.

    Expects two files from a FAERS quarterly extract:
    - drug_path: DRUGyyQq.txt (drug names, roles, routes)
    - reaction_path: REACyyQq.txt (adverse event terms per primaryid)

    These are joined on primaryid to produce drug→adverse_event claims.
    Claims are aggregated to unique (drug, adverse_event) pairs with
    report count stored in payload.

    Args:
        pipeline: IngestionPipeline instance.
        drug_path: Path to FAERS DRUG file ($ delimited).
        reaction_path: Path to FAERS REAC file ($ delimited).
        max_reports: Maximum reports to process (0 = unlimited).

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    # Phase 1: Read DRUG file — map primaryid → set of drug names
    pid_drugs: dict[str, set[str]] = {}
    with open(drug_path, "r", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or row.get("PRIMARYID") or "").strip()
            drug_name = (row.get("drugname") or row.get("DRUGNAME") or "").strip()
            role = (row.get("role_cod") or row.get("ROLE_COD") or "").upper().strip()
            if not pid or not drug_name:
                continue
            # Only include primary suspect and secondary suspect drugs
            if role not in ("PS", "SS", ""):
                continue
            pid_drugs.setdefault(pid, set()).add(drug_name)

    logger.info("FAERS DRUG: %d reports with drug names", len(pid_drugs))

    # Phase 2: Read REAC file — map primaryid → set of adverse events
    pid_reactions: dict[str, set[str]] = {}
    with open(reaction_path, "r", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="$")
        for row in reader:
            pid = (row.get("primaryid") or row.get("PRIMARYID") or "").strip()
            pt = (row.get("pt") or row.get("PT") or "").strip()
            if not pid or not pt:
                continue
            pid_reactions.setdefault(pid, set()).add(pt)

    logger.info("FAERS REAC: %d reports with reactions", len(pid_reactions))

    # Phase 3: Join on primaryid → aggregate (drug, AE) pair counts
    pair_counts: dict[tuple[str, str], int] = {}
    processed = 0
    for pid, drugs in pid_drugs.items():
        if pid not in pid_reactions:
            continue
        reactions = pid_reactions[pid]
        for drug in drugs:
            for ae in reactions:
                pair = (drug.lower(), ae.lower())
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        processed += 1
        if max_reports and processed >= max_reports:
            break

    logger.info("FAERS: %d unique drug-AE pairs from %d reports", len(pair_counts), processed)

    # Phase 4: Build entities and claims
    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []

    for (drug_name, ae_name), report_count in pair_counts.items():
        drug_eid = normalize_entity_id(f"drug_{drug_name}")
        ae_eid = normalize_entity_id(f"adverse_event_{ae_name}")
        drug_eid, drug_ext = _safe_entity_id(drug_eid)
        ae_eid, ae_ext = _safe_entity_id(ae_eid)

        if drug_eid not in entities:
            entities[drug_eid] = ("drug", drug_name, json.dumps(drug_ext) if drug_ext else "{}")
        if ae_eid not in entities:
            entities[ae_eid] = ("adverse_event", ae_name, json.dumps(ae_ext) if ae_ext else "{}")

        pred_id = "has_adverse_event"
        source_id = f"faers:{drug_name}:{ae_name}:{report_count}"
        source_type = "database_import"

        claim_id = compute_claim_id(
            drug_eid, pred_id, ae_eid,
            source_id, source_type, timestamp,
        )
        content_id = compute_content_id(drug_eid, pred_id, ae_eid)

        # Confidence scales with report count (more reports = higher signal)
        # Base is spontaneous report level (0.55), up to 0.80 for 100+ reports
        confidence = min(0.80, PHARMA_CONFIDENCE["spontaneous_report"] + report_count * 0.002)

        claim_rows.append((
            drug_eid, ae_eid,
            claim_id, content_id, pred_id, "causes",
            confidence,
            source_type, source_id,
            "", "[]", "", "", "", timestamp, "active",
        ))

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "FAERS loaded: %d drug-AE claims, %d entities in %.1fs (%.0f claims/sec)",
        len(claim_rows), len(entities), elapsed,
        len(claim_rows) / elapsed if elapsed > 0 else 0,
    )
    return result


def load_fda_safety_communications(
    pipeline,
    path: str,
) -> BatchResult:
    """Load FDA safety communications from JSON/NDJSON file.

    Expected format (one JSON object per line):
    {
        "drug": "drug name",
        "safety_issue": "description of safety issue",
        "date": "YYYY-MM-DD",
        "type": "safety_communication|black_box_warning|label_change",
        "url": "https://..."
    }

    Produces claims: drug → has_safety_signal_for → adverse_event

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to NDJSON file of safety communications.

    Returns:
        BatchResult with ingestion counts.
    """
    t0 = time.time()
    timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claim_ids: set[str] = set()

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            drug = (rec.get("drug") or "").strip()
            issue = (rec.get("safety_issue") or "").strip()
            comm_type = (rec.get("type") or "safety_communication").strip()

            if not drug or not issue:
                continue

            drug_eid = normalize_entity_id(f"drug_{drug}")
            issue_eid = normalize_entity_id(f"safety_signal_{issue}")
            drug_eid, drug_ext = _safe_entity_id(drug_eid)
            issue_eid, issue_ext = _safe_entity_id(issue_eid)

            if drug_eid not in entities:
                entities[drug_eid] = ("drug", drug, json.dumps(drug_ext) if drug_ext else "{}")
            if issue_eid not in entities:
                entities[issue_eid] = ("safety_signal", issue, json.dumps(issue_ext) if issue_ext else "{}")

            pred_id = "has_safety_signal_for"
            if comm_type == "black_box_warning":
                pred_id = "has_black_box_warning"
            elif comm_type == "label_change":
                pred_id = "label_updated_to_include"

            source_id = f"fda_safety:{drug}:{issue}"
            source_type = "database_import"
            confidence = PHARMA_CONFIDENCE.get(
                "regulatory_communication" if "warning" in comm_type else "label_revision",
                0.90,
            )

            claim_id = compute_claim_id(
                drug_eid, pred_id, issue_eid,
                source_id, source_type, timestamp,
            )
            if claim_id in seen_claim_ids:
                continue
            seen_claim_ids.add(claim_id)

            content_id = compute_content_id(drug_eid, pred_id, issue_eid)
            claim_rows.append((
                drug_eid, issue_eid,
                claim_id, content_id, pred_id, "causes",
                confidence,
                source_type, source_id,
                "", "[]", "", "", "", timestamp, "active",
            ))

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "FDA safety: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_drug_withdrawals(
    pipeline,
    path: str,
) -> BatchResult:
    """Load drug withdrawal records from CSV.

    Expected CSV format (with header):
        drug, date, jurisdiction, reason, source_url

    Produces claims: drug → indication_withdrawn → reason
    with jurisdiction as namespace.

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to withdrawals CSV file.

    Returns:
        BatchResult with ingestion counts.
    """
    from datetime import datetime, timezone

    t0 = time.time()
    fallback_timestamp = int(t0 * 1_000_000_000)

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []

    with open(path, "r", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = (row.get("drug") or "").strip()
            reason = (row.get("reason") or "").strip()
            jurisdiction = (row.get("jurisdiction") or "").strip()
            date_str = (row.get("date") or "").strip()

            if not drug or not reason:
                continue

            # Use withdrawal date as claim timestamp for temporal analysis
            timestamp = fallback_timestamp
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    timestamp = int(dt.timestamp() * 1_000_000_000)
                except ValueError:
                    pass

            drug_eid = normalize_entity_id(f"drug_{drug}")
            reason_eid = normalize_entity_id(f"withdrawal_reason_{reason}")
            drug_eid, drug_ext = _safe_entity_id(drug_eid)
            reason_eid, reason_ext = _safe_entity_id(reason_eid)

            if drug_eid not in entities:
                entities[drug_eid] = ("drug", drug, json.dumps(drug_ext) if drug_ext else "{}")
            if reason_eid not in entities:
                entities[reason_eid] = ("adverse_event", reason, json.dumps(reason_ext) if reason_ext else "{}")

            pred_id = "indication_withdrawn"
            source_id = f"withdrawal:{drug}:{jurisdiction}:{reason}"
            source_type = "database_import"
            confidence = PHARMA_CONFIDENCE["regulatory_action"]

            claim_id = compute_claim_id(
                drug_eid, pred_id, reason_eid,
                source_id, source_type, timestamp,
            )
            content_id = compute_content_id(drug_eid, pred_id, reason_eid)

            # Use jurisdiction as namespace if available
            ns = normalize_entity_id(jurisdiction) if jurisdiction else ""

            claim_rows.append((
                drug_eid, reason_eid,
                claim_id, content_id, pred_id, "contradicts",
                confidence,
                source_type, source_id,
                ns, "[]", "", "", "", timestamp, "active",
            ))

    if not claim_rows:
        return BatchResult(ingested=0)

    result = _ingest_append_direct(pipeline, entities, claim_rows, timestamp)
    elapsed = time.time() - t0
    logger.info(
        "Drug withdrawals: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result
