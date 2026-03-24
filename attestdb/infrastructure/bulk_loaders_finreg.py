"""Financial regulation data source loaders: SEC EDGAR, Federal Register.

All loaders consume real public data from US regulatory agencies and produce
claims using the standard bulk insertion pipeline.

Data sources:
- SEC EDGAR: https://efts.sec.gov/LATEST/search-index?q=%22enforcement%22
- Federal Register: https://www.federalregister.gov/api/v1
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

# Federal Register API (no key required)
FEDERAL_REGISTER_API = "https://www.federalregister.gov/api/v1"

# Confidence by source authority
FINREG_CONFIDENCE = {
    "enacted_law": 0.95,
    "final_rule": 0.90,
    "enforcement_order": 0.95,
    "proposed_rule": 0.70,
    "regulatory_guidance": 0.80,
    "stress_test": 0.85,
}

# SEC enforcement action types
SEC_ACTION_TYPES = {
    "administrative": "enforcement_action",
    "civil": "enforcement_action",
    "litigation": "enforcement_action",
    "trading_suspension": "regulatory_action",
    "delinquent_filing": "enforcement_action",
}


def load_sec_enforcement(
    pipeline,
    path: str,
    max_actions: int = 0,
) -> BatchResult:
    """Load SEC enforcement actions from JSON/NDJSON file.

    Expected format (one JSON object per line):
    {
        "respondent": "institution name",
        "date": "YYYY-MM-DD",
        "action_type": "administrative|civil|litigation",
        "violations": ["violation description", ...],
        "penalty_amount": 1000000,
        "release_number": "LR-12345",
        "url": "https://..."
    }

    Produces claims:
    - enforcement_action → results_in → institution
    - institution → violates → regulation (if violation references a rule)

    Args:
        pipeline: IngestionPipeline instance.
        path: Path to NDJSON file of enforcement actions.
        max_actions: Maximum actions to load (0 = unlimited).

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
            if max_actions and count >= max_actions:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            respondent = (rec.get("respondent") or "").strip()
            action_type = (rec.get("action_type") or "enforcement").strip()
            violations = rec.get("violations") or []
            penalty = rec.get("penalty_amount", 0)
            release = (rec.get("release_number") or "").strip()

            if not respondent:
                continue

            # Entity: institution
            inst_eid = normalize_entity_id(f"institution_{respondent}")
            inst_eid, inst_ext = _safe_entity_id(inst_eid)
            if inst_eid not in entities:
                entities[inst_eid] = ("institution", respondent, json.dumps(inst_ext) if inst_ext else "{}")

            # Entity: enforcement action
            action_id = release or f"sec_action_{count}"
            action_eid = normalize_entity_id(f"enforcement_{action_id}")
            if action_eid not in entities:
                ext = {"release_number": release} if release else {}
                entities[action_eid] = ("enforcement_action", f"SEC {action_type}: {respondent}", json.dumps(ext))

            # Claim: enforcement → results_in → institution
            pred_id = "results_in"
            source_id = f"sec_enforcement:{release or respondent}"
            source_type = "database_import"
            confidence = FINREG_CONFIDENCE["enforcement_order"]

            claim_id = compute_claim_id(
                action_eid, pred_id, inst_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claim_ids:
                seen_claim_ids.add(claim_id)
                content_id = compute_content_id(action_eid, pred_id, inst_eid)
                claim_rows.append((
                    action_eid, inst_eid,
                    claim_id, content_id, pred_id, "relates_to",
                    confidence,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

            # Claims for each violation
            for violation in violations:
                violation = violation.strip()
                if not violation:
                    continue
                viol_eid = normalize_entity_id(f"regulation_{violation}")
                viol_eid, viol_ext = _safe_entity_id(viol_eid)
                if viol_eid not in entities:
                    entities[viol_eid] = ("regulation", violation, json.dumps(viol_ext) if viol_ext else "{}")

                pred_id = "violates"
                source_id = f"sec_enforcement:{release or respondent}:{violation}"
                claim_id = compute_claim_id(
                    inst_eid, pred_id, viol_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claim_ids:
                    seen_claim_ids.add(claim_id)
                    content_id = compute_content_id(inst_eid, pred_id, viol_eid)
                    claim_rows.append((
                        inst_eid, viol_eid,
                        claim_id, content_id, pred_id, "contradicts",
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
        "SEC enforcement: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_federal_register_rules(
    pipeline,
    path: str,
    max_rules: int = 0,
) -> BatchResult:
    """Load Federal Register rules from JSON/NDJSON file.

    Expected format (one JSON object per line):
    {
        "title": "rule title",
        "document_number": "2024-12345",
        "type": "Rule|Proposed Rule|Notice",
        "agencies": [{"name": "Securities and Exchange Commission"}],
        "cfr_references": [{"title": 17, "part": 240}],
        "publication_date": "2024-01-15",
        "abstract": "...",
        "regulation_id_numbers": ["3235-AM12"]
    }

    This format matches the Federal Register API response
    (https://www.federalregister.gov/api/v1/documents).

    Produces claims:
    - regulation → implements → standard (via CFR references)
    - agency → is_supervised_by → regulation (agency publishes rule)

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
            doc_type = (rec.get("type") or "Rule").strip()
            agencies = rec.get("agencies") or []
            cfr_refs = rec.get("cfr_references") or []

            if not title or not doc_number:
                continue

            # Entity: the rule itself
            rule_eid = normalize_entity_id(f"rule_{doc_number}")
            if rule_eid not in entities:
                ext = {"document_number": doc_number, "type": doc_type}
                entities[rule_eid] = ("regulation", title[:100], json.dumps(ext))

            # Confidence by rule type
            if doc_type == "Rule":
                confidence = FINREG_CONFIDENCE["final_rule"]
            elif doc_type == "Proposed Rule":
                confidence = FINREG_CONFIDENCE["proposed_rule"]
            else:
                confidence = FINREG_CONFIDENCE["regulatory_guidance"]

            source_type = "database_import"

            # Claims: rule → implements → CFR part
            for cfr in cfr_refs:
                cfr_title = cfr.get("title", "")
                cfr_part = cfr.get("part", "")
                if not cfr_title or not cfr_part:
                    continue

                cfr_eid = normalize_entity_id(f"cfr_title_{cfr_title}_part_{cfr_part}")
                if cfr_eid not in entities:
                    entities[cfr_eid] = ("standard", f"CFR Title {cfr_title} Part {cfr_part}", "{}")

                pred_id = "implements"
                source_id = f"fedreg:{doc_number}:cfr:{cfr_title}:{cfr_part}"
                claim_id = compute_claim_id(
                    rule_eid, pred_id, cfr_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claim_ids:
                    seen_claim_ids.add(claim_id)
                    content_id = compute_content_id(rule_eid, pred_id, cfr_eid)
                    claim_rows.append((
                        rule_eid, cfr_eid,
                        claim_id, content_id, pred_id, "relates_to",
                        confidence,
                        source_type, source_id,
                        "", "[]", "", "", "", timestamp, "active",
                    ))

            # Claims: agency → publishes → rule
            for agency in agencies:
                agency_name = (agency.get("name") or "").strip()
                if not agency_name:
                    continue

                agency_eid = normalize_entity_id(f"regulator_{agency_name}")
                agency_eid, agency_ext = _safe_entity_id(agency_eid)
                if agency_eid not in entities:
                    entities[agency_eid] = ("regulator", agency_name, json.dumps(agency_ext) if agency_ext else "{}")

                pred_id = "is_supervised_by"
                source_id = f"fedreg:{doc_number}:agency:{agency_name}"
                claim_id = compute_claim_id(
                    rule_eid, pred_id, agency_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claim_ids:
                    seen_claim_ids.add(claim_id)
                    content_id = compute_content_id(rule_eid, pred_id, agency_eid)
                    claim_rows.append((
                        rule_eid, agency_eid,
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
        "Federal Register: %d claims, %d entities from %d rules in %.1fs",
        len(claim_rows), len(entities), count, elapsed,
    )
    return result
