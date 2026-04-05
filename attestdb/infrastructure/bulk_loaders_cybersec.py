"""Cybersecurity data source loaders: CISA KEV, MITRE ATT&CK.

All loaders consume real public data and produce claims using the standard
bulk insertion pipeline. KEV claims use real-world dates as timestamps
(not ingestion time) to enable temporal analysis.

Data sources:
- CISA KEV: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- MITRE ATT&CK: https://attack.mitre.org/
"""

from __future__ import annotations

import json
import logging
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

CYBERSEC_CONFIDENCE = {
    "cisa_verified": 0.95,      # CISA-confirmed exploitation
    "mitre_curated": 0.90,      # MITRE expert curation
    "nvd_analyst": 0.85,        # NVD analyst assessment
    "vendor_reported": 0.75,    # Vendor self-report
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


def load_cisa_kev(
    pipeline,
    path: str,
    max_vulns: int = 0,
) -> BatchResult:
    """Load CISA Known Exploited Vulnerabilities catalog.

    Expected format: JSON with top-level "vulnerabilities" array.
    Each entry has: cveID, vendorProject, product, vulnerabilityName,
    dateAdded, shortDescription, knownRansomwareCampaignUse, cwes[].

    Claims use dateAdded as timestamp for real temporal analysis.
    """
    t0 = time.time()

    with open(path) as f:
        data = json.load(f)

    vulns = data.get("vulnerabilities", [])
    if max_vulns > 0:
        vulns = vulns[:max_vulns]

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"
    confidence = CYBERSEC_CONFIDENCE["cisa_verified"]

    for vuln in vulns:
        cve_id = vuln.get("cveID", "").strip()
        if not cve_id:
            continue

        vendor = vuln.get("vendorProject", "").strip()
        product = vuln.get("product", "").strip()
        vuln_name = vuln.get("vulnerabilityName", "").strip()
        date_added = vuln.get("dateAdded", "")
        ransomware = vuln.get("knownRansomwareCampaignUse", "Unknown")
        cwes = vuln.get("cwes", [])
        description = vuln.get("shortDescription", "")

        timestamp = _date_to_ns(date_added)

        # Vulnerability entity
        vuln_eid = _safe_entity_id(
            normalize_entity_id(cve_id)
        )[0]
        ext = {}
        if date_added:
            ext["date_added"] = date_added
        if ransomware and ransomware != "Unknown":
            ext["ransomware_use"] = ransomware
        if description:
            ext["description"] = description[:200]
        entities[vuln_eid] = (
            "vulnerability", vuln_name or cve_id, json.dumps(ext)
        )

        # Software entity (vendor:product)
        if vendor and product:
            sw_raw = f"{vendor}:{product}"
            sw_eid = _safe_entity_id(
                normalize_entity_id(sw_raw)
            )[0]
            entities[sw_eid] = (
                "software", f"{vendor} {product}",
                json.dumps({"vendor": vendor, "product": product}),
            )

            # vulnerability → affects → software
            pred_id = "affects"
            pred_type = "causes"
            source_id = f"kev:{cve_id}"
            claim_id = compute_claim_id(
                vuln_eid, pred_id, sw_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(vuln_eid, pred_id, sw_eid)
                claim_rows.append((
                    vuln_eid, sw_eid,
                    claim_id, content_id, pred_id, pred_type,
                    confidence,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        # Weakness entities (CWE)
        for cwe_id in cwes:
            if not cwe_id or cwe_id == "NVD-CWE-noinfo":
                continue
            cwe_eid = _safe_entity_id(
                normalize_entity_id(cwe_id)
            )[0]
            entities[cwe_eid] = ("weakness", cwe_id, "{}")

            # vulnerability → classified_as → weakness
            pred_id = "classified_as"
            pred_type = "directional"
            source_id = f"kev:{cve_id}:cwe"
            claim_id = compute_claim_id(
                vuln_eid, pred_id, cwe_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    vuln_eid, pred_id, cwe_eid
                )
                claim_rows.append((
                    vuln_eid, cwe_eid,
                    claim_id, content_id, pred_id, pred_type,
                    0.90,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

    if not claim_rows:
        return BatchResult(ingested=0)

    # Use the latest timestamp for entity registration
    latest_ts = max(r[14] for r in claim_rows)
    result = _ingest_append_direct(
        pipeline, entities, claim_rows, latest_ts
    )
    elapsed = time.time() - t0
    logger.info(
        "CISA KEV: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_mitre_attack(
    pipeline,
    path: str,
    max_objects: int = 0,
) -> BatchResult:
    """Load MITRE ATT&CK Enterprise STIX 2.0 bundle.

    Extracts techniques, threat groups (intrusion-sets), malware, tools,
    campaigns, tactics, and their relationships.

    Relationship types mapped to Attest predicates:
      uses → uses (causes)
      mitigates → mitigates (opposes)
      attributed-to → attributed_to (relates_to)
      subtechnique-of → subtechnique_of (relates_to)
    """
    t0 = time.time()
    timestamp = int(t0 * _NANO)

    with open(path) as f:
        data = json.load(f)

    objects = data.get("objects", [])
    if max_objects > 0:
        objects = objects[:max_objects]

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"
    confidence = CYBERSEC_CONFIDENCE["mitre_curated"]

    # First pass: build entity index by STIX ID
    stix_to_eid: dict[str, str] = {}
    stix_to_type: dict[str, str] = {}

    type_map = {
        "attack-pattern": "technique",
        "intrusion-set": "threat_group",
        "malware": "malware",
        "tool": "tool",
        "campaign": "campaign",
        "course-of-action": "mitigation",
        "x-mitre-tactic": "tactic",
    }

    for obj in objects:
        stix_type = obj.get("type", "")
        if stix_type not in type_map:
            continue
        if obj.get("revoked", False) or obj.get("x_mitre_deprecated", False):
            continue

        stix_id = obj.get("id", "")
        name = obj.get("name", "")
        if not stix_id or not name:
            continue

        # Extract external ID (e.g., T1059, G0016, S0154)
        ext_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") in (
                "mitre-attack", "mitre-mobile-attack"
            ):
                ext_id = ref.get("external_id", "")
                break

        entity_type = type_map[stix_type]
        display = f"{ext_id}: {name}" if ext_id else name
        eid_raw = ext_id if ext_id else name
        eid = _safe_entity_id(normalize_entity_id(eid_raw))[0]

        ext = {}
        if ext_id:
            ext["external_id"] = ext_id
        desc = obj.get("description", "")
        if desc:
            ext["description"] = desc[:200]

        # Kill chain phases → tactic links
        kill_chain = obj.get("kill_chain_phases", [])
        tactic_names = [
            kc.get("phase_name", "")
            for kc in kill_chain
            if kc.get("kill_chain_name") == "mitre-attack"
        ]
        if tactic_names:
            ext["tactics"] = tactic_names

        entities[eid] = (entity_type, display, json.dumps(ext))
        stix_to_eid[stix_id] = eid
        stix_to_type[stix_id] = entity_type

    # Create tactic entities from kill chain phases
    tactic_eids: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") == "x-mitre-tactic":
            name = obj.get("name", "")
            short = obj.get("x_mitre_shortname", "")
            stix_id = obj.get("id", "")
            if name and short:
                eid = _safe_entity_id(
                    normalize_entity_id(short)
                )[0]
                entities[eid] = ("tactic", name, json.dumps({
                    "shortname": short,
                }))
                stix_to_eid[stix_id] = eid
                tactic_eids[short] = eid

    # Create technique → tactic claims from kill_chain_phases
    for obj in objects:
        if obj.get("type") != "attack-pattern":
            continue
        stix_id = obj.get("id", "")
        if stix_id not in stix_to_eid:
            continue
        tech_eid = stix_to_eid[stix_id]

        for kc in obj.get("kill_chain_phases", []):
            if kc.get("kill_chain_name") != "mitre-attack":
                continue
            phase = kc.get("phase_name", "")
            if phase not in tactic_eids:
                continue
            tactic_eid = tactic_eids[phase]

            pred_id = "part_of_tactic"
            pred_type = "directional"
            source_id = f"attack:{stix_id}:tactic"
            claim_id = compute_claim_id(
                tech_eid, pred_id, tactic_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    tech_eid, pred_id, tactic_eid
                )
                claim_rows.append((
                    tech_eid, tactic_eid,
                    claim_id, content_id, pred_id, pred_type,
                    0.95,
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

    # Second pass: process relationships
    rel_pred_map = {
        "uses": ("uses", "causes"),
        "mitigates": ("mitigates", "opposes"),
        "attributed-to": ("attributed_to", "relates_to"),
        "subtechnique-of": ("subtechnique_of", "directional"),
        "revoked-by": None,
        "detects": ("detects", "directional"),
    }

    for obj in objects:
        if obj.get("type") != "relationship":
            continue

        rel_type = obj.get("relationship_type", "")
        mapping = rel_pred_map.get(rel_type)
        if mapping is None:
            continue

        pred_id, pred_type = mapping
        source_ref = obj.get("source_ref", "")
        target_ref = obj.get("target_ref", "")

        if source_ref not in stix_to_eid or target_ref not in stix_to_eid:
            continue

        subj_eid = stix_to_eid[source_ref]
        obj_eid = stix_to_eid[target_ref]

        source_id = f"attack:{obj.get('id', '')}"
        claim_id = compute_claim_id(
            subj_eid, pred_id, obj_eid,
            source_id, source_type, timestamp,
        )
        if claim_id not in seen_claims:
            seen_claims.add(claim_id)
            content_id = compute_content_id(subj_eid, pred_id, obj_eid)
            claim_rows.append((
                subj_eid, obj_eid,
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
        "MITRE ATT&CK: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result
