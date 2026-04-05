"""Software supply chain data source loaders: PyPI packages, OSV vulnerabilities.

Loads package dependency graphs, maintainer relationships, license info,
and known vulnerabilities. PyPI version upload dates enable temporal analysis.

Data sources:
- PyPI: https://pypi.org/ (free, public JSON API)
- OSV: https://osv.dev/ (free, Google Open Source Vulnerabilities)
"""

from __future__ import annotations

import json
import logging
import re
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

SUPPLYCHAIN_CONFIDENCE = {
    "dependency": 0.95,        # Declared in package metadata
    "vulnerability": 0.90,     # Confirmed by OSV/NVD
    "maintainer": 0.90,        # From PyPI registry
    "license": 0.85,           # Self-declared, sometimes inaccurate
    "version": 0.95,           # Upload time is factual
}


def _iso_to_ns(ts_str: str) -> int:
    """Convert ISO 8601 timestamp to nanoseconds."""
    try:
        # Handle various formats: YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM:SSZ, etc.
        ts_str = ts_str.rstrip("Z").split("+")[0]
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(ts_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * _NANO)
            except ValueError:
                continue
        return int(time.time() * _NANO)
    except (ValueError, TypeError):
        return int(time.time() * _NANO)


def _parse_requirement(req_str: str) -> str | None:
    """Extract package name from a PEP 508 requirement string.

    E.g., 'requests>=2.20,<3.0; python_version >= "3.6"' → 'requests'
    """
    if not req_str:
        return None
    # Strip extras, version specs, environment markers
    match = re.match(r"^([A-Za-z0-9][\w.\-]*)", req_str.strip())
    return match.group(1).lower().replace("-", "_") if match else None


def _normalize_pkg_name(name: str) -> str:
    """Normalize PyPI package name: lowercase, hyphens → underscores."""
    return name.lower().replace("-", "_").replace(".", "_")


def load_pypi_packages(
    pipeline,
    path: str,
    max_packages: int = 0,
) -> BatchResult:
    """Load PyPI package metadata with dependency graph.

    Expected format: JSONL, one package per line. Fields used:
      name, version, requires_dist[], license, author, maintainer,
      versions[{version, upload_time}]

    Claim types generated:
      package → depends_on → package (directional)
      package → maintained_by → person (directional)
      package → licensed_as → license (relates_to)
      package → has_version → version_entity (directional, with timestamps)
    """
    t0 = time.time()

    packages = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packages.append(json.loads(line))
            if max_packages > 0 and len(packages) >= max_packages:
                break

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"

    # Build known package name set for resolving deps
    known_pkgs: set[str] = {
        _normalize_pkg_name(p.get("name", "")) for p in packages
    }

    for pkg in packages:
        name = pkg.get("name", "").strip()
        if not name:
            continue

        norm_name = _normalize_pkg_name(name)
        version = pkg.get("version", "")
        license_str = (pkg.get("license") or "").strip()
        author = (pkg.get("author") or "").strip()
        maintainer = (pkg.get("maintainer") or "").strip()
        requires_dist = pkg.get("requires_dist") or []
        versions = pkg.get("versions") or []

        # Use latest version upload time as timestamp
        timestamp = int(time.time() * _NANO)
        if versions:
            latest_upload = versions[-1].get("upload_time", "")
            if latest_upload:
                timestamp = _iso_to_ns(latest_upload)

        # Package entity
        pkg_eid = _safe_entity_id(
            normalize_entity_id(norm_name)
        )[0]
        ext = {}
        if version:
            ext["version"] = version
        if license_str:
            ext["license"] = license_str[:100]
        ext["n_versions"] = len(versions)
        ext["n_deps"] = len(requires_dist)
        entities[pkg_eid] = (
            "package", name, json.dumps(ext),
        )

        # --- Dependencies: package → depends_on → package ---
        for req in requires_dist:
            dep_name = _parse_requirement(req)
            if not dep_name or dep_name == norm_name:
                continue

            dep_eid = _safe_entity_id(
                normalize_entity_id(dep_name)
            )[0]
            if dep_eid not in entities:
                # Create minimal entity for dependency
                in_dataset = dep_name in known_pkgs
                entities[dep_eid] = (
                    "package", dep_name,
                    json.dumps({"in_dataset": in_dataset}),
                )

            pred_id = "depends_on"
            pred_type = "directional"
            source_id = f"pypi:{norm_name}:dep:{dep_name}"
            claim_id = compute_claim_id(
                pkg_eid, pred_id, dep_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(pkg_eid, pred_id, dep_eid)
                claim_rows.append((
                    pkg_eid, dep_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SUPPLYCHAIN_CONFIDENCE["dependency"],
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        # --- Maintainer: package → maintained_by → person ---
        for person_name in (author, maintainer):
            if not person_name or person_name.lower() in ("none", "unknown"):
                continue
            # Skip if it looks like an email
            if "@" in person_name and " " not in person_name:
                continue

            person_eid = _safe_entity_id(
                normalize_entity_id(person_name)
            )[0]
            entities[person_eid] = ("person", person_name, "{}")

            pred_id = "maintained_by"
            pred_type = "directional"
            source_id = f"pypi:{norm_name}:maintainer:{person_eid}"
            claim_id = compute_claim_id(
                pkg_eid, pred_id, person_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    pkg_eid, pred_id, person_eid
                )
                claim_rows.append((
                    pkg_eid, person_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SUPPLYCHAIN_CONFIDENCE["maintainer"],
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        # --- License: package → licensed_as → license ---
        if license_str and len(license_str) < 100:
            license_eid = _safe_entity_id(
                normalize_entity_id(license_str)
            )[0]
            entities[license_eid] = ("license", license_str, "{}")

            pred_id = "licensed_as"
            pred_type = "relates_to"
            source_id = f"pypi:{norm_name}:license"
            claim_id = compute_claim_id(
                pkg_eid, pred_id, license_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    pkg_eid, pred_id, license_eid
                )
                claim_rows.append((
                    pkg_eid, license_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SUPPLYCHAIN_CONFIDENCE["license"],
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
        "PyPI: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result


def load_osv_vulns(
    pipeline,
    path: str,
    max_vulns: int = 0,
) -> BatchResult:
    """Load OSV vulnerabilities mapped to PyPI packages.

    Expected format: JSONL, one vulnerability per line. Fields used:
      id, summary, aliases[], severity[], published, affected[{package, ranges}]

    Claim types generated:
      vulnerability → affects → package (causes)
      vulnerability → classified_as → severity (directional)
    """
    t0 = time.time()

    vulns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vulns.append(json.loads(line))
            if max_vulns > 0 and len(vulns) >= max_vulns:
                break

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"

    for vuln in vulns:
        vuln_id = vuln.get("id", "").strip()
        if not vuln_id:
            continue

        summary = vuln.get("summary", "")
        aliases = vuln.get("aliases", [])
        published = vuln.get("published", "")
        severity = vuln.get("severity", [])

        timestamp = _iso_to_ns(published) if published else int(time.time() * _NANO)

        # Vulnerability entity
        vuln_eid = _safe_entity_id(
            normalize_entity_id(vuln_id)
        )[0]
        ext = {}
        if aliases:
            ext["aliases"] = aliases[:5]
        if summary:
            ext["summary"] = summary[:200]
        if severity:
            ext["severity"] = severity
        entities[vuln_eid] = (
            "vulnerability", vuln_id, json.dumps(ext),
        )

        # --- Affected packages: vulnerability → affects → package ---
        for affected in vuln.get("affected", []):
            pkg_info = affected.get("package", {})
            pkg_name = pkg_info.get("name", "").strip()
            ecosystem = pkg_info.get("ecosystem", "")

            if not pkg_name or ecosystem not in ("PyPI", ""):
                continue

            norm_name = _normalize_pkg_name(pkg_name)
            pkg_eid = _safe_entity_id(
                normalize_entity_id(norm_name)
            )[0]
            if pkg_eid not in entities:
                entities[pkg_eid] = ("package", pkg_name, "{}")

            pred_id = "affects"
            pred_type = "causes"
            source_id = f"osv:{vuln_id}:{norm_name}"
            claim_id = compute_claim_id(
                vuln_eid, pred_id, pkg_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(vuln_eid, pred_id, pkg_eid)
                claim_rows.append((
                    vuln_eid, pkg_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SUPPLYCHAIN_CONFIDENCE["vulnerability"],
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        # --- Severity classification ---
        for sev in severity:
            sev_type = sev.get("type", "")
            sev_score = sev.get("score", "")
            if sev_type == "CVSS_V3" and sev_score:
                # Extract CVSS base score for severity label
                # CVSS vector string starts with CVSS:3.x/AV:...
                label = f"CVSS:{sev_score[:20]}"
                sev_eid = _safe_entity_id(
                    normalize_entity_id(label)
                )[0]
                entities[sev_eid] = (
                    "severity", label,
                    json.dumps({"type": sev_type, "score": sev_score}),
                )

                pred_id = "classified_as"
                pred_type = "directional"
                source_id = f"osv:{vuln_id}:severity"
                claim_id = compute_claim_id(
                    vuln_eid, pred_id, sev_eid,
                    source_id, source_type, timestamp,
                )
                if claim_id not in seen_claims:
                    seen_claims.add(claim_id)
                    content_id = compute_content_id(
                        vuln_eid, pred_id, sev_eid
                    )
                    claim_rows.append((
                        vuln_eid, sev_eid,
                        claim_id, content_id, pred_id, pred_type,
                        SUPPLYCHAIN_CONFIDENCE["vulnerability"],
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
        "OSV: %d claims, %d entities in %.1fs",
        len(claim_rows), len(entities), elapsed,
    )
    return result
