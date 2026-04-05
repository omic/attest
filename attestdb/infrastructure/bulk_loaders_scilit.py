"""Scientific literature data source loaders: OpenAlex works and authors.

Loads paper citation graphs, authorship, institutional affiliation, and
concept tagging from OpenAlex JSONL exports. Citation edges use publication
dates as timestamps for temporal analysis.

Data source: https://openalex.org/ (free, CC0)
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

SCILIT_CONFIDENCE = {
    "citation": 0.95,         # Citation is a verifiable fact
    "authorship": 0.95,       # Authorship is recorded by publisher
    "affiliation": 0.85,      # Affiliation can change / be partial
    "concept_tag": 0.75,      # Algorithmic concept tagging (OpenAlex ML)
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


def _openalex_short_id(url: str) -> str:
    """Extract short ID from OpenAlex URL (e.g., 'https://openalex.org/W123' → 'W123')."""
    if not url:
        return ""
    return url.rsplit("/", 1)[-1] if "/" in url else url


def load_openalex_works(
    pipeline,
    path: str,
    max_works: int = 0,
    *,
    source_label: str = "openalex",
) -> BatchResult:
    """Load OpenAlex works (papers) with citations, authors, concepts.

    Expected format: JSONL, one work per line. Fields used:
      id, title, publication_date, cited_by_count, authorships[],
      concepts[], referenced_works[], type

    Claim types generated:
      paper → cites → paper (directional, from referenced_works)
      paper → authored_by → author (directional)
      author → affiliated_with → institution (relates_to)
      paper → studies → concept (directional)
    """
    t0 = time.time()

    works = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            works.append(json.loads(line))
            if max_works > 0 and len(works) >= max_works:
                break

    entities: dict[str, tuple[str, str, str]] = {}
    claim_rows: list[tuple] = []
    seen_claims: set[str] = set()
    source_type = "database_import"

    # Build paper ID index for resolving internal citations
    paper_ids: set[str] = set()
    for work in works:
        oa_id = _openalex_short_id(work.get("id", ""))
        if oa_id:
            paper_ids.add(oa_id)

    for work in works:
        oa_id = _openalex_short_id(work.get("id", ""))
        if not oa_id:
            continue

        title = (work.get("title") or "").strip()
        pub_date = work.get("publication_date", "")
        cited_by = work.get("cited_by_count", 0)
        work_type = work.get("type", "article")
        doi = work.get("doi", "")

        timestamp = _date_to_ns(pub_date) if pub_date else int(time.time() * _NANO)

        # Paper entity
        paper_eid = _safe_entity_id(
            normalize_entity_id(oa_id)
        )[0]
        ext = {}
        if pub_date:
            ext["publication_date"] = pub_date
        if cited_by:
            ext["cited_by_count"] = cited_by
        if doi:
            ext["doi"] = doi
        if work_type:
            ext["type"] = work_type
        entities[paper_eid] = (
            "paper", title or oa_id, json.dumps(ext),
        )

        # --- Citations: paper → cites → paper ---
        for ref_url in work.get("referenced_works", []):
            ref_id = _openalex_short_id(ref_url)
            if not ref_id or ref_id not in paper_ids:
                continue  # Only cite papers in our dataset

            ref_eid = _safe_entity_id(
                normalize_entity_id(ref_id)
            )[0]
            # ref entity might not have metadata yet — register minimal
            if ref_eid not in entities:
                entities[ref_eid] = ("paper", ref_id, "{}")

            pred_id = "cites"
            pred_type = "directional"
            source_id = f"{source_label}:{oa_id}:cites:{ref_id}"
            claim_id = compute_claim_id(
                paper_eid, pred_id, ref_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(paper_eid, pred_id, ref_eid)
                claim_rows.append((
                    paper_eid, ref_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SCILIT_CONFIDENCE["citation"],
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

        # --- Authorships: paper → authored_by → author ---
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            author_oa_id = _openalex_short_id(author.get("id", ""))
            author_name = (author.get("display_name") or "").strip()
            if not author_oa_id or not author_name:
                continue

            author_eid = _safe_entity_id(
                normalize_entity_id(author_oa_id)
            )[0]
            author_ext = {}
            orcid = author.get("orcid", "")
            if orcid:
                author_ext["orcid"] = orcid
            entities[author_eid] = (
                "author", author_name, json.dumps(author_ext),
            )

            pred_id = "authored_by"
            pred_type = "directional"
            source_id = f"{source_label}:{oa_id}:author:{author_oa_id}"
            claim_id = compute_claim_id(
                paper_eid, pred_id, author_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    paper_eid, pred_id, author_eid
                )
                claim_rows.append((
                    paper_eid, author_eid,
                    claim_id, content_id, pred_id, pred_type,
                    SCILIT_CONFIDENCE["authorship"],
                    source_type, source_id,
                    "", "[]", "", "", "", timestamp, "active",
                ))

            # --- Author → affiliated_with → institution ---
            institutions = authorship.get("institutions", [])
            if institutions:
                inst = institutions[0]
                inst_oa_id = _openalex_short_id(inst.get("id", ""))
                inst_name = (inst.get("display_name") or "").strip()
                if inst_oa_id and inst_name:
                    inst_eid = _safe_entity_id(
                        normalize_entity_id(inst_oa_id)
                    )[0]
                    inst_ext = {}
                    country = inst.get("country_code", "")
                    if country:
                        inst_ext["country"] = country
                    entities[inst_eid] = (
                        "institution", inst_name,
                        json.dumps(inst_ext),
                    )

                    pred_id = "affiliated_with"
                    pred_type = "relates_to"
                    source_id = (
                        f"{source_label}:{author_oa_id}:inst:{inst_oa_id}"
                    )
                    claim_id = compute_claim_id(
                        author_eid, pred_id, inst_eid,
                        source_id, source_type, timestamp,
                    )
                    if claim_id not in seen_claims:
                        seen_claims.add(claim_id)
                        content_id = compute_content_id(
                            author_eid, pred_id, inst_eid
                        )
                        claim_rows.append((
                            author_eid, inst_eid,
                            claim_id, content_id, pred_id,
                            pred_type,
                            SCILIT_CONFIDENCE["affiliation"],
                            source_type, source_id,
                            "", "[]", "", "", "", timestamp,
                            "active",
                        ))

        # --- Concepts: paper → studies → concept ---
        for concept in work.get("concepts", []):
            concept_oa_id = _openalex_short_id(concept.get("id", ""))
            concept_name = (concept.get("display_name") or "").strip()
            score = concept.get("score", 0)
            if not concept_oa_id or not concept_name or score < 0.4:
                continue  # Skip low-confidence concept tags

            concept_eid = _safe_entity_id(
                normalize_entity_id(concept_oa_id)
            )[0]
            concept_ext = {}
            level = concept.get("level")
            if level is not None:
                concept_ext["level"] = level
            entities[concept_eid] = (
                "concept", concept_name, json.dumps(concept_ext),
            )

            pred_id = "studies"
            pred_type = "directional"
            source_id = f"{source_label}:{oa_id}:concept:{concept_oa_id}"
            claim_id = compute_claim_id(
                paper_eid, pred_id, concept_eid,
                source_id, source_type, timestamp,
            )
            if claim_id not in seen_claims:
                seen_claims.add(claim_id)
                content_id = compute_content_id(
                    paper_eid, pred_id, concept_eid
                )
                claim_rows.append((
                    paper_eid, concept_eid,
                    claim_id, content_id, pred_id, pred_type,
                    min(SCILIT_CONFIDENCE["concept_tag"], score),
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
        "OpenAlex %s: %d claims, %d entities in %.1fs",
        source_label, len(claim_rows), len(entities), elapsed,
    )
    return result
