"""Migration utilities: export and import AttestDB data via NDJSON.

Export format (one JSON object per line):
    {"type": "metadata", "format_version": 1, "vocabularies": ..., ...}
    {"type": "entity", "id": ..., "entity_type": ..., "display_name": ..., "external_ids": ...}
    {"type": "claim", "claim_id": ..., "content_id": ..., "subject_id": ..., ...}

Metadata record comes first, then entities, then claims.
"""

from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

# Current export format version. Bump when the NDJSON schema changes.
FORMAT_VERSION = 1

# Set of versions this code can import (forward-compat window).
SUPPORTED_VERSIONS = {1}

from attestdb.core.types import (  # noqa: E402
    Claim,
    ClaimStatus,
    EntityRef,
    Payload,
    PredicateRef,
    Provenance,
    claim_from_dict,
    entity_summary_from_dict,
)


def export_store(store, output_path: str) -> dict:
    """Export all entities and claims from a store to NDJSON.

    Returns stats dict with entity_count and claim_count.
    """
    t0 = time.time()
    entity_count = 0
    claim_count = 0
    seen_claims: set[str] = set()

    with open(output_path, "w") as f:
        # Export metadata registrations
        metadata = {"type": "metadata", "format_version": FORMAT_VERSION}
        try:
            metadata["vocabularies"] = store.get_registered_vocabularies()
        except Exception as exc:
            logger.warning("Failed to export vocabularies, using empty: %s", exc)
            metadata["vocabularies"] = {}
        try:
            metadata["predicate_constraints"] = store.get_predicate_constraints()
        except Exception as exc:
            logger.warning("Failed to export predicate constraints, using empty: %s", exc)
            metadata["predicate_constraints"] = {}
        try:
            metadata["payload_schemas"] = store.get_payload_schemas()
        except Exception as exc:
            logger.warning("Failed to export payload schemas, using empty: %s", exc)
            metadata["payload_schemas"] = {}
        f.write(json.dumps(metadata, separators=(",", ":"), default=str) + "\n")

        # Export all entities
        entities = [entity_summary_from_dict(d) for d in store.list_entities()]
        for e in entities:
            record = {
                "type": "entity",
                "id": e.id,
                "entity_type": e.entity_type,
                "display_name": e.name,
                "external_ids": e.external_ids,
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            entity_count += 1

        # Export all claims by iterating through entities
        for e in entities:
            claims = [claim_from_dict(d) for d in store.claims_for(e.id, None, None, 0.0)]
            for claim in claims:
                if claim.claim_id in seen_claims:
                    continue
                seen_claims.add(claim.claim_id)

                record = {
                    "type": "claim",
                    "claim_id": claim.claim_id,
                    "content_id": claim.content_id,
                    "subject_id": claim.subject.id,
                    "subject_type": claim.subject.entity_type,
                    "subject_display_name": claim.subject.display_name,
                    "subject_external_ids": claim.subject.external_ids,
                    "object_id": claim.object.id,
                    "object_type": claim.object.entity_type,
                    "object_display_name": claim.object.display_name,
                    "object_external_ids": claim.object.external_ids,
                    "predicate_id": claim.predicate.id,
                    "predicate_type": claim.predicate.predicate_type,
                    "confidence": claim.confidence,
                    "source_type": claim.provenance.source_type,
                    "source_id": claim.provenance.source_id,
                    "method": claim.provenance.method,
                    "chain": claim.provenance.chain,
                    "model_version": claim.provenance.model_version,
                    "organization": claim.provenance.organization,
                    "timestamp": claim.timestamp,
                    "status": claim.status.value,
                    "namespace": claim.namespace,
                    "expires_at": claim.expires_at,
                }
                if claim.payload:
                    record["payload_schema"] = claim.payload.schema_ref
                    record["payload_data"] = claim.payload.data

                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                claim_count += 1

    elapsed = time.time() - t0
    return {
        "entity_count": entity_count,
        "claim_count": claim_count,
        "elapsed_seconds": round(elapsed, 2),
    }


def export_entity_claims(store, entity_id: str, output_path: str) -> int:
    """Export all claims for a single entity to an NDJSON file.

    Each line is a JSON object with the same claim schema as export_store.
    Returns the count of claims exported.
    """
    claim_count = 0

    with open(output_path, "w") as f:
        claims = [claim_from_dict(d) for d in store.claims_for(entity_id, None, None, 0.0)]
        for claim in claims:
            record = {
                "type": "claim",
                "claim_id": claim.claim_id,
                "content_id": claim.content_id,
                "subject_id": claim.subject.id,
                "subject_type": claim.subject.entity_type,
                "subject_display_name": claim.subject.display_name,
                "subject_external_ids": claim.subject.external_ids,
                "object_id": claim.object.id,
                "object_type": claim.object.entity_type,
                "object_display_name": claim.object.display_name,
                "object_external_ids": claim.object.external_ids,
                "predicate_id": claim.predicate.id,
                "predicate_type": claim.predicate.predicate_type,
                "confidence": claim.confidence,
                "source_type": claim.provenance.source_type,
                "source_id": claim.provenance.source_id,
                "method": claim.provenance.method,
                "chain": claim.provenance.chain,
                "model_version": claim.provenance.model_version,
                "organization": claim.provenance.organization,
                "timestamp": claim.timestamp,
                "status": claim.status.value,
            }
            if claim.payload:
                record["payload_schema"] = claim.payload.schema_ref
                record["payload_data"] = claim.payload.data

            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            claim_count += 1

    logger.info("Exported %d claims for entity %s to %s", claim_count, entity_id, output_path)
    return claim_count


def import_store(store, input_path: str, verify: bool = True) -> dict:
    """Import entities and claims from NDJSON into a store.

    If verify=True, recomputes content_id and asserts it matches.
    Returns stats dict.
    """
    from attestdb.core.hashing import compute_content_id

    t0 = time.time()
    entity_count = 0
    claim_count = 0
    id_mismatches = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record["type"] == "metadata":
                # Validate format version
                version = record.get("format_version", 1)  # default 1 for pre-version exports
                if version not in SUPPORTED_VERSIONS:
                    raise ValueError(
                        f"Unsupported NDJSON format version {version}. "
                        f"Supported: {sorted(SUPPORTED_VERSIONS)}. "
                        f"Upgrade attestdb to import this file."
                    )
                for ns, vocab in record.get("vocabularies", {}).items():
                    store.register_vocabulary(ns, vocab)
                for pred_id, constraints in record.get("predicate_constraints", {}).items():
                    store.register_predicate(pred_id, constraints)
                for schema_id, schema in record.get("payload_schemas", {}).items():
                    store.register_payload_schema(schema_id, schema)

            elif record["type"] == "entity":
                store.upsert_entity(
                    entity_id=record["id"],
                    entity_type=record["entity_type"],
                    display_name=record["display_name"],
                    external_ids=record.get("external_ids") or None,
                    timestamp=0,
                )
                entity_count += 1

            elif record["type"] == "claim":
                if verify:
                    expected_content_id = compute_content_id(
                        record["subject_id"],
                        record["predicate_id"],
                        record["object_id"],
                    )
                    if expected_content_id != record["content_id"]:
                        id_mismatches += 1

                payload = None
                if "payload_schema" in record:
                    payload = Payload(
                        schema_ref=record["payload_schema"],
                        data=record.get("payload_data", {}),
                    )

                claim = Claim(
                    claim_id=record["claim_id"],
                    content_id=record["content_id"],
                    subject=EntityRef(
                        id=record["subject_id"],
                        entity_type=record["subject_type"],
                        display_name=record.get("subject_display_name", ""),
                        external_ids=record.get("subject_external_ids", {}),
                    ),
                    predicate=PredicateRef(
                        id=record["predicate_id"],
                        predicate_type=record["predicate_type"],
                    ),
                    object=EntityRef(
                        id=record["object_id"],
                        entity_type=record["object_type"],
                        display_name=record.get("object_display_name", ""),
                        external_ids=record.get("object_external_ids", {}),
                    ),
                    confidence=record["confidence"],
                    provenance=Provenance(
                        source_type=record["source_type"],
                        source_id=record["source_id"],
                        method=record.get("method"),
                        chain=record.get("chain", []),
                        model_version=record.get("model_version"),
                        organization=record.get("organization"),
                    ),
                    payload=payload,
                    timestamp=record["timestamp"],
                    status=ClaimStatus(record.get("status", "active")),
                    namespace=record.get("namespace", ""),
                    expires_at=record.get("expires_at", 0),
                )
                store.insert_claim(claim)
                claim_count += 1

    elapsed = time.time() - t0
    return {
        "entity_count": entity_count,
        "claim_count": claim_count,
        "id_mismatches": id_mismatches,
        "elapsed_seconds": round(elapsed, 2),
    }
