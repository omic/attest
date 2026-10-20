"""Public, local memory tools. Retrieval is by topic/file, without model calls."""

from __future__ import annotations

import uuid

from attestdb.core.normalization import normalize_entity_id
from attestdb.infrastructure.bulk_loader import _safe_entity_id, max_subject_bytes

SCHEMA = "attest/brain/v1"
PREDICATE = "brain_memory"
KINDS = {"bug", "fix", "pattern", "warning", "decision", "tip", "negative", "session"}


def _source(record: str) -> str:
    return f"brain:{record}"


# Every memory's source has one length (a uuid's hex), so every subject has one limit.
_SUBJECT_BYTES = max_subject_bytes(PREDICATE, _source(uuid.UUID(int=0).hex))


def _topic(subject: str) -> str:
    """The name a memory's subject is stored under: the subject itself, or, when its entity id
    would pass the store's limit for a subject, that id cut with a hash of the whole. A session
    summary of about 800 characters was rejected by the storage engine (2026-10-20)."""
    normalized = normalize_entity_id(subject)
    safe, _ = _safe_entity_id(normalized, max_bytes=_SUBJECT_BYTES)
    return subject if safe == normalized else safe


def remember(db, subject: str, description: str, kind: str, **metadata) -> dict:
    subject, description = subject.strip(), description.strip()
    if not subject or not description:
        raise ValueError("A nonempty subject and description are required")
    if kind not in KINDS:
        raise ValueError(f"Unknown memory kind: {kind}")
    record = uuid.uuid4().hex
    data = {"subject": subject, "description": description, "kind": kind,
            "verification": "agent_reported", **metadata}
    claim_id = db.ingest(
        subject=(_topic(subject), "brain_topic"), predicate=(PREDICATE, "relates_to"),
        object=(record, "brain_record"),
        provenance={"source_id": _source(record), "source_type": "agent"},
        confidence=0.5, payload={"schema_ref": SCHEMA, "data": data},
    )
    return {"claim_id": claim_id, **data}


def recall(db, topic: str, limit: int = 8, *, exact: bool = False) -> list[dict]:
    if not topic.strip():
        raise ValueError("A nonempty topic is required")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    normalized = normalize_entity_id(topic)
    entities = [normalize_entity_id(_topic(topic))]
    if not exact:
        entities += [e.id for e in db.search_entities(topic, top_k=20)
                     if normalized in e.id or e.id in normalized]
    found = {}
    for entity in dict.fromkeys(entities):
        for claim in db.claims_for(entity, limit=200):
            if claim.payload is not None and claim.payload.schema_ref == SCHEMA:
                if exact and claim.payload.data.get("subject") != topic:
                    continue
                found[claim.claim_id] = {
                    **claim.payload.data, "claim_id": claim.claim_id,
                    "confidence": claim.confidence, "timestamp": claim.timestamp,
                }
    return sorted(found.values(), key=lambda row: row["timestamp"], reverse=True)[:limit]


def register_tools(mcp, get_db) -> None:
    @mcp.tool()
    def attest_learned(subject: str, description: str, type: str = "pattern") -> dict:
        """Remember an agent-reported finding; it is stored, not independently verified."""
        return remember(get_db(), subject, description, type)

    @mcp.tool()
    def attest_negative_result(topic: str, finding: str) -> dict:
        """Record an unsuccessful approach so it can be recalled by topic."""
        return remember(get_db(), topic, finding, "negative")

    @mcp.tool()
    def attest_check_file(path: str) -> list[dict]:
        """Recall stored findings for a file/topic; does not read files from disk."""
        return recall(get_db(), path, exact=True)

    @mcp.tool()
    def attest_get_prior_approaches(problem: str, limit: int = 8) -> list[dict]:
        """Recall recent records by topic, including failures and reported outcomes."""
        return recall(get_db(), problem, limit)

    @mcp.tool()
    def attest_session_end(
        outcome: str, summary: str, next_steps: str = "", files_changed: list[str] | None = None,
    ) -> dict:
        """Store a reported session outcome and associate it with listed files."""
        if outcome not in {"success", "partial", "failure"}:
            raise ValueError("outcome must be success, partial, or failure")
        db = get_db()
        result = remember(db, summary, summary, "session", outcome=outcome, next_steps=next_steps)
        for path in dict.fromkeys(files_changed or []):
            remember(db, path, summary, "session", outcome=outcome, next_steps=next_steps)
        return result
