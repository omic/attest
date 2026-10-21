"""Public, local memory tools. Retrieval is by topic/file, without model calls."""

from __future__ import annotations

import uuid

from attestdb.core.normalization import normalize_entity_id

SCHEMA = "attest/brain/v1"
KINDS = {"bug", "fix", "pattern", "warning", "decision", "tip", "negative", "session"}


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
        subject=(subject, "brain_topic"), predicate=("brain_memory", "relates_to"),
        object=(record, "brain_record"),
        provenance={"source_id": f"brain:{record}", "source_type": "agent"},
        confidence=0.5, payload={"schema_ref": SCHEMA, "data": data},
    )
    return {"claim_id": claim_id, **data}


def recall(db, topic: str, limit: int = 8, *, exact: bool = False) -> list[dict]:
    if not topic.strip():
        raise ValueError("A nonempty topic is required")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    normalized = normalize_entity_id(topic)
    entities = [normalized]
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
