"""MCP server exposing AttestDB operations to AI agents.

Usage:
    attest-mcp                                    # stdio (Claude Desktop, Cursor)
    attest-mcp --transport sse --port 8892        # SSE over HTTP
    attest-mcp --transport streamable-http        # Streamable HTTP
    python -m attestdb serve --port 8892          # CLI shortcut for SSE

Environment variables:
    ATTEST_DB_PATH — database file path (default: "attest.db")
"""

import atexit
import json
import os
import time
import uuid
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("attest", instructions="Knowledge graph database with provenance-tracked claims")

# Global DB reference — set by main() or configure()
_db = None

# ---------------------------------------------------------------------------
# Auto-observe session tracking
# ---------------------------------------------------------------------------

_session_tracker: dict | None = None


def _init_session_tracker():
    """Initialize auto-observe if ATTEST_AUTO_OBSERVE is set (default: enabled)."""
    global _session_tracker
    if os.environ.get("ATTEST_AUTO_OBSERVE", "1") == "0":
        return
    _session_tracker = {
        "session_id": str(uuid.uuid4()),
        "tool_calls": [],
        "start_time": time.time(),
        "entities_queried": set(),  # entities looked up during session
        "claims_ingested": 0,  # count of claims ingested
        "learnings_recorded": 0,  # count of attest_learned calls
    }
    atexit.register(_flush_session)


def _track_tool_call(tool_name: str, args_summary: str = ""):
    """Record a tool call in the session tracker."""
    if _session_tracker is None:
        return
    _session_tracker["tool_calls"].append({
        "tool": tool_name,
        "timestamp": time.time(),
        "args_summary": args_summary[:200],
    })


def _track_entity_queried(entity_id: str):
    """Record an entity lookup."""
    if _session_tracker is not None:
        _session_tracker["entities_queried"].add(entity_id)


def _track_claims_ingested(count: int):
    """Record claims ingested."""
    if _session_tracker is not None:
        _session_tracker["claims_ingested"] += count


def _flush_session():
    """On process exit, record session summary as structured claims."""
    if _session_tracker is None or not _session_tracker["tool_calls"]:
        return
    try:
        db = _get_db()
        sid = _session_tracker["session_id"]
        prov = {"source_type": "auto_observe", "source_id": sid}
        elapsed = round(time.time() - _session_tracker["start_time"], 1)
        n_calls = len(_session_tracker["tool_calls"])
        n_ingested = _session_tracker["claims_ingested"]
        n_learned = _session_tracker["learnings_recorded"]
        entities = _session_tracker["entities_queried"]

        # Single summary claim with payload (not N claims per tool)
        tools_used = sorted({tc["tool"] for tc in _session_tracker["tool_calls"]})
        db.ingest(
            subject=(sid, "tool_session"),
            predicate=("has_status", "predicate"),
            object=("completed", "status"),
            provenance=prov,
            confidence=0.7,
            payload={
                "schema_ref": "session_summary",
                "data": {
                    "duration_s": elapsed,
                    "tool_calls": n_calls,
                    "tools_used": tools_used,
                    "claims_ingested": n_ingested,
                    "learnings_recorded": n_learned,
                    "entities_queried": sorted(entities)[:20],
                },
            },
        )
    except Exception:
        pass  # Best-effort — process is exiting


def configure(db) -> None:
    """Set the AttestDB instance used by all tools."""
    global _db
    _db = db


def _get_db():
    if _db is None:
        raise RuntimeError("Database not configured. Set ATTEST_DB_PATH env var.")
    return _db


# ---------------------------------------------------------------------------
# Tools (33)
# ---------------------------------------------------------------------------


@mcp.tool()
def ingest_claim(
    subject_id: str, subject_type: str,
    predicate_id: str, predicate_type: str,
    object_id: str, object_type: str,
    source_type: str, source_id: str,
    confidence: Optional[float] = None,
    payload: Optional[dict] = None,
) -> str:
    """Add a claim to the knowledge graph. Returns claim_id."""
    _track_tool_call("ingest_claim", f"{subject_id} {predicate_id} {object_id}")
    _track_claims_ingested(1)
    db = _get_db()
    return db.ingest(
        subject=(subject_id, subject_type),
        predicate=(predicate_id, predicate_type),
        object=(object_id, object_type),
        provenance={"source_type": source_type, "source_id": source_id},
        confidence=confidence,
        payload=payload,
    )


@mcp.tool()
def ingest_text(text: str, source_id: str = "") -> str:
    """Extract claims from unstructured text and ingest them. Returns extraction results."""
    db = _get_db()
    result = db.ingest_text(text, source_id=source_id)
    return json.dumps(result, default=str)


def _to_pair(val: str | list | tuple, default_type: str = "entity") -> tuple[str, str]:
    """Coerce subject/predicate/object to (id, type) tuple.

    Accepts:
      - "entity_name"           → ("entity_name", default_type)
      - ["entity_name", "type"] → ("entity_name", "type")
      - ("entity_name", "type") → ("entity_name", "type")
    """
    if isinstance(val, str):
        return (val, default_type)
    return (val[0], val[1])


@mcp.tool()
def ingest_batch(claims: list[dict]) -> str:
    """Bulk-ingest a list of claims. Each dict needs: subject, predicate, object, provenance."""
    from attestdb.core.types import ClaimInput

    db = _get_db()
    claim_inputs = [
        ClaimInput(
            subject=_to_pair(c["subject"]),
            predicate=_to_pair(c["predicate"], "predicate"),
            object=_to_pair(c["object"]),
            provenance=c["provenance"],
            confidence=c.get("confidence"),
            payload=c.get("payload"),
        )
        for c in claims
    ]
    result = db.ingest_batch(claim_inputs)
    _track_claims_ingested(result.ingested)
    return json.dumps({
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": result.errors,
    })


@mcp.tool()
def query_entity(entity_id: str, depth: int = 1) -> str:
    """Query the knowledge graph around an entity. Returns narrative + relationships."""
    _track_tool_call("query_entity", entity_id)
    _track_entity_queried(entity_id)
    db = _get_db()
    frame = db.query(entity_id, depth=depth)
    return json.dumps({
        "entity": frame.focal_entity.id,
        "name": frame.focal_entity.name,
        "type": frame.focal_entity.entity_type,
        "narrative": frame.narrative,
        "claim_count": frame.claim_count,
        "relationships": [
            {"predicate": r.predicate, "target": r.target.id, "confidence": r.confidence}
            for r in frame.direct_relationships[:20]
        ],
    })


@mcp.tool()
def search_entities(entity_type: Optional[str] = None, min_claims: int = 0) -> str:
    """List/filter entities by type and minimum claim count."""
    _track_tool_call("search_entities", f"type={entity_type}")
    db = _get_db()
    entities = db.list_entities(entity_type=entity_type, min_claims=min_claims)
    return json.dumps([
        {"id": e.id, "name": e.name, "type": e.entity_type, "claim_count": e.claim_count}
        for e in entities[:100]
    ])


@mcp.tool()
def get_entity(entity_id: str) -> str:
    """Get summary for a single entity."""
    db = _get_db()
    e = db.get_entity(entity_id)
    if e is None:
        return json.dumps({"error": f"Entity not found: {entity_id}"})
    return json.dumps({
        "id": e.id, "name": e.name,
        "type": e.entity_type, "claim_count": e.claim_count,
    })


@mcp.tool()
def claims_for(
    entity_id: str,
    predicate_type: Optional[str] = None,
    min_confidence: float = 0.0,
) -> str:
    """Get claims about an entity, optionally filtered."""
    db = _get_db()
    claims = db.claims_for(entity_id, predicate_type=predicate_type, min_confidence=min_confidence)
    return json.dumps([
        {
            "claim_id": c.claim_id,
            "subject": c.subject.id,
            "predicate": c.predicate.id,
            "object": c.object.id,
            "confidence": c.confidence,
            "source_type": c.provenance.source_type,
            "source_id": c.provenance.source_id,
        }
        for c in claims[:50]
    ])


@mcp.tool()
def find_paths(entity_a: str, entity_b: str, max_depth: int = 3, top_k: int = 5) -> str:
    """Find paths between two entities with per-hop metadata."""
    db = _get_db()
    paths = db.find_paths(entity_a, entity_b, max_depth=max_depth, top_k=top_k)
    return json.dumps([
        {
            "length": p.length,
            "total_confidence": p.total_confidence,
            "steps": [
                {
                    "entity": s.entity_id,
                    "type": s.entity_type,
                    "predicate": s.predicate,
                    "confidence": s.confidence,
                }
                for s in p.steps
            ],
        }
        for p in paths
    ])


@mcp.tool()
def retract_source(source_id: str, reason: str) -> str:
    """Retract all claims from a source."""
    _track_tool_call("retract_source", source_id)
    db = _get_db()
    result = db.retract(source_id, reason)
    return json.dumps({
        "source_id": result.source_id,
        "retracted_count": result.retracted_count,
        "claim_ids": result.claim_ids,
    })


@mcp.tool()
def quality_report() -> str:
    """Knowledge graph quality analysis."""
    db = _get_db()
    report = db.quality_report()
    return json.dumps(asdict(report))


@mcp.tool()
def knowledge_health() -> str:
    """Quantified health metrics for the knowledge graph."""
    db = _get_db()
    health = db.knowledge_health()
    return json.dumps(asdict(health))


@mcp.tool()
def find_bridges(top_k: int = 20) -> str:
    """Find predicted connections between currently-unlinked entities."""
    db = _get_db()
    bridges = db.find_bridges(top_k=top_k)
    return json.dumps([
        {
            "entity_a": b.entity_a,
            "entity_b": b.entity_b,
            "similarity": b.similarity,
            "bridge_type": b.bridge_type,
            "explanation": b.explanation,
        }
        for b in bridges[:top_k]
    ])


@mcp.tool()
def find_gaps(expected_patterns: dict[str, list[str]], min_claims: int = 1) -> str:
    """Find missing expected relationships in the knowledge graph."""
    db = _get_db()
    # Convert lists to sets as expected by the method
    patterns = {k: set(v) for k, v in expected_patterns.items()}
    gaps = db.find_gaps(patterns, min_claims=min_claims)
    return json.dumps(gaps, default=str)


@mcp.tool()
def schema() -> str:
    """Get the knowledge graph schema descriptor."""
    db = _get_db()
    s = db.schema()
    return json.dumps(asdict(s))


@mcp.tool()
def stats() -> str:
    """Get database statistics."""
    db = _get_db()
    return json.dumps(db.stats(), default=str)


_ASK_STOP_WORDS = frozenset({
    "what", "who", "how", "why", "where", "which", "does", "the", "are",
    "for", "and", "is", "was", "has", "have", "been", "this", "that",
    "with", "from", "about", "tell", "me", "show", "find", "get", "list",
    "of", "in", "on", "at", "by", "an", "a", "do", "did", "any", "all",
    "to", "it", "be", "not", "no", "or", "but", "if", "can", "will",
})


@mcp.tool()
def attest_ask(question: str, top_k: int = 10) -> str:
    """Answer a natural-language question using the knowledge graph.

    Returns structured answer with citations, contradictions, and gap analysis.
    """
    _track_tool_call("attest_ask", question[:100])
    db = _get_db()

    # Try full ask() first (uses LLM + embeddings if available)
    try:
        result = db.ask(question, top_k=top_k)
        if result.answer:
            return json.dumps({
                "answer": result.answer,
                "citations": [
                    {
                        "claim_id": c.claim_id, "subject": c.subject,
                        "predicate": c.predicate, "object": c.object,
                        "confidence": c.confidence, "source_type": c.source_type,
                        "source_id": c.source_id,
                    }
                    for c in result.citations
                ],
                "contradictions": result.contradictions,
                "gaps": result.gaps,
                "entities": [
                    {"id": e.id, "name": e.name, "type": e.entity_type}
                    for e in result.entities
                ],
                "meta": result.meta,
            })
    except Exception:
        pass

    # Fallback: text search when ask() returns empty (no LLM / no embeddings)
    from attestdb.core.vocabulary import knowledge_label, knowledge_sort_key

    words = [w.strip("?.,!\"'()[]{}:;") for w in question.lower().split()]
    query_terms = [w for w in words if w and len(w) >= 2 and w not in _ASK_STOP_WORDS]

    candidates = _retrieve_candidates(db, " ".join(query_terms), "", "")

    # Sort by knowledge priority (warnings first) then confidence
    candidates.sort(key=lambda c: knowledge_sort_key(c.predicate.id, c.confidence))

    lines = []
    for c in candidates[:top_k]:
        label = knowledge_label(c.predicate.id)
        lines.append(f"[{label}] {c.subject.id}: {c.object.id} (conf={c.confidence:.2f})")

    return json.dumps({
        "answer": "\n".join(lines) if lines else "No relevant claims found.",
        "citations": [
            {
                "claim_id": c.claim_id, "subject": c.subject.id,
                "predicate": c.predicate.id, "object": c.object.id,
                "confidence": c.confidence, "source_type": c.provenance.source_type,
                "source_id": c.provenance.source_id,
            }
            for c in candidates[:top_k]
        ],
        "contradictions": [],
        "gaps": [],
        "entities": [],
        "meta": {"fallback": True, "n_search_hits": len(candidates)},
    })


# ---------------------------------------------------------------------------
# New API tools (9)
# ---------------------------------------------------------------------------


@mcp.tool()
def attest_impact(source_id: str) -> str:
    """Analyze the impact of a source: what claims depend on it."""
    db = _get_db()
    report = db.impact(source_id)
    return json.dumps(asdict(report))


@mcp.tool()
def attest_blindspots(min_claims: int = 5) -> str:
    """Find single-source entities, knowledge gaps, low-confidence areas, and unresolved warnings."""
    db = _get_db()
    report = db.blindspots(min_claims=min_claims)
    return json.dumps(asdict(report))


@mcp.tool()
def attest_consensus(topic: str) -> str:
    """Analyze consensus around an entity/topic across sources."""
    db = _get_db()
    report = db.consensus(topic)
    return json.dumps(asdict(report))


@mcp.tool()
def attest_fragile(max_sources: int = 1, min_age_days: int = 0) -> str:
    """Find claims backed by few independent sources."""
    db = _get_db()
    claims = db.fragile(max_sources=max_sources, min_age_days=min_age_days)
    return json.dumps([
        {
            "claim_id": c.claim_id,
            "subject": c.subject.id,
            "predicate": c.predicate.id,
            "object": c.object.id,
            "confidence": c.confidence,
            "source_type": c.provenance.source_type,
        }
        for c in claims[:100]
    ])


@mcp.tool()
def attest_stale(days: int = 90) -> str:
    """Find claims older than the given number of days."""
    db = _get_db()
    claims = db.stale(days=days)
    return json.dumps([
        {
            "claim_id": c.claim_id,
            "subject": c.subject.id,
            "predicate": c.predicate.id,
            "object": c.object.id,
            "timestamp": c.timestamp,
        }
        for c in claims[:100]
    ])


@mcp.tool()
def attest_audit(claim_id: str) -> str:
    """Full provenance audit for a claim: corroborations, chain, dependents."""
    db = _get_db()
    trail = db.audit(claim_id)
    return json.dumps(asdict(trail))


@mcp.tool()
def attest_drift(days: int = 30) -> str:
    """Measure knowledge drift over a time period."""
    db = _get_db()
    report = db.drift(days=days)
    return json.dumps(asdict(report))


@mcp.tool()
def attest_source_reliability(source_id: Optional[str] = None) -> str:
    """Per-source corroboration and retraction rates."""
    db = _get_db()
    result = db.source_reliability(source_id=source_id)
    return json.dumps(result, default=str)


@mcp.tool()
def attest_hypothetical(
    subject_id: str, subject_type: str,
    predicate_id: str, predicate_type: str,
    object_id: str, object_type: str,
    source_type: str, source_id: str,
    confidence: Optional[float] = None,
) -> str:
    """What-if analysis: would a hypothetical claim corroborate existing knowledge or fill a gap?"""
    from attestdb.core.types import ClaimInput

    db = _get_db()
    claim = ClaimInput(
        subject=(subject_id, subject_type),
        predicate=(predicate_id, predicate_type),
        object=(object_id, object_type),
        provenance={"source_type": source_type, "source_id": source_id},
        confidence=confidence,
    )
    report = db.hypothetical(claim)
    return json.dumps(asdict(report))


# ---------------------------------------------------------------------------
# Research tools (2)
# ---------------------------------------------------------------------------


@mcp.tool()
def attest_investigate(max_questions: int = 10) -> str:
    """Autonomous gap-closing: detect blindspots, research via LLM, ingest new claims."""
    db = _get_db()
    report = db.investigate(max_questions=max_questions)
    return json.dumps({
        "questions_generated": report.questions_generated,
        "questions_researched": report.questions_researched,
        "claims_ingested": report.claims_ingested,
        "inquiries_resolved": report.inquiries_resolved,
        "blindspot_before": report.blindspot_before,
        "blindspot_after": report.blindspot_after,
    })


@mcp.tool()
def attest_research(question: str, entity_id: Optional[str] = None) -> str:
    """Research a single question via LLM and ingest discovered claims."""
    db = _get_db()
    result = db.research_question(question, entity_id=entity_id)
    return json.dumps({
        "claims_ingested": result.claims_ingested,
        "claims_rejected": result.claims_rejected,
        "inquiry_resolved": result.inquiry_resolved,
        "source": result.source,
    })


# ---------------------------------------------------------------------------
# Learning Layer tools (4)
# ---------------------------------------------------------------------------


def _generate_session_id(tenant_id: str, tool: str) -> str:
    """Generate a unique session ID: {tenant_id}:{tool}:{uuid4}."""
    return f"{tenant_id}:{tool}:{uuid.uuid4().hex[:12]}"


def _estimate_tokens(text: str) -> int:
    """Approximate token count: len / 4."""
    return max(1, len(text) // 4)


def _retrieve_candidates(db, query: str, context: str, tenant_id: str) -> list:
    """ISOLATION BOUNDARY for retrieval. Returns Claim objects.

    Searches entity names AND claim content (subject, predicate, object).
    """
    claims: list = []
    seen: set[str] = set()

    def _add_claim(claim):
        if claim.claim_id not in seen:
            prov_org = claim.provenance.organization
            if tenant_id and prov_org and prov_org != tenant_id:
                return
            seen.add(claim.claim_id)
            claims.append(claim)

    # 1. Entity search: find entities matching query, get their claims
    entities = db.search_entities(query, top_k=20)
    for entity in entities:
        for claim in db.claims_for(entity.id):
            _add_claim(claim)

    # 2. Predicate search: scan knowledge predicates for matching claims
    from attestdb.core.vocabulary import KNOWLEDGE_PREDICATES
    knowledge_predicates = KNOWLEDGE_PREDICATES
    query_lower = query.lower()
    query_tokens = {w for w in query_lower.split() if len(w) >= 3}

    if query_tokens:
        for pred in knowledge_predicates:
            try:
                for claim in db.claims_for_predicate(pred):
                    subj = claim.subject.id.lower()
                    obj = claim.object.id.lower()
                    if any(tok in subj or tok in obj for tok in query_tokens):
                        _add_claim(claim)
            except Exception:
                pass

    return claims


def _score_approach(confidence: float, outcome: str, age_days: float) -> float:
    """Score: outcome_weight x base_confidence x recency_decay."""
    outcome_weights = {
        "success": 1.0,
        "partial": 0.7,
        "failure": 0.3,
        "unknown": 0.5,
    }
    weight = outcome_weights.get(outcome, 0.5)
    decay = max(0.1, 1.0 - age_days / 365.0)
    return weight * confidence * decay


@mcp.tool()
def attest_observe_session(
    messages: list[dict],
    tool: str,
    tenant_id: str = "default",
    session_id: Optional[str] = None,
    outcome: str = "unknown",
    task_description: str = "",
    notes: str = "",
) -> str:
    """Observe an AI tool session: ingest conversation claims and track for outcome feedback.

    Args:
        messages: OpenAI-format messages [{role, content}, ...]
        tool: AI tool name (e.g. "claude-code", "cursor", "copilot")
        tenant_id: Tenant/org identifier for scoping
        session_id: Optional — auto-generated if not provided
        outcome: "success", "partial", "failure", or "unknown" (default)
        task_description: What the session was trying to accomplish
        notes: Additional context
    """
    db = _get_db()

    if not session_id:
        session_id = _generate_session_id(tenant_id, tool)

    # Ingest chat messages
    chat_result = db.ingest_chat(
        messages, conversation_id=session_id, platform=tool, extraction="heuristic",
    )

    # Ingest session metadata claim: (session_id) —[produced_by]→ (tool)
    db.ingest(
        subject=(session_id, "tool_session"),
        predicate=("produced_by", "produced_by"),
        object=(tool, "ai_tool"),
        provenance={
            "source_type": "agent_session",
            "source_id": session_id,
            "organization": tenant_id,
        },
        payload={
            "schema_ref": "session_metadata",
            "data": {
                "tool": tool,
                "platform": tool,
                "message_count": len(messages),
                "tenant_id": tenant_id,
            },
        },
    )

    # Ingest pending outcome marker: (session_id) —[awaiting_outcome]→ ("true")
    db.ingest(
        subject=(session_id, "tool_session"),
        predicate=("awaiting_outcome", "awaiting_outcome"),
        object=("true", "outcome_value"),
        provenance={
            "source_type": "agent_session",
            "source_id": session_id,
            "organization": tenant_id,
        },
    )

    outcome_status = "pending"
    # If outcome provided immediately, record it
    if outcome != "unknown":
        valid_outcomes = {"success", "partial", "failure", "unknown"}
        if outcome not in valid_outcomes:
            return json.dumps({
                "error": f"Invalid outcome: {outcome!r}. "
                f"Must be one of {sorted(valid_outcomes)}.",
            })
        _record_outcome_impl(
            db, session_id, tenant_id, outcome, task_description, notes,
        )
        outcome_status = outcome

    result_text = json.dumps({
        "session_id": session_id,
        "tenant_id": tenant_id,
        "tool": tool,
        "claims_ingested": chat_result.claims_ingested,
        "outcome_status": outcome_status,
        "token_estimate": _estimate_tokens(
            " ".join(m.get("content") or "" for m in messages)
        ),
        "message": f"Session {session_id} observed: {chat_result.claims_ingested} claims ingested",
    })
    return result_text


def _record_outcome_impl(
    db, session_id: str, tenant_id: str, outcome: str,
    task_description: str, notes: str,
) -> int:
    """Shared logic for recording an outcome against a session. Returns count of updated claims."""
    try:
        from attestdb.intelligence.ai_tools_vocabulary import AI_TOOLS_PREDICATES  # noqa: F401
    except ImportError:
        pass  # AI tools vocabulary not available

    # Ingest outcome claim: (session_id) —[had_outcome]→ (outcome)
    db.ingest(
        subject=(session_id, "tool_session"),
        predicate=("had_outcome", "had_outcome"),
        object=(outcome, "outcome_value"),
        provenance={
            "source_type": "outcome_report",
            "source_id": session_id,
            "organization": tenant_id,
        },
        payload={
            "schema_ref": "outcome_record",
            "data": {
                "outcome": outcome,
                "task_description": task_description,
                "notes": notes,
            },
        },
    )

    # Find session claims and apply confidence deltas
    deltas = {"success": 0.15, "partial": 0.05, "failure": -0.20}
    delta = deltas.get(outcome, 0.0)

    session_claims = db.claims_by_source_id(session_id)
    updated = 0
    for claim in session_claims:
        # Only adjust claims with AI tools predicates
        if claim.predicate.id not in AI_TOOLS_PREDICATES:
            continue
        new_conf = max(0.05, min(0.98, claim.confidence + delta))
        if new_conf != claim.confidence:
            # Ingest confidence_updated meta-claim
            db.ingest(
                subject=(claim.claim_id, "claim"),
                predicate=("confidence_updated", "confidence_updated"),
                object=(str(round(new_conf, 4)), "confidence_value"),
                provenance={
                    "source_type": "confidence_update",
                    "source_id": session_id,
                    "organization": tenant_id,
                },
                confidence=new_conf,
            )
            updated += 1

    # Mark outcome as resolved: (session_id) —[awaiting_outcome]→ ("false")
    db.ingest(
        subject=(session_id, "tool_session"),
        predicate=("awaiting_outcome", "awaiting_outcome"),
        object=("false", "outcome_value"),
        provenance={
            "source_type": "outcome_report",
            "source_id": session_id,
            "organization": tenant_id,
        },
    )

    return updated


@mcp.tool()
def attest_record_outcome(
    session_id: str,
    outcome: str,
    tenant_id: str = "default",
    task_description: str = "",
    notes: str = "",
) -> str:
    """Record the outcome of a previously observed session and update claim confidences.

    Args:
        session_id: Session ID from attest_observe_session
        outcome: "success", "partial", or "failure"
        tenant_id: Tenant/org identifier
        task_description: What the session was trying to accomplish
        notes: Additional context about the outcome
    """
    db = _get_db()

    if outcome not in ("success", "partial", "failure"):
        return json.dumps({
            "error": f"Invalid outcome: {outcome!r}. "
            "Must be success, partial, or failure.",
        })

    deltas = {"success": 0.15, "partial": 0.05, "failure": -0.20}
    updated = _record_outcome_impl(db, session_id, tenant_id, outcome, task_description, notes)

    result_text = json.dumps({
        "session_id": session_id,
        "tenant_id": tenant_id,
        "outcome": outcome,
        "ai_tools_claims_updated": updated,
        "confidence_delta": deltas[outcome],
        "token_estimate": _estimate_tokens(task_description + notes),
        "message": f"Outcome '{outcome}' recorded for {session_id}: {updated} claims updated",
    })
    return result_text


@mcp.tool()
def attest_get_prior_approaches(
    problem_description: str,
    tenant_id: str = "default",
    context: str = "",
    outcome_filter: Optional[str] = None,
    top_k: int = 5,
    max_tokens: int = 2000,
) -> str:
    """Retrieve prior approaches to a similar problem, ranked by outcome and recency.

    Args:
        problem_description: Description of the current problem
        tenant_id: Tenant/org identifier for scoping
        context: Additional context (e.g. codebase, language)
        outcome_filter: Filter by outcome: "success", "partial", "failure", or None (all)
        top_k: Maximum approaches to return
        max_tokens: Token budget for the response
    """
    from attestdb.core.vocabulary import KNOWLEDGE_PRIORITY

    db = _get_db()
    now_ns = time.time_ns()

    candidates = _retrieve_candidates(db, problem_description, context, tenant_id)

    # Separate into session-linked and direct knowledge claims
    session_approaches: dict[str, dict] = {}
    direct_knowledge: list[dict] = []
    session_meta_preds = {"produced_by", "awaiting_outcome", "confidence_updated", "had_outcome"}

    for claim in candidates:
        source_id = claim.provenance.source_id
        pred = claim.predicate.id

        # Direct knowledge claims (not session metadata)
        is_session_entity = claim.subject.entity_type == "tool_session"
        if pred not in session_meta_preds and not is_session_entity:
            age_days = (now_ns - claim.timestamp) / (86400 * 1e9) if claim.timestamp else 0
            # Warnings (priority 0) get full score, lower priority gets less boost
            priority = KNOWLEDGE_PRIORITY.get(pred, 9)
            priority_boost = 1.0 if priority <= 2 else 0.7 if priority <= 4 else 0.5
            base_score = _score_approach(claim.confidence, "success", age_days)
            direct_knowledge.append({
                "type": "knowledge",
                "subject": claim.subject.id,
                "predicate": pred,
                "object": claim.object.id,
                "confidence": round(claim.confidence, 4),
                "age_days": round(age_days, 1),
                "source_id": source_id,
                "score": round(base_score * priority_boost, 4),
            })
            continue

        # Session grouping
        if not source_id or source_id in session_approaches:
            continue
        session_claims = db.claims_by_source_id(source_id)
        if not session_claims:
            continue

        outcome = "unknown"
        session_confidence = 0.5
        for sc in session_claims:
            if sc.predicate.id == "had_outcome":
                outcome = sc.object.id
            session_confidence = max(session_confidence, sc.confidence)

        if outcome_filter and outcome != outcome_filter:
            continue

        ts = session_claims[0].timestamp
        age_days = (now_ns - ts) / (86400 * 1e9) if ts else 0
        score = _score_approach(session_confidence, outcome, age_days)
        predicates = [sc.predicate.id + " → " + sc.object.id for sc in session_claims[:5]]

        session_approaches[source_id] = {
            "type": "session",
            "session_id": source_id,
            "outcome": outcome,
            "score": round(score, 4),
            "confidence": round(session_confidence, 4),
            "age_days": round(age_days, 1),
            "claims": predicates,
            "claim_count": len(session_claims),
        }

    # Merge and sort by score descending, take top_k
    all_items = list(session_approaches.values()) + direct_knowledge
    approaches = sorted(all_items, key=lambda x: x["score"], reverse=True)[:top_k]

    # Enforce token budget (~150 tokens per entry)
    truncated = False
    tokens_used = 0
    kept = []
    for a in approaches:
        entry_tokens = _estimate_tokens(json.dumps(a))
        if tokens_used + entry_tokens > max_tokens:
            truncated = True
            break
        tokens_used += entry_tokens
        kept.append(a)

    result_text = json.dumps({
        "problem_description": problem_description,
        "tenant_id": tenant_id,
        "approaches": kept,
        "total_found": len(session_approaches) + len(direct_knowledge),
        "truncated": truncated,
        "token_estimate": tokens_used,
        "message": f"Found {len(kept)} results (of {len(session_approaches) + len(direct_knowledge)} total)",
    })
    return result_text


@mcp.tool()
def attest_confidence_trail(
    topic: str,
    tenant_id: str = "default",
    since_days: int = 30,
) -> str:
    """Trace how confidence in a topic evolved over time with contributing sessions.

    Args:
        topic: Entity or topic to trace
        tenant_id: Tenant/org identifier
        since_days: How far back to look (default: 30 days)
    """
    db = _get_db()

    # Check entity exists
    entity = db.get_entity(topic)
    if entity is None:
        return json.dumps({
            "error": f"Entity not found: {topic}",
            "suggestion": "Try searching with attest_get_prior_approaches or query_entity first.",
        })

    # Compute since timestamp in nanoseconds
    since_ns = time.time_ns() - (since_days * 86400 * 10**9)

    # Get current state via query
    frame = db.query(topic)
    current_state = {
        "entity_id": frame.focal_entity.id,
        "name": frame.focal_entity.name,
        "type": frame.focal_entity.entity_type,
        "claim_count": frame.claim_count,
        "relationships": len(frame.direct_relationships),
    }

    # Get evolution
    evolution = db.evolution(topic, since=since_ns)

    # Get claims and group by session (source_id)
    all_claims = db.claims_for(topic)
    sessions: dict[str, dict] = {}
    for claim in all_claims:
        # Tenant isolation: skip claims from other tenants
        prov_org = claim.provenance.organization
        if tenant_id and prov_org and prov_org != tenant_id:
            continue
        sid = claim.provenance.source_id
        if not sid:
            continue
        if sid not in sessions:
            # Look up outcome for this session
            outcome = "unknown"
            outcome_claims = db.claims_by_source_id(sid)
            for oc in outcome_claims:
                if oc.predicate.id == "had_outcome":
                    outcome = oc.object.id
                    break

            sessions[sid] = {
                "session_id": sid,
                "outcome": outcome,
                "claims_contributed": 0,
                "source_type": claim.provenance.source_type,
            }
        sessions[sid]["claims_contributed"] += 1

    # Build evolution summary
    evolution_summary = {
        "trajectory": evolution.trajectory,
        "new_claims": evolution.new_claims,
        "new_connections": len(evolution.new_connections),
        "confidence_changes": len(evolution.confidence_changes),
    }

    # Get diff
    try:
        diff = db.diff(since=since_ns)
        diff_summary = {
            "new_beliefs": len(diff.new_beliefs),
            "strengthened": len(diff.strengthened),
            "weakened": len(diff.weakened),
            "total_new_claims": diff.total_new_claims,
            "net_confidence": round(diff.net_confidence, 4),
            "summary": diff.summary,
        }
    except Exception:
        diff_summary = {}

    result = {
        "topic": topic,
        "tenant_id": tenant_id,
        "current_state": current_state,
        "sessions_contributing": list(sessions.values()),
        "evolution_summary": evolution_summary,
        "diff_summary": diff_summary,
        "token_estimate": _estimate_tokens(
            json.dumps(current_state)
            + json.dumps(list(sessions.values()))
        ),
        "message": (
            f"Confidence trail for '{topic}': "
            f"{len(sessions)} sessions, "
            f"trajectory={evolution.trajectory}"
        ),
    }
    return json.dumps(result)


@mcp.tool()
def attest_learned(
    subject: str,
    insight: str,
    insight_type: str = "pattern",
    confidence: float = 0.8,
) -> str:
    """Record a learning or finding. Simplest way to teach the knowledge graph.

    Args:
        subject: What it's about — file path, module name, concept, tool name
        insight: What you learned, in plain English
        insight_type: One of: bug, fix, pattern, decision, warning, tip, negative_result
        confidence: How confident you are (0.0-1.0, default 0.8)

    Examples:
        attest_learned("cursor.py", "batch triage fallback was calling per-claim LLM", "bug")
        attest_learned("normalization", "must strip Unicode Cf chars in both Python and Rust", "pattern")
        attest_learned("serde_json::Value", "can't use with bincode — calls deserialize_any", "warning")
    """
    db = _get_db()

    # Map insight_type to predicate
    type_to_predicate = {
        "bug": "had_bug",
        "fix": "has_fix",
        "pattern": "has_pattern",
        "decision": "has_decision",
        "warning": "has_warning",
        "tip": "has_tip",
        "negative_result": "no_evidence_for",
    }
    predicate = type_to_predicate.get(insight_type, "has_pattern")

    # Auto-detect entity type from subject
    if "/" in subject or subject.endswith(".py") or subject.endswith(".rs") or subject.endswith(".ts"):
        entity_type = "source_file"
    elif subject.endswith("()") or "::" in subject:
        entity_type = "function"
    else:
        entity_type = "concept"

    sid = _session_tracker["session_id"] if _session_tracker else "manual"
    claim_id = db.ingest(
        subject=(subject, entity_type),
        predicate=(predicate, "predicate"),
        object=(insight, insight_type),
        provenance={
            "source_type": "agent_learning",
            "source_id": sid,
        },
        confidence=confidence,
    )

    if _session_tracker is not None:
        _session_tracker["learnings_recorded"] += 1
    _track_claims_ingested(1)

    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[{insight_type}] {subject}: {insight}",
        "confidence": confidence,
    })


@mcp.tool()
def attest_check_file(file_path: str) -> str:
    """Check the knowledge graph for warnings, bugs, fixes, and patterns about a file.

    Use this before editing a file to see if there are known issues or patterns.
    This is the explicit-call equivalent of Claude Code's PreToolUse hook —
    designed for Cursor, Windsurf, Codex, and other tools without hook support.

    Args:
        file_path: Path to the file to check (absolute or relative)
    """
    _track_tool_call("attest_check_file", file_path)
    db = _get_db()

    from attestdb.core.vocabulary import (
        KNOWLEDGE_PREDICATES,
        KNOWLEDGE_PRIORITY,
        knowledge_label,
    )

    # Build search terms: basename and last two path segments
    basename = os.path.basename(file_path)
    search_terms = [basename]
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        search_terms.append("/".join(parts[-2:]))

    warnings: list[tuple[int, str]] = []  # (priority, formatted_line)
    seen: set[str] = set()

    for term in search_terms:
        entities = db.search_entities(term, top_k=3)
        for entity in entities:
            # Only include claims whose subject contains the search term
            if term.lower() not in entity.id.lower():
                continue
            for claim in db.claims_for(entity.id):
                if claim.claim_id in seen:
                    continue
                if claim.predicate.id in KNOWLEDGE_PREDICATES:
                    seen.add(claim.claim_id)
                    label = knowledge_label(claim.predicate.id)
                    pri = KNOWLEDGE_PRIORITY.get(claim.predicate.id, 9)
                    warnings.append((
                        pri,
                        f"[{label}] {claim.subject.id}: {claim.object.id}",
                    ))

    if not warnings:
        return f"No prior knowledge found for {basename}"

    # Sort by priority (warnings/patterns first)
    warnings.sort(key=lambda x: x[0])
    lines = [w[1] for w in warnings[:10]]
    return f"Attest knowledge for {basename}:\n" + "\n".join(f"  - {l}" for l in lines)


@mcp.tool()
def attest_session_end(
    outcome: str,
    summary: str,
    next_steps: str = "",
    files_changed: list[str] | None = None,
) -> str:
    """Record session outcome and what was accomplished. Call when work is done.

    Args:
        outcome: "success", "partial", or "failure"
        summary: Brief description of what was accomplished
        next_steps: What should happen next (picked up by recall in next session)
        files_changed: Key files that were modified
    """
    db = _get_db()

    if outcome not in ("success", "partial", "failure"):
        return json.dumps({"error": f"Invalid outcome: {outcome!r}. Use success/partial/failure."})

    sid = _session_tracker["session_id"] if _session_tracker else str(uuid.uuid4())

    # Record outcome with rich payload
    db.ingest(
        subject=(sid, "tool_session"),
        predicate=("had_outcome", "predicate"),
        object=(outcome, "outcome_value"),
        provenance={"source_type": "session_end", "source_id": sid},
        confidence=0.9,
        payload={
            "schema_ref": "session_outcome",
            "data": {
                "outcome": outcome,
                "summary": summary,
                "next_steps": next_steps,
                "files_changed": files_changed or [],
            },
        },
    )

    # Record next_steps as a separate claim so recall can find it
    if next_steps:
        db.ingest(
            subject=(sid, "tool_session"),
            predicate=("has_next_steps", "predicate"),
            object=(next_steps, "continuation"),
            provenance={"source_type": "session_end", "source_id": sid},
            confidence=0.85,
        )

    # Record files changed
    for f in (files_changed or []):
        db.ingest(
            subject=(sid, "tool_session"),
            predicate=("modified", "predicate"),
            object=(f, "source_file"),
            provenance={"source_type": "session_end", "source_id": sid},
            confidence=0.9,
        )

    return json.dumps({
        "session_id": sid,
        "outcome": outcome,
        "summary": summary,
        "next_steps": next_steps or "(none)",
        "files_recorded": len(files_changed or []),
        "message": f"Session {sid[:12]}... recorded as {outcome}",
    })


@mcp.tool()
def attest_negative_result(
    subject: str,
    hypothesis: str,
    search_strategy: str = "",
    confidence: float = 0.7,
) -> str:
    """Record that you investigated something and found NO evidence for it.

    This is critical for collaborative research — negative results prune the
    search tree for future sessions and other agents. Every failed investigation
    is data that prevents wasted effort.

    Args:
        subject: What entity you investigated (file, module, concept)
        hypothesis: What you were looking for and didn't find
        search_strategy: How you searched (e.g. "grep + read 5 files + tested")
        confidence: How thorough your search was (0.0-1.0, higher = more exhaustive)

    Examples:
        attest_negative_result("attest_db.py", "thread-safe concurrent writes", "tested with ThreadPoolExecutor")
        attest_negative_result("RustStore", "streaming query API", "read all 27 PyO3 methods")
        attest_negative_result("bincode", "serde_json::Value serialization support", "tested + read docs", 0.95)
    """
    db = _get_db()

    if "/" in subject or subject.endswith((".py", ".rs", ".ts")):
        entity_type = "source_file"
    elif subject.endswith("()") or "::" in subject:
        entity_type = "function"
    else:
        entity_type = "concept"

    sid = _session_tracker["session_id"] if _session_tracker else "manual"

    payload = None
    if search_strategy:
        payload = {
            "schema_ref": "negative_result",
            "data": {"search_strategy": search_strategy},
        }

    claim_id = db.ingest(
        subject=(subject, entity_type),
        predicate=("no_evidence_for", "research"),
        object=(hypothesis, "hypothesis"),
        provenance={"source_type": "agent_learning", "source_id": sid},
        confidence=confidence,
        payload=payload,
    )

    _track_claims_ingested(1)

    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[negative] {subject}: no evidence for '{hypothesis}'",
        "search_strategy": search_strategy or "(not specified)",
        "confidence": confidence,
        "note": "This will deprioritize future investigations of this hypothesis.",
    })


@mcp.tool()
def attest_research_context(entity_or_topic: str) -> str:
    """Get full research context for a topic: what's been tried, what failed, what strategies exist.

    Use this before starting a research task to avoid duplicating work.
    Returns negative results, prior findings, active strategies, and who has investigated before.

    Args:
        entity_or_topic: Entity ID or topic to get context for
    """
    _track_tool_call("attest_research_context", entity_or_topic)
    db = _get_db()

    from attestdb.infrastructure.agents import get_negative_results, get_task_context

    context = get_task_context(db, entity_or_topic)

    sections = []

    if context["negative_results"]:
        lines = []
        for nr in context["negative_results"]:
            agents = ", ".join(nr["agents"])
            strats = "; ".join(nr["search_strategies"]) if nr["search_strategies"] else "unspecified"
            lines.append(
                f"  - '{nr['hypothesis']}' — {nr['corroboration_count']} agent(s) found nothing "
                f"(searched: {strats})"
            )
        sections.append("## Dead ends (no evidence found)\n" + "\n".join(lines))

    if context["strategies"]:
        lines = []
        for s in context["strategies"]:
            lines.append(f"  - {s['strategy']} (confidence: {s['confidence']}, from: {s['agent']})")
        sections.append("## Active strategies\n" + "\n".join(lines))

    if context["findings"]:
        lines = []
        for f in context["findings"]:
            narrative = f["narrative"] or "(no narrative)"
            lines.append(f"  - [{f['agent']}] {narrative}")
        sections.append("## Prior findings\n" + "\n".join(lines))

    if context["prior_investigators"]:
        lines = []
        for inv in context["prior_investigators"]:
            lines.append(f"  - {inv['agent']} (status: {inv['status']})")
        sections.append("## Prior investigators\n" + "\n".join(lines))

    if not sections:
        return f"No research context found for '{entity_or_topic}'. This is unexplored territory."

    header = f"# Research context: {entity_or_topic}\n"
    return header + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Autoresearch tools (3)
# ---------------------------------------------------------------------------


@mcp.tool()
def autoresearch_log_experiment(
    metric_name: str,
    baseline_value: float,
    new_value: float,
    change_description: str,
    kept: bool,
    model: str = "",
    duration_seconds: float = 0.0,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> str:
    """Log an autoresearch experiment iteration: what changed, what was measured, keep or revert.

    Use this in autonomous code-generation loops to build a persistent record of
    what modifications improve or degrade a metric. Future iterations can query
    past experiments to avoid repeating failures.

    Args:
        metric_name: The metric being optimized (e.g. "pass_rate", "latency_p99", "accuracy")
        baseline_value: Metric value before the change
        new_value: Metric value after the change
        change_description: What was modified (e.g. "switched to chain-of-thought prompt")
        kept: Whether the change was kept (True) or reverted (False)
        model: LLM model used (e.g. "gpt-4o", "claude-sonnet-4-20250514")
        duration_seconds: Wall-clock time for the experiment
        tokens_in: Input tokens consumed
        tokens_out: Output tokens consumed
    """
    db = _get_db()
    sid = _session_tracker["session_id"] if _session_tracker else "manual"

    delta = new_value - baseline_value
    predicate = "improved_by" if kept and delta > 0 else "degraded_by"

    payload = {
        "schema_ref": "experiment_result",
        "data": {
            "metric_name": metric_name,
            "baseline_value": baseline_value,
            "new_value": new_value,
            "delta": delta,
            "kept": kept,
        },
    }
    if duration_seconds or tokens_in or tokens_out:
        payload["data"]["run_metrics"] = {
            "duration_seconds": duration_seconds,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }

    claim_id = db.ingest(
        subject=(f"experiment_{sid[:8]}_{int(time.time())}", "experiment_run"),
        predicate=(predicate, "relates_to"),
        object=(change_description, "code_pattern"),
        provenance={"source_type": "experiment_loop", "source_id": sid},
        confidence=0.95 if kept else 0.6,
        payload=payload,
    )
    _track_claims_ingested(1)

    # Also record the model association if provided
    if model:
        db.ingest(
            subject=(model, "model"),
            predicate=("benchmarked_at" if kept else "fails_at", "relates_to"),
            object=(metric_name, "intent_category"),
            provenance={"source_type": "experiment_loop", "source_id": sid},
            confidence=0.8,
            payload={"schema_ref": "quality_score", "data": {
                "dimension": metric_name, "score": new_value, "method": "autoresearch",
            }},
        )
        _track_claims_ingested(1)

    direction = "+" if delta >= 0 else ""
    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[{'kept' if kept else 'reverted'}] {change_description}: "
                    f"{metric_name} {baseline_value} → {new_value} ({direction}{delta:.4f})",
        "delta": delta,
        "kept": kept,
    })


@mcp.tool()
def autoresearch_get_priors(metric_name: str, limit: int = 20) -> str:
    """Get past experiment results for a metric — what worked and what didn't.

    Use this before starting a new experiment iteration to learn from prior runs.
    Results are sorted by delta (best improvements first).

    Args:
        metric_name: The metric to query (e.g. "pass_rate", "accuracy")
        limit: Maximum number of results to return (default: 20)
    """
    _track_tool_call("autoresearch_get_priors", metric_name)
    db = _get_db()

    # Search for experiment runs related to this metric
    results = db.search_entities(metric_name, limit=50)
    experiments = []

    for entity in results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            if claim.predicate.id in ("improved_by", "degraded_by"):
                payload_data = {}
                if hasattr(claim, "payload") and claim.payload:
                    pd = claim.payload if isinstance(claim.payload, dict) else {}
                    payload_data = pd.get("data", {})

                if payload_data.get("metric_name") == metric_name or not payload_data:
                    experiments.append({
                        "change": claim.object.name,
                        "effect": claim.predicate.id,
                        "delta": payload_data.get("delta", 0),
                        "baseline": payload_data.get("baseline_value"),
                        "new_value": payload_data.get("new_value"),
                        "kept": payload_data.get("kept", claim.predicate.id == "improved_by"),
                        "confidence": claim.confidence,
                        "timestamp": claim.timestamp,
                    })

    # Sort by absolute delta descending
    experiments.sort(key=lambda x: abs(x.get("delta", 0)), reverse=True)
    experiments = experiments[:limit]

    kept_count = sum(1 for e in experiments if e.get("kept"))
    reverted_count = len(experiments) - kept_count

    return json.dumps({
        "metric": metric_name,
        "total_experiments": len(experiments),
        "kept": kept_count,
        "reverted": reverted_count,
        "experiments": experiments,
        "message": f"{len(experiments)} prior experiments for '{metric_name}': "
                   f"{kept_count} kept, {reverted_count} reverted",
    })


@mcp.tool()
def autoresearch_suggest_next(
    metric_name: str,
    current_value: float,
    target_value: float = 0.0,
) -> str:
    """Suggest the next experiment based on what has and hasn't worked before.

    Analyzes past experiments to identify patterns: which changes improved the metric,
    which degraded it, and what hasn't been tried yet. Returns a prioritized list of
    suggestions.

    Args:
        metric_name: The metric being optimized
        current_value: Current metric value
        target_value: Target metric value (0 = no specific target)
    """
    _track_tool_call("autoresearch_suggest_next", metric_name)
    db = _get_db()

    # Gather past experiments
    results = db.search_entities(metric_name, limit=50)
    improvements = []
    failures = []

    for entity in results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            payload_data = {}
            if hasattr(claim, "payload") and claim.payload:
                pd = claim.payload if isinstance(claim.payload, dict) else {}
                payload_data = pd.get("data", {})

            if claim.predicate.id == "improved_by" and payload_data.get("kept"):
                improvements.append({
                    "change": claim.object.name,
                    "delta": payload_data.get("delta", 0),
                })
            elif claim.predicate.id == "degraded_by":
                failures.append({
                    "change": claim.object.name,
                    "delta": payload_data.get("delta", 0),
                })

    # Check for negative results (dead ends)
    dead_ends = []
    neg_results = db.search_entities("no_evidence_for", limit=20)
    for entity in neg_results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            if claim.predicate.id == "no_evidence_for":
                dead_ends.append(claim.object.name)

    suggestions = []

    # Suggest scaling up what worked
    for imp in sorted(improvements, key=lambda x: x["delta"], reverse=True)[:3]:
        suggestions.append({
            "action": f"Scale up or iterate on: {imp['change']}",
            "reason": f"Previously improved {metric_name} by {imp['delta']:.4f}",
            "priority": "high",
        })

    # Suggest avoiding what failed
    avoid_patterns = [f["change"] for f in failures[:5]]
    if avoid_patterns:
        suggestions.append({
            "action": f"Avoid: {'; '.join(avoid_patterns[:3])}",
            "reason": "These changes degraded the metric in past experiments",
            "priority": "info",
        })

    gap = target_value - current_value if target_value else None
    return json.dumps({
        "metric": metric_name,
        "current_value": current_value,
        "target_value": target_value or "(none)",
        "gap": gap,
        "suggestions": suggestions,
        "improvements_seen": len(improvements),
        "failures_seen": len(failures),
        "dead_ends": dead_ends[:5],
    })


# ---------------------------------------------------------------------------
# OpenClaw tools (5)
# ---------------------------------------------------------------------------


@mcp.tool()
def openclaw_ingest_action(
    agent_id: str,
    action_type: str,
    description: str,
    target: str = "",
    result: str = "",
    success: bool = True,
    metadata: str = "",
) -> str:
    """Record an autonomous agent action with full provenance.

    For self-hosted agents (OpenClaw-style) that need to build verified
    knowledge over time. Every action — code commit, message sent, API call,
    decision made — becomes a provenanced claim in the knowledge graph.

    Args:
        agent_id: Unique identifier for the agent instance
        action_type: Type of action (e.g. "code_commit", "message_sent", "api_call", "decision")
        description: What the agent did
        target: What it acted on (repo, channel, endpoint)
        result: Outcome description
        success: Whether the action succeeded
        metadata: JSON string of additional key-value pairs
    """
    db = _get_db()

    predicate = "performed" if success else "failed_at"
    confidence = 0.95 if success else 0.5

    payload_data = {
        "action_type": action_type,
        "success": success,
    }
    if result:
        payload_data["result"] = result
    if metadata:
        try:
            payload_data["metadata"] = json.loads(metadata)
        except json.JSONDecodeError:
            payload_data["metadata_raw"] = metadata

    target_entity = target or action_type
    target_type = "code_pattern" if action_type in ("code_commit", "code_review") else "intent"

    claim_id = db.ingest(
        subject=(agent_id, "model"),
        predicate=(predicate, "relates_to"),
        object=(target_entity, target_type),
        provenance={"source_type": "pipeline_run", "source_id": agent_id},
        confidence=confidence,
        payload={"schema_ref": "run_metrics", "data": payload_data},
    )
    _track_claims_ingested(1)

    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[{action_type}] {agent_id}: {description}" + (f" → {result}" if result else ""),
        "success": success,
    })


@mcp.tool()
def openclaw_query_knowledge(
    query: str,
    agent_id: str = "",
    action_type: str = "",
    limit: int = 20,
) -> str:
    """Query the knowledge graph for an autonomous agent's accumulated knowledge.

    Retrieves actions, learnings, and patterns relevant to a query. Optionally
    filter by agent_id or action_type.

    Args:
        query: What to search for (entity name, topic, or keyword)
        agent_id: Filter to a specific agent's knowledge (optional)
        action_type: Filter to a specific action type (optional)
        limit: Maximum results (default: 20)
    """
    _track_tool_call("openclaw_query_knowledge", query)
    db = _get_db()

    results = db.search_entities(query, limit=limit)
    knowledge = []

    for entity in results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            # Filter by agent_id if specified
            if agent_id and claim.provenance.source_id != agent_id:
                continue
            # Filter by action_type if specified
            if action_type:
                payload_data = {}
                if hasattr(claim, "payload") and claim.payload:
                    pd = claim.payload if isinstance(claim.payload, dict) else {}
                    payload_data = pd.get("data", {})
                if payload_data.get("action_type") != action_type:
                    continue

            knowledge.append({
                "subject": claim.subject.name,
                "predicate": claim.predicate.id,
                "object": claim.object.name,
                "confidence": claim.confidence,
                "source": claim.provenance.source_id,
                "timestamp": claim.timestamp,
            })

    knowledge.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    knowledge = knowledge[:limit]

    return json.dumps({
        "query": query,
        "results": len(knowledge),
        "knowledge": knowledge,
        "filters": {"agent_id": agent_id or "(all)", "action_type": action_type or "(all)"},
    })


@mcp.tool()
def openclaw_heartbeat_check(agent_id: str, window_minutes: int = 30) -> str:
    """Check if an autonomous agent has been active within a time window.

    Used for liveness monitoring — has the agent reported any actions recently?
    Returns the last action timestamp and a summary of recent activity.

    Args:
        agent_id: The agent to check
        window_minutes: How far back to look (default: 30 minutes)
    """
    _track_tool_call("openclaw_heartbeat_check", agent_id)
    db = _get_db()

    cutoff = time.time() - (window_minutes * 60)
    results = db.search_entities(agent_id, limit=10)

    recent_actions = []
    last_seen = 0

    for entity in results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            if claim.provenance.source_id == agent_id and claim.timestamp:
                ts = claim.timestamp
                if ts > last_seen:
                    last_seen = ts
                if ts >= cutoff:
                    recent_actions.append({
                        "predicate": claim.predicate.id,
                        "object": claim.object.name,
                        "timestamp": ts,
                    })

    alive = last_seen >= cutoff if last_seen else False

    return json.dumps({
        "agent_id": agent_id,
        "alive": alive,
        "last_seen": last_seen or None,
        "recent_actions": len(recent_actions),
        "window_minutes": window_minutes,
        "actions": recent_actions[:10],
        "message": f"{agent_id}: {'alive' if alive else 'no heartbeat'}"
                   + (f" ({len(recent_actions)} actions in last {window_minutes}m)" if alive else ""),
    })


@mcp.tool()
def openclaw_ingest_conversation(
    agent_id: str,
    platform: str,
    channel: str,
    message: str,
    author: str = "",
    intent: str = "",
) -> str:
    """Ingest a conversation message from any platform into the knowledge graph.

    For multi-platform messaging bridge agents that aggregate conversations
    from Slack, Discord, email, etc. Each message becomes a provenanced claim
    linking the author/channel to the content/intent.

    Args:
        agent_id: The agent doing the ingestion
        platform: Source platform (e.g. "slack", "discord", "email", "matrix")
        channel: Channel or thread identifier
        message: The message content (will be stored as the claim object)
        author: Message author (optional)
        intent: Detected intent or topic (optional, e.g. "bug_report", "feature_request")
    """
    db = _get_db()

    subject_name = f"{platform}/{channel}"
    if author:
        subject_name = f"{author}@{platform}/{channel}"

    object_name = intent if intent else message[:200]
    object_type = "intent" if intent else "concept"

    claim_id = db.ingest(
        subject=(subject_name, "concept"),
        predicate=("discussed", "relates_to"),
        object=(object_name, object_type),
        provenance={"source_type": "pipeline_run", "source_id": agent_id},
        confidence=0.7 if not intent else 0.85,
        payload={"data": {
            "platform": platform,
            "channel": channel,
            "message": message[:2000],
            "author": author,
        }},
    )
    _track_claims_ingested(1)

    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[{platform}/{channel}] {author or 'unknown'}: {message[:100]}...",
        "intent": intent or "(not classified)",
    })


@mcp.tool()
def openclaw_get_preferences(
    agent_id: str,
    category: str = "",
) -> str:
    """Get learned preferences and patterns for an autonomous agent.

    Aggregates the agent's historical actions to surface patterns: what it does most,
    what succeeds, what fails, preferred targets and action types.

    Args:
        agent_id: The agent to profile
        category: Optional category filter (e.g. "code_commit", "message_sent")
    """
    _track_tool_call("openclaw_get_preferences", agent_id)
    db = _get_db()

    results = db.search_entities(agent_id, limit=20)

    action_counts: dict[str, int] = {}
    success_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    targets: dict[str, int] = {}

    for entity in results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            if claim.provenance.source_id != agent_id:
                continue

            payload_data = {}
            if hasattr(claim, "payload") and claim.payload:
                pd = claim.payload if isinstance(claim.payload, dict) else {}
                payload_data = pd.get("data", {})

            action = payload_data.get("action_type", claim.predicate.id)
            if category and action != category:
                continue

            action_counts[action] = action_counts.get(action, 0) + 1
            if claim.predicate.id == "performed":
                success_counts[action] = success_counts.get(action, 0) + 1
            elif claim.predicate.id == "failed_at":
                failure_counts[action] = failure_counts.get(action, 0) + 1

            targets[claim.object.name] = targets.get(claim.object.name, 0) + 1

    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:10]

    return json.dumps({
        "agent_id": agent_id,
        "total_actions": sum(action_counts.values()),
        "top_actions": [{"action": a, "count": c} for a, c in top_actions],
        "top_targets": [{"target": t, "count": c} for t, c in top_targets],
        "success_rate": {
            action: success_counts.get(action, 0) / max(count, 1)
            for action, count in action_counts.items()
        },
        "category_filter": category or "(all)",
    })


# ---------------------------------------------------------------------------
# Resources (2)
# ---------------------------------------------------------------------------


@mcp.resource("attest://entities")
def list_all_entities() -> str:
    """Full entity list."""
    db = _get_db()
    entities = db.list_entities()
    return json.dumps([
        {"id": e.id, "name": e.name, "type": e.entity_type, "claim_count": e.claim_count}
        for e in entities
    ])


@mcp.resource("attest://schema")
def get_schema() -> str:
    """Knowledge graph schema."""
    db = _get_db()
    s = db.schema()
    return json.dumps(asdict(s))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point: attest-mcp"""
    import argparse

    parser = argparse.ArgumentParser(prog="attest-mcp")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind host for SSE/HTTP (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8892,
        help="Bind port for SSE/HTTP (default: 8892)",
    )
    parser.add_argument("--db", default=None, help="DB path (overrides ATTEST_DB_PATH)")
    args = parser.parse_args()

    global _db
    from attestdb.infrastructure.attest_db import AttestDB

    db_path = args.db or os.environ.get(
        "ATTEST_DB_PATH",
        os.environ.get("SUBSTRATE_DB_PATH", "attest.db"),
    )
    _db = AttestDB(db_path, embedding_dim=None)

    try:
        from attestdb.intelligence.ai_tools_vocabulary import register_ai_tools_vocabulary
        register_ai_tools_vocabulary(_db)
    except ImportError:
        pass  # AI tools vocabulary requires attestdb-intelligence

    try:
        from attestdb.intelligence.codegen_vocabulary import register_codegen_vocabulary
        register_codegen_vocabulary(_db)
    except ImportError:
        pass  # Codegen vocabulary requires attestdb-intelligence

    _init_session_tracker()

    def _cleanup():
        if _db is not None:
            _db.close()

    atexit.register(_cleanup)

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
