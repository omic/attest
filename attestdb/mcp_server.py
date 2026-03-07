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


def _flush_session():
    """On process exit, record the accumulated session."""
    if _session_tracker is None or not _session_tracker["tool_calls"]:
        return
    try:
        db = _get_db()
        messages = [
            {"role": "assistant", "content": f"Called {tc['tool']}: {tc['args_summary']}"}
            for tc in _session_tracker["tool_calls"]
        ]
        db.ingest_chat(
            messages,
            conversation_id=_session_tracker["session_id"],
            platform="mcp-auto-observe",
            extraction="heuristic",
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
# Tools (31)
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


@mcp.tool()
def ingest_batch(claims: list[dict]) -> str:
    """Bulk-ingest a list of claims. Each dict needs: subject, predicate, object, provenance."""
    from attestdb.core.types import ClaimInput

    db = _get_db()
    claim_inputs = [
        ClaimInput(
            subject=tuple(c["subject"]),
            predicate=tuple(c["predicate"]),
            object=tuple(c["object"]),
            provenance=c["provenance"],
            confidence=c.get("confidence"),
            payload=c.get("payload"),
        )
        for c in claims
    ]
    result = db.ingest_batch(claim_inputs)
    return json.dumps({
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": result.errors,
    })


@mcp.tool()
def query_entity(entity_id: str, depth: int = 1) -> str:
    """Query the knowledge graph around an entity. Returns narrative + relationships."""
    _track_tool_call("query_entity", entity_id)
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


@mcp.tool()
def attest_ask(question: str, top_k: int = 10) -> str:
    """Answer a natural-language question using the knowledge graph.

    Returns structured answer with citations, contradictions, and gap analysis.
    """
    _track_tool_call("attest_ask", question[:100])
    db = _get_db()
    result = db.ask(question, top_k=top_k)
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
    """Find single-source entities, knowledge gaps, and low-confidence areas."""
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


def _retrieve_candidates(db, query: str, context: str, tenant_id: str) -> list[str]:
    """ISOLATION BOUNDARY for retrieval. Returns claim IDs.

    Today: keyword search via db.search_entities() + db.claims_for().
    Tomorrow: ANN vector search. Nothing above this function changes.
    """
    claim_ids: list[str] = []
    seen: set[str] = set()
    # Search entities matching the query
    entities = db.search_entities(query, top_k=20)
    for entity in entities:
        for claim in db.claims_for(entity.id):
            if claim.claim_id not in seen:
                # Tenant filter: check provenance.organization if tenant scoping
                prov_org = claim.provenance.organization
                if tenant_id and prov_org and prov_org != tenant_id:
                    continue
                seen.add(claim.claim_id)
                claim_ids.append(claim.claim_id)
    return claim_ids


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
    db = _get_db()
    now_ns = time.time_ns()

    candidate_ids = _retrieve_candidates(db, problem_description, context, tenant_id)

    # Group candidates by session
    session_approaches: dict[str, dict] = {}
    for cid in candidate_ids:
        # candidate_ids are claim_ids — resolve the actual claim to get source_id
        # The claim's provenance.source_id is the session_id
        candidate_claims = db.claims_for(cid)
        if not candidate_claims:
            continue

        for cc in candidate_claims:
            session_id_resolved = cc.provenance.source_id
            if not session_id_resolved or session_id_resolved in session_approaches:
                continue

            # Look up all claims for this session
            session_claims = db.claims_by_source_id(session_id_resolved)
            if not session_claims:
                continue

            # Find outcome for this session
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

            # Collect predicates from session
            predicates = [sc.predicate.id + " → " + sc.object.id for sc in session_claims[:5]]

            session_approaches[session_id_resolved] = {
                "session_id": session_id_resolved,
                "outcome": outcome,
                "score": round(score, 4),
                "confidence": round(session_confidence, 4),
                "age_days": round(age_days, 1),
                "claims": predicates,
                "claim_count": len(session_claims),
            }

    # Sort by score descending, take top_k
    approaches = sorted(session_approaches.values(), key=lambda x: x["score"], reverse=True)[:top_k]

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
        "total_found": len(session_approaches),
        "truncated": truncated,
        "token_estimate": tokens_used,
        "message": f"Found {len(kept)} prior approaches (of {len(session_approaches)} total)",
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
