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
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)
from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("attest", instructions="Knowledge graph database with provenance-tracked claims")

# Global DB reference — set by main() or configure()
_db = None


def _serialize(obj: object) -> str:
    """Serialize a dataclass to JSON (used by most MCP tool return values)."""
    return json.dumps(asdict(obj))

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


class _namespace_scope:
    """Context manager that temporarily scopes the global DB to a namespace.

    FastMCP calls sync tool handlers directly on the asyncio event loop
    (no threading), so concurrent tool calls are impossible — the set-
    query-restore pattern is safe.  This wrapper ensures the namespace
    filter is always restored, even on exceptions.
    """

    __slots__ = ("_db", "_prev")

    def __init__(self, db, namespace: str):
        self._db = db
        self._prev = db.get_namespaces()
        db.set_namespace(namespace)

    def __enter__(self):
        return self._db

    def __exit__(self, *exc):
        self._db.set_namespaces(self._prev)
        return False


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
    namespace: str = "",
    ttl_seconds: int = 0,
) -> str:
    """Add a claim (subject-predicate-object triple) to the knowledge graph.

    Returns the claim_id (SHA-256 hash). Requires all fields — omitting
    source_type or source_id will raise a ProvenanceError.
    Optional namespace for team isolation (empty = global).
    Optional ttl_seconds for automatic expiry (0 = never expires).
    """
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
        namespace=namespace,
        ttl_seconds=ttl_seconds,
    )


@mcp.tool()
def ingest_text(text: str, source_id: str = "") -> str:
    """Extract claims from unstructured text and ingest them.

    Returns JSON with n_valid (claims ingested), raw_count, and warnings.
    May return 0 claims if text has no extractable relationships — this is
    normal, not an error. Requires attestdb-enterprise for LLM extraction;
    falls back to heuristic patterns without it.
    """
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
    """Bulk-ingest a list of claims. Returns {ingested, duplicates, errors}.

    Each dict needs: subject, predicate, object, provenance.
    subject/predicate/object can be a string ("entity_name") or
    a pair ["entity_name", "entity_type"]. provenance must be a dict
    with at least {source_type, source_id}.
    """
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
    """Query the knowledge graph around an entity. Returns narrative + relationships.

    Returns empty relationships if entity_id doesn't exist (not an error).
    Use search_entities first if you're unsure of the exact entity ID.
    """
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
            "source_url": _source_url(c.provenance.source_id, c.provenance.source_type),
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
    return _serialize(report)


def _source_url(source_id: str, source_type: str = "") -> str | None:
    """Resolve source_id to URL (cached import)."""
    from attestdb.core.provenance import resolve_source_url
    return resolve_source_url(source_id, source_type)


@mcp.tool()
def resolve_source_url(source_id: str, source_type: str = "") -> str:
    """Resolve a claim's source_id to a clickable URL for the original source.

    Returns the URL string or null if no URL can be derived.
    Supports 30+ sources: PubMed, UniProt, CTD, Slack, GitHub, etc.
    """
    _track_tool_call("resolve_source_url", source_id[:80])
    url = _source_url(source_id, source_type)
    return json.dumps({"source_id": source_id, "url": url})


@mcp.tool()
def knowledge_health() -> str:
    """Quantified health metrics for the knowledge graph."""
    db = _get_db()
    health = db.knowledge_health()
    return _serialize(health)


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
    return _serialize(s)


@mcp.tool()
def stats() -> str:
    """Get database statistics."""
    db = _get_db()
    return json.dumps(db.stats(), default=str)


@mcp.tool()
def set_namespace(namespace: str = "") -> str:
    """Filter all queries to a single namespace for team isolation.

    Pass empty string to clear the filter and see all namespaces.
    Claims, queries, and the change feed respect this filter.
    """
    db = _get_db()
    db.set_namespace(namespace)
    ns = db.get_namespaces()
    return json.dumps({"namespace_filter": ns, "status": "active" if ns else "all"})


@mcp.tool()
def changes(since: int = 0, limit: int = 100) -> str:
    """Poll for new claims since a cursor timestamp.

    Returns claims ingested after `since` (nanosecond timestamp).
    Use the returned `cursor` value as `since` in the next call
    for reliable cursor-based pagination. Respects namespace filter.
    """
    db = _get_db()
    claims, cursor = db.changes(since=since, limit=limit)
    return json.dumps({
        "claims": [
            {
                "claim_id": c.claim_id,
                "subject": c.subject.id,
                "predicate": c.predicate.id,
                "object": c.object.id,
                "confidence": c.confidence,
                "namespace": c.namespace,
                "timestamp": c.timestamp,
                "source_url": _source_url(
                    c.provenance.source_id if c.provenance else "",
                    c.provenance.source_type if c.provenance else "",
                ),
            }
            for c in claims
        ],
        "cursor": cursor,
        "count": len(claims),
    })


@mcp.tool()
def audit_log(
    since: int = 0,
    event_type: Optional[str] = None,
    actor: Optional[str] = None,
    limit: int = 100,
) -> str:
    """Query the mutation audit log for compliance and governance.

    Returns timestamped events (claim_ingested, source_retracted,
    batch_ingested) with actor attribution. Use `since` cursor
    for pagination.
    """
    db = _get_db()
    from dataclasses import asdict
    events = db.audit_log(since=since, event_type=event_type, actor=actor, limit=limit)
    return json.dumps([asdict(e) for e in events], default=str)


_ASK_STOP_WORDS = frozenset({
    "what", "who", "how", "why", "where", "which", "does", "the", "are",
    "for", "and", "is", "was", "has", "have", "been", "this", "that",
    "with", "from", "about", "tell", "me", "show", "find", "get", "list",
    "of", "in", "on", "at", "by", "an", "a", "do", "did", "any", "all",
    "to", "it", "be", "not", "no", "or", "but", "if", "can", "will",
})


@mcp.tool()
def attest_ask(question: str, namespace: str = "", top_k: int = 10) -> str:
    """Answer a natural-language question using the knowledge graph.

    Returns structured answer with citations, contradictions, and gap analysis.

    Pass a namespace to scope the answer to a specific session or team.
    Leave empty to search the global namespace (all claims).
    """
    _track_tool_call("attest_ask", question[:100])
    db = _get_db()

    if namespace:
        with _namespace_scope(db, namespace):
            return _attest_ask_impl(db, question, top_k)
    else:
        return _attest_ask_impl(db, question, top_k)


def _attest_ask_impl(db, question: str, top_k: int) -> str:
    """Core ask logic, operates on the given db instance."""
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
    except Exception as e:
        logger.warning("Full ask() failed, falling back to text search: %s", e)

    # Fallback: text search when ask() returns empty (no LLM / no embeddings)
    from attestdb.core.vocabulary import knowledge_label, knowledge_sort_key

    words = [w.strip("?.,!\"'()[]{}:;") for w in question.lower().split()]
    query_terms = [w for w in words if w and len(w) >= 2 and w not in _ASK_STOP_WORDS]

    candidates, sim_scores = _retrieve_candidates(db, " ".join(query_terms), "", "")

    # Sort by: semantic similarity (if available) blended with knowledge priority
    # Similarity boost: claims with high semantic match get promoted within their tier
    def _sort_key(c):
        pri_key = knowledge_sort_key(c.predicate.id, c.confidence)
        sim = sim_scores.get(c.claim_id, 0.0)
        # pri_key is (priority_tier, -confidence). Inject similarity as a tiebreaker:
        # within the same priority tier, higher similarity sorts first
        return (pri_key[0], -sim, pri_key[1])

    candidates.sort(key=_sort_key)

    lines = []
    for c in candidates[:top_k]:
        label = knowledge_label(c.predicate.id)
        sim = sim_scores.get(c.claim_id)
        sim_tag = f" sim={sim:.2f}" if sim else ""
        lines.append(f"[{label}] {c.subject.id}: {c.object.id} (conf={c.confidence:.2f}{sim_tag})")

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


@mcp.tool()
def claims_in_namespace(namespace: str, limit: int = 500) -> str:
    """Return all claims stored under the given namespace.

    Used to build a full KB snapshot for a session without relying on
    in-memory state in the client. Each ReAct3 chat session uses its
    chatId as the namespace, so this returns all claims for that session.

    An empty namespace string returns ``{"namespace": "", "count": 0, "claims": []}``.
    """
    _track_tool_call("claims_in_namespace", namespace[:100])
    if not namespace:
        return json.dumps({"namespace": "", "count": 0, "claims": []})
    db = _get_db()
    with _namespace_scope(db, namespace):
        claims = []
        for claim in db.iter_claims():
            payload_data = claim.payload.data if claim.payload else {}
            src_id = claim.provenance.source_id if claim.provenance else ""
            src_type = claim.provenance.source_type if claim.provenance else ""
            claims.append({
                "claim_id":   claim.claim_id,
                "subject":    claim.subject.id,
                "predicate":  claim.predicate.id,
                "object":     claim.object.id,
                "confidence": claim.confidence,
                "source_type": src_type,
                "source_id":   src_id,
                "source_url":  _source_url(src_id, src_type),
                "payload":    payload_data,
                "timestamp":  claim.timestamp,
            })
            if len(claims) >= limit:
                break
        return json.dumps({"namespace": namespace, "count": len(claims), "claims": claims})


# ---------------------------------------------------------------------------
# New API tools (9)
# ---------------------------------------------------------------------------


@mcp.tool()
def attest_impact(source_id: str) -> str:
    """Analyze the impact of a source: what claims depend on it."""
    db = _get_db()
    report = db.impact(source_id)
    return _serialize(report)


@mcp.tool()
def attest_blindspots(min_claims: int = 5) -> str:
    """Find single-source entities, knowledge gaps, low-confidence areas, and unresolved warnings."""
    db = _get_db()
    report = db.blindspots(min_claims=min_claims)
    return _serialize(report)


@mcp.tool()
def attest_consensus(topic: str) -> str:
    """Analyze consensus around an entity/topic across sources."""
    db = _get_db()
    report = db.consensus(topic)
    return _serialize(report)


@mcp.tool()
def attest_corroboration(min_sources: int = 2) -> str:
    """Report on corroboration status: what's independently confirmed vs single-source.

    Shows which claims have been attested by multiple independent sources (with
    confidence boost info) and which need corroboration. Use this to understand
    knowledge reliability and identify where independent confirmation is needed.

    Args:
        min_sources: Minimum independent sources to count as corroborated (default: 2)
    """
    _track_tool_call("attest_corroboration", str(min_sources))
    db = _get_db()
    report = db.corroboration_report(min_sources=min_sources)

    lines = []
    total = report["total_content_ids"]
    n_corr = report["corroborated_count"]
    n_single = report["single_source_count"]
    ratio = report["corroboration_ratio"]

    lines.append(f"## Corroboration Report")
    lines.append(f"**{n_corr}/{total}** facts corroborated ({ratio:.1%}), "
                 f"**{n_single}** single-source")
    lines.append("")

    if report["corroborated"]:
        lines.append("### Corroborated (independently confirmed)")
        for c in report["corroborated"][:15]:
            sources = ", ".join(c["source_types"])
            lines.append(
                f"  - {c['subject']} {c['predicate']} {c['object']} "
                f"({c['n_independent_sources']} sources, {c['confidence_boost']}x boost, "
                f"via: {sources})"
            )
        lines.append("")

    if report["needs_corroboration"]:
        lines.append("### Needs corroboration (single-source)")
        for c in report["needs_corroboration"][:15]:
            lines.append(
                f"  - {c['subject']} {c['predicate']} {c['object']} "
                f"(conf={c['confidence']:.2f}, source: {c['source_type']})"
            )

    return "\n".join(lines)


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
    return _serialize(trail)


@mcp.tool()
def attest_drift(days: int = 30) -> str:
    """Measure knowledge drift over a time period."""
    db = _get_db()
    report = db.drift(days=days)
    return _serialize(report)


@mcp.tool()
def attest_source_reliability(source_id: Optional[str] = None) -> str:
    """Per-source corroboration and retraction rates."""
    db = _get_db()
    result = db.source_reliability(source_id=source_id)
    return json.dumps(result, default=str)


@mcp.tool()
def attest_build_status() -> str:
    """Latest reference DB build summary: sources ok/failed, timing, claim counts."""
    db = _get_db()
    manifest = db.build_manifest()
    report = manifest.latest_build()
    if report is None:
        return json.dumps({"error": "No build manifest found"})
    from dataclasses import asdict
    return json.dumps(asdict(report))


@mcp.tool()
def attest_source_health(source_id: Optional[str] = None) -> str:
    """Per-source health: live claim count from LMDB + last build info + errors.

    If source_id is given, returns detail for that source only.
    Otherwise returns all sources.
    """
    db = _get_db()
    health = db.source_health()
    if source_id:
        health = [h for h in health if h["source_id"] == source_id]
    return json.dumps(health)


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
    return _serialize(report)


# Active sandboxes for multi-step sandbox sessions
_active_sandboxes: dict[str, object] = {}


@mcp.tool()
def attest_what_if(
    subject_id: str, subject_type: str,
    predicate_id: str, predicate_type: str,
    object_id: str, object_type: str,
    confidence: float = 0.6,
) -> str:
    """Rich hypothetical reasoning: create sandbox, analyze evidence paths, gaps, contradictions.

    Returns a SandboxVerdict with multi-hop evidence, predicted predicates,
    gap detection, follow-up suggestions, and an overall verdict — all instant,
    no LLM calls.
    """
    db = _get_db()
    verdict = db.what_if(
        (subject_id, subject_type),
        (predicate_id, predicate_type),
        (object_id, object_type),
        confidence=confidence,
    )
    return _serialize(verdict)


@mcp.tool()
def attest_sandbox_create(
    description: str,
    seed_entities: Optional[str] = None,
) -> str:
    """Create a hypothetical sandbox for multi-step reasoning.

    Returns a sandbox_id. Use attest_sandbox_analyze to test hypotheses in it.
    seed_entities: comma-separated entity IDs to preload.
    """
    db = _get_db()
    entities = [e.strip() for e in seed_entities.split(",")] if seed_entities else []
    ctx = db.hypothetical_sandbox(description, seed_entities=entities)
    sandbox_id = str(uuid.uuid4())[:8]
    _active_sandboxes[sandbox_id] = ctx
    return json.dumps({"sandbox_id": sandbox_id, "description": description,
                        "claims_loaded": ctx.sandbox.count_claims()})


@mcp.tool()
def attest_sandbox_analyze(
    sandbox_id: str,
    subject_id: str, subject_type: str,
    predicate_id: str, predicate_type: str,
    object_id: str, object_type: str,
    confidence: float = 0.6,
) -> str:
    """Test a hypothesis in an existing sandbox and get rich analysis.

    Returns SandboxVerdict with evidence paths, contradictions, gaps, and suggestions.
    """
    ctx = _active_sandboxes.get(sandbox_id)
    if ctx is None:
        return json.dumps({"error": f"Sandbox {sandbox_id} not found. Create one with attest_sandbox_create."})
    verdict = ctx.ingest_and_analyze(
        subject=(subject_id, subject_type),
        predicate=(predicate_id, predicate_type),
        object_=(object_id, object_type),
        confidence=confidence,
    )
    return _serialize(verdict)


@mcp.tool()
def attest_predict(
    entity_id: str,
    max_intermediaries: int = 100,
    min_paths: int = 3,
    top_k: int = 20,
) -> str:
    """Discover novel regulatory predictions for an entity via causal composition.

    Follows causal edges (activates, inhibits, upregulates, downregulates)
    through intermediaries and composes them to predict new relationships.
    Returns predictions ranked by convergent evidence — genuine gaps first.
    No LLM calls. Pure graph structure.

    Example: attest_predict("gene_7157") discovers genes TP53 likely
    up/downregulates based on shared chemical intermediaries across 30+ databases.
    """
    db = _get_db()
    predictions = db.predict(
        entity_id,
        max_intermediaries=max_intermediaries,
        min_paths=min_paths,
    )
    results = []
    for p in predictions[:top_k]:
        results.append({
            "target": p.target,
            "predicted_predicate": p.predicted_predicate,
            "supporting_paths": p.supporting_paths,
            "opposing_paths": p.opposing_paths,
            "consensus": round(p.consensus, 2),
            "intermediaries": p.intermediaries,
            "is_gap": p.is_gap,
            "existing_predicates": p.existing_predicates,
            "top_evidence": [
                {
                    "path": ie.path,
                    "predicates": ie.predicates,
                    "composed": ie.predicted_predicate,
                    "confidence": round(ie.confidence, 4),
                }
                for ie in p.evidence[:3]
            ],
        })
    return json.dumps({"entity": entity_id, "predictions": len(predictions),
                        "top": results}, indent=2)


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


def _retrieve_candidates(
    db, query: str, context: str, tenant_id: str,
) -> tuple[list, dict[str, float]]:
    """ISOLATION BOUNDARY for retrieval.

    Returns (claims, similarity_scores) where similarity_scores maps
    claim_id → cosine similarity (0-1, higher = more similar). Claims found
    only via text search have no entry in similarity_scores.
    """
    claims: list = []
    seen: set[str] = set()
    similarity_scores: dict[str, float] = {}

    def _add_claim(claim):
        if claim.claim_id not in seen:
            prov_org = claim.provenance.organization
            if tenant_id and prov_org and prov_org != tenant_id:
                return
            seen.add(claim.claim_id)
            claims.append(claim)

    # 1. Semantic search via embedding index (highest quality when available)
    semantic_claim_ids: set[str] = set()
    embed_idx = getattr(db, "_embedding_index", None)
    embed_fn = getattr(getattr(db, "_pipeline", None), "_embed_fn", None)
    if embed_idx and len(embed_idx) > 0 and embed_fn:
        try:
            query_vec = embed_fn(query)
            hits = embed_idx.search(query_vec, top_k=30)
            # usearch returns cosine distance (0 = identical); convert to similarity
            for claim_id, dist in hits:
                semantic_claim_ids.add(claim_id)
                similarity_scores[claim_id] = max(0.0, 1.0 - float(dist))
            # Resolve claim_ids to full Claim objects via direct lookup
            for cid in semantic_claim_ids:
                claim = db.get_claim(cid)
                if claim:
                    _add_claim(claim)
        except Exception as e:
            logger.warning("Embedding search failed: %s", e)

    # 2. Entity search: find entities matching query, get their claims
    entities = db.search_entities(query, top_k=20)
    for entity in entities:
        for claim in db.claims_for(entity.id):
            _add_claim(claim)

    # 3. Predicate search: scan knowledge predicates for matching claims
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
            except Exception as e:
                logger.warning("Predicate scan failed for %s: %s", pred, e)

    # Prune similarity scores to only include resolved claims (namespace filtering
    # may have excluded some embedding hits from other namespaces)
    if similarity_scores:
        similarity_scores = {cid: s for cid, s in similarity_scores.items() if cid in seen}

    return claims, similarity_scores


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
        from attestdb.intelligence.ai_tools_vocabulary import AI_TOOLS_PREDICATES
    except ImportError:
        AI_TOOLS_PREDICATES = set()  # AI tools vocabulary not available

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

    candidates, _sim_scores = _retrieve_candidates(db, problem_description, context, tenant_id)

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
    except Exception as e:
        logger.warning("Failed to compute diff: %s", e)
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
        insight_type: One of: bug, fix, pattern, decision, warning, tip, negative_result,
                      goal, strategy, architecture, trade_off, convention, dependency, requirement
        confidence: How confident you are (0.0-1.0, default 0.8)

    Examples:
        attest_learned("cursor.py", "batch triage fallback was calling per-claim LLM", "bug")
        attest_learned("normalization", "must strip Unicode Cf chars in both Python and Rust", "pattern")
        attest_learned("serde_json::Value", "can't use with bincode — calls deserialize_any", "warning")
        attest_learned("attestdb", "become the default memory layer for coding agents", "goal")
        attest_learned("ingestion pipeline", "validate before persist — never mix side effects", "architecture")
        attest_learned("RustStore", "single-writer with fs2 advisory locks", "convention")
        attest_learned("embedding_index", "depends on usearch for HNSW", "dependency")
        attest_learned("ClaimInput vs positional args", "ClaimInput cleaner but must keep backward compat", "trade_off")
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
        "goal": "has_goal",
        "strategy": "has_strategy",
        "architecture": "has_architecture",
        "trade_off": "has_trade_off",
        "convention": "has_convention",
        "dependency": "depends_on",
        "requirement": "has_requirement",
    }
    predicate = type_to_predicate.get(insight_type, "has_pattern")

    # Auto-detect entity type from subject
    if "/" in subject or subject.endswith((".py", ".rs", ".ts", ".js", ".go")):
        entity_type = "source_file"
    elif subject.endswith("()") or "::" in subject:
        entity_type = "function"
    elif any(kw in subject.lower() for kw in ("api", "endpoint", "route")):
        entity_type = "api"
    elif any(kw in subject.lower() for kw in ("service", "server", "worker")):
        entity_type = "service"
    elif insight_type == "goal":
        entity_type = "goal"
    elif insight_type in ("architecture", "convention", "strategy"):
        entity_type = "system"
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

    extra_claims = 0

    # Auto-link fixes to existing bugs on the same subject
    if insight_type == "fix":
        existing = db.claims_for(subject)
        already_resolved = {
            c.object.id for c in existing if c.predicate.id == "resolved"
        }
        for c in existing:
            if c.predicate.id == "had_bug" and c.object.id not in already_resolved:
                db.ingest(
                    subject=(subject, entity_type),
                    predicate=("resolved", "resolved"),
                    object=(c.object.id, "error_class"),
                    provenance={
                        "source_type": "agent_learning",
                        "source_id": sid,
                        "chain": [c.claim_id],
                    },
                    confidence=confidence,
                )
                extra_claims += 1

    # Auto-generate inverse for dependency relationships
    if insight_type == "dependency":
        db.ingest(
            subject=(insight, "concept"),
            predicate=("is_dependency_of", "predicate"),
            object=(subject, entity_type),
            provenance={
                "source_type": "agent_learning",
                "source_id": sid,
                "chain": [claim_id],
            },
            confidence=confidence,
        )
        extra_claims += 1

    total_claims = 1 + extra_claims
    if _session_tracker is not None:
        _session_tracker["learnings_recorded"] += 1
    _track_claims_ingested(total_claims)

    return json.dumps({
        "claim_id": claim_id,
        "recorded": f"[{insight_type}] {subject}: {insight}",
        "confidence": confidence,
        "extra_claims": extra_claims,
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
# Visualization tools (2)
# ---------------------------------------------------------------------------


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a Unicode bar chart segment."""
    if max_val <= 0:
        return " " * width
    filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _health_indicator(value: float, thresholds: tuple) -> str:
    """Return a status indicator based on thresholds (bad, ok, good)."""
    bad, good = thresholds
    if value >= good:
        return "[OK]"
    elif value >= bad:
        return "[~~]"
    else:
        return "[!!]"


def _sparkline(values: list[float]) -> str:
    """Render a sparkline from a list of values."""
    if not values:
        return ""
    blocks = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(blocks[min(int((v - mn) / rng * 7), 7)] for v in values)


@mcp.tool()
def attest_dashboard() -> str:
    """Visual dashboard of knowledge graph health, gaps, and development trajectory.

    Returns a rich Unicode text visualization showing:
    - Health score with component breakdown
    - Entity type distribution (bar chart)
    - Source diversity
    - Confidence distribution
    - Identified gaps and recommendations

    Use this to answer "what should we develop next?" and "where are the gaps?"
    """
    _track_tool_call("attest_dashboard", "")
    db = _get_db()

    health = db.knowledge_health()
    st = db.stats()
    blindspots = db.blindspots(min_claims=2)

    lines = []
    W = 58

    # Header
    lines.append("\u2554" + "\u2550" * W + "\u2557")
    lines.append("\u2551" + "  ATTEST KNOWLEDGE DASHBOARD".ljust(W) + "\u2551")
    lines.append("\u2560" + "\u2550" * W + "\u2563")

    # Health score
    score = health.health_score
    score_bar = _bar(score, 100, 30)
    lines.append("\u2551" + f"  Health Score: {score_bar} {score:.0f}/100".ljust(W) + "\u2551")
    lines.append("\u2551" + f"  Claims: {health.total_claims:<6} Entities: {health.total_entities:<6} Sources: {health.source_diversity}".ljust(W) + "\u2551")
    lines.append("\u2551" + " " * W + "\u2551")

    # Component scores
    lines.append("\u2551" + "  COMPONENT SCORES".ljust(W) + "\u2551")
    components = [
        ("Confidence", health.avg_confidence, (0.5, 0.8)),
        ("Freshness", health.freshness_score, (0.3, 0.7)),
        ("Corroboration", health.corroboration_ratio, (0.1, 0.3)),
        ("Multi-source", health.multi_source_ratio, (0.2, 0.5)),
        ("Density", min(health.knowledge_density / 5.0, 1.0), (0.3, 0.6)),
    ]
    for name, val, thresh in components:
        bar = _bar(val, 1.0, 15)
        ind = _health_indicator(val, thresh)
        lines.append("\u2551" + f"  {name:<14} {bar} {val:.2f} {ind}".ljust(W) + "\u2551")

    lines.append("\u2551" + " " * W + "\u2551")

    # Entity type distribution
    entity_types = st.get("entity_types", {})
    if entity_types:
        lines.append("\u2551" + "  ENTITY DISTRIBUTION".ljust(W) + "\u2551")
        sorted_types = sorted(entity_types.items(), key=lambda x: x[1], reverse=True)
        max_count = max(entity_types.values()) if entity_types else 1
        for etype, count in sorted_types[:8]:
            bar = _bar(count, max_count, 15)
            lines.append("\u2551" + f"  {etype:<16} {bar} {count}".ljust(W) + "\u2551")
        lines.append("\u2551" + " " * W + "\u2551")

    # Source type distribution
    source_types = st.get("source_types", {})
    if source_types:
        lines.append("\u2551" + "  SOURCE DISTRIBUTION".ljust(W) + "\u2551")
        sorted_sources = sorted(source_types.items(), key=lambda x: x[1], reverse=True)
        max_src = max(source_types.values()) if source_types else 1
        for stype, count in sorted_sources[:6]:
            bar = _bar(count, max_src, 15)
            lines.append("\u2551" + f"  {stype:<16} {bar} {count}".ljust(W) + "\u2551")
        lines.append("\u2551" + " " * W + "\u2551")

    # Predicate distribution (universal — works for any knowledge graph)
    pred_types = st.get("predicate_types", {})
    if pred_types:
        lines.append("\u2551" + "  RELATIONSHIP TYPES".ljust(W) + "\u2551")
        sorted_preds = sorted(pred_types.items(), key=lambda x: x[1], reverse=True)
        max_pred = max(pred_types.values()) if pred_types else 1
        for pred, count in sorted_preds[:8]:
            bar = _bar(count, max_pred, 12)
            lines.append("\u2551" + f"  {pred:<18} {bar} {count}".ljust(W) + "\u2551")
        if len(pred_types) > 8:
            lines.append("\u2551" + f"  ... and {len(pred_types) - 8} more types".ljust(W) + "\u2551")
        lines.append("\u2551" + " " * W + "\u2551")

    # Namespace & RBAC status
    ns_filter = db.get_namespaces()
    rbac_on = getattr(db, "_rbac_enabled", False)
    actor = getattr(db, "_actor", "")
    ns_line = ", ".join(ns_filter) if ns_filter else "all (no filter)"
    lines.append("\u2551" + f"  Namespace: {ns_line}".ljust(W) + "\u2551")
    lines.append("\u2551" + f"  RBAC: {'enabled' if rbac_on else 'off':<10} Actor: {actor or '(none)'}".ljust(W) + "\u2551")

    lines.append("\u2551" + " " * W + "\u2551")

    # Confidence trend
    trend_dir = "\u2191" if health.confidence_trend > 0 else ("\u2193" if health.confidence_trend < 0 else "\u2192")
    lines.append("\u2551" + f"  CONFIDENCE  avg: {health.avg_confidence:.2f}  trend: {trend_dir} {health.confidence_trend:+.3f}".ljust(W) + "\u2551")
    lines.append("\u2551" + " " * W + "\u2551")

    # Gaps and recommendations
    lines.append("\u2551" + "  GAPS & RECOMMENDATIONS".ljust(W) + "\u2551")

    recs = []
    if health.corroboration_ratio < 0.15:
        recs.append(("[!!]", f"Low corroboration ({health.corroboration_ratio:.1%} \u2014 target >20%)"))
    if health.multi_source_ratio < 0.3:
        recs.append(("[!!]", f"Few multi-source entities ({health.multi_source_ratio:.1%})"))
    if health.source_diversity < 3:
        recs.append(("[!!]", f"Low source diversity ({health.source_diversity} types)"))
    if st.get("embedding_index_size", 0) == 0:
        recs.append(("[~~]", "No embeddings indexed"))
    if health.freshness_score > 0.8:
        recs.append(("[OK]", f"High freshness ({health.freshness_score:.0%})"))
    if health.avg_confidence > 0.8:
        recs.append(("[OK]", f"Good confidence ({health.avg_confidence:.2f} avg)"))
    if health.confidence_trend > 0:
        recs.append(("[OK]", f"Confidence improving ({health.confidence_trend:+.3f})"))

    # Blindspots
    if blindspots.single_source_entities:
        n = len(blindspots.single_source_entities)
        recs.append(("[!!]", f"{n} single-source entities need corroboration"))
    if blindspots.low_confidence_areas:
        n = len(blindspots.low_confidence_areas)
        recs.append(("[~~]", f"{n} low-confidence areas"))
    if blindspots.unresolved_warnings:
        n = len(blindspots.unresolved_warnings)
        recs.append(("[!!]", f"{n} unresolved warnings/bugs"))

    if not recs:
        recs.append(("[OK]", "Knowledge graph looks healthy"))

    for indicator, msg in recs:
        lines.append("\u2551" + f"  {indicator} {msg}"[:W].ljust(W) + "\u2551")

    # Footer with trajectory suggestion
    lines.append("\u2551" + " " * W + "\u2551")

    # Priority recommendations
    priorities = []
    if health.corroboration_ratio < 0.15:
        priorities.append("Add more sources to existing claims")
    if blindspots.unresolved_warnings:
        priorities.append("Fix unresolved warnings/bugs")
    if blindspots.single_source_entities:
        top3 = blindspots.single_source_entities[:3]
        priorities.append(f"Corroborate: {', '.join(top3)}")
    if health.source_diversity < 4:
        priorities.append("Diversify ingestion sources")
    if health.total_claims > 0 and health.knowledge_density < 1.5:
        priorities.append("Add more relationships per entity")
    if len(pred_types) < 3 and health.total_claims > 20:
        priorities.append("Use more predicate types for richer knowledge")

    if priorities:
        lines.append("\u2551" + "  NEXT STEPS".ljust(W) + "\u2551")
        for i, p in enumerate(priorities[:4], 1):
            lines.append("\u2551" + f"  {i}. {p}"[:W].ljust(W) + "\u2551")

    lines.append("\u255a" + "\u2550" * W + "\u255d")

    # Auto-generate the interactive graph alongside the dashboard
    graph_result = json.loads(attest_graph())
    graph_path = graph_result.get("file", "")
    n_nodes = graph_result.get("nodes", 0)
    n_edges = graph_result.get("edges", 0)

    from urllib.parse import quote
    file_url = "file://" + quote(graph_path, safe="/")
    lines.append("")
    lines.append(f"Interactive graph ({n_nodes} entities, {n_edges} relationships):")
    lines.append(file_url)

    return "\n".join(lines)


@mcp.tool()
def attest_graph(
    center_entity: str = "",
    depth: int = 2,
    max_nodes: int = 50,
) -> str:
    """Generate an interactive HTML knowledge graph visualization.

    Writes an HTML file with an embedded force-directed graph to /tmp/attest_graph.html.
    Open it in a browser to explore entities, relationships, and confidence levels.

    Args:
        center_entity: Entity to center the graph on (empty = show all top entities)
        depth: How many hops from center entity (default: 2)
        max_nodes: Maximum nodes to include (default: 50)
    """
    _track_tool_call("attest_graph", center_entity or "(all)")
    db = _get_db()
    import tempfile

    nodes = []
    edges = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def _add_entity(eid: str, etype: str, claim_count: int = 0, is_center: bool = False):
        if eid in seen_nodes or len(seen_nodes) >= max_nodes:
            return
        seen_nodes.add(eid)
        # Color by entity type
        colors = {
            "source_file": "#4fc3f7", "concept": "#81c784", "function": "#ffb74d",
            "bug": "#ef5350", "fix": "#66bb6a", "warning": "#ffa726",
            "pattern": "#7e57c2", "decision": "#26c6da", "model": "#ec407a",
            "gene": "#4db6ac", "disease": "#f06292", "tip": "#9ccc65",
        }
        color = colors.get(etype, "#90a4ae")
        size = 20 + min(claim_count * 3, 40) if not is_center else 40
        nodes.append({
            "id": eid, "label": eid[:20], "type": etype,
            "color": color, "size": size, "claims": claim_count,
        })

    if center_entity:
        # Center on a specific entity
        try:
            frame = db.query(center_entity, depth=depth)
            fe = frame.focal_entity
            _add_entity(fe.id, fe.entity_type, fe.claim_count, is_center=True)
            for rel in frame.direct_relationships:
                _add_entity(rel.target.id, rel.target.entity_type, rel.target.claim_count)
                edge_key = f"{fe.id}-{rel.predicate}-{rel.target.id}"
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append({
                        "from": fe.id, "to": rel.target.id,
                        "label": rel.predicate, "confidence": rel.confidence,
                    })
        except Exception as e:
            logger.warning("Failed to build entity graph for %s: %s", center_entity, e)
    else:
        # Show top entities by claim count
        entities = db.list_entities()
        entities.sort(key=lambda e: e.claim_count, reverse=True)
        for entity in entities[:max_nodes]:
            _add_entity(entity.id, entity.entity_type, entity.claim_count)

        # Add edges between known entities
        for entity in entities[:max_nodes]:
            if entity.id not in seen_nodes:
                continue
            claims = db.claims_for(entity.id)
            for claim in claims[:20]:
                if claim.object.id in seen_nodes:
                    edge_key = f"{entity.id}-{claim.predicate.id}-{claim.object.id}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        edges.append({
                            "from": entity.id, "to": claim.object.id,
                            "label": claim.predicate.id,
                            "confidence": claim.confidence,
                        })

    # Generate HTML with inline JS force-directed graph (zero external deps)
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Attest Knowledge Graph</title>
<style>
body {{ margin:0; background:#1a1a2e; color:#eee; font-family:system-ui; overflow:hidden; }}
canvas {{ display:block; }}
#info {{ position:fixed; top:10px; left:10px; background:rgba(0,0,0,0.7); padding:12px; border-radius:8px; font-size:13px; }}
#tooltip {{ position:fixed; display:none; background:rgba(0,0,0,0.9); color:#fff; padding:8px 12px; border-radius:6px; font-size:12px; pointer-events:none; }}
#legend {{ position:fixed; bottom:10px; left:10px; background:rgba(0,0,0,0.7); padding:10px; border-radius:8px; font-size:11px; }}
.legend-item {{ display:flex; align-items:center; margin:2px 0; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; margin-right:6px; }}
</style></head><body>
<div id="info">{len(nodes)} entities, {len(edges)} relationships</div>
<div id="tooltip"></div>
<canvas id="c"></canvas>
<script>
const nodes = {nodes_json};
const edges = {edges_json};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
let W, H;
function resize() {{ W = canvas.width = innerWidth; H = canvas.height = innerHeight; }}
resize(); addEventListener('resize', resize);

// Initialize positions
nodes.forEach((n, i) => {{
    const a = (i / nodes.length) * Math.PI * 2;
    const r = 150 + Math.random() * 100;
    n.x = W/2 + Math.cos(a) * r;
    n.y = H/2 + Math.sin(a) * r;
    n.vx = 0; n.vy = 0;
}});
const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

let dragging = null, mx = 0, my = 0;
canvas.addEventListener('mousedown', e => {{
    const n = hitTest(e.offsetX, e.offsetY);
    if (n) {{ dragging = n; n.fixed = true; }}
}});
canvas.addEventListener('mousemove', e => {{
    mx = e.offsetX; my = e.offsetY;
    if (dragging) {{ dragging.x = mx; dragging.y = my; }}
    const n = hitTest(mx, my);
    if (n) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (mx + 15) + 'px';
        tooltip.style.top = (my - 10) + 'px';
        tooltip.innerHTML = `<b>${{n.id}}</b><br>Type: ${{n.type}}<br>Claims: ${{n.claims}}`;
    }} else {{
        tooltip.style.display = 'none';
    }}
}});
canvas.addEventListener('mouseup', () => {{ if (dragging) dragging.fixed = false; dragging = null; }});

function hitTest(x, y) {{
    for (const n of nodes) {{
        const dx = n.x - x, dy = n.y - y;
        if (dx*dx + dy*dy < (n.size/2+4)**2) return n;
    }}
    return null;
}}

function tick() {{
    // Repulsion
    for (let i = 0; i < nodes.length; i++) {{
        for (let j = i+1; j < nodes.length; j++) {{
            let dx = nodes[j].x - nodes[i].x, dy = nodes[j].y - nodes[i].y;
            let d = Math.sqrt(dx*dx + dy*dy) || 1;
            let f = 800 / (d * d);
            nodes[i].vx -= dx/d * f; nodes[i].vy -= dy/d * f;
            nodes[j].vx += dx/d * f; nodes[j].vy += dy/d * f;
        }}
    }}
    // Attraction along edges
    edges.forEach(e => {{
        const a = nodeMap[e.from], b = nodeMap[e.to];
        if (!a || !b) return;
        let dx = b.x - a.x, dy = b.y - a.y;
        let d = Math.sqrt(dx*dx + dy*dy) || 1;
        let f = (d - 120) * 0.01;
        a.vx += dx/d * f; a.vy += dy/d * f;
        b.vx -= dx/d * f; b.vy -= dy/d * f;
    }});
    // Center gravity
    nodes.forEach(n => {{
        n.vx += (W/2 - n.x) * 0.001;
        n.vy += (H/2 - n.y) * 0.001;
    }});
    // Apply velocity
    nodes.forEach(n => {{
        if (n.fixed) return;
        n.vx *= 0.9; n.vy *= 0.9;
        n.x += n.vx; n.y += n.vy;
        n.x = Math.max(20, Math.min(W-20, n.x));
        n.y = Math.max(20, Math.min(H-20, n.y));
    }});
}}

function draw() {{
    ctx.clearRect(0, 0, W, H);
    // Edges
    edges.forEach(e => {{
        const a = nodeMap[e.from], b = nodeMap[e.to];
        if (!a || !b) return;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(255,255,255,${{e.confidence * 0.4}})`;
        ctx.lineWidth = 1 + e.confidence;
        ctx.stroke();
        // Edge label
        const mx = (a.x+b.x)/2, my2 = (a.y+b.y)/2;
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.font = '9px system-ui';
        ctx.fillText(e.label, mx, my2 - 4);
    }});
    // Nodes
    nodes.forEach(n => {{
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.size/2, 0, Math.PI*2);
        ctx.fillStyle = n.color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
        // Label
        ctx.fillStyle = '#fff';
        ctx.font = '11px system-ui';
        ctx.textAlign = 'center';
        ctx.fillText(n.label, n.x, n.y + n.size/2 + 14);
    }});
    tick();
    requestAnimationFrame(draw);
}}
draw();
</script></body></html>"""

    out_path = os.path.join(tempfile.gettempdir(), "attest_graph.html")
    with open(out_path, "w") as f:
        f.write(html)

    return json.dumps({
        "file": out_path,
        "nodes": len(nodes),
        "edges": len(edges),
        "message": f"Graph written to {out_path} — open in browser to explore. "
                   f"{len(nodes)} entities, {len(edges)} relationships.",
    })


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
    results = db.search_entities(metric_name, top_k=50)
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
                        "change": claim.object.id,
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
    results = db.search_entities(metric_name, top_k=50)
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
                    "change": claim.object.id,
                    "delta": payload_data.get("delta", 0),
                })
            elif claim.predicate.id == "degraded_by":
                failures.append({
                    "change": claim.object.id,
                    "delta": payload_data.get("delta", 0),
                })

    # Check for negative results (dead ends)
    dead_ends = []
    neg_results = db.search_entities("no_evidence_for", top_k=20)
    for entity in neg_results:
        claims = db.claims_for(entity.id)
        for claim in claims:
            if claim.predicate.id == "no_evidence_for":
                dead_ends.append(claim.object.id)

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

    results = db.search_entities(query, top_k=limit)
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
                "subject": claim.subject.id,
                "predicate": claim.predicate.id,
                "object": claim.object.id,
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
    results = db.search_entities(agent_id, top_k=10)

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
                        "object": claim.object.id,
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

    results = db.search_entities(agent_id, top_k=20)

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

            targets[claim.object.id] = targets.get(claim.object.id, 0) + 1

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
# Autodidact — self-learning daemon control (5 tools)
# ---------------------------------------------------------------------------


@mcp.tool()
def autodidact_enable(
    interval: int = 3600,
    max_llm_calls_per_day: int = 100,
    max_questions_per_cycle: int = 5,
    max_cost_per_day: float = 1.00,
    sources: str = "auto",
    gap_types: str = "",
    entity_types: str = "",
    use_curator: bool = True,
) -> str:
    """Start the autodidact self-learning daemon.

    The daemon runs in a background thread, continuously detecting knowledge gaps
    and researching them via registered evidence sources (PubMed, Semantic Scholar,
    Perplexity, Serper — auto-detected from API keys).

    Args:
        interval: Seconds between cycles (default 3600 = 1 hour).
        max_llm_calls_per_day: Daily LLM call budget (resets at midnight).
        max_questions_per_cycle: Max research tasks per cycle.
        max_cost_per_day: Estimated dollar cap per day (default $1.00).
        sources: Source mode — "auto" (register sources from API keys) or "none" (gap detection only).
        gap_types: Comma-separated filter (e.g. "single_source,low_confidence"). Empty = all.
        entity_types: Comma-separated entity type filter. Empty = all.
        use_curator: Whether to triage extracted claims through the curator.
    """
    _track_tool_call("autodidact_enable", f"interval={interval}")
    db = _get_db()

    gap_list = [g.strip() for g in gap_types.split(",") if g.strip()] or None
    entity_list = [e.strip() for e in entity_types.split(",") if e.strip()] or None

    status = db.enable_autodidact(
        interval=interval,
        max_llm_calls_per_day=max_llm_calls_per_day,
        max_questions_per_cycle=max_questions_per_cycle,
        max_cost_per_day=max_cost_per_day,
        sources=sources,
        gap_types=gap_list,
        entity_types=entity_list,
        use_curator=use_curator,
    )
    return _serialize(status)


@mcp.tool()
def autodidact_disable() -> str:
    """Stop the autodidact self-learning daemon.

    Gracefully shuts down the background thread. Cycle history is preserved
    in the knowledge graph as journal claims.
    """
    _track_tool_call("autodidact_disable", "")
    db = _get_db()
    db.disable_autodidact()
    return json.dumps({"disabled": True})


@mcp.tool()
def autodidact_status() -> str:
    """Get the current status of the autodidact daemon.

    Returns enabled state, cycle count, budget usage, cost tracking,
    and details of the last completed cycle.
    """
    _track_tool_call("autodidact_status", "")
    db = _get_db()
    status = db.autodidact_status()
    return _serialize(status)


@mcp.tool()
def autodidact_run_now() -> str:
    """Trigger an immediate autodidact cycle.

    Wakes the daemon from its sleep interval and runs one research cycle
    immediately. The daemon must be enabled first via autodidact_enable.
    """
    _track_tool_call("autodidact_run_now", "")
    db = _get_db()
    if not db._autodidact:
        return json.dumps({"error": "Autodidact is not enabled. Call autodidact_enable first."})
    db.autodidact_run_now()
    return json.dumps({"triggered": True, "message": "Immediate cycle triggered."})


@mcp.tool()
def autodidact_history(limit: int = 10) -> str:
    """Get recent autodidact cycle reports.

    Returns the most recent cycle reports with details on tasks generated,
    claims ingested, negative results, cost, and blindspot changes.

    Args:
        limit: Maximum number of reports to return (default: 10, most recent first).
    """
    _track_tool_call("autodidact_history", f"limit={limit}")
    db = _get_db()
    reports = db.autodidact_history(limit=limit)
    return json.dumps({
        "count": len(reports),
        "reports": [
            {
                "cycle_number": r.cycle_number,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "tasks_generated": r.tasks_generated,
                "tasks_researched": r.tasks_researched,
                "claims_ingested": r.claims_ingested,
                "claims_rejected": r.claims_rejected,
                "negative_results": r.negative_results,
                "llm_calls": r.llm_calls,
                "estimated_cost": r.estimated_cost,
                "blindspot_before": r.blindspot_before,
                "blindspot_after": r.blindspot_after,
                "trigger": r.trigger,
                "errors": r.errors,
            }
            for r in reports
        ],
    })


# ---------------------------------------------------------------------------
# Consensus (1)
# ---------------------------------------------------------------------------


@mcp.tool()
def agent_consensus(
    question: str,
    context: str = "",
    max_rounds: int = 3,
    providers: str = "",
) -> str:
    """Get consensus from multiple AI models on a question.

    Queries all available LLM providers in parallel, shares their responses
    cross-provider, and iterates until they converge on a best answer.
    Each model's response and the final consensus are stored as verified claims.

    Args:
        question: The question to get consensus on.
        context: Optional document/chat context.
        max_rounds: Maximum cross-pollination rounds (1 = no cross-pollination).
        providers: Comma-separated provider names (empty = auto-detect all available).
    """
    _track_tool_call("agent_consensus", f"question={question[:80]}")
    db = _get_db()
    provider_list = [p.strip() for p in providers.split(",") if p.strip()] or None
    result = db.agent_consensus(
        question=question,
        context=context,
        max_rounds=max_rounds,
        providers=provider_list,
    )
    return json.dumps({
        "consensus": result.consensus,
        "confidence": result.confidence,
        "converged": result.converged,
        "rounds": result.rounds,
        "providers_used": result.providers_used,
        "dissents": result.dissents,
        "total_tokens": result.total_tokens,
        "total_cost": result.total_cost,
        "responses": [
            {
                "provider": r.provider,
                "model": r.model,
                "response": r.response[:1000],
                "round": r.round_number,
                "latency_ms": r.latency_ms,
                "error": r.error,
            }
            for r in result.responses
        ],
        "votes": [
            {
                "provider": v.provider,
                "converged": v.converged,
                "best_provider": v.best_provider,
                "rating": v.rating,
                "critique": v.critique,
            }
            for v in result.votes
        ],
    })


# ---------------------------------------------------------------------------
# Resources (2)
# ---------------------------------------------------------------------------


@mcp.resource("attest://entities")
def list_all_entities() -> str:
    """Entity list (capped at 1000 by claim count)."""
    db = _get_db()
    entities = db.list_entities(limit=1000)
    entities.sort(key=lambda e: -e.claim_count)
    result = [
        {"id": e.id, "name": e.name, "type": e.entity_type, "claim_count": e.claim_count}
        for e in entities[:1000]
    ]
    if len(entities) >= 1000:
        result.append({"_note": "Results truncated to 1000 entities. Use search_entities for specific queries."})
    return json.dumps(result)


@mcp.resource("attest://schema")
def get_schema() -> str:
    """Knowledge graph schema."""
    db = _get_db()
    s = db.schema()
    return _serialize(s)


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
    # Auto-detect embedding provider: if OPENAI_API_KEY is set, enable embeddings
    embed_dim = None
    if os.environ.get("OPENAI_API_KEY"):
        embed_dim = 768  # text-embedding-3-small default

    _db = AttestDB(db_path, embedding_dim=embed_dim)

    if embed_dim and os.environ.get("OPENAI_API_KEY"):
        try:
            _db.configure_embeddings("openai", dimensions=embed_dim)
            logger.info("Auto-embedding enabled (OpenAI text-embedding-3-small, %d dims)", embed_dim)
        except Exception as e:
            logger.warning("Could not configure embeddings: %s", e)

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
