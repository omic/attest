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

_BRAIN_INSTRUCTIONS = """\
You have a persistent knowledge brain that remembers across sessions. USE IT PROACTIVELY:

**Always do these during every session:**
1. When you discover a bug, pattern, or gotcha: call `attest_learned(subject, description, type)` \
— types: bug, fix, pattern, warning, decision, tip
2. When something fails or doesn't work: call `attest_negative_result(topic, finding)` \
— prevents repeating dead ends
3. Before editing a file you haven't touched recently: call `attest_check_file(path)` for known issues
4. Before starting a complex task: call `attest_get_prior_approaches(problem)` to see what worked before
5. When the user's task is done: call `attest_session_end(outcome, summary, next_steps, files_changed)`

**These happen automatically via hooks (no action needed):**
- Session start: prior warnings/bugs/patterns injected based on git context
- Before edits: known issues for the file surface via PreToolUse hook
- Before reading PDFs/images: token cost warning via PreToolUse hook
- After test failures: prior fixes surfaced via PostToolUse hook
- Long sessions: sprawl warning when tool calls exceed ~30

**Token discipline (proactive):**
- When a user wants to read or ingest a PDF/image: suggest converting to markdown first \
(5-20x token savings). Commands: `pandoc file.pdf -o file.md` or `markitdown file.pdf > file.md`.
- In long sessions (15+ human turns): suggest starting a fresh conversation or using \
`prompt_kit_rescue` to compress context.
- For bulk or repeated LLM tasks: call `prompt_kit_optimize(task)` for model routing advice.
- Token usage from brain LLM calls (ingest_text, attest_ask, agent_consensus) is auto-tracked.

Record knowledge liberally — anything that would save time if encountered again."""

mcp = FastMCP("attest", instructions=_BRAIN_INSTRUCTIONS)

# Global DB reference — set by main() or configure()
_db = None

# ---------------------------------------------------------------------------
# Project / agent auto-detection — set once in main()
# ---------------------------------------------------------------------------

_current_project: str | None = None
_current_agent_id: str = "claude-code"


def _detect_project() -> str | None:
    """Derive project from git remote origin, cached once per session."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            url = result.stdout.strip()
            # Normalize: "git@github.com:omic/attest.git" → "github.com/omic/attest"
            url = url.rstrip("/")
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@"):
                url = url[4:].replace(":", "/", 1)
            elif url.startswith(("https://", "http://")):
                url = url.split("://", 1)[1]
            return url
    except Exception:
        pass
    # Fallback: directory name
    return os.path.basename(os.getcwd())


def _detect_agent_id() -> str:
    """Detect which coding agent is running."""
    return os.environ.get("ATTEST_AGENT_ID", "claude-code")


# ---------------------------------------------------------------------------
# Tool category registry
# ---------------------------------------------------------------------------

# Maps every tool name → category string.  Categories follow the submodule
# split plus finer-grained subcategories for the 34 core tools.
TOOL_CATEGORIES: dict[str, str] = {
    # --- core: query (read-only data retrieval) ---
    "attest_ask": "query",
    "search_entities": "query",
    "get_entity": "query",
    "query_entity": "query",
    "claims_for": "query",
    "claims_in_namespace": "query",
    "find_bridges": "query",
    "find_gaps": "query",
    "find_paths": "query",
    "knowledge_health": "query",
    "quality_report": "query",
    "resolve_source_url": "query",
    "attest_corroboration": "query",
    "attest_diagnose_corroboration": "query",
    "attest_source_health": "query",
    "attest_source_reliability": "query",
    "attest_blindspots": "query",
    "attest_fragile": "query",
    "attest_stale": "query",
    "attest_drift": "query",
    "attest_consensus": "query",
    "attest_hypothetical": "query",
    "attest_impact": "query",
    # --- core: ingestion (write operations) ---
    "ingest_claim": "ingestion",
    "ingest_batch": "ingestion",
    "ingest_text": "ingestion",
    "retract_source": "ingestion",
    # --- core: admin ---
    "schema": "admin",
    "stats": "admin",
    "set_namespace": "admin",
    "audit_log": "admin",
    "changes": "admin",
    "attest_build_status": "admin",
    "attest_audit": "admin",
    # --- learning (mcp_tools_learning) ---
    "attest_learned": "learning",
    "attest_negative_result": "learning",
    "attest_check_file": "learning",
    "attest_get_prior_approaches": "learning",
    "attest_observe_session": "learning",
    "attest_record_outcome": "learning",
    "attest_research_context": "learning",
    "attest_confidence_trail": "learning",
    "attest_session_end": "learning",
    # --- analysis (mcp_tools_analysis) ---
    "attest_verify_claim": "analysis",
    "attest_verification_status": "analysis",
    "attest_verification_budget": "analysis",
    "attest_challenge_claim": "analysis",
    "attest_predict": "analysis",
    "attest_what_if": "analysis",
    "attest_sandbox_create": "analysis",
    "attest_sandbox_analyze": "analysis",
    "attest_create_thread": "analysis",
    "attest_resume_thread": "analysis",
    "attest_extend_thread": "analysis",
    "attest_list_threads": "analysis",
    "attest_thread_context": "analysis",
    "attest_audit_paper": "analysis",
    "attest_bulk_audit": "analysis",
    "attest_check_freshness": "analysis",
    "attest_sweep_stale": "analysis",
    "attest_archive": "analysis",
    "attest_graph_stats": "analysis",
    "attest_investigate": "analysis",
    "attest_research": "analysis",
    "attest_generate_eval": "analysis",
    "attest_score_eval": "analysis",
    "attest_register_agent": "analysis",
    "attest_agent_leaderboard": "analysis",
    "curator_cost_summary": "analysis",
    "ops_log": "admin",
    # --- autonomous (mcp_tools_autonomous) ---
    "autodidact_enable": "autonomous",
    "autodidact_disable": "autonomous",
    "autodidact_status": "autonomous",
    "autodidact_run_now": "autonomous",
    "autodidact_history": "autonomous",
    "autoresearch_log_experiment": "autonomous",
    "autoresearch_get_priors": "autonomous",
    "autoresearch_suggest_next": "autonomous",
    "agent_consensus": "autonomous",
    "openclaw_ingest_action": "autonomous",
    "openclaw_ingest_conversation": "autonomous",
    "openclaw_query_knowledge": "autonomous",
    "openclaw_heartbeat_check": "autonomous",
    "openclaw_get_preferences": "autonomous",
    # --- viz (mcp_tools_viz) ---
    "attest_dashboard": "viz",
    "attest_graph": "viz",
    # --- team (mcp_tools_team) ---
    "team_setup": "team",
    "team_configure": "team",
    "team_add_member": "team",
    "team_dashboard": "team",
    "team_digest": "team",
    "team_member_detail": "team",
    "team_commitments": "team",
    "team_check_now": "team",
    "team_meeting_prep": "team",
    "team_one_on_one_prep": "team",
    "team_performance_summary": "team",
    "team_health_report": "team",
    "team_risk_report": "team",
    "team_value_report": "team",
    "team_review_queue": "team",
    "team_generate_skills": "team",
    "team_monitor_enable": "team",
    "team_monitor_disable": "team",
    "team_monitor_status": "team",
    "team_token_usage": "team",
    "team_edit_draft": "team",
    "team_send_draft": "team",
    # --- prompt_kit (mcp_tools_prompt_kit) ---
    "prompt_kit_track": "prompt_kit",
    "prompt_kit_diagnose": "prompt_kit",
    "prompt_kit_optimize": "prompt_kit",
    "prompt_kit_report": "prompt_kit",
    "prompt_kit_audit": "prompt_kit",
    "prompt_kit_rescue": "prompt_kit",
    # --- review (attestdb/review/mcp_tools) ---
    "review_queue": "admin",
    "review_submit": "admin",
    "review_batch_approve": "admin",
    "review_stats": "admin",
    "review_dashboard": "admin",
    # --- query (attestdb/query/mcp_handler) ---
    "query_unified": "query",
    "drill_down": "query",
}

ALL_CATEGORIES = sorted(set(TOOL_CATEGORIES.values()))


def _filter_tools_by_category(allowed_categories: set[str]) -> int:
    """Remove tools not in allowed categories from the MCP server. Returns count removed."""
    all_tools = mcp._tool_manager.list_tools()
    removed = 0
    for tool in all_tools:
        cat = TOOL_CATEGORIES.get(tool.name)
        if cat and cat not in allowed_categories:
            mcp._tool_manager._tools.pop(tool.name, None)
            removed += 1
    return removed


def _serialize(obj: object) -> str:
    """Serialize a dataclass to JSON (used by most MCP tool return values)."""
    return json.dumps(asdict(obj))


def _cap_response(json_str: str, max_chars: int = 4000) -> str:
    """Truncate oversized JSON responses with a note."""
    if len(json_str) <= max_chars:
        return json_str
    # Try to truncate at a structural level by parsing
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return json_str[:max_chars] + '\n... [truncated]'
    # For dicts, try trimming list-valued fields from longest to shortest
    if isinstance(data, dict):
        for key in sorted(data, key=lambda k: len(json.dumps(data[k], default=str)), reverse=True):
            if isinstance(data[key], list) and len(data[key]) > 5:
                data[key] = data[key][:5]
                data[key].append("... truncated")
                result = json.dumps(data, default=str)
                if len(result) <= max_chars:
                    return result
    result = json.dumps(data, default=str)
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + '\n... [truncated]'

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


def _track_token_event(model: str, prompt_tokens: int, completion_tokens: int, purpose: str):
    """Accumulate a token usage event for end-of-session flushing."""
    if _session_tracker is None:
        return
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return
    _session_tracker.setdefault("token_events", []).append({
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "purpose": purpose,
        "timestamp": time.time(),
    })


def _flush_session():
    """On process exit, record session summary + token events as claims."""
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

        # Flush accumulated token events as used_tokens claims
        from attestdb.core.providers import estimate_cost
        for evt in _session_tracker.get("token_events", []):
            try:
                cost = estimate_cost(
                    evt["model"], evt["prompt_tokens"], evt["completion_tokens"],
                )
                db.ingest(
                    subject=("system", "person"),
                    predicate=("used_tokens", "token_usage"),
                    object=(evt["model"], "llm_model"),
                    provenance={
                        "source_type": "prompt_kit",
                        "source_id": f"auto:{evt['timestamp']:.6f}",
                    },
                    confidence=1.0,
                    payload={
                        "schema_ref": "token_usage/v2",
                        "data": {
                            "prompt_tokens": evt["prompt_tokens"],
                            "completion_tokens": evt["completion_tokens"],
                            "total_tokens": evt["prompt_tokens"] + evt["completion_tokens"],
                            "purpose": evt["purpose"],
                            "cost_usd": round(cost, 6),
                            "model_tier": "execution",
                            "cache_hit": False,
                        },
                    },
                )
            except Exception:
                pass
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
# Core tools
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
    verify: bool = False,
) -> str:
    """Add a claim (subject-predicate-object triple) to the knowledge graph.

    Returns the claim_id (SHA-256 hash). Requires all fields — omitting
    source_type or source_id will raise a ProvenanceError.
    Optional namespace for team isolation (empty = global).
    Optional ttl_seconds for automatic expiry (0 = never expires).
    Optional verify=True to run verification pipeline after ingest.
    Confidence is automatically calibrated for LLM sources and capped by
    source type ceiling.
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
        verify=verify,
    )


@mcp.tool()
def ingest_text(text: str, source_id: str = "") -> str:
    """Extract claims from unstructured text and ingest them.

    Returns JSON with n_valid (claims ingested), raw_count, and warnings.
    May return 0 claims if text has no extractable relationships — this is
    normal, not an error. Requires attestdb-enterprise for LLM extraction;
    falls back to heuristic patterns without it.
    """
    _track_tool_call("ingest_text", source_id or text[:80])
    db = _get_db()
    result = db.ingest_text(text, source_id=source_id)

    # Auto-track token usage from the extraction LLM call
    if hasattr(result, "prompt_tokens"):
        _track_token_event("auto", result.prompt_tokens, result.completion_tokens, "ingest_text")
    elif isinstance(result, dict):
        _track_token_event(
            "auto",
            result.get("prompt_tokens", 0),
            result.get("completion_tokens", 0),
            "ingest_text",
        )

    # Warn about raw document formats
    result_dict = json.loads(json.dumps(result, default=str))
    text_head = text[:200].lower()
    if text_head.startswith("%pdf") or "\x00" in text[:100]:
        result_dict["token_warning"] = (
            "Raw PDF detected. This costs 5-20x more tokens than markdown. "
            "Convert first: pandoc file.pdf -o file.md"
        )
    elif text_head.startswith("<!doctype") or text_head.startswith("<html"):
        result_dict["token_warning"] = (
            "Raw HTML detected. Convert to markdown first for significant token savings."
        )

    return json.dumps(result_dict)


def _to_pair(val: str | list | tuple, default_type: str = "entity") -> tuple[str, str]:
    """Coerce subject/predicate/object to (id, type) tuple.

    Accepts:
      - "entity_name"           -> ("entity_name", default_type)
      - ["entity_name", "type"] -> ("entity_name", "type")
      - ("entity_name", "type") -> ("entity_name", "type")
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


def _source_url(source_id: str, source_type: str = "") -> str | None:
    """Resolve source_id to URL (cached import)."""
    from attestdb.core.provenance import resolve_source_url
    return resolve_source_url(source_id, source_type)


@mcp.tool()
def claims_for(
    entity_id: str,
    predicate_type: Optional[str] = None,
    min_confidence: float = 0.0,
) -> str:
    """Get claims about an entity, optionally filtered."""
    db = _get_db()
    claims = db.claims_for(entity_id, predicate_type=predicate_type, min_confidence=min_confidence)[:500]
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
    return _cap_response(_serialize(report))


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
    return _cap_response(_serialize(health))


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


@mcp.tool()
def attest_ask(question: str, namespace: str = "", top_k: int = 10) -> str:
    """Answer a natural-language question using the knowledge graph.

    Returns structured answer with citations, contradictions, and gap analysis.

    Pass a namespace to scope the answer to a specific session or team.
    Leave empty to search the global namespace (all claims).
    """
    _track_tool_call("attest_ask", question[:100])
    db = _get_db()

    try:
        from attestdb.mcp_tools_learning import attest_ask_impl
    except ImportError:
        return {"error": "attest_ask requires attestdb-enterprise. Install with: pip install attestdb-enterprise"}

    if namespace:
        with _namespace_scope(db, namespace):
            result = attest_ask_impl(db, question, top_k)
    else:
        result = attest_ask_impl(db, question, top_k)
    return _cap_response(result) if isinstance(result, str) else result


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
# New API tools (thin wrappers)
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
    d = asdict(report)

    # Filter out session/metadata noise from single_source_entities
    _NOISE_PREFIXES = ("auto-stop:", "tool_session:", "outcome_value:", "status:")
    d["single_source_entities"] = [
        e for e in d.get("single_source_entities", [])
        if not any(e.lower().startswith(p) for p in _NOISE_PREFIXES)
    ][:20]

    # Cap knowledge_gaps
    d["knowledge_gaps"] = d.get("knowledge_gaps", [])[:10]

    return _cap_response(json.dumps(d, default=str))


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

    result = "\n".join(lines)
    return result[:4000] + "\n... [truncated]" if len(result) > 4000 else result


@mcp.tool()
def attest_diagnose_corroboration(content_id: str) -> str:
    """Show how corroboration is counted for a content_id. Useful for debugging inflation.

    Returns external ID clustering vs provenance overlap breakdown, making it
    visible when the same paper ingested via multiple paths inflates counts.

    Args:
        content_id: The content_id to diagnose (SHA-256 of subject+predicate+object)
    """
    _track_tool_call("attest_diagnose_corroboration", content_id)
    db = _get_db()
    result = db.diagnose_corroboration(content_id)
    return json.dumps(result, indent=2)


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
def system_pulse() -> str:
    """Heartbeat status: cycle stats, tier distribution, task backlog."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"running": False, "error": "Heartbeat not enabled. Call db.enable_heartbeat()."})
    from attestdb.intelligence.proactive_mcp_tools import system_pulse as _pulse
    return _pulse(db._heartbeat)


@mcp.tool()
def entity_health(entity_id: str) -> str:
    """Freshness, tier distribution, composite status, and gap analysis for an entity."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import entity_health as _health
    return _health(db._heartbeat, entity_id)


@mcp.tool()
def stale_entities(top_n: int = 20) -> str:
    """Entities most in need of data refresh, ranked by importance x staleness."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import stale_entities as _stale
    return _stale(db._heartbeat, top_n)


@mcp.tool()
def predicted_queries(hours: int = 24) -> str:
    """Queries the system predicts will be asked in the next N hours."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import predicted_queries as _pq
    return _pq(db._heartbeat, hours)


@mcp.tool()
def composite_status(entity_id: str) -> str:
    """Which composites exist for an entity, their versions and staleness."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import composite_status as _cs
    return _cs(db._heartbeat, entity_id)


@mcp.tool()
def trigger_synthesis(entity_id: str, composite_type: str = "entity_brief") -> str:
    """Manually trigger composite synthesis for an entity."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import trigger_synthesis as _ts
    return _ts(db._heartbeat, entity_id, composite_type)


@mcp.tool()
def knowledge_gaps(top_n: int = 10) -> str:
    """Biggest coverage gaps in the claim graph."""
    db = _get_db()
    if not db._heartbeat:
        return json.dumps({"error": "Heartbeat not enabled"})
    from attestdb.intelligence.proactive_mcp_tools import knowledge_gaps as _kg
    return _kg(db._heartbeat, top_n)


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


@mcp.tool()
def list_tool_categories() -> str:
    """List all tool categories and their tools. Always available (never filtered)."""
    by_cat: dict[str, list[str]] = {}
    for tool_name, cat in sorted(TOOL_CATEGORIES.items()):
        by_cat.setdefault(cat, []).append(tool_name)
    return json.dumps({"categories": ALL_CATEGORIES, "tools_by_category": by_cat})


# ---------------------------------------------------------------------------
# Register tool groups from submodules
# ---------------------------------------------------------------------------

try:
    from attestdb.mcp_tools_learning import register_tools as _reg_learning
    from attestdb.mcp_tools_learning import (  # re-export helpers used by tests
        _retrieve_candidates,
        _ASK_STOP_WORDS,
        attest_ask_impl as _attest_ask_impl,
    )
    _reg_learning(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.mcp_tools_viz import register_tools as _reg_viz
    _reg_viz(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.mcp_tools_autonomous import register_tools as _reg_autonomous
    _reg_autonomous(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.mcp_tools_analysis import register_tools as _reg_analysis
    _reg_analysis(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.mcp_tools_team import register_tools as _reg_team
    _reg_team(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.mcp_tools_prompt_kit import register_tools as _reg_prompt_kit
    _reg_prompt_kit(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.review.mcp_tools import register_review_tools as _reg_review
    _reg_review(mcp, _get_db)
except ImportError:
    pass

try:
    from attestdb.query.mcp_handler import register_query_tools as _reg_query
    _reg_query(mcp, _get_db)
except ImportError:
    pass

# Re-export tool functions from submodules so tests and external code can
# import them from attestdb.mcp_server (backward compatibility).
_tool_lookup = {t.name: t.fn for t in mcp._tool_manager.list_tools()}
attest_learned = _tool_lookup["attest_learned"]
attest_check_file = _tool_lookup["attest_check_file"]
attest_session_end = _tool_lookup["attest_session_end"]
attest_negative_result = _tool_lookup["attest_negative_result"]
attest_research_context = _tool_lookup["attest_research_context"]
attest_observe_session = _tool_lookup["attest_observe_session"]
attest_record_outcome = _tool_lookup["attest_record_outcome"]
attest_get_prior_approaches = _tool_lookup["attest_get_prior_approaches"]
attest_confidence_trail = _tool_lookup["attest_confidence_trail"]
attest_dashboard = _tool_lookup["attest_dashboard"]
attest_graph = _tool_lookup["attest_graph"]
autoresearch_log_experiment = _tool_lookup["autoresearch_log_experiment"]
autoresearch_get_priors = _tool_lookup["autoresearch_get_priors"]
autoresearch_suggest_next = _tool_lookup["autoresearch_suggest_next"]
openclaw_ingest_action = _tool_lookup["openclaw_ingest_action"]
openclaw_query_knowledge = _tool_lookup["openclaw_query_knowledge"]
openclaw_heartbeat_check = _tool_lookup["openclaw_heartbeat_check"]
openclaw_ingest_conversation = _tool_lookup["openclaw_ingest_conversation"]
openclaw_get_preferences = _tool_lookup["openclaw_get_preferences"]
autodidact_enable = _tool_lookup["autodidact_enable"]
autodidact_disable = _tool_lookup["autodidact_disable"]
autodidact_status = _tool_lookup["autodidact_status"]
autodidact_run_now = _tool_lookup["autodidact_run_now"]
autodidact_history = _tool_lookup["autodidact_history"]
agent_consensus = _tool_lookup["agent_consensus"]
attest_what_if = _tool_lookup["attest_what_if"]
attest_sandbox_create = _tool_lookup["attest_sandbox_create"]
attest_sandbox_analyze = _tool_lookup["attest_sandbox_analyze"]
attest_predict = _tool_lookup["attest_predict"]
attest_verify_claim = _tool_lookup["attest_verify_claim"]
attest_verification_status = _tool_lookup["attest_verification_status"]
attest_challenge_claim = _tool_lookup["attest_challenge_claim"]
attest_verification_budget = _tool_lookup["attest_verification_budget"]
attest_create_thread = _tool_lookup["attest_create_thread"]
attest_resume_thread = _tool_lookup["attest_resume_thread"]
attest_extend_thread = _tool_lookup["attest_extend_thread"]
attest_list_threads = _tool_lookup["attest_list_threads"]
attest_thread_context = _tool_lookup["attest_thread_context"]
attest_audit_paper = _tool_lookup["attest_audit_paper"]
attest_bulk_audit = _tool_lookup["attest_bulk_audit"]
attest_check_freshness = _tool_lookup["attest_check_freshness"]
attest_sweep_stale = _tool_lookup["attest_sweep_stale"]
attest_archive = _tool_lookup["attest_archive"]
attest_graph_stats = _tool_lookup["attest_graph_stats"]
attest_investigate = _tool_lookup["attest_investigate"]
attest_research = _tool_lookup["attest_research"]
prompt_kit_track = _tool_lookup.get("prompt_kit_track")
prompt_kit_diagnose = _tool_lookup.get("prompt_kit_diagnose")
prompt_kit_report = _tool_lookup.get("prompt_kit_report")
prompt_kit_optimize = _tool_lookup.get("prompt_kit_optimize")
prompt_kit_audit = _tool_lookup.get("prompt_kit_audit")
prompt_kit_rescue = _tool_lookup.get("prompt_kit_rescue")
del _tool_lookup


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
    parser.add_argument(
        "--tools",
        default=None,
        help=(
            "Comma-separated tool categories to expose (e.g. query,ingestion,admin). "
            "Also settable via ATTEST_MCP_TOOLS env var. "
            f"Available: {', '.join(ALL_CATEGORIES)}"
        ),
    )
    args = parser.parse_args()

    # Filter tools by category if requested
    tools_spec = args.tools or os.environ.get("ATTEST_MCP_TOOLS")
    if tools_spec:
        allowed = {c.strip() for c in tools_spec.split(",")}
        unknown = allowed - set(ALL_CATEGORIES)
        if unknown:
            logger.warning("Unknown tool categories ignored: %s", ", ".join(sorted(unknown)))
            allowed -= unknown
        removed = _filter_tools_by_category(allowed)
        logger.info("Tool filter: keeping %s, removed %d tools", ", ".join(sorted(allowed)), removed)

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

    global _current_project, _current_agent_id
    _current_project = _detect_project()
    _current_agent_id = _detect_agent_id()
    logger.info("Project: %s, Agent: %s", _current_project, _current_agent_id)

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
