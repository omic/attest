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
import signal
import sys
import threading
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
- In long sessions (15+ human turns): suggest starting a fresh conversation to reset context.
- Token usage from brain LLM calls (ingest_text, attest_ask) is auto-tracked.

Record knowledge liberally — anything that would save time if encountered again.
(Advanced tools — corroboration, source health, review queue, graph analytics — live in the \
`standard` profile. Set ATTEST_MCP_PROFILE=standard if you need them.)"""

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
    "attest_unanswered": "query",
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
    "gateway_savings_summary": "viz",
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
    # --- compliance (mcp_tools_compliance) ---
    "compliance_posture": "compliance",
    "compliance_gaps": "compliance",
    "compliance_evidence_for": "compliance",
    "compliance_ingest_evidence": "compliance",
    # --- audit (audit/mcp_tools) ---
    "audit_user": "audit",
    "audit_claim": "audit",
    "audit_denied": "audit",
    "compliance_report": "audit",
    "audit_export_csv": "audit",
    "audit_integrity": "audit",
    # --- predicate management ---
    "predicate_catalog": "admin",
    "predicate_infer": "admin",
    # --- reconcile (mcp_tools_reconcile) — agent action verification ---
    "verify_before_act": "reconcile",
    "reconcile_batch": "reconcile",
    "get_evidence_for": "reconcile",
    "log_agent_action": "reconcile",
    "explain_agent_action": "reconcile",
    "export_agent_imprint": "reconcile",
    "verify_agent_imprint": "reconcile",
    "claims_at": "query",
    "export_liability_ledger": "reconcile",
    "verify_liability_ledger": "reconcile",
    # --- trust (mcp_tools_trust) — composite entity trust score ---
    "trust_score": "reconcile",
    # --- transaction (mcp_tools_transaction) — agentic transaction gateway ---
    "check_transaction": "reconcile",
    # --- agent_factory (mcp_tools_agent_factory) ---
    "factory_discover_workflows": "agent_factory",
    "factory_generate_spec": "agent_factory",
    "factory_build_eval": "agent_factory",
    "factory_assemble_agent": "agent_factory",
    "factory_validate_trust": "agent_factory",
    "factory_run_pipeline": "agent_factory",
    "factory_list_workflows": "agent_factory",
    "factory_list_agents": "agent_factory",
    "factory_workflow_evolution": "agent_factory",
    "factory_export_agent": "agent_factory",
    "factory_execute_agent": "agent_factory",
    "factory_run_eval": "agent_factory",
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
    return json.dumps(asdict(obj), default=str)


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
            except BaseException:
                pass
    except BaseException:
        pass  # Best-effort — process is exiting; swallow even Rust PanicException


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

    WARNING: NOT safe for concurrent async tasks on HTTP/streaming transports
    where multiple requests may interleave. If HTTP transport with concurrent
    requests is needed, replace with task-local storage or per-request DB wrapper.
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
    """Add a subject-predicate-object claim. Returns claim_id (SHA-256).

    source_type + source_id are required (ProvenanceError if missing).
    namespace for team isolation, ttl_seconds for expiry (0=never),
    verify=True runs verification. Confidence auto-calibrated per source.
    """
    _track_tool_call("ingest_claim", f"{subject_id} {predicate_id} {object_id}")
    db = _get_db()
    try:
        claim_id = db.ingest(
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
        _track_claims_ingested(1)
        return claim_id
    except Exception as exc:
        return json.dumps({"error": f"ingest failed: {exc}"})


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
    claim_inputs = []
    parse_errors = []
    for i, c in enumerate(claims):
        try:
            claim_inputs.append(ClaimInput(
                subject=_to_pair(c["subject"]),
                predicate=_to_pair(c["predicate"], "predicate"),
                object=_to_pair(c["object"]),
                provenance=c["provenance"],
                confidence=c.get("confidence"),
                payload=c.get("payload"),
            ))
        except (KeyError, TypeError) as exc:
            parse_errors.append(f"claim[{i}]: {exc}")
    result = db.ingest_batch(claim_inputs) if claim_inputs else type(
        "R", (), {"ingested": 0, "duplicates": 0, "errors": []},
    )()
    _track_claims_ingested(result.ingested)
    all_errors = list(result.errors) + parse_errors
    return json.dumps({
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": all_errors,
    })


@mcp.tool()
def ingest_custom_data(
    rows: list[dict],
    domain_context: str = "",
    mappings: list[dict] | None = None,
    commit: bool = False,
) -> str:
    """Ingest structured rows — auto-discover mappings, preview, then commit.

    Call first with rows to get proposed mappings + sample claims. Call again
    with mappings=<validated> and commit=True to ingest. domain_context (e.g.
    "pharma CRM") improves LLM proposals.
    """
    db = _get_db()

    try:
        from attestdb.discovery.schema_reasoner import (
            preview_batch,
            execute_rows_batch,
            propose_and_validate,
            export_vocabulary_for_llm,
        )
    except ImportError:
        return json.dumps({"error": "Schema reasoner not available"})

    if not rows:
        return json.dumps({"error": "No rows provided"})

    # Discovery mode: propose + validate
    if mappings is None:
        mappings, preview = propose_and_validate(
            rows, domain_context=domain_context, db=db,
        )
        return json.dumps({
            "mode": "preview",
            "mappings": mappings,
            "preview": {
                "total_rows": preview.total_rows,
                "expected_claims": preview.expected_claims,
                "produced_claims": preview.produced_claims,
                "drop_rate": f"{preview.drop_rate:.0%}",
                "field_coverage": {
                    k: f"{v:.0%}" for k, v in preview.field_coverage.items()
                },
                "sample_claims": preview.sample_claims,
                "warnings": preview.warnings,
            },
            "next_step": "Review the mappings and sample claims above. "
            "If they look correct, call ingest_custom_data again with "
            "mappings=<these mappings> and commit=True to ingest.",
        })

    # Preview mode with explicit mappings
    if not commit:
        preview = preview_batch(rows, mappings)
        return json.dumps({
            "mode": "preview",
            "mappings": mappings,
            "preview": {
                "total_rows": preview.total_rows,
                "expected_claims": preview.expected_claims,
                "produced_claims": preview.produced_claims,
                "drop_rate": f"{preview.drop_rate:.0%}",
                "field_coverage": {
                    k: f"{v:.0%}" for k, v in preview.field_coverage.items()
                },
                "sample_claims": preview.sample_claims,
                "warnings": preview.warnings,
            },
            "next_step": "Add commit=True to ingest these claims.",
        })

    # Commit mode: ingest for real
    claims = execute_rows_batch(
        rows, mappings, include_payload=True,
        source_type="custom_import",
        source_id=domain_context or "custom",
    )
    if not claims:
        return json.dumps({"error": "No claims produced from mappings", "mappings": mappings})

    # Auto-register payload schemas from the data
    registered_schemas: list[str] = []
    seen_schemas: set[str] = set()
    for c in claims:
        if c.payload and "schema_ref" in c.payload:
            ref = c.payload["schema_ref"]
            if ref not in seen_schemas:
                seen_schemas.add(ref)
                # Build schema from first claim's payload data keys
                field_types = {}
                for k, v in c.payload.get("data", {}).items():
                    if isinstance(v, bool):
                        field_types[k] = "boolean"
                    elif isinstance(v, int):
                        field_types[k] = "integer"
                    elif isinstance(v, float):
                        field_types[k] = "number"
                    else:
                        field_types[k] = "string"
                try:
                    db.register_payload_schema(ref, {
                        "schema_ref": ref,
                        "fields": field_types,
                        "source_type": "custom_import",
                    })
                    registered_schemas.append(ref)
                except Exception as exc:
                    logger.warning("Failed to register schema %s: %s", ref, exc)

    result = db.ingest_batch(claims)
    _track_claims_ingested(result.ingested)

    # Post-ingestion audit: reconcile a sample against existing graph
    audit = {}
    try:
        from attestdb.intelligence.reconciler import Reconciler
        reconciler = Reconciler(db)
        sample = claims[:20]
        observations = [
            {"entity": c.subject[0], "predicate": c.predicate[0], "object": c.object[0]}
            for c in sample
        ]
        batch_result = reconciler.reconcile_batch(observations)
        audit = {
            "verified": batch_result.verified,
            "unverified": batch_result.unverified,
            "contradicted": batch_result.contradicted,
            "stale": batch_result.stale,
            "total_checked": batch_result.total,
        }
        # Surface contradictions as warnings
        contradictions = [
            f"{r.entity} --{r.predicate}--> {r.object}: {r.evidence_summary}"
            for r in batch_result.results if r.verdict == "contradicted"
        ]
        if contradictions:
            audit["contradiction_details"] = contradictions
    except Exception as exc:
        logger.warning("Post-ingestion reconciliation failed: %s", exc)

    return json.dumps({
        "mode": "committed",
        "ingested": result.ingested,
        "duplicates": result.duplicates,
        "errors": list(result.errors),
        "total_rows": len(rows),
        "total_mappings": len(mappings),
        "registered_schemas": registered_schemas,
        "audit": audit,
    })


@mcp.tool()
def search_by_payload(
    schema_ref: str = "",
    record_id: str = "",
    field: str = "",
    value: str = "",
    limit: int = 20,
) -> str:
    """Search claims by source record payload (schema_ref, record_id, or field/value).

    schema_ref: source system like "salesforce/opportunity" or "jira/bug".
    record_id: specific source record. field+value: match payload data field.
    """
    db = _get_db()
    claims = db.claims_by_payload(
        schema_ref=schema_ref or None,
        record_id=record_id or None,
        field=field or None,
        value=value or None,
        limit=limit,
    )
    results = []
    for c in claims:
        entry = {
            "claim_id": c.claim_id,
            "subject": c.subject.id,
            "predicate": c.predicate.id,
            "object": c.object.id,
            "confidence": c.confidence,
        }
        if c.payload:
            entry["payload"] = {
                "schema_ref": c.payload.schema_ref,
                "record_id": c.payload.data.get("record_id"),
                "data": c.payload.data,
            }
        results.append(entry)
    return json.dumps({"claims": results, "total": len(results)})


@mcp.tool()
def run_connector(
    connector: str,
    config: dict | None = None,
) -> str:
    """Run a data connector to fetch from an external system and ingest claims.
    connector: slack|gmail|jira|github|salesforce|hubspot|linear|zendesk|servicenow|
    pagerduty|postgres|mysql|csv|notion|confluence|gdocs|teams|s3|elasticsearch|
    mongodb|airtable|google_sheets|zoho|http. config: connector-specific dict.
    """
    db = _get_db()
    cfg = config or {}

    try:
        from attestdb.connectors import connect, CONNECTOR_REGISTRY
    except ImportError:
        return json.dumps({"error": "Connectors module not available"})

    if connector not in CONNECTOR_REGISTRY:
        available = sorted(CONNECTOR_REGISTRY.keys())
        return json.dumps({"error": f"Unknown connector: {connector}", "available": available})

    try:
        conn = connect(connector, **cfg)
        result = conn.run(db)
        return json.dumps({
            "connector": connector,
            "claims_ingested": result.claims_ingested,
            "claims_skipped": result.claims_skipped,
            "errors": result.errors[:10],
            "elapsed_seconds": round(result.elapsed_seconds, 1),
        })
    except Exception as exc:
        return json.dumps({"error": f"Connector {connector} failed: {exc}"})


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
def predicate_catalog() -> str:
    """List registered predicates with descriptions, allowed subject/object types,
    data patterns, directionality, opposition pairs, and observation counts.
    """
    db = _get_db()
    store = getattr(db, "_predicate_store", None)
    if store is None:
        return json.dumps({"error": "PredicateStore not available"})
    return json.dumps(store.export_catalog(), indent=2)


@mcp.tool()
def predicate_infer(min_observations: int = 20) -> str:
    """Crystallize predicate type constraints (subject_types/object_types) from
    observed usage. min_observations: minimum claims required before inferring.
    """
    db = _get_db()
    store = getattr(db, "_predicate_store", None)
    if store is None:
        return json.dumps({"error": "PredicateStore not available"})
    results = store.infer_constraints(min_observations=min_observations)
    return json.dumps({
        "inferred": results,
        "count": len(results),
    }, indent=2)


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
def attest_ask(question: str, namespace: str = "", top_k: int = 10,
               engine: str = "v2") -> str:
    """Answer a NL question against the knowledge graph. Returns answer with
    citations, contradictions, and gap analysis.

    namespace: scope to session/team ("" = global). engine: v2 (default) | v3
    (Agent SDK, needs ANTHROPIC_API_KEY) | shadow (runs both).
    """
    _track_tool_call("attest_ask", question[:100])
    db = _get_db()

    try:
        from attestdb.mcp_tools_learning import attest_ask_impl
    except ImportError:
        return {"error": "attest_ask requires attestdb-enterprise. Install with: pip install attestdb-enterprise"}

    if namespace:
        with _namespace_scope(db, namespace):
            result = attest_ask_impl(db, question, top_k, engine=engine)
    else:
        result = attest_ask_impl(db, question, top_k, engine=engine)
    return _cap_response(result) if isinstance(result, str) else result


@mcp.tool()
def claims_in_namespace(namespace: str, limit: int = 500) -> str:
    """Return all claims stored under a namespace — full session KB snapshot.
    Empty namespace returns an empty result (no global listing).
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
def attest_unanswered(
    limit: int = 20,
    reason: str | None = None,
    since_days: float | None = None,
) -> str:
    """Demand-driven gap signal: questions ask_engine couldn't answer well.

    Distinct from attest_blindspots (structural / graph-level) — this surfaces
    what users actually asked but didn't get back. Returns a per-reason summary
    and the most recent unanswered queries.

    reason: filter to one of {no_entities, no_evidence, low_confidence,
    no_answer, fallback}; None returns all.
    since_days: only include records from the last N days (default: all time).
    """
    import time as _time
    db = _get_db()
    since = (_time.time() - since_days * 86400) if since_days else None
    result = db.unanswered(limit=limit, reason=reason, since=since)
    return _cap_response(json.dumps(result, default=str))


@mcp.tool()
def attest_consensus(topic: str) -> str:
    """Analyze consensus around an entity/topic across sources."""
    db = _get_db()
    report = db.consensus(topic)
    return _serialize(report)


@mcp.tool()
def attest_corroboration(min_sources: int = 2) -> str:
    """Corroboration report: which claims are independently confirmed vs single-source.
    min_sources: minimum independent sources to count as corroborated (default 2).
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
    """Debug corroboration inflation: external-ID clustering vs provenance overlap
    breakdown for a content_id (SHA-256 of subject+predicate+object).
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
# ATTEST_MCP_PROFILE gates which groups are registered to keep the per-turn
# tool-list footprint small. Each tool's name+description+schema is sent to
# the LLM on every turn, so 165 tools ≈ 24K tokens of fixed overhead.
#
# Profiles (comma-sep groups also accepted):
#   core     — lean allowlist of hot-path tools   (~16 tools, ~2K tokens)
#   standard — core + analysis + viz + audit      (~60 tools, ~10K tokens)
#   full     — all registered groups              (~165 tools, ~24K tokens)
#
# Default is "core". Set ATTEST_MCP_PROFILE=full to restore the full surface,
# or e.g. ATTEST_MCP_PROFILE=core,team to opt specific groups in.
#
# The `core` profile additionally applies a post-registration allowlist
# (_LEAN_CORE_TOOLS) so that tools registered at module-level in mcp_server.py
# are pruned unless they're on the allowlist. Selection is driven by actual
# usage telemetry from Claude Code transcripts.
import os as _os

_GROUP_PROFILES = {
    "core": {"learning", "query", "review"},
    "standard": {"learning", "query", "review", "analysis", "viz", "audit"},
    "full": {
        "learning", "query", "review", "analysis", "viz", "audit",
        "autonomous", "team", "prompt_kit", "compliance", "reconcile",
        "trust", "transaction", "agent_factory", "narrative", "novelty",
    },
}

# Tools kept in the `core` profile. Selection: every tool with recorded usage
# in Claude Code transcripts, plus the canonical write/read/admin surface that
# the MCP instructions promote. Everything else is available via `standard`.
_LEAN_CORE_TOOLS = frozenset({
    # write path
    "ingest_claim", "ingest_text", "ingest_batch", "ingest_custom_data",
    # read path
    "query_entity", "search_entities", "get_entity", "attest_ask",
    # learning loop (promoted in MCP instructions)
    "attest_learned", "attest_negative_result", "attest_check_file",
    "attest_session_end", "attest_get_prior_approaches",
    # basic admin
    "stats", "changes", "schema", "list_tool_categories",
})

def _resolve_enabled_groups() -> set[str]:
    raw = _os.environ.get("ATTEST_MCP_PROFILE", "core").strip().lower()
    if raw in _GROUP_PROFILES:
        return set(_GROUP_PROFILES[raw])
    # Treat as comma-separated list — may include profile names or bare groups.
    enabled: set[str] = set()
    for token in (t.strip() for t in raw.split(",") if t.strip()):
        if token in _GROUP_PROFILES:
            enabled |= _GROUP_PROFILES[token]
        else:
            enabled.add(token)
    return enabled or set(_GROUP_PROFILES["core"])

_ENABLED_GROUPS = _resolve_enabled_groups()

def _register_group(name: str, fn) -> None:
    if name not in _ENABLED_GROUPS:
        return
    try:
        fn()
    except ImportError:
        pass


def _reg_learning_group():
    from attestdb.mcp_tools_learning import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_viz_group():
    from attestdb.mcp_tools_viz import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_autonomous_group():
    from attestdb.mcp_tools_autonomous import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_analysis_group():
    from attestdb.mcp_tools_analysis import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_team_group():
    from attestdb.mcp_tools_team import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_prompt_kit_group():
    from attestdb.mcp_tools_prompt_kit import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_review_group():
    from attestdb.review.mcp_tools import register_review_tools as _reg
    _reg(mcp, _get_db)

def _reg_query_group():
    from attestdb.query.mcp_handler import register_query_tools as _reg
    _reg(mcp, _get_db)

def _reg_compliance_group():
    from attestdb.mcp_tools_compliance import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_audit_group():
    from attestdb.audit.mcp_tools import register_audit_tools as _reg
    _reg(mcp, lambda: _get_db()._audit)

def _reg_reconcile_group():
    from attestdb.mcp_tools_reconcile import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_trust_group():
    from attestdb.mcp_tools_trust import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_transaction_group():
    from attestdb.mcp_tools_transaction import register_tools as _reg
    _reg(mcp, _get_db)

def _reg_agent_factory_group():
    from attestdb.mcp_tools_agent_factory import register_tools as _reg
    _reg(mcp, _get_db)


def _reg_narrative_group():
    from attestdb.mcp_tools_narrative import register_tools as _reg
    _reg(mcp, _get_db)


def _reg_novelty_group():
    from attestdb.mcp_tools_novelty import register_tools as _reg
    _reg(mcp, _get_db)


def _reg_recommender_group():
    from attestdb.mcp_tools_recommender import register_tools as _reg
    _reg(mcp, _get_db)


# learning is always needed for test re-exports; register then check.
try:
    from attestdb.mcp_tools_learning import (  # re-export helpers used by tests
        _retrieve_candidates,
        _ASK_STOP_WORDS,
        attest_ask_impl as _attest_ask_impl,
    )
except ImportError:
    pass

_register_group("learning", _reg_learning_group)
_register_group("viz", _reg_viz_group)
_register_group("autonomous", _reg_autonomous_group)
_register_group("analysis", _reg_analysis_group)
_register_group("team", _reg_team_group)
_register_group("prompt_kit", _reg_prompt_kit_group)
_register_group("review", _reg_review_group)
_register_group("query", _reg_query_group)
_register_group("compliance", _reg_compliance_group)
_register_group("audit", _reg_audit_group)
_register_group("agent_factory", _reg_agent_factory_group)
_register_group("narrative", _reg_narrative_group)
_register_group("novelty", _reg_novelty_group)
_register_group("recommender", _reg_recommender_group)

# reconcile / trust / transaction register UNCONDITIONALLY so the gateway's
# curation layer and direct-import demos always see them. Under the `core`
# profile the lean-core allowlist below prunes them; under any broader
# profile they stay exposed. (Steve's original flat-registration pattern.)
try:
    _reg_reconcile_group()
except ImportError:
    pass
try:
    _reg_trust_group()
except ImportError:
    pass
try:
    _reg_transaction_group()
except ImportError:
    pass

# When the user explicitly chose exactly the `core` profile, prune everything
# outside the lean allowlist. Any broader profile (standard/full/custom groups)
# keeps the full registered surface.
if _os.environ.get("ATTEST_MCP_PROFILE", "core").strip().lower() == "core":
    _mgr = mcp._tool_manager
    for _name in list(_mgr._tools.keys()):
        if _name not in _LEAN_CORE_TOOLS:
            _mgr._tools.pop(_name, None)

# Re-export tool functions from submodules so tests and external code can
# import them from attestdb.mcp_server (backward compatibility).
_tool_lookup = {t.name: t.fn for t in mcp._tool_manager.list_tools()}
attest_learned = _tool_lookup.get("attest_learned")
attest_check_file = _tool_lookup.get("attest_check_file")
attest_session_end = _tool_lookup.get("attest_session_end")
attest_negative_result = _tool_lookup.get("attest_negative_result")
attest_research_context = _tool_lookup.get("attest_research_context")
attest_observe_session = _tool_lookup.get("attest_observe_session")
attest_record_outcome = _tool_lookup.get("attest_record_outcome")
attest_get_prior_approaches = _tool_lookup.get("attest_get_prior_approaches")
attest_confidence_trail = _tool_lookup.get("attest_confidence_trail")
attest_dashboard = _tool_lookup.get("attest_dashboard")
attest_graph = _tool_lookup.get("attest_graph")
autoresearch_log_experiment = _tool_lookup.get("autoresearch_log_experiment")
autoresearch_get_priors = _tool_lookup.get("autoresearch_get_priors")
autoresearch_suggest_next = _tool_lookup.get("autoresearch_suggest_next")
openclaw_ingest_action = _tool_lookup.get("openclaw_ingest_action")
openclaw_query_knowledge = _tool_lookup.get("openclaw_query_knowledge")
openclaw_heartbeat_check = _tool_lookup.get("openclaw_heartbeat_check")
openclaw_ingest_conversation = _tool_lookup.get("openclaw_ingest_conversation")
openclaw_get_preferences = _tool_lookup.get("openclaw_get_preferences")
autodidact_enable = _tool_lookup.get("autodidact_enable")
autodidact_disable = _tool_lookup.get("autodidact_disable")
autodidact_status = _tool_lookup.get("autodidact_status")
autodidact_run_now = _tool_lookup.get("autodidact_run_now")
autodidact_history = _tool_lookup.get("autodidact_history")
agent_consensus = _tool_lookup.get("agent_consensus")
attest_what_if = _tool_lookup.get("attest_what_if")
attest_sandbox_create = _tool_lookup.get("attest_sandbox_create")
attest_sandbox_analyze = _tool_lookup.get("attest_sandbox_analyze")
attest_predict = _tool_lookup.get("attest_predict")
attest_verify_claim = _tool_lookup.get("attest_verify_claim")
attest_verification_status = _tool_lookup.get("attest_verification_status")
attest_challenge_claim = _tool_lookup.get("attest_challenge_claim")
attest_verification_budget = _tool_lookup.get("attest_verification_budget")
attest_create_thread = _tool_lookup.get("attest_create_thread")
attest_resume_thread = _tool_lookup.get("attest_resume_thread")
attest_extend_thread = _tool_lookup.get("attest_extend_thread")
attest_list_threads = _tool_lookup.get("attest_list_threads")
attest_thread_context = _tool_lookup.get("attest_thread_context")
attest_audit_paper = _tool_lookup.get("attest_audit_paper")
attest_bulk_audit = _tool_lookup.get("attest_bulk_audit")
attest_check_freshness = _tool_lookup.get("attest_check_freshness")
attest_sweep_stale = _tool_lookup.get("attest_sweep_stale")
attest_archive = _tool_lookup.get("attest_archive")
attest_graph_stats = _tool_lookup.get("attest_graph_stats")
attest_investigate = _tool_lookup.get("attest_investigate")
attest_research = _tool_lookup.get("attest_research")
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
# Version watchdog — auto-restart when package is updated
# ---------------------------------------------------------------------------

_RESTART_SENTINEL = os.path.join(os.path.expanduser("~"), ".attest", ".mcp-restart")
_VERSION_CHECK_INTERVAL = 10  # seconds


def _get_installed_version() -> str | None:
    """Read the current installed package version from metadata (not cached module)."""
    try:
        from importlib.metadata import version
        return version("attestdb")
    except Exception:
        return None


def _start_version_watchdog():
    """Background thread that exits the server when a restart is signaled.

    Two triggers:
    1. Sentinel file ~/.attest/.mcp-restart exists (touched by `attest-mcp install`)
    2. Installed package version differs from what we started with (new wheel installed)

    On either trigger, the server exits cleanly (exit code 0).
    Claude Code automatically relaunches MCP servers that exit, so the new
    version picks up immediately.
    """
    startup_version = _get_installed_version()
    logger.info("Version watchdog started (v=%s)", startup_version)

    def _watchdog():
        while True:
            time.sleep(_VERSION_CHECK_INTERVAL)
            try:
                # Check 1: restart sentinel file
                if os.path.exists(_RESTART_SENTINEL):
                    try:
                        os.unlink(_RESTART_SENTINEL)
                    except OSError:
                        pass
                    logger.info("Restart sentinel detected — exiting for relaunch")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return

                # Check 2: package version changed on disk
                if startup_version:
                    # Clear cached metadata so we read fresh from disk
                    try:
                        from importlib.metadata import distributions
                        # Force re-read by clearing any caches
                        import importlib
                        importlib.invalidate_caches()
                    except Exception:
                        pass
                    current = _get_installed_version()
                    if current and current != startup_version:
                        logger.info(
                            "Package updated: %s → %s — exiting for relaunch",
                            startup_version, current,
                        )
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            except Exception:
                pass  # watchdog must never crash the server

    t = threading.Thread(target=_watchdog, daemon=True, name="mcp-version-watchdog")
    t.start()


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

    # Agent-reconcile mode: curated tool subset for agent behavioral memory
    # lifecycle (verify-before-act + evidence lookup). Triggered by
    # ATTEST_RECONCILE_MODE=1.
    if os.environ.get("ATTEST_RECONCILE_MODE") == "1":
        reconcile_categories = {"query", "ingestion", "learning", "admin", "reconcile"}
        removed = _filter_tools_by_category(reconcile_categories)
        logger.info(
            "Reconcile mode: keeping %s, removed %d tools",
            ", ".join(sorted(reconcile_categories)),
            removed,
        )

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

    # Start version watchdog — auto-restarts when package is updated
    _start_version_watchdog()

    # Handle SIGTERM gracefully (sent by watchdog or system)
    def _sigterm_handler(signum, frame):
        logger.info("SIGTERM received — shutting down for relaunch")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        mcp.run(transport=args.transport)
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    except Exception:
        logger.exception("MCP server crashed — exiting for relaunch")
        sys.exit(1)


if __name__ == "__main__":
    main()
