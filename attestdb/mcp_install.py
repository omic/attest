"""Universal MCP installer for coding tools.

Configures the Attest MCP server for Claude Code, Cursor, Windsurf,
Codex, Gemini CLI, and other MCP-compatible coding tools.

Usage:
    attest-mcp install                      # Auto-detect tools, project scope
    attest-mcp install --tool claude        # Claude Code only
    attest-mcp install --tool cursor        # Cursor only
    attest-mcp install --global             # User scope (all projects)
    attest-mcp recall                       # SessionStart hook output
    attest-mcp uninstall                    # Remove from all tools
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Default memory DB path — shared across all projects and sessions
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_DIR = Path.home() / ".attest"
DEFAULT_MEMORY_DB = DEFAULT_MEMORY_DIR / "memory.attest"
HOOK_METRICS_LOG = DEFAULT_MEMORY_DIR / "hook_metrics.jsonl"

# ---------------------------------------------------------------------------
# Tool configuration templates
# ---------------------------------------------------------------------------

TOOLS = {
    "claude": {
        "name": "Claude Code",
        "detect": lambda: shutil.which("claude") is not None,
        "config_paths": {
            "project": lambda: Path(".mcp.json"),
            "user": lambda: Path.home() / ".claude.json",
        },
        "hooks_path": {
            "project": lambda: Path(".claude") / "settings.json",
            "user": lambda: Path.home() / ".claude" / "settings.json",
        },
    },
    "cursor": {
        "name": "Cursor",
        "detect": lambda: (
            Path.home() / ".cursor").exists() or shutil.which("cursor") is not None,
        "config_paths": {
            "project": lambda: Path(".cursor") / "mcp.json",
            "user": lambda: Path.home() / ".cursor" / "mcp.json",
        },
    },
    "windsurf": {
        "name": "Windsurf",
        "detect": lambda: (
            Path.home() / ".codeium" / "windsurf").exists(),
        "config_paths": {
            "user": lambda: (
                Path.home() / ".codeium" / "windsurf" / "mcp_config.json"),
        },
    },
    "codex": {
        "name": "Codex (OpenAI)",
        "detect": lambda: shutil.which("codex") is not None,
        "config_paths": {
            "project": lambda: Path(".mcp.json"),  # Codex reads .mcp.json
        },
    },
    "gemini": {
        "name": "Gemini CLI",
        "detect": lambda: shutil.which("gemini") is not None,
        "config_paths": {
            "project": lambda: Path(".gemini") / "settings.json",
            "user": lambda: Path.home() / ".gemini" / "settings.json",
        },
    },
}


def _find_attest_mcp() -> str:
    """Find the attest-mcp entry point co-located with the running Python.

    Prefers the entry point in the same bin/Scripts dir as sys.executable
    (the Python that installed attestdb) so hooks always use the matching
    version, even when a stale attest-mcp exists elsewhere on PATH.
    """
    exe_dir = Path(sys.executable).parent
    for name in ("attest-mcp", "attest-mcp.exe"):
        candidate = exe_dir / name
        if candidate.exists():
            return str(candidate)
    # Fall back to PATH (but may find stale version)
    found = shutil.which("attest-mcp")
    return found or ""


def _load_cloud_config() -> dict | None:
    """Load cloud config from ~/.attest/cloud.json if it exists."""
    cloud_path = DEFAULT_MEMORY_DIR / "cloud.json"
    if not cloud_path.exists():
        return None
    try:
        return json.loads(cloud_path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def _mcp_server_entry() -> dict:
    """Build the MCP server config entry.

    If ~/.attest/cloud.json exists, returns a remote MCP config pointing
    to the cloud endpoint.  Otherwise returns a local stdio config.
    """
    cloud = _load_cloud_config()
    if cloud and cloud.get("endpoint") and cloud.get("api_key"):
        # Cloud mode — point MCP client at the cloud endpoint
        endpoint = cloud["endpoint"].rstrip("/")
        return {
            "url": f"{endpoint}/mcp",
            "headers": {
                "Authorization": f"Bearer {cloud['api_key']}",
            },
        }

    # Local mode — launch attest-mcp as a subprocess
    attest_mcp = _find_attest_mcp()
    if not attest_mcp:
        # Fall back to module invocation
        attest_mcp = sys.executable
        args = ["-m", "attestdb.mcp_server"]
    else:
        args = []

    db_path = str(DEFAULT_MEMORY_DB)

    entry = {
        "command": attest_mcp,
        "args": args + ["--db", db_path],
        "env": {
            "ATTEST_AUTO_OBSERVE": "1",
        },
    }
    return entry


def _read_json(path: Path) -> dict:
    """Read a JSON file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write JSON with atomic tmp+rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(str(tmp), str(path))


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def install(tools: list[str] | None = None, scope: str = "project") -> list[str]:
    """Install Attest MCP server config for coding tools.

    Returns list of tools configured.
    """
    # Ensure memory directory exists
    DEFAULT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    entry = _mcp_server_entry()
    configured = []

    targets = tools or list(TOOLS.keys())
    for tool_key in targets:
        tool = TOOLS.get(tool_key)
        if not tool:
            print(f"  Unknown tool: {tool_key}. Valid options: {sorted(TOOLS.keys())}", file=sys.stderr)
            continue

        # Auto-detect: skip tools that aren't installed (unless explicitly requested)
        if tools is None and not tool["detect"]():
            continue

        # When explicitly requested, warn if tool is not installed
        if tools is not None and not tool["detect"]():
            print(f"  Warning: {tool['name']} not detected on this system. Installing config anyway.", file=sys.stderr)

        config_paths = tool.get("config_paths", {})
        config_path_fn = config_paths.get(scope) or config_paths.get("user") or config_paths.get("project")
        if not config_path_fn:
            continue

        config_path = config_path_fn()
        existing = _read_json(config_path)
        servers = existing.setdefault("mcpServers", {})
        servers["brain"] = entry
        _write_json(config_path, existing)
        configured.append(tool_key)
        print(f"  {tool['name']}: {config_path}")

    # Claude Code hooks (SessionStart for prior approach retrieval)
    if "claude" in configured:
        _install_claude_hooks(scope)

    # Generate agent instructions
    _install_agent_instructions()

    return configured


def _install_claude_hooks(scope: str) -> None:
    """Add SessionStart, PreToolUse, PostToolUse, and Stop hooks."""
    hook_paths = TOOLS["claude"].get("hooks_path", {})
    hook_path_fn = hook_paths.get(scope) or hook_paths.get("user")
    if not hook_path_fn:
        return

    hook_path = hook_path_fn()
    existing = _read_json(hook_path)
    hooks = existing.setdefault("hooks", {})

    # Find attest-mcp binary co-located with the running Python
    attest_mcp = _find_attest_mcp()
    if attest_mcp:
        recall_cmd = f"{attest_mcp} recall"
        pre_edit_cmd = f"{attest_mcp} pre-edit-check"
        pre_read_cmd = f"{attest_mcp} pre-read-check"
        post_test_cmd = f"{attest_mcp} post-test-check"
        stop_cmd = f"{attest_mcp} stop"
    else:
        base = f"{sys.executable} -m attestdb.mcp_install"
        recall_cmd = f"{base} recall"
        pre_edit_cmd = f"{base} pre-edit-check"
        pre_read_cmd = f"{base} pre-read-check"
        post_test_cmd = f"{base} post-test-check"
        stop_cmd = f"{base} stop"

    def _is_attest_hook(h: dict) -> bool:
        """Check if a hook entry is an attest hook."""
        return any(
            "attest-mcp" in (hk.get("command", "") or "")
            or "mcp_install" in (hk.get("command", "") or "")
            for hk in h.get("hooks", [])
        )

    # --- SessionStart: recall ---
    start_hooks = hooks.get("SessionStart", [])
    start_hooks = [h for h in start_hooks if not _is_attest_hook(h)]
    start_hooks.append({
        "matcher": "",
        "hooks": [{"type": "command", "command": recall_cmd}],
    })
    hooks["SessionStart"] = start_hooks

    # --- PreToolUse: warn before editing files with known issues ---
    pre_hooks = hooks.get("PreToolUse", [])
    pre_hooks = [h for h in pre_hooks if not _is_attest_hook(h)]
    pre_hooks.append({
        "matcher": "Edit|Write",
        "hooks": [{"type": "command", "command": pre_edit_cmd, "timeout": 10}],
    })
    pre_hooks.append({
        "matcher": "Read",
        "hooks": [{"type": "command", "command": pre_read_cmd, "timeout": 3}],
    })
    hooks["PreToolUse"] = pre_hooks

    # --- PostToolUse: surface prior fixes when tests fail ---
    post_hooks = hooks.get("PostToolUse", [])
    post_hooks = [h for h in post_hooks if not _is_attest_hook(h)]
    post_hooks.append({
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": post_test_cmd, "timeout": 10}],
    })
    hooks["PostToolUse"] = post_hooks

    # --- Stop: lightweight session summary signal ---
    stop_hooks = hooks.get("Stop", [])
    stop_hooks = [h for h in stop_hooks if not _is_attest_hook(h)]
    stop_hooks.append({
        "matcher": "",
        "hooks": [{"type": "command", "command": stop_cmd, "timeout": 5}],
    })
    hooks["Stop"] = stop_hooks

    _write_json(hook_path, existing)
    print(f"  Claude Code hooks: {hook_path}")


def _install_agent_instructions() -> None:
    """Generate agent instruction files for each tool that supports them."""
    instructions = _build_instructions()

    # CLAUDE.md — only if one doesn't already exist (don't overwrite user's file)
    # Instead, write to a separate file that CLAUDE.md can reference
    attest_md = Path(".attest-instructions.md")
    attest_md.write_text(instructions + "\n")
    print(f"  Agent instructions: {attest_md}")

    # .cursorrules — append if not already present
    cursorrules = Path(".cursorrules")
    if cursorrules.exists():
        existing = cursorrules.read_text()
        if "attest_get_prior_approaches" not in existing:
            cursorrules.write_text(existing.rstrip() + "\n\n" + instructions + "\n")
            print(f"  Cursor rules: {cursorrules} (appended)")
    # Don't create .cursorrules if it doesn't exist — only append


def _build_instructions() -> str:
    """Build the agent instructions markdown."""
    return """## Attest Memory (MCP Learning Layer)

You have a persistent knowledge graph that remembers across sessions.
Three things happen automatically — no action needed from you:

1. **Session start**: Prior knowledge is injected based on git context
2. **Before edits**: Known warnings/bugs for the file appear as context
3. **After test failures**: Prior fixes for failing tests are surfaced

### Recording knowledge (during work)
When you discover something worth remembering, call `attest_learned`:
- `attest_learned("file.py", "the real bug was X not Y", "bug")`
- `attest_learned("module_name", "always do X before Y", "pattern")`
- `attest_learned("serde_json::Value", "can't use with bincode", "warning")`

Types: `bug`, `fix`, `pattern`, `decision`, `warning`, `tip`

This is the primary way to teach the knowledge graph. Use it whenever you
discover something that would save time if you encountered it again.

### Ending a session
When the user's task is done, call `attest_session_end`:
- `outcome`: "success", "partial", or "failure"
- `summary`: Brief description of what was accomplished
- `next_steps`: What should happen next (shown at start of next session)
- `files_changed`: Key files that were modified

### Querying knowledge
- `attest_ask`: Natural-language questions ("What bugs were found in X?")
- `attest_get_prior_approaches`: Find relevant past work ranked by outcome
- `attest_confidence_trail`: See how confidence evolved on a topic"""


# ---------------------------------------------------------------------------
# Recall — SessionStart hook that outputs prior approaches
# ---------------------------------------------------------------------------

def _merge_session_end(db_path: Path) -> bool:
    """Merge session-end sidecar into the real DB.

    Written by the Stop hook. Contains outcome, summary, files_changed.
    Produces the same claims as attest_session_end() would.
    Returns True if merged successfully.
    """
    session_end_path = DEFAULT_MEMORY_DIR / "session_end.json"
    if not session_end_path.exists():
        return False

    try:
        with open(session_end_path) as f:
            data = json.load(f)
    except Exception:
        return False

    outcome = data.get("outcome", "success")
    summary = data.get("summary", "")
    files_changed = data.get("files_changed", [])

    if not summary:
        session_end_path.unlink(missing_ok=True)
        return False

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        # Unrecoverable — no point retrying on next session
        session_end_path.unlink(missing_ok=True)
        return False

    try:
        db = AttestDB(str(db_path), embedding_dim=None)
    except Exception:
        return False

    try:
        import uuid

        sid = f"auto-stop:{uuid.uuid4().hex[:12]}"
        project = data.get("project")

        prov = {"source_type": "session_end", "source_id": sid}
        if project:
            prov["project"] = project

        db.ingest(
            subject=(sid, "tool_session"),
            predicate=("had_outcome", "had_outcome"),
            object=(outcome, "outcome_value"),
            provenance=prov,
            confidence=0.9,
            payload={
                "schema_ref": "session_outcome",
                "data": {
                    "outcome": outcome,
                    "summary": summary,
                    "files_changed": files_changed,
                    "project": project or "",
                },
            },
        )

        # Record files changed so recall can surface them
        for fname in files_changed[:10]:
            try:
                db.ingest(
                    subject=(sid, "tool_session"),
                    predicate=("modified_file", "predicate"),
                    object=(fname, "source_file"),
                    provenance=prov,
                    confidence=0.7,
                )
            except Exception:
                continue

    finally:
        db.close()

    session_end_path.unlink(missing_ok=True)
    return True


def _merge_auto_learnings(db_path: Path) -> int:
    """Merge auto-extracted learnings from the sidecar into the real DB.

    Called at SessionStart (recall), before the MCP server acquires the lock.
    The sidecar is written by the Stop hook and contains JSONL entries with
    {timestamp, file, description, type}.

    Returns number of learnings merged.
    """
    auto_path = DEFAULT_MEMORY_DIR / "auto_learnings.jsonl"
    if not auto_path.exists():
        return 0

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return 0

    entries: list[dict] = []
    try:
        with open(auto_path) as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return 0

    if not entries:
        # Empty file — clean up
        auto_path.unlink(missing_ok=True)
        return 0

    # Open the real DB — MCP server shouldn't be holding the lock yet
    try:
        db = AttestDB(str(db_path), embedding_dim=None)
    except Exception:
        return 0

    count = 0
    try:
        for entry in entries:
            file_name = entry.get("file", "")
            description = entry.get("description", "")
            kind = entry.get("type", "pattern")

            if not file_name or not description:
                continue

            # Map type to predicate
            pred_map = {
                "bug": "had_bug",
                "fix": "has_fix",
                "pattern": "has_pattern",
                "decision": "has_decision",
                "warning": "has_warning",
                "tip": "has_tip",
            }
            predicate = pred_map.get(kind, "has_pattern")

            try:
                db.ingest(
                    subject=file_name,
                    predicate=predicate,
                    object=description,
                    source_id="auto_extraction",
                    source_type="hook",
                    confidence=0.6,  # Lower confidence for auto-extracted
                )
                count += 1
            except Exception:
                continue
    finally:
        db.close()

    # Clear the sidecar after successful merge
    if count > 0:
        auto_path.unlink(missing_ok=True)

    return count


def recall(task: str | None = None, db_path: str | None = None) -> None:
    """Query the memory DB and print context-aware knowledge to stdout.

    Output goes to the AI's context window via SessionStart hook.
    Uses git context (modified files, recent commits) to surface relevant knowledge.
    Scoped to the current project when possible (old untagged claims pass through).
    """
    memory_db = Path(db_path) if db_path else DEFAULT_MEMORY_DB
    if not memory_db.exists():
        return

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return

    # Merge session-end data and auto-extracted learnings from the Stop hook
    _merge_session_end(memory_db)
    _merge_auto_learnings(memory_db)

    # Auto-detect project for scoping
    project = _detect_recall_project()

    # Open read-only to avoid fs2 lock conflicts with the MCP server.
    # The Rust engine copies the DB to a temp file internally.
    try:
        db = AttestDB.open_read_only(str(memory_db))
    except Exception:
        return

    try:
        from attestdb.core.vocabulary import (
            KNOWLEDGE_HIGH_PRIORITY_THRESHOLD,
            knowledge_label,
        )

        stats = db.stats()
        total_claims = stats.get("total_claims", 0)
        if total_claims == 0:
            return

        # Gather git context for relevance filtering
        context_files, context_terms = _git_context()

        lines = []
        lines.append("## Attest Memory")
        lines.append(f"*{total_claims} claims, "
                     f"{stats.get('entity_count', 0)} entities*\n")

        # 1. Continue from previous session (next_steps)
        try:
            next_claims = db.claims_for_predicate("has_next_steps")
            if next_claims:
                # Filter by project — old untagged claims (no project) pass through
                if project:
                    next_claims = [
                        c for c in next_claims
                        if not c.provenance.project or c.provenance.project == project
                    ]
                if next_claims:
                    latest = max(next_claims, key=lambda c: c.timestamp)
                    age = _format_age(latest.timestamp)
                    lines.append(f"### Continue from previous session ({age})")
                    lines.append(f"{latest.object.id}\n")
        except Exception:
            pass

        # 2. Context-relevant knowledge (based on git status)
        # Includes warnings, patterns, decisions, tips, bugs, fixes — prioritized
        relevant_items = _find_relevant_knowledge(db, context_files, context_terms, project)

        # Split by priority threshold into actionable vs historical
        high_pri = [r for r in relevant_items if r[4] <= KNOWLEDGE_HIGH_PRIORITY_THRESHOLD]
        low_pri = [r for r in relevant_items if r[4] > KNOWLEDGE_HIGH_PRIORITY_THRESHOLD]

        if high_pri:
            lines.append("### Warnings & patterns (relevant to current work)")
            for pred, subj, obj, conf, _pri in high_pri[:8]:
                lines.append(f"- **[{knowledge_label(pred)}]** `{subj}`: {obj}")
            lines.append("")

        if low_pri:
            lines.append("### Known issues in these files")
            for pred, subj, obj, conf, _pri in low_pri[:5]:
                lines.append(f"- [{knowledge_label(pred)}] `{subj}`: {obj}")
            lines.append("")

        # 3. Non-context learnings (warnings/patterns not matched by git context)
        # Only show if they weren't already surfaced above
        context_claim_ids = {(r[1], r[2]) for r in relevant_items}
        learning_predicates = ["has_warning", "has_pattern", "has_decision"]
        extra_learnings: list[tuple] = []
        for pred in learning_predicates:
            try:
                claims = db.claims_for_predicate(pred)
            except Exception:
                continue
            for c in claims:
                if (c.subject.id, c.object.id) not in context_claim_ids:
                    extra_learnings.append((
                        knowledge_label(pred), c.subject.id, c.object.id,
                        c.timestamp,
                    ))

        if extra_learnings:
            extra_learnings.sort(key=lambda x: x[3], reverse=True)
            lines.append("### Other learnings")
            for kind, subj, obj, _ts in extra_learnings[:5]:
                lines.append(f"- [{kind}] `{subj}`: {obj}")
            lines.append("")

        # 4. Recent session outcomes (brief)
        try:
            outcome_claims = db.claims_for_predicate("had_outcome")
            if project:
                outcome_claims = [
                    c for c in outcome_claims
                    if not c.provenance.project or c.provenance.project == project
                ]
        except Exception:
            outcome_claims = []

        if outcome_claims:
            recent = sorted(outcome_claims, key=lambda c: c.timestamp, reverse=True)[:5]
            lines.append(f"### Recent sessions ({len(outcome_claims)} total)")
            for c in recent:
                summary = ""
                if c.payload and hasattr(c.payload, "data") and c.payload.data:
                    summary = c.payload.data.get("summary", "")
                age = _format_age(c.timestamp)
                icon = "+" if "success" in c.object.id else "-" if "failure" in c.object.id else "~"
                desc = f" — {summary}" if summary else ""
                lines.append(f"- [{icon}] {c.object.id} ({age}){desc}")
            lines.append("")

        # 5. Smart recommendations based on gaps and learnings
        recs = _smart_recommendations(db, project)
        if recs:
            lines.append("### Recommended next")
            lines.extend(recs)
            lines.append("")

        # Token discipline tips (always shown)
        lines.append("### Token discipline")
        token_lines = _session_efficiency_summary()
        if token_lines:
            lines.extend(token_lines)
        else:
            lines.append("- Before reading PDFs: `pandoc file.pdf -o file.md` (saves 5-20x tokens)")
            lines.append("- Every ~15 turns: /clear or /compact to avoid context bloat")
        lines.append("")

        lines.append("*Use `attest_learned` to record findings. "
                     "Use `attest_session_end` when done.*")

        # Auto-regenerate skills if claims changed (hash check makes this fast)
        try:
            from attestdb.intelligence.skill_generator import generate_skills
            generate_skills(db, output_dir=".claude/skills")
        except Exception:
            pass  # skill regeneration is optional

        print("\n".join(lines))
    finally:
        db.close()


def _smart_recommendations(db, project: str | None = None) -> list[str]:
    """Mine the knowledge graph for actionable recommendations."""
    def _proj_ok(c) -> bool:
        if not project:
            return True
        cp = c.provenance.project if c.provenance else None
        return not cp or cp == project

    recs: list[str] = []
    try:
        # 1. Unresolved bugs: had_bug without a corresponding has_fix
        bugs = []
        fixes_for: set[str] = set()
        try:
            for c in db.claims_for_predicate("has_fix"):
                if _proj_ok(c):
                    fixes_for.add(c.subject.id)
            for c in db.claims_for_predicate("had_bug"):
                if _proj_ok(c) and c.subject.id not in fixes_for:
                    bugs.append((c.subject.id, c.object.id, c.timestamp))
        except Exception:
            pass

        if bugs:
            bugs.sort(key=lambda b: b[2], reverse=True)
            for subj, desc, _ts in bugs[:3]:
                short = desc[:80] + "..." if len(desc) > 80 else desc
                recs.append(f"- **Fix needed**: `{subj}` — {short}")

        # 2. Incomplete sessions (partial/failure outcomes with next_steps)
        try:
            outcomes = db.claims_for_predicate("had_outcome")
            incomplete = [
                c for c in outcomes
                if ("partial" in c.object.id or "failure" in c.object.id)
                and _proj_ok(c)
            ]
            if incomplete:
                latest = max(incomplete, key=lambda c: c.timestamp)
                summary = ""
                if latest.payload and hasattr(latest.payload, "data") and latest.payload.data:
                    summary = latest.payload.data.get("summary", "")
                if summary:
                    age = _format_age(latest.timestamp)
                    short = summary[:80] + "..." if len(summary) > 80 else summary
                    recs.append(f"- **Unfinished** ({age}): {short}")
        except Exception:
            pass

        # 3. Unresolved warnings (high-priority, no corresponding fix/decision)
        try:
            resolved_subjects = set()
            for pred in ("has_fix", "has_decision", "resolved"):
                try:
                    for c in db.claims_for_predicate(pred):
                        resolved_subjects.add(c.subject.id)
                except Exception:
                    pass

            for c in db.claims_for_predicate("has_warning"):
                if c.subject.id not in resolved_subjects and _proj_ok(c) and len(recs) < 5:
                    short = c.object.id[:80] + "..." if len(c.object.id) > 80 else c.object.id
                    recs.append(f"- **Open warning**: `{c.subject.id}` — {short}")
        except Exception:
            pass

        # 4. Recent negative results (dead ends to avoid)
        try:
            negs = db.claims_for_predicate("has_negative_result")
            if negs:
                recent_negs = sorted(negs, key=lambda c: c.timestamp, reverse=True)[:2]
                for c in recent_negs:
                    short = c.object.id[:80] + "..." if len(c.object.id) > 80 else c.object.id
                    recs.append(f"- **Dead end**: `{c.subject.id}` — {short}")
        except Exception:
            pass

    except Exception:
        pass

    return recs[:6]


def _session_efficiency_summary() -> list[str]:
    """Compute session efficiency stats from hook_metrics.jsonl for recall output."""
    try:
        if not HOOK_METRICS_LOG.exists():
            return []

        raw_lines = HOOK_METRICS_LOG.read_text().strip().split("\n")
        if len(raw_lines) < 5:
            return []

        timestamps: list[float] = []
        pdf_reads = 0
        for line in raw_lines[-200:]:
            try:
                m = json.loads(line)
                ts = _parse_metric_timestamp(m.get("timestamp", 0))
                if ts > 0:
                    timestamps.append(ts)
                if m.get("hook") == "pre_read_check" and m.get("fired"):
                    pdf_reads += 1
            except (json.JSONDecodeError, KeyError):
                continue

        if not timestamps:
            return []

        # Count sessions (gaps > 30 min)
        sessions = 1
        session_lengths: list[float] = []
        session_start = timestamps[0]
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i - 1] > 1800:
                session_lengths.append(timestamps[i - 1] - session_start)
                session_start = timestamps[i]
                sessions += 1
        session_lengths.append(timestamps[-1] - session_start)

        avg_min = (sum(session_lengths) / len(session_lengths)) / 60
        total_events = len(timestamps)

        tips = []
        tips.append(
            f"- {sessions} sessions tracked  ·  avg {avg_min:.0f} min  ·  "
            f"{total_events} tool calls"
        )
        if pdf_reads > 0:
            tips.append(
                f"- {pdf_reads} raw PDF/image reads last period — "
                f"run `pandoc file.pdf -o file.md` before reading"
            )
        if avg_min > 30:
            tips.append(
                "- Sessions averaging >30 min — use /clear every ~15 turns"
            )
        return tips
    except Exception:
        return []


def _detect_recall_project() -> str | None:
    """Detect project from git remote for recall scoping."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            url = result.stdout.strip().rstrip("/")
            if url.endswith(".git"):
                url = url[:-4]
            if url.startswith("git@"):
                url = url[4:].replace(":", "/", 1)
            elif url.startswith(("https://", "http://")):
                url = url.split("://", 1)[1]
            return url
    except Exception:
        pass
    return os.path.basename(os.getcwd())


def _git_context() -> tuple[list[str], set[str]]:
    """Extract context from git: modified files and recent commit terms."""
    import subprocess

    files: list[str] = []
    terms: set[str] = set()

    try:
        # Modified/staged files
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    path = line[3:].strip().split(" -> ")[-1]  # handle renames
                    files.append(path)
                    # Extract terms from file path
                    for part in path.replace("/", " ").replace("_", " ").replace(".", " ").split():
                        if len(part) >= 3:
                            terms.add(part.lower())
    except Exception:
        pass

    try:
        # Recent commit messages
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--no-decorate"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    # Skip the hash, get message words
                    msg = line.split(" ", 1)[-1] if " " in line else ""
                    for word in msg.split():
                        w = word.strip(".,;:!?()[]{}\"'").lower()
                        if len(w) >= 3 and w not in {"the", "and", "for", "fix", "add", "update"}:
                            terms.add(w)
    except Exception:
        pass

    return files, terms


def _find_relevant_knowledge(db, files: list[str], terms: set[str], project: str | None = None) -> list[tuple]:
    """Find knowledge claims relevant to the current git context.

    Returns (predicate, subject, object, confidence, priority) tuples,
    sorted by priority (warnings/patterns first, then bugs, then fixes).
    Filters by project when set — old untagged claims (no project) pass through.
    """
    if not files and not terms:
        return []

    from attestdb.core.vocabulary import KNOWLEDGE_PREDICATES, KNOWLEDGE_PRIORITY

    def _project_matches(claim) -> bool:
        """True if claim belongs to this project or is untagged (graceful degradation)."""
        if not project:
            return True
        claim_project = claim.provenance.project if claim.provenance else None
        return not claim_project or claim_project == project

    # Filter to predicates relevant for recall (skip session-level ones)
    recall_predicates = {
        p for p in KNOWLEDGE_PREDICATES
        if p not in {"had_outcome", "produced_by", "has_next_steps"}
    }
    all_predicates = recall_predicates

    relevant: list[tuple] = []
    seen: set[str] = set()

    # Direct file matches (highest relevance)
    for f in files:
        entities = db.search_entities(f, top_k=5)
        for entity in entities:
            for claim in db.claims_for(entity.id):
                if claim.claim_id not in seen and claim.predicate.id in all_predicates and _project_matches(claim):
                    seen.add(claim.claim_id)
                    relevant.append((
                        claim.predicate.id, claim.subject.id, claim.object.id,
                        claim.confidence,
                        KNOWLEDGE_PRIORITY.get(claim.predicate.id, 9),
                    ))

    # Term-based matches
    for pred in all_predicates:
        try:
            claims = db.claims_for_predicate(pred)
        except Exception:
            continue
        for c in claims:
            if c.claim_id in seen:
                continue
            if not _project_matches(c):
                continue
            subj_lower = c.subject.id.lower()
            obj_lower = c.object.id.lower()
            if any(t in subj_lower or t in obj_lower for t in terms):
                seen.add(c.claim_id)
                relevant.append((
                    c.predicate.id, c.subject.id, c.object.id,
                    c.confidence,
                    KNOWLEDGE_PRIORITY.get(c.predicate.id, 9),
                ))

    # Sort by priority (warnings first), then by confidence (highest first)
    relevant.sort(key=lambda x: (x[4], -x[3]))
    return relevant


def _format_age(ts: int) -> str:
    """Format a nanosecond timestamp as a relative age string."""
    if ts <= 0:
        return "unknown age"
    age_days = int((time.time() - ts / 1e9) / 86400)
    if age_days < 0:
        return "just now"
    if age_days == 0:
        return "today"
    if age_days == 1:
        return "yesterday"
    return f"{age_days}d ago"


def _brain_banner(title: str, body: str) -> str:
    """Wrap hook output in a branded Attest Brain banner."""
    header = f"🧠 Attest Brain  ›  {title}"
    bar = "─" * len(header)
    lines = [f"  {l}" for l in body.strip().split("\n")]
    return f"{header}\n{bar}\n" + "\n".join(lines)


def _log_hook_metric(hook: str, file: str, fired: bool, n_items: int, latency_ms: float) -> None:
    """Append a single metric line to the hook metrics JSONL log."""
    try:
        from datetime import datetime, timezone

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hook": hook,
            "file": file,
            "fired": fired,
            "n_items": n_items,
            "latency_ms": round(latency_ms, 2),
        }
        DEFAULT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(HOOK_METRICS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never let instrumentation break the hooks


# ---------------------------------------------------------------------------
# PreToolUse hook — warn about expensive document reads
# ---------------------------------------------------------------------------

_EXPENSIVE_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg",
}


def pre_read_check() -> None:
    """PreToolUse hook for Read: warn about expensive PDF/image formats.

    Fast — only checks the file extension, no DB query needed.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return

    file_path = hook_input.get("tool_input", {}).get("file_path", "")
    if not file_path:
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _EXPENSIVE_EXTENSIONS:
        return  # Text/code files — no output, zero cost

    _t0 = time.monotonic()
    stem = file_path.rsplit(".", 1)[0]

    basename = os.path.basename(file_path)
    if ext == ".pdf":
        body = (
            f"{basename} will cost 5-20x more tokens as raw PDF.\n"
            f"\n"
            f"Run one of these first, then read the .md instead:\n"
            f"  $ pandoc \"{file_path}\" -o \"{stem}.md\"\n"
            f"  $ markitdown \"{file_path}\" > \"{stem}.md\"\n"
            f"\n"
            f"A 4,500-word PDF: ~50K tokens raw vs ~5K as markdown."
        )
        warning = _brain_banner("Token Warning", body)
    else:
        body = (
            f"{basename} is expensive in context as a raw image.\n"
            f"\n"
            f"Options:\n"
            f"  Ask me to describe what you need from it\n"
            f"  Extract text with OCR first: tesseract \"{file_path}\" stdout"
        )
        warning = _brain_banner("Token Warning", body)

    result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": warning,
        }
    }
    print(json.dumps(result))
    _log_hook_metric(
        hook="pre_read_check",
        file=file_path,
        fired=True,
        n_items=1,
        latency_ms=(time.monotonic() - _t0) * 1000,
    )


# ---------------------------------------------------------------------------
# PreToolUse hook — surface warnings before editing a file
# ---------------------------------------------------------------------------

def _check_session_sprawl() -> str | None:
    """Check hook_metrics.jsonl for session sprawl. Returns warning or None."""
    try:
        if not HOOK_METRICS_LOG.exists():
            return None

        lines = HOOK_METRICS_LOG.read_text().strip().split("\n")[-100:]
        now = time.time()

        timestamps: list[float] = []
        for line in lines:
            try:
                m = json.loads(line)
                ts = _parse_metric_timestamp(m.get("timestamp", 0))
                if ts > 0:
                    timestamps.append(ts)
            except (json.JSONDecodeError, KeyError):
                continue

        if len(timestamps) < 5:
            return None

        # Find session boundary: last gap > 30 min
        session_start = timestamps[0]
        for i in range(len(timestamps) - 1, 0, -1):
            if timestamps[i] - timestamps[i - 1] > 1800:
                session_start = timestamps[i]
                break

        session_events = sum(1 for ts in timestamps if ts >= session_start)
        elapsed_min = (now - session_start) / 60

        if session_events >= 80:
            severity = "critical"
        elif session_events >= 50:
            severity = "high"
        elif session_events >= 30:
            severity = "moderate"
        else:
            return None

        return (
            f"~{session_events} tool calls  ·  {elapsed_min:.0f} min  ·  {severity}\n"
            f"\n"
            f"Every turn resends the full conversation history.\n"
            f"Turn 5 costs ~2K tokens. Turn 30 costs ~40K+.\n"
            f"\n"
            f"Actions:\n"
            f"  /clear               start fresh (fastest)\n"
            f"  /compact             compress context in-place\n"
            f"  prompt_kit_rescue    extract key decisions, then /clear"
        )
    except Exception:
        return None


def pre_edit_check() -> None:
    """PreToolUse hook: check knowledge graph for warnings about the file being edited.

    Reads JSON from stdin (Claude Code hook format), queries the memory DB for
    the file, and outputs additionalContext if there are relevant warnings.
    """
    import sys

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    if not file_path:
        return

    memory_db = DEFAULT_MEMORY_DB
    if not memory_db.exists():
        return

    # Extract filename and module-level identifiers for search
    basename = os.path.basename(file_path)
    # Also try the relative path (e.g., "attestdb/mcp_server.py")
    search_terms = [basename]
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) >= 2:
        search_terms.append("/".join(parts[-2:]))

    try:
        from attestdb.core.vocabulary import KNOWLEDGE_PREDICATES, knowledge_label
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return

    try:
        db = AttestDB.open_read_only(str(memory_db))
    except Exception:
        return

    try:
        warnings: list[str] = []
        seen: set[str] = set()

        _t0 = time.monotonic()
        for term in search_terms:
            entities = db.search_entities(term, top_k=3)
            for entity in entities:
                # Only include claims whose subject contains the search term
                # to avoid fuzzy matches returning unrelated files
                if term.lower() not in entity.id.lower():
                    continue
                for claim in db.claims_for(entity.id):
                    if claim.claim_id in seen:
                        continue
                    if claim.predicate.id in KNOWLEDGE_PREDICATES:
                        seen.add(claim.claim_id)
                        label = knowledge_label(claim.predicate.id)
                        warnings.append(
                            f"[{label}] {claim.subject.id}: {claim.object.id}"
                        )
        _latency_ms = (time.monotonic() - _t0) * 1000

        # Check for session sprawl (cheap — reads metrics file, no DB)
        sprawl_warning = _check_session_sprawl()

        if warnings or sprawl_warning:
            parts = []
            if sprawl_warning:
                parts.append(_brain_banner("Session Sprawl", sprawl_warning))
            if warnings:
                body = "\n".join(f"- {w}" for w in warnings[:5])
                parts.append(_brain_banner(f"Known Issues: {basename}", body))
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": "\n".join(parts),
                }
            }
            print(json.dumps(result))

        # Instrumentation
        _log_hook_metric(
            hook="pre_edit_check",
            file=file_path,
            fired=len(warnings) > 0 or sprawl_warning is not None,
            n_items=len(warnings) + (1 if sprawl_warning else 0),
            latency_ms=_latency_ms,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# PostToolUse hook — detect test failures, surface prior fixes
# ---------------------------------------------------------------------------

def post_test_check() -> None:
    """PostToolUse hook: detect test failures and surface prior fixes.

    Reads JSON from stdin (Claude Code hook format). If a Bash command ran
    tests and failed, checks the knowledge graph for prior fixes.
    """
    import sys

    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        return

    tool_name = hook_input.get("tool_name", "")
    if tool_name != "Bash":
        return

    tool_input = hook_input.get("tool_input", {})
    command = tool_input.get("command", "")

    # Only trigger on test commands
    test_indicators = ("pytest", "cargo test", "python -m pytest", "uv run pytest")
    if not any(ind in command for ind in test_indicators):
        return

    tool_response = hook_input.get("tool_response", {})
    stdout = tool_response.get("stdout", "")
    stderr = tool_response.get("stderr", "")
    output = stdout + "\n" + stderr

    # Check for test failures
    if "FAILED" not in output and "ERRORS" not in output and "error[" not in output:
        return  # Tests passed, nothing to do

    # Extract failed test names
    failed_tests: list[str] = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("FAILED "):
            # "FAILED tests/unit/test_foo.py::test_bar - AssertionError"
            test_path = line[7:].split(" - ")[0].split("::")[0]
            test_name = line[7:].split(" - ")[0]
            if test_path not in failed_tests:
                failed_tests.append(test_path)

    if not failed_tests:
        return

    memory_db = DEFAULT_MEMORY_DB
    if not memory_db.exists():
        return

    try:
        from attestdb.core.vocabulary import knowledge_label
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return

    try:
        db = AttestDB.open_read_only(str(memory_db))
    except Exception:
        return

    try:
        hints: list[str] = []
        seen: set[str] = set()

        _t0 = time.monotonic()
        for test_path in failed_tests[:3]:
            basename = os.path.basename(test_path)
            # Search by test file name AND derived source module name
            # e.g., test_mcp_server.py → mcp_server.py, mcp_server
            search_terms = [basename]
            source_name = basename.replace("test_", "").replace("test", "")
            if source_name:
                search_terms.append(source_name)
                search_terms.append(source_name.replace(".py", ""))

            for term in search_terms:
                entities = db.search_entities(term, top_k=3)
                for entity in entities:
                    for claim in db.claims_for(entity.id):
                        if claim.claim_id in seen:
                            continue
                        if claim.predicate.id in ("has_fix", "had_bug", "has_tip", "has_warning"):
                            seen.add(claim.claim_id)
                            label = knowledge_label(claim.predicate.id)
                            hints.append(
                                f"[{label}] {claim.subject.id}: {claim.object.id}"
                            )
        _latency_ms = (time.monotonic() - _t0) * 1000

        if hints:
            context = (
                "⚠ Attest found prior knowledge about these test failures:\n"
                + "\n".join(f"  - {h}" for h in hints[:5])
            )
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": context,
                }
            }
            print(json.dumps(result))

        # Instrumentation
        _log_hook_metric(
            hook="post_test_check",
            file=command,
            fired=len(hints) > 0,
            n_items=len(hints),
            latency_ms=_latency_ms,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Stop hook — session signal + auto-extraction of learnings
# ---------------------------------------------------------------------------

def _parse_metric_timestamp(ts) -> float:
    """Parse a metric timestamp (ISO string or epoch float) to epoch float."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(ts)
            return dt.timestamp()
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _extract_test_fix_cycles() -> list[dict]:
    """Scan recent hook metrics for test-fail → test-pass patterns.

    Returns a list of {file, description} dicts for auto-ingestion.
    A cycle is: post_test_check fired (test failed) on file X, then later
    post_test_check ran but didn't fire (test passed) on overlapping files.
    """
    if not HOOK_METRICS_LOG.exists():
        return []

    # Read last 200 metrics entries (enough for one session)
    entries: list[dict] = []
    try:
        with open(HOOK_METRICS_LOG) as f:
            lines = f.readlines()
        for line in lines[-200:]:
            try:
                entries.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    except Exception:
        return []

    # Find test-fail → test-pass transitions in the current session
    # (entries within the last 30 minutes)
    now = time.time()
    session_cutoff = now - 1800  # 30 minutes

    failures: dict[str, float] = {}  # command → timestamp of failure
    fixes: list[dict] = []

    for entry in entries:
        ts = _parse_metric_timestamp(entry.get("timestamp", 0))
        if ts < session_cutoff:
            continue

        if entry.get("hook") != "post_test_check":
            continue

        cmd = entry.get("file", "")
        if entry.get("fired"):
            # Test failure detected
            failures[cmd] = ts
        elif not entry.get("fired") and failures:
            # Tests passed — check if we had a prior failure with same/similar cmd
            for fail_cmd, fail_ts in list(failures.items()):
                if ts > fail_ts:
                    # Extract test file names from the pytest command
                    test_files = _extract_test_files(cmd)
                    if test_files:
                        fixes.append({
                            "file": test_files[0],
                            "description": (
                                f"test-fail-fix cycle detected: tests in "
                                f"{', '.join(test_files)} failed then passed"
                            ),
                        })
                    del failures[fail_cmd]
                    break

    return fixes


def _extract_test_files(command: str) -> list[str]:
    """Extract test file paths from a pytest command string."""
    parts = command.split()
    files: list[str] = []
    for part in parts:
        if "test" in part.lower() and part.endswith(".py"):
            files.append(os.path.basename(part))
        elif part.startswith("tests/"):
            files.append(os.path.basename(part))
    return files


def _extract_edit_patterns() -> list[dict]:
    """Scan recent pre_edit_check metrics for repeatedly edited files.

    If the same file was edited 3+ times in a session, it's worth noting
    as a hot-spot.
    """
    if not HOOK_METRICS_LOG.exists():
        return []

    now = time.time()
    session_cutoff = now - 1800

    file_counts: dict[str, int] = {}
    try:
        with open(HOOK_METRICS_LOG) as f:
            for line in f.readlines()[-200:]:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                ts = _parse_metric_timestamp(entry.get("timestamp", 0))
                if ts < session_cutoff:
                    continue
                if entry.get("hook") == "pre_edit_check":
                    fname = entry.get("file", "")
                    if fname:
                        file_counts[fname] = file_counts.get(fname, 0) + 1
    except Exception:
        return []

    patterns: list[dict] = []
    for fname, count in file_counts.items():
        if count >= 3:
            patterns.append({
                "file": os.path.basename(fname),
                "description": (
                    f"hot-spot: {os.path.basename(fname)} was edited "
                    f"{count} times in this session"
                ),
            })
    return patterns


def _auto_ingest_learnings(learnings: list[dict]) -> int:
    """Ingest auto-extracted learnings into the memory DB.

    Returns count of successfully ingested claims.
    """
    if not learnings:
        return 0

    memory_db = DEFAULT_MEMORY_DB
    if not memory_db.exists():
        return 0

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return 0

    # Unlike read-only hooks, the stop hook writes — so it needs the real DB.
    # But the MCP server might hold the lock. Copy + write + merge back?
    # Simpler: use a temp copy, write there, and on next recall it merges.
    # Actually simplest: use a separate "auto-learnings" sidecar file that
    # recall() checks and merges on next session start.
    auto_path = DEFAULT_MEMORY_DIR / "auto_learnings.jsonl"
    try:
        with open(auto_path, "a") as f:
            for learning in learnings:
                entry = {
                    "timestamp": time.time(),
                    "file": learning.get("file", ""),
                    "description": learning.get("description", ""),
                    "type": learning.get("type", "pattern"),
                }
                f.write(json.dumps(entry) + "\n")
        return len(learnings)
    except Exception:
        return 0


def _count_session_events() -> dict:
    """Count hook events in the current session for activity estimation."""
    try:
        if not HOOK_METRICS_LOG.exists():
            return {}

        lines = HOOK_METRICS_LOG.read_text().strip().split("\n")[-100:]
        now = time.time()

        timestamps: list[float] = []
        by_hook: dict[str, int] = {}
        for line in lines:
            try:
                m = json.loads(line)
                ts = _parse_metric_timestamp(m.get("timestamp", 0))
                if ts > 0:
                    timestamps.append(ts)
                    hook = m.get("hook", "unknown")
                    by_hook[hook] = by_hook.get(hook, 0) + 1
            except (json.JSONDecodeError, KeyError):
                continue

        if not timestamps:
            return {}

        # Find session boundary (last gap > 30 min)
        session_start = timestamps[0]
        for i in range(len(timestamps) - 1, 0, -1):
            if timestamps[i] - timestamps[i - 1] > 1800:
                session_start = timestamps[i]
                break

        session_events = sum(1 for ts in timestamps if ts >= session_start)
        elapsed_min = (now - session_start) / 60

        return {
            "total": session_events,
            "elapsed_min": round(elapsed_min, 1),
            **{k: v for k, v in by_hook.items()},
        }
    except Exception:
        return {}


def _infer_session_outcome() -> str:
    """Infer session outcome from hook metrics.

    Heuristic: if any post-test-check fired (test failure detected) with no
    subsequent success, outcome is "partial". If no test failures at all,
    outcome is "success". If only failures, "failure".
    """
    if not HOOK_METRICS_LOG.exists():
        return "success"  # No data = assume success (lightweight sessions)

    test_failures = 0
    test_checks = 0
    try:
        with open(HOOK_METRICS_LOG) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if entry.get("hook") == "post-test-check":
                    test_checks += 1
                    if entry.get("fired"):
                        test_failures += 1
    except Exception:
        return "success"

    if test_failures == 0:
        return "success"
    if test_failures < test_checks:
        return "partial"  # Some failures followed by passes
    return "failure"  # All test checks were failures


def _build_session_summary(
    learnings: list[dict], changed_files: list[str], outcome: str,
) -> str:
    """Build a brief session summary from auto-extracted data."""
    parts: list[str] = []

    if changed_files:
        parts.append(f"Modified {len(changed_files)} file(s)")

    bug_count = sum(1 for l in learnings if l.get("type") == "bug")
    fix_count = sum(1 for l in learnings if l.get("type") == "fix")
    pattern_count = sum(1 for l in learnings if l.get("type") in ("pattern", "tip"))

    if bug_count or fix_count:
        parts.append(f"{fix_count} fix(es), {bug_count} bug(s) detected")
    if pattern_count:
        parts.append(f"{pattern_count} pattern(s) extracted")

    if not parts:
        return f"Session ended ({outcome})"
    return f"{outcome.capitalize()}: {'. '.join(parts)}."


def stop_session_summary() -> None:
    """Stop hook: auto-extract learnings, write session-end sidecar, log signal.

    Fires when Claude Code finishes responding. Reads hook metrics to
    detect patterns (test-fail-fix cycles, hot-spot files), auto-ingests
    them as learnings, writes a session-end sidecar for recall to merge,
    and logs changed files via git.
    """
    try:
        import subprocess

        try:
            hook_input = json.loads(sys.stdin.read())
        except (json.JSONDecodeError, ValueError):
            hook_input = {}

        cwd = hook_input.get("cwd", os.getcwd())

        _t0 = time.monotonic()

        # --- Auto-extract learnings from this session ---
        learnings: list[dict] = []
        learnings.extend(_extract_test_fix_cycles())
        learnings.extend(_extract_edit_patterns())

        n_auto = _auto_ingest_learnings(learnings)

        # --- Get recently changed files (quick git check) ---
        changed_files: list[str] = []
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD~1"],
                capture_output=True, text=True, timeout=3, cwd=cwd,
            )
            if result.returncode == 0:
                changed_files = [
                    f for f in result.stdout.strip().split("\n") if f.strip()
                ]
        except Exception:
            pass

        # --- Write session-end sidecar for recall to merge ---
        outcome = _infer_session_outcome()
        summary = _build_session_summary(learnings, changed_files, outcome)

        # Count session activity from hook metrics
        activity = _count_session_events()

        # Auto-detect project for provenance tagging
        project = None
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=3, cwd=cwd,
            )
            if result.returncode == 0:
                url = result.stdout.strip().rstrip("/")
                if url.endswith(".git"):
                    url = url[:-4]
                if url.startswith("git@"):
                    url = url[4:].replace(":", "/", 1)
                elif url.startswith(("https://", "http://")):
                    url = url.split("://", 1)[1]
                project = url
        except Exception:
            pass
        if not project:
            project = os.path.basename(cwd)

        session_end_path = DEFAULT_MEMORY_DIR / "session_end.json"
        try:
            session_data = {
                "timestamp": time.time(),
                "outcome": outcome,
                "summary": summary,
                "files_changed": changed_files[:20],
                "learnings_count": n_auto,
                "cwd": cwd,
                "activity_estimate": activity,
                "project": project,
            }
            with open(session_end_path, "w") as f:
                json.dump(session_data, f)
        except Exception:
            pass

        _latency_ms = (time.monotonic() - _t0) * 1000

        # Print visible session summary
        _print_session_summary(activity, outcome, changed_files, n_auto)

        _log_hook_metric(
            hook="stop",
            file=json.dumps(changed_files[:20]) if changed_files else "[]",
            fired=n_auto > 0,
            n_items=n_auto,
            latency_ms=_latency_ms,
        )
    except Exception:
        pass  # Never fail — this is a lightweight signal


def _print_session_summary(
    activity: dict, outcome: str, changed_files: list[str], n_auto: int,
) -> None:
    """Print a visible session activity summary via Stop hook output."""
    try:
        total = activity.get("total", 0)
        elapsed = activity.get("elapsed_min", 0)
        edits = activity.get("pre_edit_check", 0)
        pdf_reads = activity.get("pre_read_check", 0)

        if total < 3:
            return  # Too short to summarize

        parts = [f"{outcome}"]
        parts.append(f"{total} tool calls in {elapsed:.0f} min")
        if changed_files:
            parts.append(f"{len(changed_files)} files changed")
        if edits:
            parts.append(f"{edits} edits")
        if pdf_reads:
            parts.append(f"{pdf_reads} PDF/image reads")
        if n_auto:
            parts.append(f"{n_auto} patterns auto-extracted")

        if total >= 50:
            parts.append("consider shorter sessions")

        reason = f"🧠 Attest Brain  ›  {' · '.join(parts)}"

        print(json.dumps({"stopReason": reason}))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metrics — summarize hook instrumentation data
# ---------------------------------------------------------------------------

def metrics() -> None:
    """Read the hook metrics JSONL log and print a summary."""
    if not HOOK_METRICS_LOG.exists():
        print("No hook metrics recorded yet.")
        print(f"  Log path: {HOOK_METRICS_LOG}")
        return

    by_hook: dict[str, list[dict]] = {}
    line_count = 0
    parse_errors = 0

    with open(HOOK_METRICS_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                parse_errors += 1
                continue
            hook = entry.get("hook", "unknown")
            by_hook.setdefault(hook, []).append(entry)

    if line_count == 0:
        print("No hook metrics recorded yet.")
        return

    print(f"Hook Metrics Summary ({HOOK_METRICS_LOG})")
    print(f"{'=' * 60}")
    print(f"Total entries: {line_count}" + (f" ({parse_errors} parse errors)" if parse_errors else ""))
    print()

    for hook, entries in sorted(by_hook.items()):
        total = len(entries)
        fired = sum(1 for e in entries if e.get("fired"))
        fire_rate = (fired / total * 100) if total else 0

        items_when_fired = [e.get("n_items", 0) for e in entries if e.get("fired")]
        avg_items = (sum(items_when_fired) / len(items_when_fired)) if items_when_fired else 0

        latencies = [e.get("latency_ms", 0) for e in entries]
        avg_latency = (sum(latencies) / len(latencies)) if latencies else 0

        print(f"  {hook}:")
        print(f"    Invocations:    {total}")
        print(f"    Fired:          {fired} ({fire_rate:.1f}%)")
        print(f"    Avg items/fire: {avg_items:.1f}")
        print(f"    Avg latency:    {avg_latency:.1f} ms")
        print()


# ---------------------------------------------------------------------------
# Benchmark — quantify the knowledge graph's impact
# ---------------------------------------------------------------------------

def benchmark(db_path: str | None = None) -> None:
    """Measure the knowledge graph's impact on coding sessions.

    Computes concrete metrics that answer: "how much does AttestDB help?"
    """
    memory_db = Path(db_path) if db_path else DEFAULT_MEMORY_DB
    if not memory_db.exists():
        print("No memory database found. Start using attest-mcp to build knowledge.")
        return

    try:
        from attestdb.core.vocabulary import (
            KNOWLEDGE_PREDICATES,
            KNOWLEDGE_PRIORITY,
            knowledge_label,
        )
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        print("attestdb not installed.")
        return

    # Open DB in read-only mode (copies to temp internally, no lock conflict)
    try:
        db = AttestDB.open_read_only(str(memory_db))
    except Exception as e:
        print(f"Failed to open memory database: {e}")
        return

    try:
        stats = db.stats()
        total_claims = stats.get("total_claims", 0)
        entity_count = stats.get("entity_count", 0)

        # --- 1. Knowledge inventory ---
        knowledge_claims: dict[str, list] = {}
        for pred in KNOWLEDGE_PREDICATES:
            try:
                claims = db.claims_for_predicate(pred)
                if claims:
                    knowledge_claims[pred] = claims
            except Exception:
                pass

        total_knowledge = sum(len(v) for v in knowledge_claims.values())

        # Count by priority tier
        tier_0 = sum(len(knowledge_claims.get(p, []))
                     for p in ("has_warning", "has_vulnerability"))
        tier_1 = sum(len(knowledge_claims.get(p, []))
                     for p in ("has_pattern", "has_decision"))
        tier_2 = len(knowledge_claims.get("has_tip", []))
        tier_3 = sum(len(knowledge_claims.get(p, []))
                     for p in ("had_bug", "has_issue", "failed_on"))
        tier_4 = sum(len(knowledge_claims.get(p, []))
                     for p in ("has_fix", "resolved"))

        # --- 2. Coverage map ---
        # Which files have knowledge vs total entities
        file_entities: set[str] = set()
        covered_files: set[str] = set()
        for entity_dict in db._store.list_entities():
            eid = entity_dict.get("id", "")
            etype = entity_dict.get("entity_type", "")
            if etype == "source_file":
                file_entities.add(eid)
                # Check if this file has actionable knowledge
                for claim in db.claims_for(eid):
                    if claim.predicate.id in KNOWLEDGE_PREDICATES:
                        covered_files.add(eid)
                        break

        # --- 3. Bug/fix resolution rate ---
        bug_subjects: set[str] = set()
        fix_subjects: set[str] = set()
        for c in knowledge_claims.get("had_bug", []):
            bug_subjects.add(c.subject.id)
        for c in knowledge_claims.get("has_issue", []):
            bug_subjects.add(c.subject.id)
        for c in knowledge_claims.get("has_fix", []):
            fix_subjects.add(c.subject.id)
        for c in knowledge_claims.get("resolved", []):
            fix_subjects.add(c.subject.id)

        resolved = bug_subjects & fix_subjects
        unresolved = bug_subjects - fix_subjects

        # --- 4. Session outcomes ---
        try:
            outcome_claims = db.claims_for_predicate("had_outcome")
        except Exception:
            outcome_claims = []

        n_success = sum(1 for c in outcome_claims if "success" in c.object.id)
        n_partial = sum(1 for c in outcome_claims if "partial" in c.object.id)
        n_failure = sum(1 for c in outcome_claims if "failure" in c.object.id)
        n_sessions = len(outcome_claims)

        # --- 5. Hook metrics ---
        hook_stats: dict[str, dict] = {}
        if HOOK_METRICS_LOG.exists():
            with open(HOOK_METRICS_LOG) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    hook = entry.get("hook", "unknown")
                    h = hook_stats.setdefault(hook, {
                        "total": 0, "fired": 0, "items": 0,
                        "latency_sum": 0.0,
                    })
                    h["total"] += 1
                    if entry.get("fired"):
                        h["fired"] += 1
                        h["items"] += entry.get("n_items", 0)
                    h["latency_sum"] += entry.get("latency_ms", 0)

        # --- 6. Recall token efficiency ---
        # Compare knowledge graph output vs flat MEMORY.md
        recall_lines = []
        for pred, claims in sorted(
            knowledge_claims.items(),
            key=lambda x: KNOWLEDGE_PRIORITY.get(x[0], 9),
        ):
            for c in claims:
                recall_lines.append(
                    f"[{knowledge_label(pred)}] {c.subject.id}: {c.object.id}"
                )
        recall_text = "\n".join(recall_lines)
        recall_tokens = len(recall_text.split())  # rough word count

        # Check MEMORY.md size for comparison
        memory_md_tokens = 0
        memory_md_path = ""
        # Search Claude Code's project memory directories
        claude_projects = Path.home() / ".claude" / "projects"
        if claude_projects.exists():
            for candidate in claude_projects.glob("**/MEMORY.md"):
                try:
                    tokens = len(candidate.read_text().split())
                    if tokens > memory_md_tokens:
                        memory_md_tokens = tokens
                        memory_md_path = str(candidate)
                except Exception:
                    pass
        # Also check user-level
        user_memory = Path.home() / ".claude" / "MEMORY.md"
        if user_memory.exists():
            try:
                tokens = len(user_memory.read_text().split())
                if tokens > memory_md_tokens:
                    memory_md_tokens = tokens
                    memory_md_path = str(user_memory)
            except Exception:
                pass

        # --- Print report ---
        print("=" * 60)
        print("  ATTEST KNOWLEDGE GRAPH BENCHMARK")
        print("=" * 60)
        print()

        print(f"  Database: {memory_db}")
        print(f"  Claims: {total_claims}  |  Entities: {entity_count}")
        print(f"  Sessions recorded: {n_sessions}")
        print()

        print("  KNOWLEDGE INVENTORY")
        print(f"  {'─' * 50}")
        print(f"  Warnings/vulnerabilities (P0):  {tier_0:>3}")
        print(f"  Patterns/decisions (P1):        {tier_1:>3}")
        print(f"  Tips (P2):                      {tier_2:>3}")
        print(f"  Bugs/issues (P3):               {tier_3:>3}")
        print(f"  Fixes/resolutions (P4):         {tier_4:>3}")
        print(f"  {'─' * 50}")
        print(f"  Total actionable knowledge:     {total_knowledge:>3}")
        print()

        print("  BUG RESOLUTION")
        print(f"  {'─' * 50}")
        if bug_subjects:
            pct = len(resolved) / len(bug_subjects) * 100
            print(f"  Files with bugs:     {len(bug_subjects)}")
            print(f"  Files with fixes:    {len(resolved)}")
            print(f"  Unresolved:          {len(unresolved)}")
            print(f"  Resolution rate:     {pct:.0f}%")
        else:
            print("  No bugs recorded yet.")
        print()

        print("  FILE COVERAGE")
        print(f"  {'─' * 50}")
        print(f"  Files tracked:       {len(file_entities)}")
        print(f"  Files with knowledge:{len(covered_files):>3}")
        if file_entities:
            pct = len(covered_files) / len(file_entities) * 100
            print(f"  Coverage rate:       {pct:.0f}%")
        print()

        print("  SESSION SUCCESS RATE")
        print(f"  {'─' * 50}")
        if n_sessions:
            success_rate = n_success / n_sessions * 100
            print(f"  Success:  {n_success:>3}  ({success_rate:.0f}%)")
            print(f"  Partial:  {n_partial:>3}")
            print(f"  Failure:  {n_failure:>3}")
        else:
            print("  No sessions recorded yet.")
        print()

        print("  HOOK EFFECTIVENESS")
        print(f"  {'─' * 50}")
        if hook_stats:
            for hook, h in sorted(hook_stats.items()):
                fire_rate = h["fired"] / h["total"] * 100 if h["total"] else 0
                avg_lat = h["latency_sum"] / h["total"] if h["total"] else 0
                items_per = h["items"] / h["fired"] if h["fired"] else 0
                print(f"  {hook}:")
                print(f"    {h['total']} calls, {h['fired']} fired ({fire_rate:.0f}%), "
                      f"{items_per:.1f} items/fire, {avg_lat:.1f}ms avg")
        else:
            print("  No hook metrics recorded yet.")
        print()

        print("  TOKEN EFFICIENCY (knowledge graph vs flat file)")
        print(f"  {'─' * 50}")
        print(f"  Knowledge graph:  {recall_tokens:>5} tokens "
              f"({total_knowledge} facts)")
        if memory_md_tokens:
            print(f"  MEMORY.md:        {memory_md_tokens:>5} tokens")
            if recall_tokens > 0:
                ratio = memory_md_tokens / recall_tokens
                print(f"  Compression:      {ratio:.1f}x "
                      f"(graph is {ratio:.1f}x more dense)")
        else:
            print(f"  MEMORY.md:        not found")
        print()

        # --- Value score ---
        # Composite score: knowledge density × coverage × resolution rate
        knowledge_per_session = total_knowledge / max(n_sessions, 1)
        # Normalize density: 10+ facts/session = 100%
        density_score = min(knowledge_per_session / 10.0, 1.0)
        coverage_rate = len(covered_files) / max(len(file_entities), 1)
        resolution_rate = len(resolved) / max(len(bug_subjects), 1)
        success_rate_f = n_success / max(n_sessions, 1)

        value_score = (
            density_score * 0.3
            + coverage_rate * 0.2
            + resolution_rate * 0.25
            + success_rate_f * 0.25
        ) * 100

        print(f"  VALUE SCORE: {value_score:.0f}/100")
        print(f"  {'─' * 50}")
        print(f"  Knowledge density: {knowledge_per_session:.1f} facts/session (weight 30%)")
        print(f"  File coverage:     {coverage_rate * 100:.0f}% (weight 20%)")
        print(f"  Bug resolution:    {resolution_rate * 100:.0f}% (weight 25%)")
        print(f"  Session success:   {success_rate_f * 100:.0f}% (weight 25%)")
        print()
        print("=" * 60)

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

def uninstall(tools: list[str] | None = None) -> list[str]:
    """Remove Attest MCP server config from coding tools."""
    removed = []
    targets = tools or list(TOOLS.keys())

    for tool_key in targets:
        tool = TOOLS.get(tool_key)
        if not tool:
            continue
        for scope_paths in tool.get("config_paths", {}).values():
            path = scope_paths()
            if not path.exists():
                continue
            data = _read_json(path)
            servers = data.get("mcpServers", {})
            if "brain" in servers:
                del servers["brain"]
                _write_json(path, data)
                if tool_key not in removed:
                    removed.append(tool_key)
                print(f"  Removed from {tool['name']}: {path}")

    # Remove hooks (SessionStart, PreToolUse, PostToolUse, Stop)
    def _is_attest_hook(h: dict) -> bool:
        return any(
            "attest-mcp" in (hk.get("command", "") or "")
            or "mcp_install" in (hk.get("command", "") or "")
            for hk in h.get("hooks", [])
        )

    for scope in ("project", "user"):
        hook_paths = TOOLS.get("claude", {}).get("hooks_path", {})
        hook_fn = hook_paths.get(scope)
        if not hook_fn:
            continue
        hook_path = hook_fn()
        if not hook_path.exists():
            continue
        data = _read_json(hook_path)
        hooks_dict = data.get("hooks", {})
        changed = False
        for event in ("SessionStart", "PreToolUse", "PostToolUse", "Stop"):
            event_hooks = hooks_dict.get(event, [])
            filtered = [h for h in event_hooks if not _is_attest_hook(h)]
            if len(filtered) != len(event_hooks):
                hooks_dict[event] = filtered
                changed = True
        if changed:
            _write_json(hook_path, data)
            print(f"  Removed Claude Code hooks: {hook_path}")

    # Remove instructions file
    for f in (".attest-instructions.md",):
        p = Path(f)
        if p.exists():
            p.unlink()
            print(f"  Removed {p}")

    return removed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI for attest-mcp install/recall/uninstall."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="attest-mcp",
        description="Attest MCP server — persistent memory for coding agents",
    )
    sub = parser.add_subparsers(dest="command")

    # install
    p_install = sub.add_parser("install", help="Configure Attest for your coding tools")
    p_install.add_argument(
        "--tool", choices=list(TOOLS.keys()),
        action="append", dest="tools",
        help="Specific tool to configure (default: auto-detect all)",
    )
    p_install.add_argument(
        "--global", dest="scope", action="store_const", const="user", default="project",
        help="Install for all projects (user scope) instead of current project",
    )
    p_install.add_argument(
        "--db", default=str(DEFAULT_MEMORY_DB),
        help=f"Memory database path (default: {DEFAULT_MEMORY_DB})",
    )

    # recall (used by SessionStart hooks)
    p_recall = sub.add_parser("recall", help="Output prior approaches (for SessionStart hooks)")
    p_recall.add_argument("--task", help="Current task description")
    p_recall.add_argument("--db", default=str(DEFAULT_MEMORY_DB))

    # pre-edit-check (used by PreToolUse hooks)
    sub.add_parser("pre-edit-check", help="Check knowledge graph before file edits (PreToolUse hook)")

    # pre-read-check (used by PreToolUse hooks for Read)
    sub.add_parser("pre-read-check", help="Warn about expensive PDF/image reads (PreToolUse hook)")

    # post-test-check (used by PostToolUse hooks)
    sub.add_parser("post-test-check", help="Surface prior fixes on test failure (PostToolUse hook)")

    # stop (used by Stop hooks)
    sub.add_parser("stop", help="Stop hook handler (lightweight session signal)")

    # metrics
    sub.add_parser("metrics", help="Show hook instrumentation metrics summary")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Quantify knowledge graph impact")
    p_bench.add_argument("--db", default=str(DEFAULT_MEMORY_DB))

    # uninstall
    p_uninstall = sub.add_parser("uninstall", help="Remove Attest from coding tools")
    p_uninstall.add_argument(
        "--tool", choices=list(TOOLS.keys()),
        action="append", dest="tools",
    )

    # If the first real arg looks like an MCP server flag (--transport, --db, --host, --port)
    # or there are no args at all (stdio default), delegate to the MCP server.
    # This preserves backward compat for: attest-mcp, attest-mcp --transport stdio, etc.
    raw_args = sys.argv[1:]
    mcp_flags = {"--transport", "--host", "--port", "--db", "stdio", "sse", "streamable-http"}
    subcommands = {"install", "recall", "uninstall", "pre-edit-check", "pre-read-check", "post-test-check", "stop", "metrics", "benchmark"}
    if not raw_args or (raw_args[0] not in subcommands and raw_args[0] != "-h" and raw_args[0] != "--help"):
        from attestdb.mcp_server import main as mcp_main
        mcp_main()
        return

    args, unknown = parser.parse_known_args()

    if args.command == "install":
        print("Installing Attest MCP server...")
        configured = install(tools=args.tools, scope=args.scope)
        if configured:
            print(f"\nConfigured for: {', '.join(TOOLS[t]['name'] for t in configured)}")
            print(f"Memory DB: {DEFAULT_MEMORY_DB}")
            print("\nThe MCP server will start automatically when you open your coding tool.")
        else:
            print("\nNo coding tools detected. Install with --tool to configure manually:")
            print("  attest-mcp install --tool claude")
            print("  attest-mcp install --tool cursor")

    elif args.command == "recall":
        recall(task=args.task, db_path=args.db)

    elif args.command == "pre-edit-check":
        pre_edit_check()

    elif args.command == "pre-read-check":
        pre_read_check()

    elif args.command == "post-test-check":
        post_test_check()

    elif args.command == "stop":
        stop_session_summary()

    elif args.command == "metrics":
        metrics()

    elif args.command == "benchmark":
        benchmark(db_path=args.db)

    elif args.command == "uninstall":
        removed = uninstall(tools=args.tools)
        if removed:
            print(f"\nRemoved from: {', '.join(TOOLS[t]['name'] for t in removed)}")
        else:
            print("Nothing to remove.")


if __name__ == "__main__":
    main()
