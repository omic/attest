"""Universal MCP installer for coding tools.

Configures the Attest MCP server for Claude Code, Cursor, Windsurf,
Codex, and other MCP-compatible coding tools.

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
import tempfile
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
}


def _mcp_server_entry() -> dict:
    """Build the MCP server config entry."""
    attest_mcp = shutil.which("attest-mcp")
    if not attest_mcp:
        # Fall back to module invocation
        attest_mcp = sys.executable
        args = ["-m", "attestdb.mcp_server"]
    else:
        args = []

    db_path = str(DEFAULT_MEMORY_DB)

    entry = {
        "type": "stdio",
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
            print(f"  Unknown tool: {tool_key}", file=sys.stderr)
            continue

        # Auto-detect: skip tools that aren't installed (unless explicitly requested)
        if tools is None and not tool["detect"]():
            continue

        config_paths = tool.get("config_paths", {})
        config_path_fn = config_paths.get(scope) or config_paths.get("user") or config_paths.get("project")
        if not config_path_fn:
            continue

        config_path = config_path_fn()
        existing = _read_json(config_path)
        servers = existing.setdefault("mcpServers", {})
        servers["attest"] = entry
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
    """Add SessionStart, PreToolUse, and PostToolUse hooks."""
    hook_paths = TOOLS["claude"].get("hooks_path", {})
    hook_path_fn = hook_paths.get(scope) or hook_paths.get("user")
    if not hook_path_fn:
        return

    hook_path = hook_path_fn()
    existing = _read_json(hook_path)
    hooks = existing.setdefault("hooks", {})

    # Find attest-mcp binary for hook commands
    attest_mcp = shutil.which("attest-mcp")
    if attest_mcp:
        recall_cmd = f"{attest_mcp} recall"
        pre_edit_cmd = f"{attest_mcp} pre-edit-check"
        post_test_cmd = f"{attest_mcp} post-test-check"
    else:
        base = f"{sys.executable} -m attestdb.mcp_install"
        recall_cmd = f"{base} recall"
        pre_edit_cmd = f"{base} pre-edit-check"
        post_test_cmd = f"{base} post-test-check"

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
    hooks["PreToolUse"] = pre_hooks

    # --- PostToolUse: surface prior fixes when tests fail ---
    post_hooks = hooks.get("PostToolUse", [])
    post_hooks = [h for h in post_hooks if not _is_attest_hook(h)]
    post_hooks.append({
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": post_test_cmd, "timeout": 10}],
    })
    hooks["PostToolUse"] = post_hooks

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

def recall(task: str | None = None, db_path: str | None = None) -> None:
    """Query the memory DB and print context-aware knowledge to stdout.

    Output goes to the AI's context window via SessionStart hook.
    Uses git context (modified files, recent commits) to surface relevant knowledge.
    """
    memory_db = Path(db_path) if db_path else DEFAULT_MEMORY_DB
    if not memory_db.exists():
        return

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        return

    # Copy the DB file (and WAL if present) to a temp location to avoid
    # fs2 lock conflicts. The MCP server may hold the advisory lock on the
    # original file, and claims recorded during the session live in the WAL
    # until close() checkpoints them into the main file.
    tmp_copy = None
    tmp_wal = None
    try:
        tmp_fd, tmp_copy = tempfile.mkstemp(suffix=".attest")
        os.close(tmp_fd)
        shutil.copy2(str(memory_db), tmp_copy)
        # Also copy the WAL — it contains claims not yet checkpointed
        wal_path = str(memory_db) + ".wal"
        if os.path.exists(wal_path):
            tmp_wal = tmp_copy + ".wal"
            shutil.copy2(wal_path, tmp_wal)
        db = AttestDB(tmp_copy, embedding_dim=None)
    except Exception:
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)
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
                # Most recent next_steps
                latest = max(next_claims, key=lambda c: c.timestamp)
                age = _format_age(latest.timestamp)
                lines.append(f"### Continue from previous session ({age})")
                lines.append(f"{latest.object.id}\n")
        except Exception:
            pass

        # 2. Context-relevant knowledge (based on git status)
        # Includes warnings, patterns, decisions, tips, bugs, fixes — prioritized
        relevant_items = _find_relevant_knowledge(db, context_files, context_terms)

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

        lines.append("*Use `attest_learned` to record findings. "
                     "Use `attest_session_end` when done.*")

        print("\n".join(lines))
    finally:
        db.close()
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)


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


def _find_relevant_knowledge(db, files: list[str], terms: set[str]) -> list[tuple]:
    """Find knowledge claims relevant to the current git context.

    Returns (predicate, subject, object, confidence, priority) tuples,
    sorted by priority (warnings/patterns first, then bugs, then fixes).
    """
    if not files and not terms:
        return []

    from attestdb.core.vocabulary import KNOWLEDGE_PREDICATES, KNOWLEDGE_PRIORITY

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
                if claim.claim_id not in seen and claim.predicate.id in all_predicates:
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
# PreToolUse hook — surface warnings before editing a file
# ---------------------------------------------------------------------------

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

    tmp_copy = None
    tmp_wal = None
    try:
        tmp_fd, tmp_copy = tempfile.mkstemp(suffix=".attest")
        os.close(tmp_fd)
        shutil.copy2(str(memory_db), tmp_copy)
        wal_path = str(memory_db) + ".wal"
        if os.path.exists(wal_path):
            tmp_wal = tmp_copy + ".wal"
            shutil.copy2(wal_path, tmp_wal)
        db = AttestDB(tmp_copy, embedding_dim=None)
    except Exception:
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)
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

        if warnings:
            context = (
                f"⚠ Attest knowledge for {basename}:\n"
                + "\n".join(f"  - {w}" for w in warnings[:5])
            )
            result = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": context,
                }
            }
            print(json.dumps(result))

        # Instrumentation
        _log_hook_metric(
            hook="pre_edit_check",
            file=file_path,
            fired=len(warnings) > 0,
            n_items=len(warnings),
            latency_ms=_latency_ms,
        )
    finally:
        db.close()
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)


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

    tmp_copy = None
    tmp_wal = None
    try:
        tmp_fd, tmp_copy = tempfile.mkstemp(suffix=".attest")
        os.close(tmp_fd)
        shutil.copy2(str(memory_db), tmp_copy)
        wal_path = str(memory_db) + ".wal"
        if os.path.exists(wal_path):
            tmp_wal = tmp_copy + ".wal"
            shutil.copy2(wal_path, tmp_wal)
        db = AttestDB(tmp_copy, embedding_dim=None)
    except Exception:
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)
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
        if tmp_copy and os.path.exists(tmp_copy):
            os.unlink(tmp_copy)
        if tmp_wal and os.path.exists(tmp_wal):
            os.unlink(tmp_wal)


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
            if "attest" in servers:
                del servers["attest"]
                _write_json(path, data)
                if tool_key not in removed:
                    removed.append(tool_key)
                print(f"  Removed from {tool['name']}: {path}")

    # Remove hooks (SessionStart, PreToolUse, PostToolUse)
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
        for event in ("SessionStart", "PreToolUse", "PostToolUse"):
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

    # post-test-check (used by PostToolUse hooks)
    sub.add_parser("post-test-check", help="Surface prior fixes on test failure (PostToolUse hook)")

    # metrics
    sub.add_parser("metrics", help="Show hook instrumentation metrics summary")

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
    subcommands = {"install", "recall", "uninstall", "pre-edit-check", "post-test-check", "metrics"}
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

    elif args.command == "post-test-check":
        post_test_check()

    elif args.command == "metrics":
        metrics()

    elif args.command == "uninstall":
        removed = uninstall(tools=args.tools)
        if removed:
            print(f"\nRemoved from: {', '.join(TOOLS[t]['name'] for t in removed)}")
        else:
            print("Nothing to remove.")


if __name__ == "__main__":
    main()
