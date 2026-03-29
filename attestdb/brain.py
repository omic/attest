"""Attest Brain — personal knowledge system for AI agents that gets smarter over time.

Two commands to install:
    pip install attestdb
    attest brain install

The brain automatically:
- Records findings with provenance and confidence scores
- Warns before editing files with known issues
- Recalls what worked (and what didn't) for similar problems
- Decays stale knowledge and flags contradictions
- Detects knowledge gaps

Works with Claude Code, Cursor, Windsurf, Codex, and Gemini CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from attestdb.mcp_install import (
    DEFAULT_MEMORY_DB,
    DEFAULT_MEMORY_DIR,
    TOOLS,
    _find_attest_mcp,
    _mcp_server_entry,
    _read_json,
    _write_json,
    install as mcp_install,
)


def install(tool: str | None = None, scope: str = "user") -> None:
    """Install Attest Brain into AI coding tools.

    This is a friendlier wrapper around the MCP installer that:
    1. Installs the MCP server config
    2. Configures lifecycle hooks (recall, pre-edit, stop)
    3. Prints usage instructions
    """
    print("Installing Attest Brain...\n")

    # Ensure memory directory exists
    DEFAULT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    # Detect tools
    tools_arg = [tool] if tool else None
    configured = mcp_install(tools=tools_arg, scope=scope)

    if not configured:
        detected = [k for k, v in TOOLS.items() if v["detect"]()]
        if detected:
            print(f"\n  Detected: {', '.join(detected)}")
            print("  But no tools were configured. Try: attest brain install --tool claude")
        else:
            print("\n  No coding tools detected.")
            print("  Install Claude Code, Cursor, or Windsurf first, then re-run.")
            print("  Or install manually: attest brain install --tool claude")
        return

    print(f"\n  Brain installed for: {', '.join(TOOLS[t]['name'] for t in configured)}")
    print(f"  Memory database: {DEFAULT_MEMORY_DB}")
    print()
    _print_usage()


def uninstall(tool: str | None = None) -> None:
    """Remove Attest Brain from AI coding tools."""
    print("Removing Attest Brain...\n")

    targets = [tool] if tool else list(TOOLS.keys())

    for tool_key in targets:
        tool_info = TOOLS.get(tool_key)
        if not tool_info:
            continue

        # Remove MCP config
        for scope in ("project", "user"):
            config_path_fn = tool_info.get("config_paths", {}).get(scope)
            if not config_path_fn:
                continue
            config_path = config_path_fn()
            if not config_path.exists():
                continue
            existing = _read_json(config_path)
            servers = existing.get("mcpServers", {})
            if "attest" in servers:
                del servers["attest"]
                _write_json(config_path, existing)
                print(f"  Removed MCP config: {config_path}")

        # Remove hooks (Claude Code only)
        if tool_key == "claude":
            for scope in ("project", "user"):
                hook_paths = tool_info.get("hooks_path", {})
                hook_path_fn = hook_paths.get(scope)
                if not hook_path_fn:
                    continue
                hook_path = hook_path_fn()
                if not hook_path.exists():
                    continue
                existing = _read_json(hook_path)
                hooks = existing.get("hooks", {})
                changed = False
                for event in ("SessionStart", "PreToolUse", "PostToolUse", "Stop"):
                    if event in hooks:
                        before = len(hooks[event])
                        hooks[event] = [
                            h for h in hooks[event]
                            if not any(
                                "attest-mcp" in (hk.get("command", "") or "")
                                or "mcp_install" in (hk.get("command", "") or "")
                                for hk in h.get("hooks", [])
                            )
                        ]
                        if len(hooks[event]) < before:
                            changed = True
                        if not hooks[event]:
                            del hooks[event]
                if changed:
                    _write_json(hook_path, existing)
                    print(f"  Removed hooks: {hook_path}")

    # Remove agent instructions
    instructions_path = Path(".attest-instructions.md")
    if instructions_path.exists():
        instructions_path.unlink()
        print(f"  Removed: {instructions_path}")

    print("\n  Brain uninstalled. Memory database preserved at:")
    print(f"  {DEFAULT_MEMORY_DB}")
    print("  (Delete manually if you want to erase all knowledge.)")


def status() -> None:
    """Show brain status — claims, entities, sessions, confidence distribution."""
    print("Attest Brain Status\n")

    if not DEFAULT_MEMORY_DB.exists():
        print("  Not initialized. Run: attest brain install")
        return

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        print("  Error: attestdb not importable")
        return

    try:
        db = AttestDB(str(DEFAULT_MEMORY_DB), embedding_dim=None)
    except Exception as e:
        print(f"  Error opening brain: {e}")
        return

    try:
        stats = db.stats()
        total = stats.get("total_claims", 0)
        entities = stats.get("entity_count", stats.get("total_entities", 0))

        print(f"  Database:   {DEFAULT_MEMORY_DB}")
        if DEFAULT_MEMORY_DB.exists():
            size_mb = DEFAULT_MEMORY_DB.stat().st_size / (1024 * 1024)
            print(f"  Size:       {size_mb:.1f} MB")
        print(f"  Claims:     {total}")
        print(f"  Entities:   {entities}")

        if total == 0:
            print("\n  Brain is empty. Start using your coding agent — it will")
            print("  learn automatically via hooks and MCP tools.")
            return

        # Count knowledge by type
        knowledge_counts = {}
        for pred in ["has_warning", "had_bug", "has_fix", "has_pattern",
                      "has_decision", "has_tip", "no_evidence_for"]:
            try:
                claims = db.claims_for_predicate(pred)
                if claims:
                    knowledge_counts[pred] = len(claims)
            except Exception:
                continue

        if knowledge_counts:
            print("\n  Knowledge breakdown:")
            label_map = {
                "has_warning": "Warnings",
                "had_bug": "Bugs found",
                "has_fix": "Fixes recorded",
                "has_pattern": "Patterns",
                "has_decision": "Decisions",
                "has_tip": "Tips",
                "no_evidence_for": "Negative results",
            }
            for pred, count in sorted(knowledge_counts.items(),
                                       key=lambda x: x[1], reverse=True):
                label = label_map.get(pred, pred)
                print(f"    {label}: {count}")

        # Session count
        try:
            outcomes = db.claims_for_predicate("had_outcome")
            if outcomes:
                successes = sum(1 for c in outcomes if "success" in c.object.id)
                failures = sum(1 for c in outcomes if "failure" in c.object.id)
                print(f"\n  Sessions:   {len(outcomes)} total "
                      f"({successes} success, {failures} failure)")
        except Exception:
            pass

        # Installed tools
        print("\n  Installed in:")
        found_any = False
        for key, tool_info in TOOLS.items():
            for scope in ("user", "project"):
                config_path_fn = tool_info.get("config_paths", {}).get(scope)
                if not config_path_fn:
                    continue
                config_path = config_path_fn()
                if config_path.exists():
                    existing = _read_json(config_path)
                    if "attest" in existing.get("mcpServers", {}):
                        print(f"    {tool_info['name']} ({scope}): {config_path}")
                        found_any = True
        if not found_any:
            print("    (none — run: attest brain install)")

    finally:
        db.close()


def _print_usage() -> None:
    """Print usage instructions after install."""
    print("  How it works:")
    print("  1. Start coding — the brain recalls relevant knowledge automatically")
    print("  2. Before edits, it warns about known bugs and patterns")
    print("  3. Record findings with attest_learned(subject, description, type)")
    print("  4. Record failures with attest_negative_result(topic, finding)")
    print("  5. End sessions with attest_session_end(outcome, summary)")
    print()
    print("  The brain gets smarter because:")
    print("  - Findings have confidence scores that update based on evidence")
    print("  - Stale knowledge decays over time")
    print("  - Multiple sources confirming the same thing = higher confidence")
    print("  - Negative results prevent repeating past mistakes")
    print("  - Contradictions get flagged automatically")
    print()
    print("  Learn more: https://attestdb.com/brain.html")


def _cli_main() -> None:
    """Entry point for `attest-brain` standalone command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="attest-brain",
        description="Attest Brain — personal knowledge system for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    p_install = sub.add_parser("install", help="Install brain into coding tools")
    p_install.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: auto-detect)",
    )
    p_install.add_argument(
        "--project", dest="scope", action="store_const", const="project", default="user",
        help="Install for current project only (default: user-wide)",
    )

    sub.add_parser("status", help="Show brain stats and health")

    p_uninstall = sub.add_parser("uninstall", help="Remove brain from coding tools")
    p_uninstall.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: all)",
    )

    args = parser.parse_args()

    if args.command == "install":
        install(tool=args.tool, scope=args.scope)
    elif args.command == "status":
        status()
    elif args.command == "uninstall":
        uninstall(tool=getattr(args, "tool", None))
    else:
        parser.print_help()
