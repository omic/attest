"""Attest Manager — team activity intelligence for engineering managers.

Two commands to install:
    pip install -U attestdb
    attest manager install

Monitors Slack, GitHub, Jira/Linear activity, detects anomalies
(blockers, silence, velocity drops, stale PRs), recognizes wins
(shipping streaks), and drafts supportive check-in messages.
Nothing is sent without the manager's approval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from attestdb.mcp_install import (
    DEFAULT_MEMORY_DB,
    DEFAULT_MEMORY_DIR,
    TOOLS,
    _read_json,
    _write_json,
    install as mcp_install,
)


def install(tool: str | None = None, scope: str = "user") -> None:
    """Install Attest Manager MCP tools into IDE."""
    print("Installing Attest Manager...\n")

    DEFAULT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    tools_arg = [tool] if tool else None
    configured = mcp_install(tools=tools_arg, scope=scope)

    if not configured:
        print("\n  No coding tools detected.")
        print("  Install Claude Code, Cursor, or Windsurf first, then re-run.")
        return

    print(f"\n  Manager installed for: {', '.join(TOOLS[t]['name'] for t in configured)}")
    print(f"  Memory database: {DEFAULT_MEMORY_DB}")
    print()
    print("  Setup your team:")
    print("    1. Call team_setup(team_name, members_json) to add your team")
    print("    2. Call team_configure(team_name, slack_token=...) for Slack DMs")
    print("    3. Call team_monitor_enable(team_name) to start monitoring")
    print()
    print("  The monitor will:")
    print("    - Track commits, PRs, tickets, and Slack activity")
    print("    - Detect blockers, silence, velocity drops, and stale PRs")
    print("    - Recognize shipping streaks and high velocity")
    print("    - Draft supportive check-in messages for your review")
    print("    - Never send anything without your approval")


def uninstall(tool: str | None = None) -> None:
    """Remove Attest Manager from coding tools."""
    from attestdb.brain import uninstall as brain_uninstall
    brain_uninstall(tool=tool)


def status() -> None:
    """Show manager status."""
    print("Attest Manager Status\n")

    if not DEFAULT_MEMORY_DB.exists():
        print("  Not initialized. Run: attest manager install")
        return

    try:
        from attestdb.infrastructure.attest_db import AttestDB
    except ImportError:
        print("  Error: attestdb not importable")
        return

    try:
        db = AttestDB(str(DEFAULT_MEMORY_DB), embedding_dim=None)
    except Exception as e:
        print(f"  Error: {e}")
        return

    try:
        stats = db.stats()
        print(f"  Database: {DEFAULT_MEMORY_DB}")
        print(f"  Claims:   {stats.get('total_claims', 0)}")
        print(f"  Entities: {stats.get('entity_count', stats.get('total_entities', 0))}")

        # Count team members
        try:
            team_claims = db.claims_for_predicate("has_member")
            if team_claims:
                members = set(c.object.id for c in team_claims)
                teams = set(c.subject.id for c in team_claims)
                print(f"\n  Teams:    {len(teams)}")
                print(f"  Members:  {len(members)}")
        except Exception:
            pass

        # Count anomalies and drafts
        try:
            anomaly_claims = db.claims_for_predicate("has_anomaly")
            if anomaly_claims:
                print(f"  Anomalies detected: {len(anomaly_claims)}")
        except Exception:
            pass

        # Pending drafts
        try:
            entities = db.search_entities("draft:")
            pending = 0
            for e in entities:
                if not e.id.startswith("draft:"):
                    continue
                claims = db.claims_for(e.id)
                for c in claims:
                    payload = c.payload or {}
                    data = payload.data if hasattr(payload, "data") else payload.get("data", {})
                    if isinstance(data, dict) and data.get("status") == "pending_review":
                        pending += 1
            if pending > 0:
                print(f"  Pending drafts: {pending}")
        except Exception:
            pass

    finally:
        db.close()


def _cli_main() -> None:
    """Entry point for `attest-manager` standalone command."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="attest-manager",
        description="Attest Manager — team activity intelligence",
    )
    sub = parser.add_subparsers(dest="command")

    p_install = sub.add_parser("install", help="Install manager into coding tools")
    p_install.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: auto-detect)",
    )

    sub.add_parser("status", help="Show manager stats")

    p_uninstall = sub.add_parser("uninstall", help="Remove manager from coding tools")
    p_uninstall.add_argument("--tool", help="Specific tool (default: all)")

    args = parser.parse_args()

    if args.command == "install":
        install(tool=args.tool)
    elif args.command == "status":
        status()
    elif args.command == "uninstall":
        uninstall(tool=getattr(args, "tool", None))
    else:
        parser.print_help()
