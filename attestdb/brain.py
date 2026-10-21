"""Local persistent memory for coding clients, installed with attestdb[mcp]."""

from __future__ import annotations

from attestdb.mcp_install import (
    DEFAULT_MEMORY_DB,
    DEFAULT_MEMORY_DIR,
    TOOLS,
    _read_mcp_config,
    _server_key,
)
from attestdb.mcp_install import (
    install as mcp_install,
)


def install(tool: str | None = None, scope: str = "user") -> None:
    """Install MCP config and usage instructions; no automatic hooks or model calls."""
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

    from attestdb.mcp_install import uninstall as mcp_uninstall

    mcp_uninstall([tool] if tool else None)
    print(f"Memory database preserved at: {DEFAULT_MEMORY_DB}")


def status() -> None:
    """Show brain status — claims, entities, sessions, confidence distribution."""
    print("Attest Brain Status\n")

    if not DEFAULT_MEMORY_DB.exists():
        print("  Not initialized. Run: attest brain install")
        return

    from attestdb.infrastructure.attest_db import AttestDB

    db = AttestDB(str(DEFAULT_MEMORY_DB), embedding_dim=None)
    try:
        print(f"  Database: {DEFAULT_MEMORY_DB}")
        print(f"  Claims: {db.stats().get('total_claims', 0)}")
        print("  Installed in:")
        for tool_info in TOOLS.values():
            for scope, path_fn in tool_info.get("config_paths", {}).items():
                path = path_fn()
                if path.exists() and "brain" in _read_mcp_config(path).get(_server_key(path), {}):
                    print(f"    {tool_info['name']} ({scope}): {path}")
    finally:
        db.close()


def _print_usage() -> None:
    """Print usage instructions after install."""
    print("  Restart your coding client to load the MCP server.")
    print("  Remember: attest_learned(subject, description, type)")
    print("  Recall: attest_get_prior_approaches(problem) or attest_check_file(path)")
    print("  Record outcomes: attest_session_end(outcome, summary)")
    print("  Storage and topic/file recall are local; no model or API key is required.")
    print("  Automatic lifecycle hooks and semantic recall are not included.")


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
