"""CLI entry point for Attest — python -m attestdb <command>.

Commands:
    chat              Interactive multi-LLM chat with round-robin consensus
    ingest <path>     Ingest claims from chat logs (.zip, .json, .txt, .md)
    stats <db>        Show database statistics
    query <db> <e>    Query knowledge around an entity
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _register_vocabs(db, vocabs):
    """Register vocabularies, skipping if intelligence package not installed."""
    _vocab_map = {
        "bio": ("attestdb.intelligence.bio_vocabulary", "register_bio_vocabulary"),
        "devops": ("attestdb.intelligence.devops_vocabulary", "register_devops_vocabulary"),
        "ml": ("attestdb.intelligence.ml_vocabulary", "register_ml_vocabulary"),
        "codegen": ("attestdb.intelligence.codegen_vocabulary", "register_codegen_vocabulary"),
    }
    for v in vocabs:
        if v not in _vocab_map:
            continue
        module_path, func_name = _vocab_map[v]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            getattr(mod, func_name)(db)
        except ImportError:
            print(
                f"Warning: '{v}' vocabulary requires "
                "attestdb-intelligence (pip install attestdb-enterprise)"
            )


def cmd_ingest(args):
    """Ingest claims from a chat log file into an Attest database."""
    from attestdb.infrastructure.attest_db import AttestDB

    db_path = args.db
    extraction = args.extraction

    print(f"Opening database: {db_path}")
    db = AttestDB(db_path, embedding_dim=None)

    # Register vocabularies based on domain
    _register_vocabs(db, args.vocab or ["bio"])

    if extraction == "llm":
        curator_model = args.curator or "heuristic"
        db.configure_curator(curator_model)

    use_curator = extraction == "llm"

    for path in args.files:
        print(f"\nIngesting: {path}")
        t0 = time.monotonic()
        try:
            results = db.ingest_chat_file(
                path,
                platform=args.platform,
                use_curator=use_curator,
                extraction=extraction,
            )
            elapsed = time.monotonic() - t0
            for r in results:
                print(f"  {r.summary}")
                if r.turns_skipped:
                    print(f"    ({r.turns_skipped} short turns skipped)")
                if r.warnings:
                    for w in r.warnings:
                        print(f"    Warning: {w}")
            total_ingested = sum(r.claims_ingested for r in results)
            total_extracted = sum(r.claims_extracted for r in results)
            print(f"  {len(results)} conversations, {total_extracted} extracted, "
                  f"{total_ingested} ingested in {elapsed:.1f}s")
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    stats = db.stats()
    print(f"\nDatabase totals: {stats.get('total_claims', 0)} claims, "
          f"{stats.get('total_entities', 0)} entities")
    db.close()


def cmd_stats(args):
    """Show database statistics."""
    from attestdb.infrastructure.attest_db import AttestDB

    db = AttestDB(args.db, embedding_dim=None)
    stats = db.stats()

    print(f"Database: {args.db}")
    print(f"  Claims:     {stats.get('total_claims', 0)}")
    print(f"  Entities:   {stats.get('total_entities', 0)}")
    if stats.get("embedding_index_size"):
        print(f"  Embeddings: {stats['embedding_index_size']}")
    db.close()


def cmd_schema(args):
    """Show knowledge graph schema — entity types, predicates, relationship patterns."""
    from attestdb.infrastructure.attest_db import AttestDB

    db = AttestDB(args.db, embedding_dim=None)
    desc = db.schema()
    print(desc)
    db.close()


def cmd_serve(args):
    """Start the MCP server with SSE or streamable-http transport."""
    import sys as _sys

    # Inject args so mcp_server.main()'s argparse sees them
    _sys.argv = [
        "attest-mcp",
        "--transport", args.transport,
        "--host", args.host,
        "--port", str(args.port),
        "--db", args.db,
    ]
    from attestdb.mcp_server import main as mcp_main
    mcp_main()


# ---- Cloud commands ----


def _cloud_config_path():
    from pathlib import Path
    return Path.home() / ".attest" / "cloud.json"


def _load_cloud_config():
    import json
    from pathlib import Path
    path = _cloud_config_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError):
        return None


def _save_cloud_config(config: dict):
    import json
    path = _cloud_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")


def cmd_cloud_login(args):
    """Store API key + endpoint for cloud access."""
    try:
        import requests
    except ImportError:
        print("Error: requests package required. pip install requests", file=sys.stderr)
        sys.exit(1)

    endpoint = args.endpoint.rstrip("/")
    api_key = args.api_key

    # Validate by hitting /health
    try:
        r = requests.get(f"{endpoint}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        print(f"Connected to {endpoint} — status: {health.get('status', 'ok')}")
    except Exception as e:
        print(f"Error: Cannot reach {endpoint}/health — {e}", file=sys.stderr)
        sys.exit(1)

    # Validate API key by hitting /api/v1/account/keys
    try:
        r = requests.get(
            f"{endpoint}/api/v1/account/keys",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if r.status_code == 401:
            print("Error: Invalid API key.", file=sys.stderr)
            sys.exit(1)
        r.raise_for_status()
        print(f"Authenticated successfully.")
    except requests.exceptions.HTTPError as e:
        if r.status_code != 401:
            print(f"Warning: Could not validate API key — {e}")

    _save_cloud_config({"endpoint": endpoint, "api_key": api_key})
    print(f"Saved to {_cloud_config_path()}")


def cmd_cloud_push(args):
    """Upload a local .attest database to the cloud."""
    from pathlib import Path
    try:
        import requests
    except ImportError:
        print("Error: requests package required. pip install requests", file=sys.stderr)
        sys.exit(1)

    config = _load_cloud_config()
    if not config:
        print("Error: Not logged in. Run: attest login --api-key <key>", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: File not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Uploading {db_path} ({size_mb:.1f} MB) to {config['endpoint']}...")

    with open(db_path, "rb") as f:
        r = requests.post(
            f"{config['endpoint']}/api/v1/account/databases/upload",
            headers={"Authorization": f"Bearer {config['api_key']}"},
            files={"file": (db_path.name, f, "application/octet-stream")},
            timeout=300,
        )

    if r.status_code != 200:
        print(f"Error: Upload failed — {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    result = r.json()
    entities = result.get('entity_count', result.get('total_entities', 0))
    print(f"Uploaded: {result.get('total_claims', 0)} claims, {entities} entities")


def cmd_cloud_pull(args):
    """Download the cloud database to a local file."""
    from pathlib import Path
    try:
        import requests
    except ImportError:
        print("Error: requests package required. pip install requests", file=sys.stderr)
        sys.exit(1)

    config = _load_cloud_config()
    if not config:
        print("Error: Not logged in. Run: attest login --api-key <key>", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    print(f"Downloading from {config['endpoint']} to {db_path}...")

    r = requests.get(
        f"{config['endpoint']}/api/v1/account/databases/download",
        headers={"Authorization": f"Bearer {config['api_key']}"},
        stream=True,
        timeout=300,
    )

    if r.status_code != 200:
        print(f"Error: Download failed — {r.status_code} {r.text}", file=sys.stderr)
        sys.exit(1)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(db_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            f.write(chunk)

    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB to {db_path}")


def cmd_cloud_status(args):
    """Show cloud connection info and DB stats."""
    try:
        import requests
    except ImportError:
        print("Error: requests package required. pip install requests", file=sys.stderr)
        sys.exit(1)

    config = _load_cloud_config()
    if not config:
        print("Not connected. Run: attest login --api-key <key>")
        return

    print(f"Endpoint: {config['endpoint']}")
    print(f"API Key:  {config['api_key'][:12]}...{config['api_key'][-4:]}")

    # Fetch stats
    try:
        r = requests.get(
            f"{config['endpoint']}/api/v1/stats",
            headers={"Authorization": f"Bearer {config['api_key']}"},
            timeout=10,
        )
        if r.status_code == 200:
            stats = r.json()
            print(f"Claims:   {stats.get('total_claims', 0)}")
            print(f"Entities: {stats.get('entity_count', stats.get('total_entities', 0))}")
        else:
            print(f"Stats:    (unavailable — {r.status_code})")
    except Exception as e:
        print(f"Stats:    (unreachable — {e})")


def cmd_chat(args):
    """Launch interactive multi-LLM chat."""
    from attestdb.infrastructure.attest_db import AttestDB

    db = AttestDB(args.db, embedding_dim=None)
    providers = [p.strip() for p in args.providers.split(",")] if args.providers else None

    mode = getattr(args, "mode", "browser")

    if mode == "api":
        from attestdb.core.chat import MultiChat
        chat = MultiChat(
            db=db,
            providers=providers,
            primary=args.primary,
            cwd=os.getcwd(),
        )
        try:
            chat.run()
        finally:
            db.close()
    else:
        # Browser mode (default)
        from attestdb.core.browser_chat import run_browser_chat
        # Parse --models flag: "chatgpt=o3,claude=opus" → dict
        models = None
        if getattr(args, "models", None):
            models = {}
            for spec in args.models.split(","):
                if "=" in spec:
                    prov, model = spec.split("=", 1)
                    models[prov.strip().lower()] = model.strip().lower()
        try:
            run_browser_chat(
                db=db,
                providers=providers,
                cwd=os.getcwd(),
                models=models,
                auto_followup=getattr(args, "auto_followup", False),
            )
        finally:
            db.close()


def cmd_query(args):
    """Query knowledge around an entity."""
    from attestdb.infrastructure.attest_db import AttestDB

    db = AttestDB(args.db, embedding_dim=None)

    # Register vocabularies
    _register_vocabs(db, args.vocab or ["bio"])

    try:
        frame = db.query(args.entity, depth=args.depth)
        print(f"\n{frame.focal_entity.name} ({frame.focal_entity.entity_type})")
        print(f"  {frame.claim_count} claims, confidence: {frame.confidence_range}")
        for rel in frame.direct_relationships:
            print(f"  --[{rel.predicate}]--> {rel.target.name} (conf={rel.confidence:.2f})")
        if frame.narrative:
            print(f"\n  Narrative: {frame.narrative}")
    except Exception as e:
        print(f"Entity not found or query failed: {e}", file=sys.stderr)
        sys.exit(1)

    db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="attest",
        description="Attest — the database for knowledge you can verify",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Interactive multi-LLM chat with consensus")
    p_chat.add_argument(
        "--mode", choices=["browser", "api"], default="browser",
        help="browser (default): Playwright automation of ChatGPT/Claude/Gemini. "
             "api: use provider APIs directly (requires API keys)",
    )
    p_chat.add_argument(
        "--providers", default=None,
        help="Comma-separated providers. "
             "Browser mode: chatgpt,claude,gemini (default: all three). "
             "API mode: openai,gemini,deepseek,etc (default: auto-detect from keys)",
    )
    p_chat.add_argument(
        "--primary", default=None,
        help="Primary provider for summarization (API mode only)",
    )
    p_chat.add_argument(
        "--db", default="attest.db",
        help="Database path (default: attest.db)",
    )
    p_chat.add_argument(
        "--models", default=None,
        help="Model tiers per provider: chatgpt=o3,claude=opus,gemini=pro",
    )
    p_chat.add_argument(
        "--auto-followup", action="store_true", default=False,
        help="Route provider follow-up questions to other providers before asking you",
    )

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Ingest claims from chat logs")
    p_ingest.add_argument("files", nargs="+", help="Chat log files (.zip, .json, .txt, .md)")
    p_ingest.add_argument(
        "--db", default="attest.db",
        help="Database path (default: attest.db)",
    )
    p_ingest.add_argument(
        "--extraction", choices=["llm", "heuristic", "smart"], default="heuristic",
        help=(
            "Extraction mode: heuristic (default, no API key),"
            " llm, or smart (heuristic + LLM for novel)"
        ),
    )
    p_ingest.add_argument(
        "--curator", default=None,
        help="LLM curator model (e.g. groq, openai). Only used with --extraction=llm",
    )
    p_ingest.add_argument(
        "--platform", choices=["auto", "chatgpt", "openai", "generic"], default="auto",
        help="Chat format hint (default: auto-detect)",
    )
    p_ingest.add_argument(
        "--vocab", nargs="+", choices=["bio", "devops", "ml"],
        help="Vocabularies to register (default: bio)",
    )

    # --- stats ---
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("db", help="Database path")

    # --- query ---
    p_query = sub.add_parser("query", help="Query knowledge around an entity")
    p_query.add_argument("db", help="Database path")
    p_query.add_argument("entity", help="Entity to query")
    p_query.add_argument(
        "--depth", type=int, default=2,
        help="Traversal depth (default: 2)",
    )
    p_query.add_argument(
        "--vocab", nargs="+", choices=["bio", "devops", "ml"],
        help="Vocabularies to register (default: bio)",
    )

    # --- schema ---
    p_schema = sub.add_parser("schema", help="Show knowledge graph schema")
    p_schema.add_argument("db", help="Database path")

    # --- install ---
    p_install = sub.add_parser("install", help="Configure Attest MCP for coding tools")
    p_install.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex"],
        action="append", dest="tools",
        help="Specific tool (default: auto-detect all)",
    )
    p_install.add_argument(
        "--global", dest="scope", action="store_const", const="user", default="project",
        help="Install for all projects (user scope)",
    )

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start MCP server (SSE/HTTP)")
    p_serve.add_argument(
        "--transport",
        choices=["sse", "streamable-http"],
        default="sse",
        help="Transport protocol (default: sse)",
    )
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8892, help="Bind port (default: 8892)")
    p_serve.add_argument("--db", default="attest.db", help="Database path (default: attest.db)")

    # --- extension ---
    p_ext = sub.add_parser(
        "extension",
        help="Launch browser with AttestDB capture extension (meta-chat across ChatGPT/Claude/Gemini)",
    )
    p_ext.add_argument(
        "--providers", default=None,
        help="Comma-separated providers to open (default: chatgpt,claude,gemini)",
    )
    p_ext.add_argument(
        "--port", type=int, default=8893,
        help="Ingest server port (default: 8893)",
    )
    p_ext.add_argument(
        "--db", default="attest.db",
        help="Database path (default: attest.db)",
    )
    p_ext.add_argument(
        "--headless", action="store_true",
        help="Server only — use with your own Chrome (prints manual install steps)",
    )

    # --- login ---
    p_login = sub.add_parser("login", help="Connect to AttestDB Cloud")
    p_login.add_argument("--endpoint", default="https://api.attestdb.com", help="API endpoint URL")
    p_login.add_argument("--api-key", required=True, help="API key (attest_xxx...)")

    # --- push ---
    p_push = sub.add_parser("push", help="Upload local DB to cloud")
    p_push.add_argument(
        "--db", default=str(__import__("pathlib").Path.home() / ".attest" / "memory.attest"),
        help="Local DB path (default: ~/.attest/memory.attest)",
    )

    # --- pull ---
    p_pull = sub.add_parser("pull", help="Download cloud DB to local")
    p_pull.add_argument(
        "--db", default=str(__import__("pathlib").Path.home() / ".attest" / "memory.attest"),
        help="Local path to save (default: ~/.attest/memory.attest)",
    )

    # --- brain ---
    p_brain = sub.add_parser(
        "brain",
        help="Personal knowledge brain for AI agents (install/status/uninstall)",
    )
    brain_sub = p_brain.add_subparsers(dest="brain_command", help="Brain commands")
    p_brain_install = brain_sub.add_parser("install", help="Install brain into coding tools")
    p_brain_install.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: auto-detect)",
    )
    p_brain_install.add_argument(
        "--project", dest="scope", action="store_const", const="project", default="user",
        help="Install for current project only (default: user-wide)",
    )
    brain_sub.add_parser("status", help="Show brain stats and health")
    p_brain_uninstall = brain_sub.add_parser("uninstall", help="Remove brain from coding tools")
    p_brain_uninstall.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: all)",
    )

    # --- status ---
    sub.add_parser("status", help="Show cloud connection info and DB stats")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "brain":
        from attestdb.brain import install as brain_install, uninstall as brain_uninstall, status as brain_status
        if args.brain_command == "install":
            brain_install(tool=args.tool, scope=args.scope)
        elif args.brain_command == "uninstall":
            brain_uninstall(tool=getattr(args, "tool", None))
        elif args.brain_command == "status":
            brain_status()
        else:
            p_brain.print_help()
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "schema":
        cmd_schema(args)
    elif args.command == "install":
        from attestdb.mcp_install import install
        install(tools=args.tools, scope=args.scope)
    elif args.command == "extension":
        from attestdb.extension_launcher import run as run_extension
        providers = args.providers.split(",") if args.providers else None
        run_extension(
            providers=providers,
            port=args.port,
            db_path=args.db,
            headless=args.headless,
        )
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "login":
        cmd_cloud_login(args)
    elif args.command == "push":
        cmd_cloud_push(args)
    elif args.command == "pull":
        cmd_cloud_pull(args)
    elif args.command == "status":
        cmd_cloud_status(args)


if __name__ == "__main__":
    main()
