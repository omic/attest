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


def cmd_discover(args):
    """Run schema discovery on connected data sources."""
    import json as _json

    from attestdb.discovery.sampler import analyze_fields, sample_source
    from attestdb.discovery.analyzer import infer_semantics, detect_deprecated_fields
    from attestdb.discovery.schema_map import SchemaMap
    from attestdb.discovery.aligner import align_schemas
    from attestdb.discovery.claim_templates import generate_claim_templates, templates_to_report

    source_ids = [s.strip() for s in args.sources.split(",")]
    tenant_id = args.tenant or "default"

    print(f"Discovering schemas for: {', '.join(source_ids)}")

    # Build mock connectors from --connector-type if no real connectors
    schema_maps: list[SchemaMap] = []

    for source_id in source_ids:
        print(f"\n--- Sampling {source_id} ---")

        # Try to load as an existing SchemaMap JSON file
        try:
            sm = SchemaMap.load(source_id)
            print(f"  Loaded existing schema from {source_id}")
            schema_maps.append(sm)
            continue
        except Exception:
            pass

        # Try to use the connector framework
        try:
            from attestdb.connectors import connect
            connector = connect(args.connector_type or source_id)
        except Exception:
            print(f"  Error: Cannot connect to '{source_id}'. "
                  "Provide a SchemaMap JSON path or a valid connector name.",
                  file=sys.stderr)
            continue

        samples = sample_source(connector, sample_size=args.sample_size, tenant_id=tenant_id)
        print(f"  Sampled {len(samples)} records")

        profiles = analyze_fields(samples, parent_object=source_id, tenant_id=tenant_id)
        print(f"  Profiled {len(profiles)} fields")

        mappings = infer_semantics(profiles, tenant_id=tenant_id)
        deprecated = detect_deprecated_fields(profiles)
        if deprecated:
            print(f"  Deprecated fields: {', '.join(deprecated)}")

        auto_count = sum(1 for m in mappings if m.review_status == "auto_mapped")
        review_count = sum(1 for m in mappings if m.review_status == "needs_review")
        unmapped_count = sum(1 for m in mappings if m.review_status == "unmapped")
        print(f"  Mappings: {auto_count} auto, {review_count} review, {unmapped_count} unmapped")

        sm = SchemaMap(
            source_id=source_id,
            source_type=args.connector_type or source_id,
            tenant_id=tenant_id,
            field_profiles=profiles,
            semantic_mappings=mappings,
            deprecated_fields=deprecated,
        )
        schema_maps.append(sm)

        # Save individual schema map
        out_path = f"{source_id}_schema.json"
        sm.save(out_path)
        print(f"  Saved to {out_path}")

    if len(schema_maps) < 2:
        print("\nNeed at least 2 sources for cross-source alignment.")
        for sm in schema_maps:
            print(f"\n{sm.to_review_report()}")
        return

    # Cross-source alignment
    print("\n--- Cross-Source Alignment ---")
    unified = align_schemas(schema_maps, tenant_id=tenant_id)
    print(unified.to_report())

    # Generate claim templates
    print("\n--- Claim Templates ---")
    templates = generate_claim_templates(unified, schema_maps, tenant_id=tenant_id)
    print(templates_to_report(templates))

    # Output as JSON if requested
    if args.output:
        package = {
            "schemas": [sm.to_dict() for sm in schema_maps],
            "unified_schema": unified.to_dict(),
            "templates": [
                {
                    "claim_type": t.claim_type,
                    "entity_key_field": t.entity_key_field,
                    "value_field": t.value_field,
                    "source_configs": t.source_configs,
                    "confidence": t.confidence,
                }
                for t in templates
            ],
        }
        with open(args.output, "w") as f:
            _json.dump(package, f, indent=2)
        print(f"\nReview package saved to {args.output}")

    if args.auto_approve:
        approved = sum(1 for t in templates if t.confidence >= 0.90)
        print(f"\n--auto-approve: {approved}/{len(templates)} templates above 0.90 confidence")


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

    # --- quickstart ---
    p_qs = sub.add_parser("quickstart", help="60-second demo with sample data")
    p_qs.add_argument("--db", default=None, help="Database path (default: temp)")

    # --- mcp-config ---
    p_mcp_cfg = sub.add_parser("mcp-config", help="Generate MCP config for Claude Code / Cursor")
    p_mcp_cfg.add_argument("--print-only", action="store_true", help="Print JSON without writing")
    p_mcp_cfg.add_argument("--cursor", action="store_true", help="Write Cursor config instead")

    # --- doctor ---
    sub.add_parser("doctor", help="Run health checks and diagnose issues")

    # --- trial ---
    p_trial = sub.add_parser("trial", help="7-day Pro trial (no credit card)")
    trial_sub = p_trial.add_subparsers(dest="trial_command", help="Trial commands")
    p_trial_start = trial_sub.add_parser("start", help="Start a 7-day Pro trial")
    p_trial_start.add_argument("--email", help="Email address")
    trial_sub.add_parser("status", help="Show trial status")

    # --- upgrade ---
    p_upgrade = sub.add_parser("upgrade", help="Upgrade to a paid plan")
    p_upgrade.add_argument(
        "--tier", choices=["pro", "growth", "team"], default="pro",
        help="Plan tier (default: pro)",
    )
    p_upgrade.add_argument(
        "--interval", choices=["month", "year"], default="month",
        help="Billing interval (default: month)",
    )

    # --- share ---
    p_share = sub.add_parser("share", help="Generate a shareable claim graph link")
    p_share.add_argument("--db", default=None, help="Database path")
    p_share.add_argument("--private", action="store_true", help="Open locally without sharing")

    # --- telemetry ---
    p_telem = sub.add_parser("telemetry", help="Manage anonymous usage telemetry")
    telem_sub = p_telem.add_subparsers(dest="telemetry_action", help="Telemetry commands")
    telem_sub.add_parser("on", help="Enable telemetry")
    telem_sub.add_parser("off", help="Disable telemetry")
    telem_sub.add_parser("status", help="Show telemetry status")

    # --- stats ---
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("db", nargs="?", default=None, help="Database path (default: ~/.attest/memory.attest)")

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

    # --- discover ---
    p_discover = sub.add_parser("discover", help="Auto-discover schemas from data sources")
    p_discover.add_argument(
        "sources",
        help="Comma-separated source IDs or SchemaMap JSON paths",
    )
    p_discover.add_argument(
        "--connector-type", default=None,
        help="Connector type for all sources (e.g., salesforce, postgres)",
    )
    p_discover.add_argument(
        "--sample-size", type=int, default=1000,
        help="Number of records to sample per source (default: 1000)",
    )
    p_discover.add_argument(
        "--tenant", default="default",
        help="Tenant ID (default: default)",
    )
    p_discover.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file for the review package",
    )
    p_discover.add_argument(
        "--auto-approve", action="store_true", default=False,
        help="Auto-approve high-confidence mappings (for CI/CD)",
    )
    p_discover.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Show what would be created without writing",
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

    # --- manager ---
    p_manager = sub.add_parser(
        "manager",
        help="Team activity intelligence (install/status/uninstall)",
    )
    manager_sub = p_manager.add_subparsers(dest="manager_command", help="Manager commands")
    p_mgr_install = manager_sub.add_parser("install", help="Install manager into coding tools")
    p_mgr_install.add_argument(
        "--tool", choices=["claude", "cursor", "windsurf", "codex", "gemini"],
        help="Specific tool (default: auto-detect)",
    )
    manager_sub.add_parser("status", help="Show manager stats")
    p_mgr_uninstall = manager_sub.add_parser("uninstall", help="Remove manager from coding tools")
    p_mgr_uninstall.add_argument("--tool", help="Specific tool (default: all)")

    # --- status ---
    sub.add_parser("status", help="Show cloud connection info and DB stats")

    args = parser.parse_args()
    if args.command is None:
        print("AttestDB — claim-native knowledge graph for AI agents")
        print()
        print("  Quick start:  attestdb quickstart    (60-second demo with sample data)")
        print("  Try cloud:    attestdb trial start    (7-day Pro trial, no credit card)")
        print("  Agent memory: attestdb mcp-config     (give your AI agent persistent memory)")
        print("  Health check: attestdb doctor")
        print("  Full docs:    https://attestdb.com/quickstart.html")
        print()
        parser.print_help()
        sys.exit(0)

    # --- Trial expiry nudge (runs on every CLI invocation) ---
    try:
        from attestdb import config as _cfg
        import datetime as _dt
        _trial = _cfg.get_trial_info()
        if _trial and _trial.get("expires_at"):
            _exp = _dt.datetime.fromisoformat(_trial["expires_at"])
            _remaining = (_exp - _dt.datetime.now(_dt.timezone.utc)).days
            if _remaining <= 0:
                print(f"Your Pro trial has ended. Your local instance and all claims are intact.")
                print(f"  Restore cloud access: attestdb upgrade")
                print()
            elif _remaining <= 2:
                print(f"Your Pro trial ends in {_remaining} day{'s' if _remaining != 1 else ''}.")
                print(f"  Keep cloud running: attestdb upgrade")
                print()
    except Exception:
        pass

    # --- New launch CLI commands ---
    if args.command == "quickstart":
        from attestdb.cli_commands import cmd_quickstart
        cmd_quickstart(args)
        return
    elif args.command == "mcp-config":
        from attestdb.cli_commands import cmd_mcp_config
        cmd_mcp_config(args)
        return
    elif args.command == "doctor":
        from attestdb.cli_commands import cmd_doctor
        cmd_doctor(args)
        return
    elif args.command == "trial":
        from attestdb.cli_commands import cmd_trial_start, cmd_trial_status
        if getattr(args, "trial_command", None) == "start":
            cmd_trial_start(args)
        elif getattr(args, "trial_command", None) == "status":
            cmd_trial_status(args)
        else:
            print("Usage: attest trial {start|status}")
        return
    elif args.command == "upgrade":
        from attestdb.cli_commands import cmd_upgrade
        cmd_upgrade(args)
        return
    elif args.command == "share":
        from attestdb.cli_commands import cmd_share
        cmd_share(args)
        return
    elif args.command == "telemetry":
        from attestdb.cli_commands import cmd_telemetry
        cmd_telemetry(args)
        return

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
    elif args.command == "manager":
        from attestdb.manager import install as mgr_install, uninstall as mgr_uninstall, status as mgr_status
        if args.manager_command == "install":
            mgr_install(tool=getattr(args, "tool", None))
        elif args.manager_command == "uninstall":
            mgr_uninstall(tool=getattr(args, "tool", None))
        elif args.manager_command == "status":
            mgr_status()
        else:
            p_manager.print_help()
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "stats":
        if getattr(args, "db", None):
            cmd_stats(args)
        else:
            from attestdb.cli_commands import cmd_stats_new
            cmd_stats_new(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "discover":
        cmd_discover(args)
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
