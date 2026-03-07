"""CLI entry point for Attest — python -m attestdb <command>.

Commands:
    ingest <path>   Ingest claims from chat logs (.zip, .json, .txt, .md)
    stats <db>      Show database statistics
    query <db> <e>  Query knowledge around an entity
"""

from __future__ import annotations

import argparse
import sys
import time


def _register_vocabs(db, vocabs):
    """Register vocabularies, skipping if intelligence package not installed."""
    _vocab_map = {
        "bio": ("attestdb.intelligence.bio_vocabulary", "register_bio_vocabulary"),
        "devops": ("attestdb.intelligence.devops_vocabulary", "register_devops_vocabulary"),
        "ml": ("attestdb.intelligence.ml_vocabulary", "register_ml_vocabulary"),
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

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "schema":
        cmd_schema(args)
    elif args.command == "serve":
        cmd_serve(args)


if __name__ == "__main__":
    main()
