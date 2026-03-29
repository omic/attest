# Attest

[![PyPI - attestdb](https://img.shields.io/pypi/v/attestdb?label=attestdb)](https://pypi.org/project/attestdb/)
[![PyPI - attest-py](https://img.shields.io/pypi/v/attest-py?label=attest-py)](https://pypi.org/project/attest-py/)
[![Tests](https://github.com/omic/attest/actions/workflows/test.yml/badge.svg)](https://github.com/omic/attest/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/attestdb)](https://pypi.org/project/attestdb/)
[![License: BSL-1.1](https://img.shields.io/badge/license-BSL--1.1-blue)](LICENSE)

> Licensed under the [Business Source License 1.1](LICENSE). Free to use, modify, and self-host. Cannot be offered as a competing managed service. Converts to Apache 2.0 after 4 years.

**A brain for AI agents that actually learns.** Give Claude Code, Cursor, OpenClaw, and any MCP-compatible agent a persistent knowledge system that tracks confidence, records what failed, detects gaps, and gets smarter every session. Free, open source, runs 100% locally.

```bash
pip install attestdb
attest brain install
```

## What It Does

Your agent forgets everything between sessions. Attest Brain fixes that:

- **Session recall** -- prior knowledge (warnings, patterns, dead ends) injected at session start
- **Pre-edit warnings** -- known bugs and patterns surface before you edit a file
- **Post-test fixes** -- prior solutions appear when tests fail
- **Negative results** -- records what was tried and *didn't* work, so nobody repeats the search
- **Confidence scoring** -- every fact has a 0-1 score that updates as evidence arrives
- **Contradiction resolution** -- conflicting claims resolved with principled reasoning
- **Gap detection** -- finds what the brain doesn't know and flags it

Works with **Claude Code**, **OpenClaw**, **Cursor**, **Windsurf**, **Codex**, and **Gemini CLI**.

### Without Brain vs With Brain

| | Without | With Brain |
|---|---|---|
| Same bug, second time | Debug from scratch (40 min) | Brain recalls the fix (3 min) |
| Preferences | Asks the same questions every time | Remembers and applies automatically |
| Dead ends | Re-investigates failed approaches | Skips known failures |
| Knowledge | Resets to zero each session | Compounds over time |

## Quick Start

```python
import attestdb

db = attestdb.open("my_knowledge.db")

# Ingest a claim with provenance
db.ingest(
    subject=("BRCA1", "gene"),
    predicate=("associated_with", "relates_to"),
    object=("Breast Cancer", "disease"),
    provenance={"source_type": "database_import", "source_id": "PMID:12345"},
    confidence=0.9,
)

# Query the knowledge graph
frame = db.query("BRCA1", depth=2)
print(frame.focal_entity.name)
for rel in frame.direct_relationships:
    print(f"  --[{rel.predicate}]--> {rel.target.name} (conf={rel.confidence:.2f})")

db.close()
```

## MCP Tools (84 total)

The brain exposes 84 MCP tools. Key ones for everyday use:

```python
# Record knowledge
attest_learned("redis client", "v7 needs decode_responses=True", "warning")
attest_negative_result("session auth", "msgpack doesn't work for Redis 7 serialization")
attest_session_end("success", "Fixed auth tests after Redis upgrade")

# Recall knowledge
attest_get_prior_approaches("Redis serialization error")
attest_check_file("payment_handler.py")
attest_research_context("session management")

# Analyze knowledge
attest_blindspots()           # Find gaps
attest_confidence_trail("redis client")  # See confidence evolution
attest_predict("entity_a", "entity_b")   # Causal prediction via graph
```

## CLI

```bash
attest brain install            # Install brain into your coding tools
attest brain status             # View brain statistics
attest brain uninstall          # Remove brain from coding tools
attest stats my.db              # Show database statistics
attest query my.db BRCA1        # Query knowledge around an entity
attest serve --port 8892        # Start MCP server (SSE/HTTP)
```

## Under the Hood

Attest isn't a thin wrapper over SQLite or a vector store. It's a purpose-built claim-native database with a Rust storage engine.

- **1.3M claims/sec** ingestion, **8us** entity query (LMDB via heed)
- **Atomic claims** -- every fact is a (subject, predicate, object) triple with provenance
- **Dual IDs** -- `claim_id` (SHA-256, globally unique) + `content_id` (SHA-256 of S+P+O, for corroboration)
- **13 validation rules** on every write
- **Merkle audit chain** -- tamper-evident append-only log
- **Graph algorithms** -- PageRank, betweenness centrality, SVD embeddings, causal prediction

## Core Capabilities

- **Provenance-tracked claims** -- every fact has a source chain
- **Confidence scoring** -- evidence-weighted, updates on corroboration
- **Confidence decay** -- stale knowledge loses confidence over time
- **Retraction with cascade** -- `db.retract("source_123")` propagates downstream
- **Contradiction resolution** -- evidence-weighted voting across sources
- **Snapshot/Restore** -- `db.snapshot(path)` for backups
- **Embedding index** -- HNSW similarity search via usearch
- **Multi-agent research** -- agent registration, task queue, federation
- **Negative results** -- first-class claims that prune the search tree
- **Enterprise RBAC** -- groups, policies, entitlements for 10K+ orgs (enterprise)

## Intelligence Layer (Enterprise)

LLM-powered features (curation, text extraction, connectors, insight engine) are available in `attestdb-enterprise`:

```bash
pip install attestdb-enterprise
```

## Common Errors

**`ProvenanceError`** -- Every claim needs `provenance={"source_type": "...", "source_id": "..."}`. By design: claims without provenance can't be verified.

**`ImportError: requires attestdb-enterprise`** -- Methods like `ingest_chat()` and `connect()` need the enterprise package. Use `db.ingest()` for structured claims without it.

**`database is locked`** -- Only one process can write to a `.attest` file. Kill stale processes or use `:memory:` for testing.

## Running Tests

```bash
pip install attestdb[dev]
pytest tests/unit/ tests/integration/   # ~1100 tests, <75s
cd rust && cargo test                   # Rust unit + golden vectors
```

## Documentation

- [attestdb.com/brain.html](https://attestdb.com/brain.html) -- Brain landing page
- [attestdb.com/quickstart.html](https://attestdb.com/quickstart.html) -- Quick start guide
- `docs/02_architecture.md` -- Technical architecture
- `docs/07_design_decisions.md` -- Design decisions with rationale

## License

[Business Source License 1.1](LICENSE) -- free to use, modify, and self-host. Converts to Apache 2.0 after 4 years.
