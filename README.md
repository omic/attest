# Attest

[![PyPI - attestdb](https://img.shields.io/pypi/v/attestdb?label=attestdb)](https://pypi.org/project/attestdb/)
[![PyPI - attest-py](https://img.shields.io/pypi/v/attest-py?label=attest-py)](https://pypi.org/project/attest-py/)
[![Tests](https://github.com/omic/attest/actions/workflows/test.yml/badge.svg)](https://github.com/omic/attest/actions/workflows/test.yml)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/attestdb)](https://pypi.org/project/attestdb/)
[![License: BSL-1.1](https://img.shields.io/badge/license-BSL--1.1-blue)](LICENSE)

> Licensed under the [Business Source License 1.1](LICENSE). Free to use, modify, and self-host. Cannot be offered as a competing managed service. Converts to Apache 2.0 after 4 years.

**The database for agents.** Persistent memory that compounds across sessions -- every fact sourced, every claim confidence-weighted, every failure a data point that prunes the search tree for everyone. `pip install`, single file, no server.

## Install

```bash
pip install attestdb
```

## Quick Start

```python
import attestdb

db = attestdb.open("my_knowledge.db")

# Ingest a claim with provenance
db.ingest(
    subject=("BRCA1", "gene"),
    predicate=("associated_with", "relates_to"),
    object=("Breast Cancer", "disease"),
    provenance={"source_type": "literature_extraction", "source_id": "PMID:12345"},
    confidence=0.9,
)

# Query the knowledge graph
frame = db.query("BRCA1", depth=2)
print(frame.focal_entity.name)
for rel in frame.direct_relationships:
    print(f"  --[{rel.predicate}]--> {rel.target.name} (conf={rel.confidence:.2f})")

db.close()
```

## AI Agent Memory

Attest gives AI coding agents persistent, cross-session memory. Install the MCP server and your agent learns from every session -- warnings, patterns, dead ends, and fixes are recalled automatically.

```bash
pip install attestdb[mcp]
attest-mcp install          # Auto-detect Claude Code, Cursor, Windsurf, Codex
```

What it does:

- **Recall hook** -- injects prior knowledge at session start (warnings, patterns, next steps from last session)
- **Pre-edit check** -- surfaces known bugs/warnings before you edit a file
- **Post-test check** -- surfaces prior fixes when tests fail
- **Negative results** -- records what you investigated and *didn't* find, so future sessions don't repeat the search
- **Research context** -- shows what's been tried, what failed, and active strategies before starting work

```python
# Record a finding
attest_learned("ClaimInput API", "must use provenance= dict, not source_id= kwarg", "warning")

# Record a dead end
attest_negative_result("RustStore", "streaming query API", "read all 27 PyO3 methods")

# Get research context before starting work
attest_research_context("bincode serialization")
# Returns: dead ends, prior findings, active strategies, prior investigators
```

36 MCP tools total. Works with Claude Code, Cursor, Windsurf, and Codex.

## Multi-Agent Collaborative Research

Multiple agents can collaborate on knowledge expansion -- submitting research, claiming tasks from a gap-driven queue, and sharing negative results so failed investigations prune the search tree for everyone.

```python
from attestdb.infrastructure.agents import (
    register_agent, submit_research, submit_negative_result,
    generate_tasks, claim_task, complete_task, get_task_context,
)

# Register agents
register_agent(db, "agent-alpha", capabilities=["literature"], model="gpt-4o")

# Submit research with gaps discovered
submit_research(db, "agent-alpha", claims=[...],
    topic="tp53", gaps_discovered=["mutation spectrum in rare cancers"])

# Record what didn't work (prunes the search tree for other agents)
submit_negative_result(db, "agent-alpha",
    subject=("tp53", "gene"),
    hypothesis=("direct kinase activity", "mechanism"),
    search_strategy="PubMed + STRING + Reactome")

# Task queue auto-generates from knowledge gaps
tasks = generate_tasks(db)  # Entities with 3+ negative results get deprioritized

# Get context before starting (dead ends, strategies, prior findings)
context = get_task_context(db, "tp53")

# Federation: sync claims between instances via NDJSON
export_claims_since(db, since_ns=0, stream=output_file)
import_claims_from_stream(db2, stream=input_file)
```

Content-addressed claims (`content_id` = SHA-256 of subject+predicate+object) enable conflict-free dedup across federated instances. Same fact from different agents = automatic corroboration.

## Rust Backend

For 100x faster ingestion (1M+ claims/sec), install the Rust storage engine:

```bash
pip install attest-py
```

Attest uses the Rust backend automatically when available.

## Core Capabilities

- **Provenance-tracked claims** -- every fact has a source chain
- **Confidence scoring** -- Tier 1 (direct evidence) + Tier 2 (corroboration)
- **Retraction with cascade** -- `db.retract("source_123")` propagates downstream
- **Time travel** -- `db.at(timestamp)` for point-in-time views
- **Snapshot/Restore** -- `db.snapshot(path)` and `AttestDB.restore(path)`
- **MCP server** -- 36 tools for AI agent integration
- **Embedding index** -- HNSW similarity search via usearch
- **Audit chain** -- tamper-evident Merkle hash chain on append-only log
- **Multi-agent research** -- agent registration, task queue, federation
- **Negative results** -- first-class claims that prune the search tree

## Intelligence Layer (Enterprise)

LLM-powered features (curation, text extraction, chat ingestion, connectors, insight engine) are available in `attestdb-enterprise`:

```bash
pip install attestdb-enterprise
```

## CLI

```bash
attest stats my.db              # Show database statistics
attest query my.db BRCA1        # Query knowledge around an entity
attest schema my.db             # Show knowledge graph schema
attest serve --port 8892        # Start MCP server
attest ingest file.json --db my.db  # Ingest claims from file
attest-mcp install              # Install MCP for your coding tool
attest-mcp metrics              # View hook performance metrics
```

## Common Errors

**`ProvenanceError`** -- Every claim needs `provenance={"source_type": "...", "source_id": "..."}`. This is by design: claims without provenance can't be verified.

**`ImportError: requires attestdb-enterprise`** -- Methods like `ingest_chat()`, `ingest_text()`, and `connect()` need the enterprise package. Use `db.ingest()` for structured claims without it.

**`ValueError: Unknown vocabulary`** -- Valid vocab names are `bio`, `devops`, `ml`, `ai_tools`, or `all`. Not `biology`.

**`database is locked`** -- Only one process can write to a `.attest` file. Kill stale processes or use `:memory:` for testing.

See `docs/10_getting_started.md` for a full troubleshooting guide.

## Documentation

See `docs/` for full architecture and design documentation:
- `docs/02_architecture.md` -- Full technical architecture
- `docs/06_api_spec.md` -- API contract and validation rules
- `docs/07_design_decisions.md` -- Critical decisions with rationale
- `docs/10_getting_started.md` -- Getting started + troubleshooting

## Running Tests

```bash
pip install attestdb[dev]
pytest tests/unit/ tests/integration/   # ~35 tests, <30s
cd rust && cargo test                   # Rust unit + golden vectors
```

## License

[Business Source License 1.1](LICENSE) -- free to use, modify, and self-host. Converts to Apache 2.0 after 4 years.
