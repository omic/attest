# Attest

> Licensed under the [Business Source License 1.1](LICENSE). Free to use, modify, and self-host. Cannot be offered as a competing managed service. Converts to Apache 2.0 after 4 years.

A **claim-native database** for building reality models -- structured, provenanced, continuously-learning representations of organizational knowledge.

Claims are the atomic primitive. The graph is derived, not primary. Everything can be rebuilt from the claim log. Provenance is structural -- the engine rejects writes without a valid provenance chain.

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

## Rust Backend

For 100x faster ingestion (1M+ claims/sec), install the optional Rust engine:

```bash
pip install attest-py
```

Attest uses the Rust storage engine for all operations.

## Core Capabilities

- **Provenance-tracked claims** -- every fact has a source chain
- **Confidence scoring** -- Tier 1 (direct evidence) + Tier 2 (corroboration)
- **Retraction with cascade** -- `db.retract("source_123")` propagates downstream
- **Time travel** -- `db.at(timestamp)` for point-in-time views
- **Snapshot/Restore** -- `db.snapshot(path)` and `AttestDB.restore(path)`
- **MCP server** -- expose your knowledge graph to AI agents
- **Embedding index** -- HNSW similarity search via usearch
- **Audit chain** -- tamper-evident Merkle hash chain on append-only log

## Intelligence Layer (Enterprise)

LLM-powered features (curation, text extraction, chat ingestion, connectors, insight engine) are available in `attestdb-enterprise`:

```bash
pip install attestdb-enterprise
```

## Optional Dependencies

```bash
pip install attestdb[mcp]   # MCP server for AI agents
pip install attestdb[all]   # everything
```

## CLI

```bash
attest stats my.db              # Show database statistics
attest query my.db BRCA1        # Query knowledge around an entity
attest schema my.db             # Show knowledge graph schema
attest serve --port 8892        # Start MCP server
attest ingest file.json --db my.db  # Ingest claims from file
```

## Documentation

See `docs/` for full architecture and design documentation:
- `docs/02_architecture.md` -- Full technical architecture
- `docs/06_api_spec.md` -- API contract and validation rules
- `docs/07_design_decisions.md` -- Critical decisions with rationale
- `docs/06_api_spec.md` -- API contract and validation rules

## Running Tests

```bash
pip install attestdb[dev]
pytest tests/unit/ tests/integration/   # ~35 tests, <30s
cd rust && cargo test                   # Rust unit + golden vectors
```

## License

[Business Source License 1.1](LICENSE) -- free to use, modify, and self-host. Converts to Apache 2.0 after 4 years.
