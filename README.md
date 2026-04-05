# AttestDB

The truth layer for AI agents. Every fact has a source, a confidence score, and gets flagged when the source is wrong.

[![PyPI](https://img.shields.io/pypi/v/attestdb)](https://pypi.org/project/attestdb/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: BSL-1.1](https://img.shields.io/badge/license-BSL--1.1-green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/omic/attest)](https://github.com/omic/attest)

```bash
pip install attestdb
```

---

## The Problem

Every AI system has a trust problem. Agents hallucinate. Data sources contradict each other. Nobody can trace where a "fact" came from or when it went stale.

Vector databases find similar text. Graph databases store edges. Neither tracks whether what they're storing is *still true*.

AttestDB stores **claims** — facts with sources, confidence scores, and timestamps. When a source is wrong, one call retracts it and everything downstream is flagged automatically. When two sources disagree, both claims coexist until evidence resolves them. When data goes stale, the system knows.

## See It Work

```python
from attestdb import AttestDB

db = AttestDB("demo.attest", embedding_dim=None)

# Ingest claims from different sources
db.ingest(
    subject=("api-gateway", "service"),
    predicate=("depends_on", "relates_to"),
    object=("redis", "service"),
    provenance={"source_id": "k8s-manifest-v2.3", "source_type": "infrastructure"},
    confidence=1.0,
)

db.ingest(
    subject=("api-gateway", "service"),
    predicate=("handles", "relates_to"),
    object=("10k-requests-per-second", "metric"),
    provenance={"source_id": "load-test-march-2024", "source_type": "benchmark"},
    confidence=0.95,
)

# Query the knowledge graph
frame = db.query("api-gateway", depth=2)
for rel in frame.direct_relationships:
    print(f"  {rel.predicate} → {rel.target.name}  (conf: {rel.confidence})")

# The load test had a bug. Retract everything from that source.
result = db.retract("load-test-march-2024", reason="Bug in test harness")
print(f"Retracted {result.retracted_count} claims")
# Claims from k8s-manifest-v2.3 are untouched. Only the bad source is gone.

# Time travel — see what the graph looked like before the retraction
import time
snapshot = db.at(int(time.time() * 1e9) - int(60 * 1e9))  # 1 minute ago
old_frame = snapshot.query("api-gateway", depth=1)
```

## Why Not a Vector DB?

|  | AttestDB | Pinecone / Weaviate | Neo4j | PostgreSQL |
|---|---|---|---|---|
| Atomic unit | **Sourced claim** | Vector embedding | Edge | Row |
| Provenance | **Required on every write** | Optional metadata | Optional property | Not built-in |
| Retraction cascade | **Automatic** | — | Manual | Manual |
| Contradiction handling | **Evidence-weighted** | Last write wins | Last write wins | Last write wins |
| Confidence scoring | **Built-in (0–1)** | Similarity score | — | — |
| Query latency | **~12µs** | ~10ms | ~5ms | ~1ms |
| MCP tools | **106** | — | — | — |

## Features

### Retraction Cascades

One bad source? One call fixes everything downstream.

```python
result = db.retract("buggy-sensor-feed", reason="Calibration error discovered")
# Every claim from that source is retracted.
# Claims corroborated by independent sources survive.
```

### Corroboration

The same fact from three independent sources is stronger than one source at high confidence.

```python
# Same fact, different sources — confidence compounds
db.ingest(
    subject=("acme-corp", "company"), predicate=("headquartered_in", "relates_to"),
    object=("san-francisco", "city"),
    provenance={"source_id": "crunchbase", "source_type": "database"}, confidence=0.9,
)
db.ingest(
    subject=("acme-corp", "company"), predicate=("headquartered_in", "relates_to"),
    object=("san-francisco", "city"),
    provenance={"source_id": "linkedin", "source_type": "database"}, confidence=0.85,
)

report = db.corroboration_report(min_sources=2)
```

### Time Travel

Query any point in the past. No backup restores.

```python
# What did we know last Tuesday?
snapshot = db.at(last_tuesday_ns)
frame = snapshot.query("customer-123", depth=2)
```

### Graph Traversal

BFS path finding, profiling, and explanation.

```python
# Find connections between entities
paths = db.find_paths("drug-a", "disease-b", max_depth=3, top_k=5)
for path in paths:
    print(f"Confidence: {path.total_confidence}")

# Profile a query
frame, profile = db.explain("BRCA1", depth=2)
print(f"Query: {profile.elapsed_ms:.1f}ms")
```

### Batch Ingestion

Load millions of claims from any source.

```python
from attestdb.core.types import ClaimInput

claims = [
    ClaimInput(
        subject=("entity-a", "type"), predicate=("rel", "type"), object=("entity-b", "type"),
        provenance={"source_id": "dataset-v3", "source_type": "bulk"}, confidence=0.88,
    )
    for _ in range(100_000)
]
result = db.ingest_batch(claims)
print(f"Ingested: {result.ingested}, Duplicates: {result.duplicates}")
```

### Tamper-Evident Audit

Merkle hash chain on every write. Tamper-evident by construction.

```python
schema = db.schema()
print(f"{schema.total_claims} claims, {schema.total_entities} entities")
print(f"Entity types: {schema.entity_types}")
```

## Give Your AI Agent a Brain

Three commands. Your agent remembers bugs, patterns, and dead ends across sessions.

```bash
pip install attestdb
attestdb mcp-config
# Restart Claude Code — your agent now has persistent memory
```

Works with Claude Code, Cursor, Windsurf, Codex, and any MCP-compatible agent. 106 tools available out of the box — search, ingest, retract, verify, predict, and more.

## Under the Hood

```
Ingestion:     1.3M claims/sec (Rust engine, single-threaded)
Query latency: ~12µs indexed lookups (in-memory), ~122µs LMDB
Storage:       Single file — no server, no config, like SQLite
Connectors:    30 (Slack, GitHub, Gmail, Jira, Salesforce, Postgres, etc.)
MCP tools:     106
Production:    85M+ claims, 13M+ entities on reference database
```

**Rust engine.** The storage layer is written in Rust (LMDB via heed), exposed to Python through PyO3 bindings. Atomic writes, file locking, CRC32 crash recovery. The Rust crate (`attest-py`) ships as pre-built wheels for Linux, macOS, and Windows.

**Dual ID system.** Every claim gets two hashes: `claim_id` (unique per assertion — includes source and timestamp) and `content_id` (groups the same fact across sources — enables corroboration).

**13 validation rules** on every write. Provenance is structural — the engine rejects writes without a valid source chain.

## Architecture

```
attestdb/
  core/            — Types, normalization (locked), hashing (locked), confidence, vocabulary
  infrastructure/  — AttestDB, ingestion, query engine, embedding index, migration
rust/
  attest-core/     — Locked invariants in Rust (normalization, hashing, types)
  attest-store/    — LMDB storage engine (append-only claim log, entity store, indexes)
  attest-py/       — PyO3 bindings via maturin
tests/
  unit/            — Unit tests
  integration/     — Integration tests
  cross_lang/      — Golden vectors (Python ↔ Rust verification)
```

Entity normalization is locked and identical across Python and Rust: NFKD unicode → lowercase → collapse whitespace → Greek letters spelled out. 118 golden test vectors verify cross-language consistency.

## Pricing

|  | Open Source | Cloud ($49/mo) | Team ($249/mo) | Enterprise (from $2,500/mo) |
|---|---|---|---|---|
| Engine | Full Rust engine | Full Rust engine | Full Rust engine | Full Rust engine |
| Claims | Unlimited | 500K | 10M | Unlimited |
| Queries | Unlimited | 500K | 10M | Unlimited |
| Storage | Unlimited | 5 GB | 100 GB | Unlimited |
| Connectors | All 30 | All 30 | All 30 | All 30 |
| MCP tools | 106 | 106 | 106 | 106 |
| Living Database | — | — | Freshness, composites, auto-discovery | Everything in Team |
| RBAC / SSO | — | — | — | SSO/SAML, claim-level ACL |
| Support | Community | Email | Slack (priority) | Dedicated engineer |

The open-source install is the full product — not a demo, not time-limited. Same engine that runs in production.

[Full pricing details →](https://attestdb.com/pricing.html)

## Links

- [Documentation](https://attestdb.com/docs.html) — API reference, connectors, MCP tools
- [Quick Start](https://attestdb.com/quickstart.html) — Up and running in 60 seconds
- [Live Demo](https://attestdb.com/demo/) — 85M+ claims, query anything
- [Enterprise](https://attestdb.com/enterprise.html) — Auto-discovery, entity resolution, ACL
- [Manifesto](https://attestdb.com/manifesto.html) — Why we built this

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

The intelligence layer (LLM-powered features, connectors) is available separately as `attestdb-enterprise`. The open-source repo contains the full engine, core types, ingestion pipeline, query engine, and all Rust code.

## License

BSL-1.1 — free for non-production use. Converts to Apache 2.0 after 4 years. See [LICENSE](LICENSE) for details.
