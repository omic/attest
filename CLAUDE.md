# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Attest is a **claim-native database** for building reality models -- structured, provenanced, continuously-learning representations of organizational knowledge. The Rust engine is the sole backend.

The first vertical is computational biology (Omic). Distribution model follows SQLite/DuckDB: `pip install attestdb`, zero infrastructure.

## Architecture (Key Concepts)

**Claims are the atomic primitive.** The graph is derived, not primary. Everything can be rebuilt from the claim log. Provenance is structural -- the engine rejects writes without a valid provenance chain.

**Dual ID system:**
- `claim_id` = SHA-256(subject + predicate + object + source_id + source_type + timestamp) -- globally unique per assertion
- `content_id` = SHA-256(subject + predicate + object) -- corroboration grouping key

**Entity normalization** is locked and must be identical across Python and Rust: NFKD unicode -> lowercase -> collapse whitespace -> Greek letters spelled out (see `docs/02_architecture.md` lines 69-80).

**Critical invariants** (see `docs/07_design_decisions.md`): entity normalization, claim/content ID computation, required provenance, immutable append-only claims, 13 validation rules on every write.

## Codebase Structure

```
attestdb/
  core/           -- Types, normalization (locked), hashing (locked), confidence, vocabulary
  infrastructure/ -- ingestion, query_engine, embedding_index,
                    bulk_loader, attest_db, snapshot, migration
rust/
  Cargo.toml      -- Workspace manifest
  attest-core/    -- Locked invariants in Rust (normalization, hashing, types, errors, confidence, vocabulary)
  attest-store/   -- Storage engine: append-only claim log, entity store, union-find, file persistence
  attest-py/      -- PyO3 bindings via maturin
tests/
  unit/           -- Unit tests
  integration/    -- Integration tests
  cross_lang/     -- Golden test vectors (110 vectors for Python<->Rust verification)
  fixtures/       -- Synthetic claims, test data
examples/         -- quickstart.py, bio_curation.py, devops_quickstart.py
docs/             -- Design docs
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/unit/ tests/integration/           # Fast -- <30s
cd rust && cargo test                           # Rust unit + golden vectors -- ~3s, 77 tests
python scripts/generate_golden_vectors.py       # Regenerate golden vectors from Python
```

## Key Implementation Details

**Ingestion pipeline** separates validation from persistence:
- `_validate_and_build(claim_input)` -> `(Claim, embedding)` -- applies all 13 rules, no side effects except existence checks
- `_persist(claim, embedding)` -- writes to store + embedding index
- `ingest()` calls both + corroboration tracking; `ingest_batch()` calls both without corroboration (logging only)

**Rust engine:**
- 27 PyO3 methods including temporal queries and text search
- File locking via `fs2`, atomic writes (`.tmp` + rename), CRC32 crash recovery
- 1.3M claims/sec insert, 8us entity query, 15us BFS
- Rust store only persists to disk on `close()`. Snapshot does close+reopen to flush.

### Build Steps for attest-py
```bash
cd rust/attest-py && maturin build --release
pip install rust/target/wheels/attest_py-*.whl --force-reinstall
```

## Test Quality

**No filler tests.** Every test must verify real behavior -- parsing real data, claims landing in a real DB, actual error paths. Never mock the thing you're testing. Mock only external boundaries (LLM API calls, HTTP requests to third-party services).

## Intelligence Layer (Enterprise)

LLM-powered features (curator, text extraction, chat ingestion, connectors, insight engine, vocabularies) are available separately in `attestdb-enterprise`. Methods on `AttestDB` that require intelligence (e.g. `configure_curator()`, `ingest_chat()`, `ingest_text()`, `connect()`) will raise `ImportError` with installation instructions if the enterprise package is not installed.
