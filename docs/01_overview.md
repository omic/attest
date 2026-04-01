# Attest: Overview

## One-Liner

Attest is a claim-native database for building reality models — structured, provenanced, continuously-learning representations of what an organization knows, how it knows it, and where the gaps are.

## The Problem

Organizations accumulate knowledge everywhere — agents, documents, conversations, databases, experiments, incident reports — but have no infrastructure for turning that raw knowledge into a coherent, trustworthy model of reality. Every agent starts from zero. Every analyst re-discovers what a colleague already learned. Operational lessons live in Slack threads that nobody will ever search. Research insights die in notebooks when people leave.

Existing tools address fragments:

| Category | Examples | What they do | What they lack |
|----------|----------|-------------|----------------|
| Vector DBs | Pinecone, Weaviate | Embedding retrieval | No structure, no provenance, no contradictions |
| Graph DBs | Neo4j, FalkorDB | Relational structure | Schema-first, manual curation, provenance is optional metadata |
| Agent memory | Mem0, Zep/Graphiti, Letta | Temporal KGs from conversations | Optimized for retrieval, not active learning. No derivation chains |
| KG construction | Cognee, GraphRAG, LightRAG | Entity/relationship extraction | Static snapshots, not learning systems |
| RAG pipelines | LangChain, LlamaIndex | Chunk retrieval | Can't distinguish experimental result from hallucination |

None of these build a **reality model** — a living, multi-scale representation of organizational knowledge where topics emerge, connect, and evolve over time. None track WHERE knowledge came from, none actively identify what's MISSING, and none help you see how different domains of knowledge CONNECT.

## Why It Matters

1. **Any organization deploying agents needs this within 2-3 years.** Without it, agents are perpetual amnesiacs and organizational knowledge degrades instead of compounds.
2. **The learning advantage compounds and is nearly impossible to replicate.** An organization running Attest for 2 years has accumulated evidence a competitor can't speed-run.
3. **Cross-domain discovery is the highest-value output.** The connection between TREM2 and lipid metabolism in neurodegeneration, or between database connection pool exhaustion and authentication failures, or between semiconductor export restrictions and AI infrastructure spending — these connections only emerge when knowledge is structured, connected, and actively analyzed across domain boundaries.

## What Attest Is

A general-purpose, embeddable database for organizational reality models. Four properties differentiate it:

**1. Claim-native storage with structural provenance.** The atomic unit is an evidence claim — `(subject, predicate, object, confidence, provenance_chain, embedding, timestamp)` — not a node or edge. Provenance is a storage-level invariant (the engine rejects writes without it), not optional metadata. The graph is a derived view over claims, enabling time-travel, recomputation, and confidence re-weighting.

**2. A typed claim format with pluggable vocabularies.** Entity types, predicate types, and payload schemas are registered and enforced at the storage layer. Extensible without migration — register new types, don't alter a schema. A biology lab registers proteins and binding affinities; a DevOps team registers services and incidents; an investment fund registers companies and market events. Same engine, different vocabularies.

**3. Co-located graph + vector + text + temporal indexes.** A single query blends structural traversal, semantic similarity, keyword matching, and temporal filtering with zero serialization overhead. Purpose-built for the hybrid queries that reality models need.

**4. Emergent knowledge topology.** Topics, domains, and cross-domain connections emerge automatically from the graph structure. The system doesn't just store what you know — it shows you the SHAPE of your knowledge: where it's deep, where it's shallow, where domains connect, and where surprising gaps exist.

**5. Export to any target, audit any database.** Attest's claim log is the canonical record — export materialized views to Neo4j, SQL databases, vector stores, or flat files. Because every fact carries provenance, Attest can audit external databases: which facts are well-supported, which lack evidence, which are contradicted by newer claims.

## How It Works (Three Examples)

### Computational Biology (Omic — first vertical)

A research team runs computational drug discovery agents. Each agent run produces claims: "compound X inhibits protein Y with IC50 = 5nM (computational docking)." The curator triages: store significant findings, flag contradictions, resolve entities to canonical IDs. Over weeks, the graph grows. Topics emerge: "kinase inhibition," "neuroinflammation," "blood-brain barrier penetration." The insight engine notices: the neuroinflammation subgraph and the lipid metabolism subgraph are structurally disconnected but contain semantically similar entities — a potential cross-domain connection worth investigating.

### DevOps Incident Learning

An engineering team's monitoring agents, incident responders, and postmortem processes generate claims: "deployment v2.3 caused latency spike in auth service (observation, monitoring_agent)," "connection pool exhaustion was root cause (human_annotation, postmortem)," "increasing pool size to 200 resolved the issue (observation, monitoring_agent)." Over months, topics emerge: "database performance," "deployment reliability," "authentication failures." The insight engine notices: auth service failures always follow database connection issues, but there's no documented dependency — a gap in the architecture documentation.

### Investment Research

Analysts and research agents produce claims about companies, markets, and regulatory changes. "TSMC announced $40B Arizona fab expansion (document_extraction, press_release)," "US semiconductor export controls restrict EUV sales to China (document_extraction, federal_register)," "NVIDIA H100 supply constrained through Q3 (llm_inference, analyst_agent)." Topics emerge: "semiconductor supply chain," "AI infrastructure spending," "US-China trade." The insight engine notices: the "energy grid capacity" topic has growing connections to "AI infrastructure" but the team has almost no evidence on power availability at major data center regions — a gap that could affect investment theses.

## What Attest Is Not

- Not an agent memory system (Mem0, Zep compete there — different category, narrower scope)
- Not a knowledge graph construction tool (Cognee, GraphRAG extract from text — we build reality models)
- Not a vector database (we have vectors, but claims are the primitive)
- Not a RAG pipeline (we provide ContextFrames and topic maps, not text chunks)

## The Product

**Open source (the engine):** Claim-native storage engine with structural provenance enforcement, typed claim format with vocabulary registration, ContextFrame query interface, entity resolution, Python client library. Runs on a purpose-built Rust engine (1.3M claims/sec).

**Commercial (the intelligence layer):** Domain-specific curators (LLM-powered editorial triage), insight engine (topology-aware bridge prediction, gap identification, confidence alerts), narrative summaries, hosted cloud version.

**First vertical: computational biology (Omic).** Validates the core engine against the most demanding use case: provenance-sensitive, cross-domain, high-volume, with established ground truth databases (Hetionet, DisGeNET, STRING, Reactome) for evaluation. Drug repurposing demo running on full Hetionet.

## Distribution Model

SQLite/DuckDB playbook: `pip install attestdb` or embed the Rust binary. Zero infrastructure. Ubiquity of the engine creates the ecosystem; value capture at the intelligence layer.

## Current Status

Phase 1 and Phase 2 are **complete**. The intelligence layer, Rust engine, and full API surface are built, tested, and validated on full-scale biomedical data. ~790 Python tests passing, 77 Rust tests passing.

### What's Built and Validated

**Core engine (Python + Rust):** Claim-native storage with all 13 validation rules enforced. Entity normalization, claim/content ID computation, and provenance DAG validation are locked — identical in Python and Rust, verified by 110 golden test vectors.

**API surface:** First-class retraction with cascading provenance (`db.retract()`, `db.retract_cascade()`), symmetric predicate traversal, time-travel queries (`db.at(timestamp)`), query profiling (`db.explain()`), inquiry tracking with ContextFrame surfacing (`db.ingest_inquiry()`, `db.check_inquiry_matches()`), curation API (`db.curate()`, `db.configure_curator()`, `db.ingest_text()`), quality reporting (`db.quality_report()`), insight engine methods (`db.find_bridges()`, `db.find_gaps()`, `db.find_confidence_alerts()`), provenance tracing (`db.trace_downstream()`), topic queries (`db.query_topic()`), LLM narrative generation (`db.query(..., llm_narrative=True)`).

**Intelligence layer:** Curator (98% triage accuracy vs. expert labels), insight engine (bridge prediction with hub filtering, vocabulary-driven gap identification, confidence alerts), text extractor (LLM-powered claim extraction from free text), narrative ContextFrame synthesis (template + LLM-generated summaries), topic membership via Leiden community detection, eval harness (edge holdout, curator scoring, Track B export), graph embeddings (SVD + damped random walk), supervised ensemble scorer (4-signal pipeline, 8% precision@50).

**Multi-source knowledge graph:** Bulk loader ingests Hetionet (47K nodes, 2.3M edges), DisGeNET, STRING, and Reactome into a unified claim graph with full provenance. Drug repurposing demo predicts novel ALS candidates — 4/10 top predictions independently validated in PubMed (including a 2025 Dasatinib paper), 6/10 genuinely novel.

**Domain vocabularies:** Biology (Omic), DevOps (service/deployment/incident), and ML experiment tracking (model/dataset/experiment) vocabularies ship with the engine. All importable from top-level `attestdb` module.

**Rust engine (Phase 2 — complete):** Three crates: `attest-core` (locked invariants), `attest-store` (27-method API with file locking, crash recovery, atomic writes), `attest-py` (PyO3 bindings). Benchmarks: 1.3M claims/sec insert, ~12µs entity query, 15µs BFS. NDJSON migration tooling for export/import.

### Quantitative Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Curator triage accuracy | >80% | 98% |
| Edge recovery (Hetionet holdout) | >15% withheld edges | 17.35% |
| Query latency (ContextFrame) | <2s | 12ms |
| Ingestion throughput (Rust) | >100 claims/sec | 1.3M claims/sec |
| Python tests | — | ~790 passing |
| Rust tests | — | 77 passing |

### Shipped Features

| # | Feature | Tests |
|---|---------|-------|
| 1 | `db.retract()` + `db.retract_cascade()` with provenance degradation | 9 |
| 2 | Symmetric predicate traversal | 3 |
| 3 | `db.at(timestamp)` time travel | 4 |
| 4 | Narrative ContextFrames (template + LLM synthesis) | 8 |
| 5 | Topic membership + `db.query_topic()` | 5 |
| 6 | `db.explain()` query profiling | 4 |
| 7 | DevOps vocabulary + quickstart example | 2 |
| 8 | ML vocabulary template | 2 |
| 9 | Inquiry tracking + `db.check_inquiry_matches()` | 8 |
| 10 | Curation API (`db.curate()`, `db.configure_curator()`) | 6 |
| 11 | Text extraction (`db.ingest_text()`) | 6 |
| 12 | Quality reporting (`db.quality_report()`) | 7 |
| 13 | Insight engine API (`db.find_bridges()`, `db.find_gaps()`, `db.find_confidence_alerts()`) | 3 |
| 14 | Provenance tracing (`db.trace_downstream()`) | 2 |
| 15 | Module exports (lazy imports for all public types) | 7 |

### Remaining Work

| Item | Status | Notes |
|------|--------|-------|
| BM25 text index | Designed | Phase 3 |
| Named vector spaces | Designed | Phase 3 |
| Transitive/inverse predicates | Designed | Phase 3 |
| Tier 2 confidence wired into queries | Code exists | `tier2_confidence()` works but queries still use Tier 1 weights |
| Enterprise features | Phase 3 | Multi-tenant, RBAC, managed cloud |

## Build Strategy

**Phase 1 (complete):** Validated the intelligence layer against Omic's biology use case. All quantitative exit criteria met.

**Phase 2 (complete):**
- **2A:** Locked invariants ported to Rust — normalization, hashing, types, confidence, vocabulary. Bit-identical with Python, verified by golden vectors.
- **2B-E:** Rust storage engine with 27-method API, file locking, crash recovery, atomic writes, maintained indexes. PyO3 bindings. Migration tooling.
- **2F:** Full API surface — retraction cascade, curation API, quality reporting, text extraction, insight engine, provenance tracing, topic queries, LLM narrative, inquiry alerting. ~790 tests passing.

**Phase 3 (future):** BM25 text index, named vector spaces, transitive/inverse predicates, query planner, Tier 2 confidence wiring, enterprise features, managed cloud.
