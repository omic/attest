# Attest: Architecture

> **Reading guide:** This document describes both what's built and what's designed but not yet implemented. Sections are marked: **[BUILT]** = working and tested, **[PARTIAL]** = scaffolded or partially implemented, **[DESIGNED]** = specified but not yet built. The design sections are retained because they constrain Phase 2+ implementation.

## Design Principles

1. **Claims are the atomic primitive.** The graph is derived, not primary. Everything can be rebuilt from the claim log.
2. **Provenance is structural, not metadata.** The engine rejects writes without a valid provenance chain. Contamination is impossible by construction.
3. **Types are registered, not hardcoded.** The claim schema is rigid. The vocabulary is extensible. No migration required to add domain concepts.
4. **Append-only, never mutate.** Claims are immutable. Confidence, summaries, and the graph are mutable derived views. Time-travel and recomputation are free.
5. **Embeddable, single binary.** No server, no infrastructure. SQLite deployment model.
6. **Zero-config to first query.** Sensible defaults for everything. Vocabulary registration, predicate constraints, and payload schemas are power-user features, not prerequisites. `attestdb.open()` → `db.ingest()` must work with zero setup.

## The Claim Schema [BUILT]

Every write to Attest is a claim. The schema is enforced by the storage engine (13 validation rules, all tested):

```
Claim:
  claim_id:    ContentHash     # SHA-256(subject + predicate + object + source_id + source_type + timestamp)
  content_id:  ContentHash     # SHA-256(subject + predicate + object) — corroboration grouping key
  subject:     EntityRef       # typed reference, validated against vocabulary
  predicate:   PredicateRef    # from registered vocabulary, type-checked
  object:      EntityRef | ScalarValue
  confidence:  f64 [0.0, 1.0]  # computed from provenance via tiered confidence system
  provenance:  ProvenanceChain # required — engine rejects writes without this
  embedding:   Option<Vec<f32>> # fixed dimensionality per named embedding space
  payload:     Option<TypedPayload> # schema-validated structured data
  timestamp:   i64             # nanosecond precision, immutable
  status:      enum [active, archived, tombstoned, provenance_degraded]
```

Two claims asserting the same relationship (same subject + predicate + object) from different sources share the same `content_id` but have different `claim_id`s. This is how corroboration is detected without dedup collision.

### Supporting Types

```
EntityRef:
  id:            String          # normalized canonical (lowercase, whitespace-collapsed, Greek spelled out)
  entity_type:   TypeTag         # must exist in registered vocabulary
  display_name:  String          # original casing preserved for humans (first writer wins, curator overrides)
  external_ids:  Map<Namespace, String>  # optional anchoring to external systems

PredicateRef:
  id:            String          # canonical predicate identifier
  predicate_type: TypeTag        # must exist in registered vocabulary

ScalarValue:
  type:          enum [float, int, string, bool, datetime, json]
  value:         bytes
  unit:          Option<String>  # "nM", "ms", "USD", etc.

ProvenanceChain:
  source_type:   TypeTag         # must exist in registered source vocabulary
  source_id:     String          # originating system/paper/agent/human
  method:        Option<String>  # specific method used
  chain:         Vec<ClaimID>    # upstream dependencies (must exist, DAG-validated)
  model_version: Option<String>  # if computational, exact version
  organization:  Option<String>  # originating organization (enables future federated sharing)

TypedPayload:
  schema_ref:    SchemaID        # references a registered payload schema
  data:          bytes           # validated against schema on write
```

### Entity Normalization [BUILT — locked, identical in Python and Rust]

Every entity ID is normalized on ingestion. The normalization function is deterministic and identical in Phase 1 (Python) and Phase 2 (Rust):

```python
def normalize_entity_id(raw: str) -> str:
    s = unicodedata.normalize("NFKD", raw)
    s = s.lower()
    s = " ".join(s.split())
    s = s.strip()
    greek = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
             "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu",
             "τ": "tau", "ω": "omega"}
    for char, name in greek.items():
        s = s.replace(char, name)
    return s
```

"TREM2", "trem2", and "Trem2" all normalize to "trem2" and hit the same graph node.

### Entity Aliases (same_as) [BUILT]

When two canonical IDs refer to the same real-world entity, a `same_as` claim links them:

```python
db.ingest(subject=("q9nzc2", "protein"), predicate=("same_as", "same_as"),
          object=("trem2", "protein"), provenance={...}, confidence=1.0)
```

Aliases are resolved at query time via an in-memory union-find index. Old claims are never rewritten. The alias index is rebuilt from same_as/not_same_as claims on startup.

### Claim Identity Functions [BUILT — locked, identical in Python and Rust]

```python
def compute_claim_id(subject_canonical, predicate_id, object_canonical,
                     source_id, source_type, timestamp) -> str:
    payload = f"{subject_canonical}|{predicate_id}|{object_canonical}|{source_id}|{source_type}|{timestamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def compute_content_id(subject_canonical, predicate_id, object_canonical) -> str:
    payload = f"{subject_canonical}|{predicate_id}|{object_canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

## The Type System [BUILT]

Attest sits between rigid schemas (SQL) and schemaless (MongoDB): **the claim structure is rigid, the vocabulary is extensible, and payloads are schema-validated.**

### Vocabulary Registration [BUILT]

The engine ships with a built-in vocabulary sufficient for general use without any registration:

```
Built-in entity types:    entity, event, metric, document, agent, system
Built-in predicate types: relates_to, caused, observed, derived_from, contradicts,
                          same_as, not_same_as, retracted, contradiction_resolved, inquiry
Built-in source types:    observation, computation, document_extraction,
                          llm_inference, human_annotation
```

Predicates with special engine semantics:
- `same_as` / `not_same_as`: entity alias resolution (union-find index)
- `contradicts` / `contradiction_resolved`: contradiction tracking (confidence Tier 3)
- `retracted`: source retraction (cascading tombstone)
- `inquiry`: open questions (tracked by insight engine, matched against incoming evidence)

**Zero-config mode:** A developer can immediately ingest claims using only the built-in vocabulary. No `register_vocabulary()` call is required. The built-in types cover generic use cases (DevOps incidents, ML experiments, audit trails). Domain-specific vocabularies are an optional upgrade path.

```python
# This works immediately — no setup required
db = attestdb.open("./my.db")
db.ingest(
    subject=("deployment_v2.3", "event"),
    predicate=("caused", "caused"),
    object=("latency_spike", "event"),
    provenance={"source_type": "observation", "source_id": "monitoring_agent"}
)
```

**Registered vocabulary mode:** Applications register domain vocabularies for stronger type checking:

```python
# Optional — enables domain-specific type validation
db.register_vocabulary("ops", {
    "entity_types": ["service", "deployment", "incident", "config_change"],
    "predicate_types": ["depends_on", "triggered", "resolved_by", "degraded"],
    "source_types": ["monitoring", "log_analysis", "postmortem", "runbook"]
})
```

When a domain vocabulary is registered, the engine validates claims against it. Unknown types are rejected. Without registration, only built-in types are validated. New types are registered without migration — no ALTER TABLE equivalent needed.

**Strict mode (opt-in):** `db.open("./my.db", strict=True)` rejects any claim using a type not in a registered vocabulary. For production deployments where type discipline matters.

### Predicate Constraints [BUILT — type constraints + symmetric traversal]

Predicates can optionally declare type constraints and logical properties:

```python
db.register_predicate("binds", {
    "subject_types": ["protein", "compound"],
    "object_types": ["protein", "compound", "dna_sequence"],
    "symmetric": True,              # A binds B → B binds A (query engine searches both directions)
    "transitive": False,
    "inverse": None,
    "expected_payload": "binding_affinity"
})

db.register_predicate("causes", {
    "subject_types": ["event", "entity"],
    "object_types": ["event", "entity"],
    "symmetric": False,
    "transitive": True,             # A causes B, B causes C → A causes C (derived at query time)
    "inverse": "caused_by",         # auto-generated inverse for bidirectional traversal
    "expected_payload": None
})
```

Logical properties:
- **symmetric:** if true, the query engine traverses the predicate in both directions. No duplicate claims stored — the engine infers the inverse.
- **transitive:** if true, the query engine can follow chains of this predicate to compute indirect relationships. Depth-bounded by the query's `depth` parameter.
- **inverse:** names the inverse predicate. "A causes B" automatically generates "B caused_by A" at query time.

**Status:** Subject/object type constraints are enforced on ingestion. Symmetric traversal is implemented — predicates marked `symmetric: True` are automatically traversed in both directions by the query engine. Transitivity and inverse generation are Phase 3.

The engine rejects `("United States", "binds", "GDP")` because "binds" doesn't accept those entity types. Lightweight schema validation — type checking on the claim structure, not foreign keys.

### Payload Schemas [BUILT]

Structured data attached to claims is schema-validated:

```python
db.register_payload_schema("binding_affinity", {
    "type": "object",
    "properties": {
        "metric": {"enum": ["Kd", "Ki", "IC50", "EC50"]},
        "value": {"type": "number"},
        "unit": {"enum": ["nM", "uM", "mM", "M"]},
        "conditions": {"type": "object"}
    },
    "required": ["metric", "value", "unit"]
})

db.register_payload_schema("latency_measurement", {
    "type": "object",
    "properties": {
        "p50": {"type": "number"},
        "p99": {"type": "number"},
        "unit": {"enum": ["ms", "us", "s"]},
        "sample_size": {"type": "integer"}
    },
    "required": ["p99", "unit"]
})
```

Claims with a payload reference a schema. The engine validates on write.

### Where Things Live: Core vs. Vertical

| Component | Core (open source) | Vertical (commercial) |
|-----------|-------------------|----------------------|
| Claim schema | Enforced by engine | — |
| Built-in vocabulary | Minimal (entity, event, etc.) | Domain-specific (protein, service, etc.) |
| Vocabulary registration API | Engine provides | Verticals register |
| Predicate constraints | Engine validates | Verticals define |
| Payload schemas | Engine validates | Verticals define |
| Confidence function | Pluggable framework + default | Domain-calibrated weights |
| Entity resolution | Deterministic (content-addressed, exact match) | LLM-powered fuzzy resolution |
| Ontology linking | External ID storage | Domain ontology plugins (UniProt, etc.) |
| Multi-resolution summaries | Framework (grouping API) | Domain-specific levels |
| Curator | — | LLM-powered, domain-tuned |
| Insight engine | — | Bridge prediction, gap ID, alerts |

## Storage Engine

### Append-Only Claim Log [BUILT — Kuzu-backed; Rust native log is Phase 2B]

All claims are appended to an immutable log. Content-addressed hashing provides automatic dedup — identical claims from different agents hash to the same ID (corroboration detection).

**Current implementation:** Kuzu property graph with claims as edges, entities as nodes. Application-layer validation enforces all 13 ingestion rules. ~36 claims/sec throughput with entity/claim caching.

**Phase 2B target:** Native Rust claim log with dictionary encoding for predicates/source_types, delta encoding for timestamps, quantized embeddings. Target: >100 claims/sec, ~100 bytes/claim compressed.

### Indexes [PARTIAL — graph + embedding built, text + temporal designed]

**Graph index [BUILT]:** Kuzu's native graph storage with Cypher traversal. Application layer provides `get_adjacency_list()` and `get_weighted_adjacency()` for in-memory graph operations (used by insight engine, ensemble scorer). Phase 2B replaces with CSR format.

**Embedding index [BUILT]:** usearch HNSW, in-process alongside Kuzu. Supports add/search/persist. Structural embeddings stored via synthetic `_struct_{entity_id}` keys. Single embedding space per database (fixed dimensionality).

**Temporal index [DESIGNED]:** Time-partitioned access. Rust engine has `claims_in_range(min_ts, max_ts)`. Query-level `db.at(timestamp)` wrapper is on the priority list.

**Inverted text index [DESIGNED]:** BM25 keyword search over entity names, predicate types, payload text fields. Not yet implemented. Phase 2B.

A single query will blend all four indexes in one pass. Currently, graph + embedding are separate lookups joined in the Python application layer.

### Hybrid Query Fusion [DESIGNED]

When a ContextFrame query uses multiple indexes (common case), results are fused using configurable algorithms:

- **Graph-first:** Start from focal entity, traverse graph, score by confidence. Default for entity-centric queries. **This is what's currently implemented.**
- **Embedding-first:** Start from semantic search, retrieve top-K similar entities, then expand graph neighborhoods. Default for exploratory queries.
- **Keyword-first:** Start from BM25 text match, then expand graph context. Default for exact-term queries.
- **Parallel fusion:** Run graph + embedding + keyword in parallel, merge via Reciprocal Rank Fusion (RRF). Default for hybrid queries.

The query planner selects strategy based on query parameters and index statistics (see Query Planner below).

### Query Planner [PARTIAL — db.explain() built, full statistics-driven planner is Phase 3]

The engine maintains a statistics catalog, updated incrementally on each claim ingestion:

```
Statistics:
  entity_count_by_type:     {protein: 45000, compound: 120000, ...}
  claim_count_by_predicate: {binds: 500000, inhibits: 30000, ...}
  avg_degree_by_type:       {protein: 12.5, compound: 3.2, ...}
  source_type_distribution: {experimental: 40%, computational: 35%, ...}
  embedding_coverage:       0.85  # fraction of entities with embeddings
  text_index_coverage:      0.95  # fraction of claims with indexed text
```

The query planner uses these statistics to:
1. Choose traversal starting point (most selective index first — learned from Neo4j's anchor node optimization)
2. Estimate cost of graph traversal vs. embedding scan vs. text search
3. Set query budget allocation across indexes
4. Decide parallel vs. sequential execution

**Query debugging:**

```python
# Show execution plan without running
plan = db.explain(focal_entity="TREM2", depth=2, min_confidence=0.5)
# Returns: ExecutionPlan { strategy: "graph_first", estimated_claims: 450, 
#          indexes_used: ["graph", "temporal"], estimated_ms: 120 }

# Run and show actual execution stats
frame, stats = db.profile(focal_entity="TREM2", depth=2, min_confidence=0.5)
# Returns: frame + ProfileStats { actual_claims: 523, actual_ms: 145,
#          index_hits: {graph: 523, embedding: 0, text: 0, temporal: 523},
#          cache_hits: 412 }
```

### Named Vector Spaces [DESIGNED]

A single database can maintain multiple embedding indexes for different representation types:

```python
db = attestdb.open("./bio.db", embeddings={
    "structural": 768,    # protein structure embeddings (ESM-2)
    "functional": 256,    # GO term embeddings
    "literature": 1024    # paper text embeddings (PubMedBERT)
})

# Ingest with multiple embeddings
db.ingest(
    subject=("TREM2", "protein"),
    predicate=("binds", "binds"),
    object=("APOE", "protein"),
    provenance={...},
    embeddings={
        "structural": [0.1, 0.2, ...],   # 768-dim
        "functional": [0.3, 0.4, ...],    # 256-dim
    }
)

# Search in a specific embedding space
results = db.search(query_embedding=[...], embedding_space="structural", top_k=20)
```

Each named space has its own HNSW index with the declared dimensionality. Embeddings are optional per-claim — a claim can have embeddings in some spaces but not others. The default single-space configuration (`embedding_dim=768`) is syntactic sugar for a single unnamed space.

### Scaling

| Scale | Claim Log | Graph Index | Embedding Index | Text Index |
|-------|-----------|-------------|-----------------|------------|
| 1M claims | ~100MB disk | ~12MB RAM | ~192MB (int8) | ~50MB |
| 100M claims | ~10GB disk | ~1.2GB RAM | Partitioned, ~20GB | ~5GB |
| 1B claims | ~100GB disk | ~12GB RAM | Partitioned, ~200GB | ~50GB |

Fits on a single machine up to ~10B claims with memory-mapped files. Text index uses LSM-tree segmented storage (similar to Weaviate's bucket architecture) for efficient writes and compaction.

## Confidence System [PARTIAL — Tier 1 active + retraction built, Tier 2 not wired into queries]

### Three-Tier Confidence

Confidence is computed in tiers. Each tier adds complexity. Tier 1 is active on all ingestion and queries. Retraction with cascading confidence recomputation is built. Tier 2 code exists (`tier2_confidence()`, `count_independent_sources()`) but is not yet wired into query-time confidence on ContextFrames.

**Tier 1: Source-Type Confidence [BUILT — active on all claims]**

Pure function of provenance metadata. No queries, no dependencies, computed on ingestion.

```python
SOURCE_TYPE_WEIGHTS = {
    "observation": 0.70, "computation": 0.50, "document_extraction": 0.60,
    "llm_inference": 0.30, "human_annotation": 0.90,
    # Bio vocabulary adds:
    "experimental": 0.85, "crystallography": 0.95, "mass_spec": 0.80,
    "docking": 0.40, "literature": 0.65,
}

def tier1_confidence(claim) -> float:
    return SOURCE_TYPE_WEIGHTS.get(claim.provenance.source_type, 0.50)
```

**Tier 2: Corroboration-Adjusted [PARTIAL — code exists, not wired into queries]**

Adds corroboration boost via content_id grouping and recency decay. Computed lazily on read, cached with dirty flag.

```python
def tier2_confidence(claim, db) -> float:
    base = SOURCE_TYPE_WEIGHTS.get(claim.provenance.source_type, 0.50)
    n_independent = db.count_independent_sources(claim.content_id)
    corroboration_boost = min(1.0 + 0.3 * math.log2(max(n_independent, 1)), 1.7)
    age_days = (now() - claim.timestamp) / 86400
    recency = 0.5 ** (age_days / 365)
    return min(base * corroboration_boost * recency, 1.0)
```

"Supporting claims" precisely defined: claims sharing the same content_id with independent provenance (no shared ancestors in provenance DAG).

**Tier 3: Contradiction-Aware [DESIGNED]**

Adds penalty for curator-flagged contradictions. "Contradicting claims" precisely defined: claims where the curator has written a `contradicts` predicate linking two content_ids.

### Circular Evidence Detection (Hard Constraint)

Two claims are "independent" only if neither appears in the other's provenance chain AND they don't share a common ancestor claim. This is a DAG check, enforced at the engine level. If claim Y lists claim X in its chain, Y cannot count as independent corroboration of X.

### Retraction (First-Class Operation) [BUILT]

First-class retraction with cascading confidence recomputation. The single strongest differentiator vs. every competitor — none of them can answer "a paper was retracted, what do we no longer trust?"

```python
result = db.retract(source_id="PMID:12345678", reason="Paper retracted by journal")
# 1. Tombstones all claims from that source
# 2. Marks downstream claims (provenance.chain references tombstoned claims) as "provenance_degraded"
# 3. Triggers cascading confidence recomputation
# 4. Writes audit claim: (source_id, "retracted", reason)
# Returns: RetractResult { tombstoned: int, degraded: int, recomputed: int }
```

### Confidence Recomputation

Because the claim log is immutable and confidence is a derived view, you can:
- Recompute the entire belief graph with different confidence weights
- Ask "what would our knowledge look like if we weighted experimental evidence 3x?"
- Register a new confidence function and rebuild without data loss

## Compaction [DESIGNED]

The append-only log grows unboundedly. A deterministic compaction process manages the working set:

```
Active:              Claims <6 months old, OR confidence > threshold, OR
                     referenced by other active claims.
                     → Materialized in graph and embedding indexes.

Provenance_degraded: Claims whose provenance chain includes tombstoned claims.
                     → Still active and queryable, but flagged. Confidence recomputed
                       excluding tombstoned upstream claims.

Archived:            Claims >6 months, low confidence, not referenced.
                     → Removed from indexes, retained in cold log for provenance.

Tombstoned:          Claims from retracted/buggy sources (via db.retract()).
                     → Provenance preserved, claim excluded from all indexes,
                       downstream claims marked provenance_degraded.
```

The graph and embedding indexes **only materialize active claims**. Working set is bounded even though history grows. Compaction is deterministic: same log + same rules = same active set.

## Query Interface: ContextFrames [BUILT]

Attest serves consumers through structured ContextFrames — not text dumps or raw subgraphs. Entity-centric queries are built and returning 12ms latency with LLM-generated narrative summaries, topic membership, inquiry surfacing, and query profiling. Topic-centric queries depend on the full knowledge topology hierarchy (Phase 3).

### Entity ContextFrame [BUILT]

```python
frame = db.query(
    focal_entity="deployment_v2.3",
    depth=2,
    min_confidence=0.5,
    exclude_source_types=["llm_inference"],
    topic="deployment reliability",             # optional — restrict to claims within this topic
    max_tokens=4000
)
```

Returns:

```
ContextFrame:
  focal_entity:         {id, name, type, external_ids}
  direct_relationships: [{predicate, target, confidence, n_sources, source_types}, ...]
  quantitative_data:    [{predicate, target, value, unit, metric, confidence}, ...]
  contradictions:       [{claim_a, claim_b, evidence_each, status}, ...]
  knowledge_gaps:       ["No data on X interaction with Y", ...]      # from insight engine (empty if not running)
  narrative:            "Deployment v2.3 is connected to..."           # from intelligence layer (empty if not running)
  topic_membership:     [{topic_id, topic_label, level, strength}, ...]
  provenance_summary:   {observation: 60%, computation: 30%, ...}
  open_inquiries:       [{question, inquirer, priority}, ...]          # from inquiry claims (empty if none)
  claim_count:          int
  confidence_range:     (min, max)
```

The `narrative` field transforms a structured database result into a reality model explanation. The query engine synthesizes frame contents into a human-readable paragraph via LLM (using the same OpenAI-compatible API infrastructure as the curator). This is the difference between "47 claims about TREM2" and "TREM2 sits at the intersection of neurodegeneration and innate immunity..."

The `topic_membership` field shows which emergent topics the focal entity belongs to, enabling navigation: "TREM2 is in the neurodegeneration topic (L2) and the innate immunity topic (L2), which meet at the microglial signaling topic (L1)."

The `open_inquiries` field surfaces outstanding questions about this entity — things someone has asked but the graph doesn't yet answer.

### Topic ContextFrame [DESIGNED — depends on Knowledge Topology]

For queries about a domain rather than an entity:

```python
topic_frame = db.query_topic(
    topic="neurodegeneration",
    level=2,                                    # domain level
    max_tokens=8000
)
```

Returns:

```
TopicContextFrame:
  topic:                {id, label, level, entity_count, claim_count}
  key_entities:         [{entity, centrality_score, claim_count}, ...]     # most connected within topic
  sub_topics:           [{id, label, entity_count}, ...]                   # children in topic hierarchy
  parent_topics:        [{id, label}, ...]                                 # parents in topic hierarchy
  cross_domain_bridges: [{target_topic, connector_entities, claim_count, confidence}, ...]
  knowledge_density:    float                                              # claims per entity pair
  frontier_gaps:        ["No evidence connecting X to Y despite structural similarity", ...]
  contradictions:       [{claim_a, claim_b, evidence_each, status}, ...]   # topic-level contradictions
  narrative:            "Neurodegeneration research in this graph covers..."
  provenance_summary:   {experimental: 40%, computational: 35%, ...}
  temporal_trend:       [{month, claims_added, entities_added}, ...]       # growth over time
  open_inquiries:       [{question, inquirer, priority}, ...]
```

### Cross-Domain Query [DESIGNED]

```python
bridges = db.cross_domain(
    topic_a="neurodegeneration",
    topic_b="lipid metabolism",
    max_depth=3
)
# Returns: ranked paths connecting the two topics, with confidence and novelty scores
```

### Time Travel [BUILT]

```python
# What did we know as of January?
frame = db.at("2026-01-15").query(focal_entity="service_x")

# What claims depend on this retracted paper?
downstream = db.trace_downstream(claim_id="abc123")

# What would knowledge look like excluding all claims from agent_7?
db.recompute_confidence(exclude_sources=["agent_7"])
```

## Knowledge Topology [PARTIAL — topic membership built, full hierarchy Phase 3]

Topics are not manually defined. They emerge from the graph structure via community detection, producing a multi-scale, overlapping hierarchy of knowledge domains. Basic topic membership via Leiden community detection (CPM) is built and surfaced in ContextFrames. Full hierarchy construction, density mapping, and cross-domain connector identification are Phase 3.

### Why Topology Matters

A flat graph of claims is useful for answering "what do we know about entity X?" But a reality model needs to answer higher-order questions:

- "What are the major domains of knowledge in this graph?" (topic discovery)
- "Where is our knowledge deep vs. shallow?" (density mapping)
- "Which domains are well-connected and which are isolated?" (cross-domain analysis)
- "Where are the surprising gaps — topics that SHOULD connect but don't?" (structural gap detection)

These questions can't be answered by entity-centric queries. They require understanding the SHAPE of the knowledge graph.

### Topic Hierarchy

```
TopicNode:
  id:                  ContentHash       # deterministic from constituent entity set
  label:               String            # auto-generated or curator-assigned
  level:               int               # 0=entity, 1=local, 2=domain, 3=mega-domain
  entities:            Set<EntityRef>    # members at this resolution
  parent_topics:       Set<TopicID>      # higher-level topics (DAG — overlapping membership)
  child_topics:        Set<TopicID>      # lower-level topics
  density:             float             # claims per entity pair (knowledge depth)
  boundary_entities:   Set<EntityRef>    # entities also in other topics (cross-domain connectors)
  dominant_source_types: list[str]       # what kind of evidence dominates
  created_at:          timestamp
  last_updated:        timestamp
```

Topics are a **derived index** — rebuilt periodically from the claim graph, not stored as claims. Like the graph index or embedding index, they are a materialized view over the claim log.

### Computation

```
1. COMMUNITY DETECTION
   Leiden algorithm at multiple resolution parameters, producing topics at levels 1-3.
   Each entity gets an overlapping membership vector.
   Runs on the CSR graph index — no external dependencies.

2. TOPIC LABELING
   Phase 1: template from dominant entity types + predicate types
   Phase 2: LLM-generated concise label from top entities in the topic
   Curator can override any auto-generated label.

3. HIERARCHY CONSTRUCTION
   Topics at different resolutions form a DAG (not a tree — overlapping membership).
   Parent-child relationships based on entity subset containment.

4. CROSS-DOMAIN CONNECTOR IDENTIFICATION
   Entities with high membership in multiple topics are connectors.
   These are the highest-value targets for insight engine bridge prediction.

5. DENSITY MAPPING
   Per topic: internal density (claims per entity pair within topic),
   boundary density (claims connecting topic to neighbors),
   gap score (expected connections missing based on type patterns).
```

### Example: Biology Graph After 3 Months

```
L3: "cancer biology"
├── L2: "oncogene signaling"          density: 0.42 (deep)
│   ├── L1: "BRAF V600E pathway"
│   ├── L1: "RAS-MAPK cascade"
│   └── L1: "PI3K-AKT-mTOR"
├── L2: "tumor immunology"            density: 0.31 (moderate)
│   ├── L1: "checkpoint inhibitors"
│   └── L1: "tumor microenvironment"
└── L2: "cancer metabolism"            density: 0.08 (shallow — frontier)
    └── L1: "Warburg effect"

Cross-domain connectors:
  - "p53" bridges oncogene signaling ↔ tumor immunology ↔ cancer metabolism
  - "HIF-1α" bridges tumor microenvironment ↔ cancer metabolism

Structural gap: cancer metabolism ↔ oncogene signaling has only 3 claims
  despite p53 connecting both. The insight engine flags this.
```

### Example: DevOps Graph After 6 Months

```
L3: "platform reliability"
├── L2: "deployment pipeline"          density: 0.55 (deep)
│   ├── L1: "CI/CD failures"
│   └── L1: "rollback procedures"
├── L2: "database operations"          density: 0.38 (moderate)
│   ├── L1: "connection pool management"
│   └── L1: "query performance"
└── L2: "authentication system"        density: 0.12 (shallow)
    └── L1: "token refresh failures"

Cross-domain connectors:
  - "connection pool" bridges database operations ↔ authentication system
  
Structural gap: authentication system ↔ deployment pipeline has weak links.
  Postmortems show auth failures after deployments but no documented dependency.
```

## Ingestion [BUILT]

### Structured Claims (Primary Path) [BUILT]

Direct API ingestion with full provenance, as described in the claim schema. This is the path for agents, database imports, and programmatic sources. All 13 validation rules enforced. Batch mode with warm caches (~36 claims/sec). Bulk loaders for Hetionet, DisGeNET, STRING, Reactome, and ChEMBL.

### Text-to-Claims (Secondary Path) [DESIGNED]

Most organizational knowledge starts as unstructured text. The text ingestion pipeline converts text → candidate claims → curator triage → stored claims.

```python
results = db.ingest_text(
    text="The deployment of v2.3 caused a latency spike in the auth service. "
         "Root cause was connection pool exhaustion in the shared Postgres instance.",
    source_type="document_extraction",
    source_id="slack://C04N8BXYZ/1234567890",
    extraction_model="sonnet"
)
# Internally:
# 1. LLM extracts candidate claims from text:
#    - (deployment_v2.3, caused, latency_spike_auth) 
#    - (connection_pool_exhaustion, caused, latency_spike_auth)
#    - (connection_pool_exhaustion, located_in, shared_postgres)
# 2. Entity normalization applied to each candidate
# 3. Curator triages: store/skip/flag for each candidate
# 4. Valid claims ingested with provenance chain pointing to text source
# Returns: TextIngestionResult { extracted: 3, stored: 2, skipped: 1, errors: [] }
```

The extraction model is configurable. Provenance is explicit: every text-extracted claim has `source_type: "document_extraction"` and a chain pointing to the source text. Consumers can filter by source type to exclude text-extracted claims when higher-confidence evidence is needed.

### Inquiry Ingestion [BUILT]

Questions — things you want to know but don't yet — are stored as claims with the built-in `inquiry` predicate:

```python
db.ingest(
    subject=("trem2", "protein"),
    predicate=("inquiry", "inquiry"),
    object=("complement system", "pathway"),
    provenance={"source_type": "human_annotation", "source_id": "researcher_jane"},
    payload={"schema": "inquiry", "data": {
        "question": "Does TREM2 interact with the complement system?",
        "priority": "high",
        "context": "Relevant to AD therapeutic hypothesis"
    }}
)
```

Inquiry claims are ingested, stored, and surfaced in ContextFrames via the `open_inquiries` field. The query engine retrieves open inquiries for the focal entity and includes them in query results.

## File Format [DESIGNED — Phase 2B]

### Versioning

The Attest file format follows SQLite/DuckDB's approach to backward compatibility:

```
File Header (first 64 bytes):
  magic:              [u8; 8]    # b"ATTEST\0\0" — identifies an Attest file
  format_version:     u16        # major format version (1, 2, ...)
  min_reader_version: u16        # minimum reader version that can open this file
  embedding_config:   [u8; 32]   # named spaces + dimensionalities (serialized)
  flags:              u32        # strict mode, compression type, etc.
  created_at:         i64        # database creation timestamp
  checksum:           u32        # header integrity check
```

**Compatibility contract:** All Attest v1.x releases read v1 format files. Format changes that break backward compatibility increment the major version. The `min_reader_version` field allows forward compatibility — a v1.3 file can declare it needs at least a v1.2 reader if it uses features added in v1.2.

This is declared before open-source launch. Format stability = trust. Without it, early adopters risk losing their data on upgrade.

### Single-File Distribution

One `.attest` file contains: claim log, all indexes (graph, embedding, text, temporal), vocabulary registry, payload schemas, statistics catalog, and configuration. Sidecar files (WAL, lock) exist only during active use and are cleaned up on close — identical to SQLite's behavior.

## Extension Architecture [DESIGNED — Phase 3]

Extensions are designed from day one, even if the loading mechanism ships later. The engine defines extension points — interfaces that external code can implement:

### Extension Points

| Extension Point | What it does | Example |
|----------------|-------------|---------|
| **Vocabulary plugin** | Registers entity types, predicate types, source types, predicate constraints, payload schemas as a bundle | `attest-bio`, `attest-devops` |
| **Confidence function** | Custom scoring logic for the confidence system | Biology-calibrated weights, DevOps-calibrated weights |
| **Ingestion adapter** | Reads external formats and produces claim streams | Parquet adapter, CSV adapter, JSON-LD adapter, Neo4j importer |
| **Export formatter** | Serializes claims/ContextFrames to external formats | JSON-LD exporter, RDF/Turtle exporter, Parquet exporter |
| **Embedding provider** | Generates embeddings on ingest when not provided | OpenAI, Cohere, local model (ONNX) |
| **Text analyzer** | Tokenization and stemming for the inverted text index | Language-specific analyzers, domain-specific tokenizers |

### Distribution Model (Phase 3)

Following DuckDB's extension template approach:
- Extensions are loadable shared libraries (.so/.dylib/.dll) or WASM modules
- Signed with Attest keys for security
- Distributed via a registry (`attestdb install bio`)
- Auto-loaded when referenced (`db.open("./my.db", vocabulary="bio")` auto-installs the bio vocabulary plugin)

In Phase 2, extension points exist as Rust traits/interfaces. The bio vocabulary is the first "extension" even though it's statically linked. The dynamic loading mechanism ships in Phase 3.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────┐
│                Attest (Rust binary)                │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │           Claim Ingestion API                │    │
│  │  (validates schema, provenance, types,       │    │
│  │   payload schemas, predicate constraints)    │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌──────────────────▼───────────────────────────┐    │
│  │         Append-Only Claim Log                │    │
│  │  (content-addressed, compressed,             │    │
│  │   tiered: active / archived / tombstoned)    │    │
│  └──────────────────┬───────────────────────────┘    │
│                     │                                 │
│  ┌────────┐ ┌───────▼──┐ ┌──────────┐ ┌──────────┐  │
│  │ Graph  │ │ Embedding│ │ Temporal │ │ Inverted │  │
│  │ Index  │ │  Index   │ │  Index   │ │ Text Idx │  │
│  │ (CSR)  │ │  (HNSW)  │ │(partitnd)│ │  (BM25)  │  │
│  └───┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│      └───────────┼────────────┼─────────────┘        │
│                  │            │                        │
│  ┌───────────────▼────────────▼──────────────────┐   │
│  │    Query Planner + ContextFrame Engine        │   │
│  │  (statistics-driven strategy selection,       │   │
│  │   hybrid fusion, query budgets, EXPLAIN,      │   │
│  │   time-travel, confidence recomputation)      │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  Vocabulary registry │ Confidence functions           │
│  Payload schemas     │ Statistics catalog              │
│  Compaction process  │ Extension points                │
│  File format v1      │ Named vector spaces             │
└──────────────────────────────────────────────────────┘

          ▲                           ▲
          │                           │
   Python/TS client              Application layer
   (pip install attestdb)       (curator, insight engine,
                                  domain-specific logic)
```

The core binary is the database. The curator, insight engine, and domain-specific intelligence are application-layer processes that use the database through its API — they are not embedded in the binary.

## Future Architecture (Phase 3+)

These are not built in Phase 1 or 2, but the architecture accounts for them to avoid costly retrofits.

### Multi-Tenancy

Weaviate's per-tenant shard architecture is the model. Each tenant gets an isolated namespace with its own claims, indexes, vocabulary, and configuration. Tenant states: ACTIVE (loaded, queryable), INACTIVE (on disk, fast to activate), OFFLOADED (cold storage).

**Architectural hook:** The file format supports multiple namespaces within a single `.attest` file. Each namespace has its own claim log segment, index partitions, and vocabulary registry. Phase 2 ships with a single default namespace. Phase 3 adds multi-namespace support and tenant lifecycle management.

### Auto-Vectorization

Embedding providers (OpenAI, Cohere, local ONNX models) can be registered to automatically generate embeddings on ingest when not provided:

```python
db = attestdb.open("./my.db", vectorizer="text-embedding-3-small")
# Claims ingested without embeddings are auto-vectorized
```

**Architectural hook:** The embedding field in the claim schema is already optional. The ingestion API checks: if embedding is provided, store it; if not and a vectorizer is configured, generate it; if neither, store without embedding. Phase 2 supports pre-computed embeddings only. Phase 3 adds the vectorizer integration.

## Phase 1: Kuzu Implementation [COMPLETE]

Phase 1 validated the intelligence layer using Kuzu as infrastructure. All exit criteria met: curator 98% accuracy, edge recovery 17.35%, query latency 12ms. The Rust engine (described above) is the Phase 2 target architecture. The API surface is identical in both phases (see 06_api_spec.md).

### Claim → Kuzu Mapping

Kuzu is a property graph database. Claims don't map 1:1 to its native model. The mapping:

**Entities → Kuzu nodes:**
```cypher
CREATE NODE TABLE Entity (
    id STRING PRIMARY KEY,
    entity_type STRING,
    name STRING,
    external_ids STRING,          -- JSON: {"uniprot": "Q9NZC2", ...}
    created_at INT64
)
```

**Claims → Kuzu edges:**
```cypher
CREATE REL TABLE Claim (
    FROM Entity TO Entity,
    claim_id STRING,              -- content-addressed hash
    predicate_type STRING,
    confidence DOUBLE,
    source_type STRING,
    source_id STRING,
    method STRING,
    provenance_chain STRING,      -- JSON array of upstream claim IDs
    model_version STRING,
    payload STRING,               -- JSON, schema-validated in application layer
    embedding DOUBLE[768],        -- if Kuzu supports fixed-length arrays; otherwise separate table
    timestamp INT64
)
```

**Vocabulary + schemas → Kuzu metadata tables:**
```cypher
CREATE NODE TABLE Vocabulary (
    namespace STRING,
    type_category STRING,         -- "entity_type", "predicate_type", "source_type"
    type_name STRING,
    PRIMARY KEY (namespace, type_category, type_name)
)

CREATE NODE TABLE PayloadSchema (
    schema_id STRING PRIMARY KEY,
    schema_json STRING
)
```

**What Kuzu gives us for free:** ACID transactions, crash recovery, Cypher queries, bulk loading (COPY FROM), concurrent read access, buffer pool management.

**What we enforce at the application layer in Phase 1:**
- Provenance validation (required fields, chain existence check, DAG check)
- Vocabulary type checking
- Payload schema validation
- Content-addressed ID computation
- Predicate type constraints

### Embedding Index

Kuzu does not have native vector search. In Phase 1, we run an in-process HNSW library (usearch or hnswlib) alongside Kuzu:

- On claim ingestion: insert embedding into HNSW index (in-memory, persisted to disk as sidecar file)
- On semantic search: query HNSW, get entity IDs, then query Kuzu for full claim data
- Two separate indexes, coordinated by the Python application layer
- Acceptable for Phase 1 validation; the co-located index in the Rust engine eliminates this seam

### Known Limitations (current Kuzu-backed implementation)

- Provenance enforcement is bypassable (anyone with Cypher access can write directly to Kuzu)
- No co-located index queries (graph + vector are separate lookups joined in Python)
- No compaction (Kuzu manages its own storage; we filter by status field instead of removing claims)
- Embedding index is a sidecar file, not integrated into the database file
- Ingestion throughput caps at ~36 claims/sec (Kuzu per-claim overhead)
- Confidence is Tier 1 only at query time (Tier 2 code exists but not integrated)
- Transitivity and inverse predicate generation not yet implemented (symmetric works)
- No BM25 text index (keyword search not available)
- Full knowledge topology hierarchy (density mapping, cross-domain connectors) not yet built — basic topic membership works

These are addressed by Phase 2B Rust engine and Phase 3.

## Export Adapters & Database Audit [DESIGNED]

### Single Source of Truth

Attest's claim log is the canonical record. The graph, embeddings, and any external system's copy are derived views. When Attest exports to Neo4j, it's materializing a view — the claims remain authoritative. This means Attest can serve as the single source of truth that feeds downstream systems, and can audit those systems for consistency.

### Export Adapters

Attest exports materialized views of its claim graph to external systems and formats:

| Target | Format | Method | Notes |
|--------|--------|--------|-------|
| Neo4j | Cypher | `db.export_cypher(path)` | CREATE/MERGE statements with provenance as node properties |
| SQL / RDBMS | SQL DDL + INSERT | `db.export_sql(path, dialect)` | Entities, claims, provenance as normalized tables |
| Vector DBs | Embeddings + metadata | `db.export_embeddings(path)` | Entity embeddings with metadata for Pinecone/Weaviate/Chroma |
| CSV/TSV | Flat files | `db.export_csv(path)` | Entities and relationships for spreadsheet/BI tools |
| NDJSON | Streaming JSON | `export_store()` | **Already built** — full round-trip fidelity via migration module |
| REST API | HTTP endpoints | Console `/api/` routes | **Already available** via attest-console |

Every export carries provenance. A Neo4j node created by Attest export includes the source chain, confidence, and timestamp as properties — not just the entity name and type. This is what makes the export auditable: you can always trace an exported fact back to its original claim.

### Database Audit

Because Attest tracks provenance chains and confidence, it can evaluate whether an external database's facts are well-supported. The audit API compares Attest's grounded claims against an external system:

```python
# Designed API:
audit = db.audit_against(
    external_source="neo4j://localhost:7687",  # or a CSV/NDJSON dump
    scope={"entity_types": ["service", "team"]},
)

# Returns:
#   audit.in_attest_only     — claims with provenance that target system lacks
#   audit.in_external_only      — facts the external system has that Attest doesn't
#   audit.confidence_divergence — same fact, different confidence/provenance quality
#   audit.stale_external        — external facts contradicted by newer Attest claims
```

The audit concept: facts in the external system that have no Attest claim backing them are flagged as unsupported. Facts in Attest that are missing from the external system represent sync gaps. Facts that exist in both but with different confidence levels indicate divergence. This turns Attest into a provenance auditor for any downstream database.

## Evaluation Harness [BUILT]

Automated evaluation is built from the start. It's the quantitative backbone for every decision.

### Edge Holdout Test (Insight Engine) [BUILT — 17.35% recall achieved]

1. Load Hetionet into Kuzu via ego network sampling (`_snowball_sample`: medium-degree entity, 1-hop neighborhood, ~200 entities, ~5K edges)
2. Withhold 10% of edges as ground truth
3. Run damped random walk scoring on normalized adjacency D^{-1/2} A D^{-1/2}, per-entity Hits@K (K=40)
4. Measure recall: what percentage of withheld edges recovered?

Result: **17.35% recall** (target was >15%). SVD structural embeddings + random walk scoring.

### Curator Accuracy Test [BUILT — 98% accuracy achieved]

1. Collect 200+ actual Omic agent outputs with expert labels
2. Run curator on same outputs
3. Compare: accuracy, false positive rate, false negative rate

Result: **98% accuracy** (target was >80%). Uses OpenAI-compatible APIs (Groq, DeepSeek, Grok).

### Agent Quality Comparison [DESIGNED]

1. Select 20 research queries that Omic agents currently handle
2. Run each query twice: once with ContextFrame access, once without
3. Domain experts blind-rate both outputs on: accuracy, completeness, novelty, citation quality
4. Statistical test for significant difference

Not yet run — requires the narrative ContextFrame feature (priority #2) to be meaningful.

### Drug Repurposing Validation [BUILT]

End-to-end demo loading 4-source biomedical graph (Hetionet + DisGeNET + STRING + Reactome), traversing Drug→Gene→Disease paths, ranking by multi-source corroboration, and validating against PubMed. ALS demo: 4/10 top predictions have independent literature support (including a 2025 Dasatinib paper); 6/10 are genuinely novel.

### Continuous Evaluation (Phase 2+) [DESIGNED]

All evaluations run automatically on a weekly cadence. Metrics are tracked over time. Regressions trigger alerts. New seeded data improves the holdout test baseline. Curator eval set grows as more expert labels are collected.
