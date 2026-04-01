# Attest: API Specification

This is the shared contract between the infrastructure and intelligence layers. Both engineers build against these interfaces from day 1. Implemented on the Rust engine (LMDB via heed).

## Database Lifecycle

```python
import attestdb

# Zero-config: works immediately with built-in vocabulary
db = attestdb.open("./my_evidence.db")

# Or configure single embedding space
db = attestdb.open("./bio.db", embedding_dim=768)

# Or strict mode with registered vocabularies
db = attestdb.open("./strict.db", strict=True)

# Optional: register domain vocabulary (not required for basic use)
db.register_vocabulary("bio", {
    "entity_types": ["protein", "gene", "compound", "disease", "pathway"],
    "predicate_types": ["binds", "inhibits", "upregulates", "expressed_in"],
    "source_types": ["experimental", "computational", "literature", "llm_inference"]
})

# Optional: register predicate constraints (enforced if present)
db.register_predicate("binds", {
    "subject_types": ["protein", "compound"],
    "object_types": ["protein", "compound", "gene"],
    "symmetric": True,                        # A binds B ↔ B binds A
    "expected_payload": "binding_affinity"
})

# Optional: register payload schemas (enforced if referenced)
db.register_payload_schema("binding_affinity", {
    "type": "object",
    "properties": {
        "metric": {"enum": ["Kd", "Ki", "IC50", "EC50"]},
        "value": {"type": "number"},
        "unit": {"enum": ["nM", "uM", "mM", "M"]}
    },
    "required": ["metric", "value", "unit"]
})

# Close
db.close()
```

## Claim Ingestion

### Single Claim

```python
claim_id = db.ingest(
    subject=("TREM2", "protein"),           # (id, entity_type)
    predicate=("binds", "binds"),            # (id, predicate_type)
    object=("APOE", "protein"),              # (id, entity_type)
    provenance={
        "source_type": "experimental",
        "source_id": "PMID:12345678",
        "method": "surface_plasmon_resonance",
        "chain": [],                          # no upstream claims
        "model_version": None
    },
    confidence=0.85,                          # optional — computed from provenance if omitted
    embedding=[0.1, 0.2, ...],                # optional — single embedding space
    payload={                                  # optional — validated against schema if predicate has expected_payload
        "schema_ref": "binding_affinity",      # references registered payload schema
        "data": {"metric": "Kd", "value": 5.2, "unit": "nM"}
    },
    external_ids={                            # optional — anchors entities to external databases
        "subject": {"uniprot": "Q9NZC2"},
        "object": {"uniprot": "P02649"}
    },
    namespace="",                             # optional — namespace isolation
    ttl_seconds=0,                            # optional — auto-expiry (0 = never)
)
# Returns: claim_id (content-addressed hash)
# Raises: ProvenanceError, VocabularyError, SchemaValidationError, DuplicateClaimError
```

### Batch Ingestion

```python
results = db.ingest_batch(claims=[...])       # list of ClaimInput objects
# Returns: BatchResult { ingested: int, duplicates: int, errors: list[Error] }
```

### Text Ingestion

```python
results = db.ingest_text(
    text="The deployment of v2.3 caused a latency spike in the auth service.",
    source_id="slack://C04N8BXYZ/1234567890",
    use_curator=True                          # run curator on extracted claims (default True)
)
# Returns: ExtractionResult { claims, rejected, warnings, raw_count, n_valid }
```

### Batch Text Ingestion

```python
results = db.ingest_texts(
    texts=[
        {"text": "...", "source_id": "slack://..."},
        {"text": "...", "source_id": "confluence://..."},
    ],
    use_curator=True
)
```

### Ingestion Validation Rules (enforced on every write)

1. Entity IDs are normalized via `normalize_entity_id()` before any other processing
2. `claim_id` computed from (subject + predicate + object + source_id + source_type + timestamp)
3. `content_id` computed from (subject + predicate + object) — used for corroboration grouping
4. If `claim_id` already exists → DuplicateClaimError
5. If other claims share `content_id` → log corroboration, increment count
6. `provenance.source_type` must exist in registered vocabulary (or built-in) — warning by default, error in strict mode
7. `subject.entity_type` and `object.entity_type` must exist in registered vocabulary (or built-in) — warning by default, error in strict mode
8. `predicate.predicate_type` must exist in registered vocabulary (or built-in) — warning by default, error in strict mode
9. If predicate has registered constraints, subject/object types must match
10. If payload references a schema, data is validated against it
11. If `provenance.chain` is non-empty, all referenced claim IDs must exist
12. Provenance chain must be a DAG (no circular references)
13. If embedding is provided, dimensionality must match database config

## Querying: ContextFrames

### Basic Query

```python
frame = db.query(
    focal_entity="TREM2",                    # entity ID
    depth=2,                                  # graph traversal hops
    min_confidence=0.5,                       # filter low-confidence claims
    exclude_source_types=["llm_inference"],   # contamination control
    max_claims=500,                           # query budget
    max_tokens=4000,                          # token budget for serialized output
    predicate_types=["binds", "inhibits"],    # optional — restrict to these predicates
    llm_narrative=False,                      # optional — generate narrative via LLM
)
```

### ContextFrame Schema

```python
@dataclass
class ContextFrame:
    focal_entity: EntitySummary
    direct_relationships: list[Relationship]
    quantitative_data: list[QuantitativeClaim]
    contradictions: list[Contradiction]
    knowledge_gaps: list[str]                 # populated by blindspots(), empty from query()
    narrative: str                            # from intelligence layer (empty if not running)
    topic_membership: list[str]               # community IDs this entity belongs to
    open_inquiries: list[str]                 # claim IDs of outstanding inquiries
    provenance_summary: dict[str, float]      # source_type → percentage of evidence
    claim_count: int                          # total claims in frame
    confidence_range: tuple[float, float]     # min, max confidence in frame

@dataclass
class EntitySummary:
    id: str
    name: str
    entity_type: str
    external_ids: dict[str, str]
    claim_count: int

@dataclass
class Relationship:
    predicate: str
    target: EntitySummary
    confidence: float
    n_independent_sources: int                # DAG-checked: circular dependencies excluded
    source_types: list[str]
    latest_claim_timestamp: int
    payload: dict | None                      # structured data if present
    is_symmetric: bool                        # True if predicate is registered as symmetric

@dataclass
class Contradiction:
    claim_a: str                              # claim ID
    claim_b: str                              # claim ID
    description: str
    evidence_a: int                           # number of supporting claims for A
    evidence_b: int                           # number of supporting claims for B
    status: str                               # "unresolved", "a_preferred", "b_preferred"

@dataclass
class QuantitativeClaim:
    predicate: str
    target: str
    value: float
    unit: str
    metric: str
    source_type: str
    confidence: float
```

### Semantic Search

```python
results = db.search(
    query_embedding=[0.1, 0.2, ...],
    top_k=20
)
# Returns: list of (entity_id, similarity_score)
```

### Text Search

```python
# BM25 entity name search enriched with claims
results = db.text_search(
    query="CRISPR Cas9 gene editing",
    entity_type="document",                    # optional type filter
    min_confidence=0.5,
    top_k=20,
)
# Returns: list[Claim]
```

### Hybrid Search

```python
# Combines text and embedding search with reciprocal rank fusion
results = db.hybrid_search(
    query="TREM2 binding partners in neurodegeneration",
    alpha=0.7,                                 # 0.0 = pure text, 1.0 = pure vector
    entity_type="protein",
    top_k=20
)
# Returns: list[tuple[Claim, float]]  (claim, fused_score)
# Falls back to text-only if no embedding provider configured
```

### Query Debugging

```python
# Show execution plan without running the query
plan = db.explain(focal_entity="TREM2", depth=2, min_confidence=0.5)
# Returns: ExecutionPlan {
#   strategy: "graph_first",
#   indexes_used: ["graph", "temporal"],
#   estimated_claims: 450,
#   estimated_ms: 120
# }

# Run query and return execution statistics
frame, stats = db.profile(focal_entity="TREM2", depth=2, min_confidence=0.5)
# Returns: frame + ProfileStats {
#   actual_claims: 523,
#   actual_ms: 145,
#   index_hits: {graph: 523, embedding: 0, text: 0, temporal: 523},
#   cache_hits: 412,
#   fusion_strategy: "graph_first"
# }
```

### Time Travel

```python
# Snapshot at a point in time
frame = db.at("2026-01-15").query(focal_entity="TREM2")

# Trace downstream dependencies of a claim
downstream = db.trace_downstream(claim_id="abc123")
# Returns: list of claim IDs that depend on this claim via provenance chain
```

### Graph Statistics

```python
stats = db.stats()
# Returns:
# {
#   "total_claims": int,
#   "total_entities": int,
#   "entity_types": dict[str, int],       # type → count
#   "predicate_types": dict[str, int],    # type → count
#   "source_types": dict[str, int],       # type → count
#   "embedding_index_size": int,
# }
```

## Curator Interface

The curator is an application-layer process. It interacts with Attest through the same API, plus a few curator-specific patterns:

```python
# Configure curator (heuristic or LLM-powered)
db.configure_curator(model="heuristic")       # no LLM cost
db.configure_curator(model="groq")            # LLM triage via Groq
db.configure_curator(model="groq", api_key="...", env_path=".env")

# Triage and ingest claims through curator
result = db.curate(claims=[...], agent_id="pipeline_v1")
# Returns: CuratorResult { stored: list[str], skipped: list[str], flagged: list[str], errors: list[str] }
```

The curator does NOT get special APIs. It uses the same ingestion and query interfaces. Its claims are distinguishable by `source_id: "curator_v1"` and traceable via provenance chain.

## Insight Engine Interface

```python
# Bridge prediction: find semantically similar but structurally disconnected entity pairs
bridges = db.find_bridges(entity_type="protein", min_claims=2, top_k=50)
# Returns: list[BridgePrediction] { entity_a, entity_b, similarity, explanation }

# Vocabulary-driven gap identification
from attestdb.intelligence.insight_engine import build_expected_patterns  # enterprise package
from attestdb.intelligence.bio_vocabulary import PREDICATE_CONSTRAINTS    # enterprise package
patterns = build_expected_patterns(PREDICATE_CONSTRAINTS)
gaps = db.find_gaps(patterns, entity_type="gene", min_claims=1)
# Returns: list[GapResult] { entity_id, entity_type, missing_predicate_types, explanation }

# Confidence alerts: single-source entities, stale evidence, mixed quality
alerts = db.find_confidence_alerts(min_claims=2, stale_threshold=0)
# Returns: list[EntityConfidenceAlert] { entity_id, alert_type, explanation }

# Blindspot analysis: database-wide gap and vulnerability scan
blindspots = db.blindspots(min_claims=5)
# Returns: BlindspotMap { single_source_entities, knowledge_gaps, low_confidence_areas, unresolved_warnings }
```

### Additional Query Primitives

```python
# Check if path exists between two entities
db.path_exists(entity_a: str, entity_b: str, max_depth: int = 3) -> bool

# List entities with filters
db.list_entities(
    entity_type: str = None,
    min_claims: int = 0,
    offset: int = 0,
    limit: int | None = None,
) -> list[EntitySummary]

# Get claims for a specific entity
db.claims_for(
    entity_id: str,
    predicate_type: str = None,
    source_type: str = None,
    min_confidence: float = 0.0
) -> list[Claim]
```

### Embedding Similarity

```python
similarity = db.embedding_similarity(entity_a="TREM2", entity_b="BACE1") -> float
# Cosine similarity between entity embeddings (structural or averaged claim embeddings)
```

### Weak Foundations

```python
foundations = db.weak_foundations(
    max_source_confidence=0.3,
    min_downstream=5,
) -> list[Claim]
# Claims with low source confidence but many downstream dependents
```

## Entity Resolution

```python
# Resolve an entity ID through alias chain
canonical = db.resolve("q9nzc2")
# Returns: "trem2" (if same_as claim exists), or "q9nzc2" (if no alias)

# Get all claims asserting the same content (corroboration group)
claims = db.claims_by_content_id(content_id="abc123def...")
# Returns: list of Claim objects sharing this content_id
```

## Retraction

```python
result = db.retract(source_id="PMID:12345678", reason="Paper retracted by journal")
# Returns: RetractResult {
#   tombstoned: int,      # claims from this source
#   degraded: int,        # downstream claims marked provenance_degraded
#   recomputed: int       # claims whose confidence was recomputed
# }

# Retract with cascade: tombstone source claims + mark downstream as degraded
cascade = db.retract_cascade(source_id="paper_123", reason="Retracted")
# Returns: CascadeResult { source_retract: RetractResult, degraded_claim_ids, degraded_count }

# Trace all claims that depend on a specific claim
tree = db.trace_downstream(claim_id="abc123")
# Returns: DownstreamNode { claim_id, dependents: list[DownstreamNode] }
```

## Namespace Isolation

```python
# Set active namespace — all queries scoped to this namespace
db.set_namespace("team_a")

# Multi-namespace view
db.set_namespaces(["team_a", "team_b"])

# Get current namespace filter
db.get_namespaces()

# Clear namespace filter (see all claims)
db.set_namespace("")
```

## Change Feed

```python
# Cursor-based polling for new claims
claims, new_cursor = db.changes(since=cursor, limit=100)
# Respects active namespace filter
```

## Audit Log

```python
# Append-only JSONL mutation log with actor attribution
db.set_actor("researcher_jane")

# Query audit log
entries = db.audit_log(since=timestamp, event_type="claim_ingested", actor="researcher_jane", limit=100)
# Events: claim_ingested, batch_ingested, source_retracted, rbac_grant, fork_created, fork_merged
```

## RBAC

```python
# Per-namespace role-based access control
db.enable_rbac()
db.grant(principal="jane", namespace="team_a", role="writer")
db.revoke(principal="jane", namespace="team_a")

# Roles: admin (full), writer (ingest+query), reader (query only)
# Wildcard: namespace="*" grants access to all namespaces
# Enforced on: ingest(), ingest_batch(), retract()
```

## Fork / Merge

```python
# Create copy-on-write branch
fork = db.fork("experiment_branch")

# Work on fork independently...
fork.ingest(...)

# Merge new claims back with conflict detection
report = db.merge(fork, dry_run=False)
# Returns: MergeReport { unique_claims, shared_beliefs, conflicts, entity_overlap }
```

## Knowledge Topology

```python
# Compute topology (must be called before topics/density_map/cross_domain_bridges)
db.compute_topology()

# Get the topic hierarchy
topics = db.topics(level=2)                   # optional level filter
# Returns: list of TopicNode

# Query a topic — returns ContextFrames for entities in that community
topic_frames = db.query_topic(
    topic_id="community_42",                  # community ID from topics()
    depth=2,
    min_confidence=0.0,
    max_claims=500,
)
# Returns: list[ContextFrame]

# Cross-domain exploration
bridges = db.cross_domain_bridges(top_k=20)
# Returns: list[CrossDomainBridge]

# Knowledge density map
density = db.density_map()
# Returns: list[DensityMapEntry] {
#   topic: TopicNode,
#   internal_density: float,
#   boundary_density: float,
#   gap_score: float
# }
```

## Quality Report

```python
report = db.quality_report(stale_threshold=0, expected_patterns=patterns)
# Returns: QualityReport {
#   total_claims, total_entities, entity_type_counts,
#   single_source_entity_count, stale_entity_count,
#   gap_count, confidence_alert_count, avg_claims_per_entity,
#   source_type_distribution, predicate_distribution
# }
```

## Inquiry Tracking

```python
# Register an inquiry as a claim
db.ingest(
    subject=("trem2", "protein"),
    predicate=("inquiry", "inquiry"),
    object=("complement system", "pathway"),
    provenance={"source_type": "human_annotation", "source_id": "researcher_jane"},
    payload={"schema_ref": "inquiry", "data": {
        "question": "Does TREM2 interact with the complement system?",
        "priority": "high",
        "context": "Relevant to AD therapeutic hypothesis"
    }}
)

# Check if new data matches open inquiries
matches = db.check_inquiry_matches(subject_id="trem2", object_id="complement system")
# Returns: list of matching inquiry claim_ids
```

## Connector Library

```python
# Factory method — returns a Connector, call .run(db) to execute
conn = db.connect("slack", token="xoxb-...", save=True)
result = conn.run(db)

# Direct construction
from attestdb import SlackConnector  # requires enterprise package
conn = SlackConnector(token="xoxb-...")
result = conn.run(db)

# Database connector with column mapping
conn = db.connect("postgres",
    dsn="postgresql://user:pass@host/db",
    query="SELECT gene, relation, target FROM interactions",
    mapping={"subject": "gene", "predicate": "relation", "object": "target"},
)
result = conn.run(db)

# Continuous ingestion
db.sync("slack", interval=300)                 # poll every 5 minutes
```

Available connectors (21): `slack`, `teams`, `gmail`, `gdocs`, `postgres`, `mysql`, `notion`, `confluence`, `csv`, `sqlite`, `github`, `jira`, `zoho`, `linear`, `pagerduty`, `http`, `airtable`, `mongodb`, `elasticsearch`, `s3`, `google_sheets`.

`save=True` persists all connection parameters to a Fernet-encrypted token store (`{db_path}.tokens`). Requires the `cryptography` package.

Chat connectors (Slack, Teams) → `ingest_chat()`. Text connectors (Gmail, GDocs, Notion, Confluence, Zoho) → `ingest_text()`. SQL connectors (Postgres, MySQL) → `fetch()` yield claim dicts. File connectors (CSV, SQLite) → `fetch()` zero deps. API connectors (GitHub, Jira, Linear, PagerDuty, Airtable, HTTP) → `fetch()` structured claims. Data connectors (MongoDB, Elasticsearch, S3, Google Sheets) → `fetch()` with field mapping.

## Snapshot / Restore

```python
db.snapshot("./backup.attest")
restored = attestdb.restore("./backup.attest")
```

## Autodidact (Self-Learning Daemon)

```python
# Start background daemon — detects gaps, searches sources, ingests evidence
db.enable_autodidact(
    interval=3600,                    # seconds between cycles
    max_llm_calls_per_day=200,        # budget cap
    max_cost_per_day=1.00,            # USD cap
    search_fn=my_search,              # custom evidence source
    sources="auto",                   # or "none" to register manually
    connectors=["slack", "github"],   # search configured connectors
    gap_types=["single_source", "low_confidence", "decayed"],
)

# Monitor
status = db.autodidact_status()       # AutodidactStatus
history = db.autodidact_history(10)   # list[CycleReport]
db.autodidact_run_now()               # trigger immediate cycle

# Stop
db.disable_autodidact()
```

Each cycle: detect gaps via `generate_tasks()` → search evidence sources by priority → extract claims via `ingest_text()` → measure blindspot reduction → journal the cycle. Skips entities already researched with negative results (3+ failures).

## Confidence Decay

```python
# Time-weighted confidence — older evidence loses weight at query time
db.configure_decay(
    half_life_days=365,
    predicate_half_lives={"has_status": 30, "interacts_with": 730},
)

# Stored claims are never modified. Decay is computed at query time.
# The autodidact daemon prioritizes "decayed" entities for refresh.
```

Per-predicate half-lives: operational predicates (`has_status`: 30 days) decay fast, durable scientific knowledge (`interacts_with`: 730 days) decays slowly.

## Temporal Analysis

```python
# Detect when evidence patterns shifted
result = db.temporal_analyze("brca1", "regime_shifts")
# TemporalResult(num_shifts=2, regime_shifts=[RegimeShift(index=47, direction="up")])

result = db.temporal_analyze("brca1", "velocity")
# TemporalResult(velocity=VelocityStats(mean_velocity=0.23, max_velocity=1.4))

result = db.temporal_analyze("brca1", "cycles", min_period=7)
# TemporalResult(cycles=[DetectedCycle(period=7.0, power=0.89)])

# All three at once
result = db.temporal_summary("brca1")
```

Builds time series from entity claims (auto-bucketed by day/week/month), then runs:
- **Regime shifts**: AR(1) divergence on sliding window
- **Velocity**: gradient + exponential smoothing
- **Cycles**: Welch's PSD + peak detection

## Build Observability

```python
# Build manifest — per-source tracking for multi-source bulk builds
manifest = db.build_manifest()
report = manifest.latest_build()
# Returns: BuildReport { build_id, started_at, completed_at, builder, sources, total_claims }

# Source health — live LMDB counts + build history
health = db.source_health()
# Returns: list[SourceReport] per source with claim counts and build status
```

## Error Types

```python
class AttestError(Exception): pass
class ProvenanceError(AttestError): pass       # missing or invalid provenance
class VocabularyError(AttestError): pass       # unknown entity/predicate/source type
class SchemaValidationError(AttestError): pass # payload doesn't match registered schema
class PredicateConstraintError(AttestError): pass  # subject/object types don't match predicate
class DuplicateClaimError(AttestError): pass   # content-addressed ID already exists
class CircularProvenanceError(AttestError): pass  # provenance chain contains cycles
class DimensionalityError(AttestError): pass   # embedding dim doesn't match db config
class QueryBudgetExceeded(AttestError): pass   # query hit max_claims or max_tokens limit
```

## Implementation Summary

| API Surface | Status | Notes |
|-------------|--------|-------|
| `db.open()` | ✅ Built | Creates `.attest` LMDB file |
| `db.ingest()` | ✅ Built | 13 validation rules, Rust engine |
| `db.ingest_batch()` | ✅ Built | Rust batch append, 1.3M claims/sec |
| `db.ingest_text()` | ✅ Built | LLM extraction + optional curator |
| `db.query()` → ContextFrame | ✅ Built | BFS traversal, confidence/source filters |
| `db.search()` | ✅ Built | HNSW via usearch |
| `db.explain()` / `db.profile()` | ✅ Built | Query plan + execution stats |
| `db.at()` time travel | ✅ Built | Native temporal index |
| `db.retract()` / `db.retract_cascade()` | ✅ Built | Rust-native status overlay, BFS cascade |
| `db.curate()` | ✅ Built | 7 LLM providers + heuristic mode |
| `db.quality_report()` | ✅ Built | Single-pass quality scan |
| `db.find_bridges()` | ✅ Built | 3-phase bridge prediction |
| `db.find_gaps()` | ✅ Built | Vocabulary-driven gap detection |
| `db.find_confidence_alerts()` | ✅ Built | Provenance alert scan |
| `db.blindspots()` | ✅ Built | Database-wide gap + vulnerability scan |
| `db.trace_downstream()` | ✅ Built | BFS provenance tree |
| `db.check_inquiry_matches()` | ✅ Built | Filter open inquiries |
| `db.topics()` / `db.query_topic()` | ✅ Built | Leiden community detection |
| `db.density_map()` | ✅ Built | Statistics over topic hierarchy |
| `db.cross_domain_bridges()` | ✅ Built | Topology-aware bridge entities |
| `db.path_exists()` | ✅ Built | BFS graph traversal |
| `db.connect()` | ✅ Built | 30 connectors + encrypted token store |
| `db.sync()` | ✅ Built | Continuous ingestion with ConnectorScheduler |
| `db.set_namespace()` | ✅ Built | Per-namespace isolation |
| `db.changes()` | ✅ Built | Cursor-based change feed |
| `db.audit_log()` | ✅ Built | Append-only JSONL mutation log |
| `db.enable_rbac()` / `db.grant()` | ✅ Built | Per-namespace role-based access control |
| `db.fork()` / `db.merge()` | ✅ Built | CoW branching with conflict detection |
| `db.snapshot()` / `restore()` | ✅ Built | Backup/recovery |
| `db.build_manifest()` | ✅ Built | Build observability + --resume |
| `db.source_health()` | ✅ Built | Live LMDB counts + build history |
| Predicate `symmetric` | ✅ Built | Bidirectional traversal |
| `db.ingest_texts()` (batch) | ✅ Built | Batch text ingestion |
| `db.text_search()` | ✅ Built | BM25 entity search enriched with claims |
| `db.hybrid_search()` | ✅ Built | RRF fusion of text + embedding search |
| `db.embedding_similarity()` | ✅ Built | Cosine similarity between entity embeddings |
| `db.weak_foundations()` | ✅ Built | Low-provenance claims with many downstream |
| `db.recompute_confidence()` | ✅ Built | Recompute confidence excluding sources |
| Named vector spaces | ✅ Built | MultiSpaceEmbeddingIndex with manifest |
| Predicate `transitive` | ✅ Built | Post-BFS transitive synthesis with composition |
| Predicate `inverse` | ✅ Built | Bidirectional inverse flipping at query time |
