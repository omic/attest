# Attest: API Specification

This is the shared contract between the infrastructure and intelligence layers. Both engineers build against these interfaces from day 1. Implemented on the Rust engine.

## Database Lifecycle

```python
import attestdb

# Zero-config: works immediately with built-in vocabulary
db = attestdb.open("./my_evidence.db")

# Or configure embedding spaces
db = attestdb.open("./bio.db", embeddings={
    "structural": 768,
    "literature": 1024
})

# Or single embedding space (shorthand)
db = attestdb.open("./simple.db", embedding_dim=768)

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
    "transitive": False,
    "inverse": None,
    "expected_payload": "binding_affinity"
})

db.register_predicate("causes", {
    "subject_types": ["event", "entity"],
    "object_types": ["event", "entity"],
    "symmetric": False,
    "transitive": True,                       # A causes B, B causes C → A causes C
    "inverse": "caused_by",                   # auto-generated inverse predicate
    "expected_payload": None
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
    embedding=[0.1, 0.2, ...],                # optional — single unnamed space
    # OR named embeddings for multi-space databases:
    embeddings={                              # optional — one or more named spaces
        "structural": [0.1, 0.2, ...],
        "literature": [0.3, 0.4, ...],
    },
    payload={                                  # optional — validated against schema if predicate has expected_payload
        "schema": "binding_affinity",
        "data": {"metric": "Kd", "value": 5.2, "unit": "nM"}
    },
    external_ids={                            # optional — anchors entities to external databases
        "subject": {"uniprot": "Q9NZC2"},
        "object": {"uniprot": "P02649"}
    }
)
# Returns: claim_id (content-addressed hash)
# Raises: ProvenanceError, VocabularyError, SchemaValidationError, DuplicateClaimError
```

### Batch Ingestion

```python
results = db.ingest_batch(
    claims=[...],                             # list of claim dicts (same structure as single)
    mode="deferred_index"                     # skip per-claim index updates, rebuild at end
)
# Returns: BatchResult { ingested: int, duplicates: int, errors: list[Error] }
```

### Ingestion Validation Rules (enforced on every write)

1. Entity IDs are normalized via `normalize_entity_id()` before any other processing
2. `claim_id` computed from (subject + predicate + object + source_id + source_type + timestamp)
3. `content_id` computed from (subject + predicate + object) — used for corroboration grouping
4. If `claim_id` already exists → DuplicateClaimError
5. If other claims share `content_id` → log corroboration, increment count
6. `provenance.source_type` must exist in registered vocabulary (or built-in)
7. `subject.entity_type` and `object.entity_type` must exist in registered vocabulary (or built-in)
8. `predicate.predicate_type` must exist in registered vocabulary (or built-in)
9. If predicate has registered constraints, subject/object types must match
10. If payload references a schema, data is validated against it
11. If `provenance.chain` is non-empty, all referenced claim IDs must exist
12. Provenance chain must be a DAG (no circular references)
13. If embedding is provided, dimensionality must match database config for the named space

## Querying: ContextFrames

### Basic Query

```python
frame = db.query(
    focal_entity="TREM2",                    # entity ID
    depth=2,                                  # graph traversal hops
    min_confidence=0.5,                       # filter low-confidence claims
    exclude_source_types=["llm_inference"],   # contamination control
    topic="neurodegeneration",                # optional — restrict to claims within this topic
    max_claims=500,                           # query budget
    max_tokens=4000                           # token budget for serialized output
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
    knowledge_gaps: list[str]                 # from insight engine (empty if not running)
    narrative: str                            # from intelligence layer (empty if not running)
    topic_membership: list[TopicMembership]   # which topics this entity belongs to
    open_inquiries: list[Inquiry]             # outstanding questions about this entity
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
# Single embedding space
results = db.search(
    query_embedding=[0.1, 0.2, ...],
    entity_type="protein",                     # optional type filter
    min_confidence=0.5,
    top_k=20
)
# Returns: list of (entity_id, similarity_score, EntitySummary)

# Named embedding space
results = db.search(
    query_embedding=[0.1, 0.2, ...],
    embedding_space="structural",              # which named space to search
    entity_type="protein",
    top_k=20
)
```

### Text Search (BM25)

```python
results = db.text_search(
    query="CRISPR Cas9 gene editing",
    fields=["entity_name", "payload"],         # which fields to search (default: all indexed)
    entity_type="document",                    # optional type filter
    min_confidence=0.5,
    top_k=20,
    boost={"entity_name": 3.0}                 # optional field boosting (BM25F)
)
# Returns: list of (entity_id, bm25_score, EntitySummary)
```

### Hybrid Search

```python
results = db.hybrid_search(
    query="TREM2 binding partners in neurodegeneration",
    embedding_space="literature",              # for the vector component
    alpha=0.7,                                 # 0.0 = pure keyword, 1.0 = pure vector
    fusion="rrf",                              # "rrf" (reciprocal rank fusion) or "rsf" (relative score fusion)
    entity_type="protein",
    top_k=20
)
# Returns: list of (entity_id, fused_score, EntitySummary)
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

# Recompute confidence excluding specific sources
db.recompute_confidence(
    exclude_sources=["agent_7"],              # exclude by source_id
    exclude_source_types=["llm_inference"],   # exclude by type
    weights={"experimental": 1.0, "computational": 0.5}  # override source type weights
)
```

### Graph Statistics

```python
stats = db.stats()
# Returns:
# {
#   "total_claims": int,
#   "active_claims": int,
#   "archived_claims": int,
#   "tombstoned_claims": int,
#   "entity_count": int,
#   "entity_types": dict[str, int],       # type → count
#   "predicate_types": dict[str, int],    # type → count
#   "source_types": dict[str, int],       # type → count
#   "avg_confidence": float,
#   "embedding_index_size": int,
#   "graph_index_size": int,
#   "ingestion_rate": float,              # claims/sec (trailing average)
# }
```

## Curator Interface

The curator is an application-layer process. It interacts with Attest through the same API, plus a few curator-specific patterns:

```python
# Curator reads neighborhood to assess novelty
neighborhood = db.query(focal_entity=candidate_entity, depth=1, max_claims=100)

# Curator checks for existing contradictions
existing = db.search(query_embedding=claim_embedding, top_k=5)

# Curator writes curated claim (its own provenance)
db.ingest(
    subject=...,
    predicate=...,
    object=...,
    provenance={
        "source_type": "llm_inference",
        "source_id": "curator_v1",
        "method": "significance_triage",
        "chain": [original_agent_claim_id],   # traces back to the agent output
        "model_version": "haiku-20260101"
    }
)
```

The curator does NOT get special APIs. It uses the same ingestion and query interfaces. Its claims are distinguishable by `source_id: "curator_v1"` and traceable via provenance chain.

## Insight Engine Interface

Also application-layer, same API:

```python
# Bridge prediction: find semantically similar but structurally disconnected entity pairs
# This is a compound query the insight engine builds from primitives:

# 1. Get all entities of type X
entities = db.list_entities(entity_type="protein", min_claims=10)

# 2. For each pair, check structural connectivity
connected = db.path_exists(entity_a="TREM2", entity_b="APOE", max_depth=3)

# 3. Check embedding similarity
similarity = db.embedding_similarity(entity_a="TREM2", entity_b="BACE1")

# 4. Write prediction as a claim
db.ingest(
    subject=("TREM2", "protein"),
    predicate=("predicted_bridge", "relates_to"),
    object=("BACE1", "protein"),
    provenance={
        "source_type": "computation",
        "source_id": "insight_engine_v1",
        "method": "bridge_prediction",
        "chain": [],                          # derived from graph structure, not from specific claims
        "model_version": "v0.1"
    },
    confidence=0.3                            # low confidence — it's a prediction
)
```

### Additional Query Primitives (needed by insight engine)

```python
# Check if path exists between two entities
db.path_exists(entity_a: str, entity_b: str, max_depth: int) -> bool

# Get embedding similarity between entities
db.embedding_similarity(entity_a: str, entity_b: str) -> float

# List entities with filters
db.list_entities(
    entity_type: str = None,
    min_claims: int = 0,
    min_confidence: float = 0.0
) -> list[EntitySummary]

# Get claims for a specific entity
db.claims_for(
    entity_id: str,
    predicate_type: str = None,
    source_type: str = None,
    min_confidence: float = 0.0
) -> list[Claim]

# Confidence alerts: find claims with weak provenance that are upstream of many beliefs
db.weak_foundations(
    max_source_confidence: float = 0.3,      # source type weight threshold
    min_downstream: int = 5,                  # minimum downstream dependent claims
    max_age_months: int = 6
) -> list[Claim]
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
```

## Knowledge Topology

```python
# Get the topic hierarchy
topics = db.topics(level=2)
# Returns: list of TopicNode at the specified level

# Query a topic
topic_frame = db.query_topic(
    topic="neurodegeneration",              # topic label or ID
    level=2,                                # hierarchy level
    max_tokens=8000
)
# Returns: TopicContextFrame (see 02_architecture.md for schema)

# Cross-domain exploration
bridges = db.cross_domain(
    topic_a="neurodegeneration",
    topic_b="lipid metabolism",
    max_depth=3
)
# Returns: list of CrossDomainPath {
#   path: list[EntityRef],                  # entity chain connecting topics
#   confidence: float,                      # min confidence along path
#   novelty: float,                         # how unexpected this connection is
#   claim_ids: list[str]                    # claims forming the path
# }

# Knowledge density map
density = db.density_map(level=2)
# Returns: list of TopicDensity {
#   topic: TopicNode,
#   internal_density: float,                # claims per entity pair within topic
#   boundary_density: float,                # claims connecting to adjacent topics
#   gap_score: float                        # expected connections missing
# }

# Recompute topics (normally runs as background process)
db.recompute_topics()
```

Topic computation is Phase 2 (weeks 5-6). Phase 1 does not have topology queries.

## Text Ingestion

```python
results = db.ingest_text(
    text="The deployment of v2.3 caused a latency spike in the auth service.",
    source_type="document_extraction",
    source_id="slack://C04N8BXYZ/1234567890",
    extraction_model="sonnet",                # which LLM extracts claims
    auto_curate=True                          # run curator on extracted claims (default True)
)
# Returns: TextIngestionResult {
#   extracted: int,       # candidate claims extracted from text
#   stored: int,          # claims that passed curation and were stored
#   skipped: int,         # claims curator filtered out
#   errors: list[Error]
# }

# Batch text ingestion
results = db.ingest_texts(
    texts=[
        {"text": "...", "source_id": "slack://...", "source_type": "document_extraction"},
        {"text": "...", "source_id": "confluence://...", "source_type": "document_extraction"},
    ],
    extraction_model="sonnet",
    auto_curate=True
)
```

Text ingestion is Phase 2 (weeks 7-8). Phase 1 uses structured claim ingestion only.

## Inquiry Tracking

```python
# Record an open question
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

# List open inquiries
inquiries = db.open_inquiries(
    entity_id="trem2",                       # optional — filter to specific entity
    priority="high"                          # optional — filter by priority
)
# Returns: list of Inquiry { question, inquirer, priority, created_at, entity_pair }

# The insight engine monitors inquiries and alerts when evidence arrives
# that connects the inquiry's subject and object entities.
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

## Curation API

```python
# Configure curator (heuristic or LLM-powered)
db.configure_curator(model="heuristic")       # no LLM cost
db.configure_curator(model="groq")            # LLM triage via Groq
db.configure_curator(model="groq", api_key="...", env_path=".env")

# Triage and ingest claims through curator
result = db.curate(claims=[...], agent_id="pipeline_v1")
# Returns: CuratorResult { stored: list[str], skipped: list[str], flagged: list[str], errors: list[str] }

# Extract claims from text and ingest (optional curator triage)
result = db.ingest_text(text="...", source_id="PMID:12345", use_curator=True)
# Returns: ExtractionResult { claims, rejected, warnings, raw_count }
```

## Insight Engine API

```python
# Bridge prediction: find structurally disconnected but related entity pairs
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

## Provenance Tracing

```python
# Retract with cascade: tombstone source claims + mark downstream as degraded
cascade = db.retract_cascade(source_id="paper_123", reason="Retracted")
# Returns: CascadeResult { source_retract: RetractResult, degraded_claim_ids, degraded_count }

# Trace all claims that depend on a specific claim
tree = db.trace_downstream(claim_id="abc123")
# Returns: DownstreamNode { claim_id, dependents: list[DownstreamNode] }
```

## Inquiry Alerting

```python
# Register an inquiry
db.ingest_inquiry(question="...", subject=("drug_x", "compound"), object=("target_y", "protein"),
                  predicate_hint="inhibits")

# Check if new data matches open inquiries
matches = db.check_inquiry_matches(subject_id="drug_x", object_id="target_y")
matches = db.check_inquiry_matches(predicate_id="inhibits")
# Returns: list of matching inquiry claim_ids
```

## Connector Library [BUILT]

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
```

Available connectors: `slack`, `gmail`, `gdocs`, `postgres`, `mysql`, `notion`, `confluence`.

`save=True` persists all connection parameters to a Fernet-encrypted token store (`{db_path}.tokens`). Requires the `cryptography` package.

Chat connectors (Slack) override `run()` → `db.ingest_chat()`. Text connectors (Gmail, GDocs, Notion, Confluence) override `run()` → `db.ingest_text()`. SQL connectors (Postgres, MySQL) use `fetch()` → yield claim dicts.

## Implementation Summary

| API Surface | Rust Engine |
|-------------|-------------|
| `db.open()` | Creates `.attest` file |
| `db.ingest()` | Validates in Rust, writes to claim log |
| `db.query()` → ContextFrame | Native engine blending indexes |
| `db.search()` | HNSW search |
| `db.explain()` | Returns (ContextFrame, QueryProfile) |
| `db.at()` time travel | Native temporal index |
| `db.retract()` | Tombstones + audit trail |
| `db.retract_cascade()` | BFS provenance degradation |
| `db.curate()` | Curator triage + ingest |
| `db.ingest_text()` | LLM extraction + optional curator |
| `db.quality_report()` | Single-pass quality scan |
| `db.find_bridges()` | 3-phase bridge prediction |
| `db.find_gaps()` | Vocabulary-driven gap detection |
| `db.find_confidence_alerts()` | Provenance alert scan |
| `db.trace_downstream()` | BFS provenance tree |
| `db.check_inquiry_matches()` | Filter open inquiries |
| `db.topics()` / `db.query_topic()` | Leiden community detection |
| `db.density_map()` | Statistics over topic hierarchy |
| `db.cross_domain_bridges()` | Topology-aware bridge entities |
| `db.path_exists()` | CSR graph traversal |
| `db.connect()` | Connector factory (Slack, Gmail, GDocs, Postgres, MySQL, Notion, Confluence) |
| `db.ingest_batch()` | Rust batch append |
| Predicate `symmetric` | CSR bidirectional traversal |
| `db.text_search()` | Phase 3: BM25F inverted index |
| `db.hybrid_search()` | Phase 3: parallel fusion (RRF/RSF) |
| Predicate `transitive` | Phase 3: chain traversal |
| Predicate `inverse` | Phase 3: auto-generated at query time |
| Named vector spaces | Phase 3: multiple named HNSW indexes |
