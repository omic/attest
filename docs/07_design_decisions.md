# Attest: Design Decisions

Each section resolves one identified design risk. These decisions are binding — they constrain implementation in both Phase 1 and Phase 2. Engineers should treat disagreement with any decision as a blocking conversation, not a silent deviation.

---

## 1. Entity Identity

### The Problem

No canonical ID strategy. "TREM2", "Q9NZC2", "trem2", and "triggering receptor expressed on myeloid cells 2" could all become separate nodes in the graph. The insight engine can't find bridges between entities that are secretly the same entity under different names.

### The Decision

**Entity IDs are normalized lowercase strings. External IDs are the authority layer. Aliases are first-class.**

```
Entity identity has three layers:

1. canonical_id:  Normalized string. Lowercase, whitespace-collapsed, ASCII-transliterated.
                  "TREM2" → "trem2", "β-amyloid" → "beta-amyloid"
                  This is the graph node ID. Two claims about "TREM2" and "Trem2" 
                  hit the same node.

2. display_name:  Original casing preserved for human display.
                  Stored as entity property. First writer wins; curator can override.

3. external_ids:  Authoritative cross-references.
                  {"uniprot": "Q9NZC2", "ncbi_gene": "54209", "hgnc": "17761"}
                  Indexed for reverse lookup: "give me the entity with uniprot Q9NZC2"
```

**Normalization function (locked on day 1, never changes):**

```python
import unicodedata

def normalize_entity_id(raw: str) -> str:
    """Deterministic entity ID normalization. Once shipped, never changes."""
    # 1. Unicode normalize (NFKD decomposition)
    s = unicodedata.normalize("NFKD", raw)
    # 2. Lowercase
    s = s.lower()
    # 3. Collapse whitespace
    s = " ".join(s.split())
    # 4. Strip leading/trailing whitespace
    s = s.strip()
    # 5. Replace common Greek letters with spelled-out forms
    greek = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
             "ε": "epsilon", "κ": "kappa", "λ": "lambda", "μ": "mu",
             "τ": "tau", "ω": "omega"}
    for char, name in greek.items():
        s = s.replace(char, name)
    return s
```

**Alias mechanism:**

When the curator (or a human) discovers that two canonical_ids refer to the same real-world entity, they write an alias claim:

```python
db.ingest(
    subject=("q9nzc2", "protein"),        # the alias
    predicate=("same_as", "same_as"),     # built-in predicate
    object=("trem2", "protein"),          # the canonical target
    provenance={
        "source_type": "human_annotation",
        "source_id": "curator_v1",
        "method": "entity_resolution",
        "chain": []
    },
    confidence=1.0
)
```

`same_as` is a built-in predicate with special semantics:
- Transitive: if A same_as B and B same_as C, then A same_as C
- The query engine resolves aliases at read time. ContextFrame assembly transparently merges aliased entities.
- The alias graph is materialized as a union-find structure in memory (rebuilt from same_as claims on startup). O(1) lookup.
- Alias resolution is an index, not a mutation. Old claims are never rewritten.
- `db.resolve(entity_id)` returns the canonical target after following alias chains.

**What this means for Phase 1:**
- Normalization function runs on every entity ID at ingestion time
- Seeding pipelines (Hetionet, ChEMBL) normalize on import
- same_as predicate is available from day 1
- Union-find alias index is built in memory on db.open()
- ContextFrame assembly calls db.resolve() before traversal

**What this does NOT solve (deferred to Phase 2):**
- Fuzzy entity resolution ("triggering receptor expressed on myeloid cells 2" → "trem2") requires LLM. Phase 1 relies on exact normalized match + external_ids.
- Cross-type aliases ("TREM2 gene" vs "TREM2 protein") are distinct entities with a `encodes` relationship, not aliases.

---

## 2. Claim ID and Corroboration

### The Problem

The docs say content-addressed hashing provides "automatic dedup/corroboration" but include source_id in the hash. These goals contradict. If source_id is in the hash, two agents reporting the same finding produce different claim IDs (no automatic corroboration). If source_id is NOT in the hash, the second agent's report is rejected as a duplicate.

### The Decision

**Two separate identifiers. content_id for corroboration grouping. claim_id for unique identity.**

```
Claim:
  claim_id:    SHA-256(subject.canonical_id + predicate.id + object.canonical_id 
                       + provenance.source_id + provenance.source_type + timestamp)
               → Globally unique. Never collides. This is the primary key.

  content_id:  SHA-256(subject.canonical_id + predicate.id + object.canonical_id)
               → Same for all claims asserting the same relationship.
               → The corroboration grouping key.
```

**Behavior on ingestion:**

```
1. Compute content_id from (subject, predicate, object)
2. Compute claim_id from (subject, predicate, object, source_id, source_type, timestamp)
3. Check: does this exact claim_id already exist?
   → Yes: DuplicateClaimError (exact same claim from same source at same time)
   → No: proceed
4. Check: do other claims with this content_id exist?
   → Yes: This is corroboration. Store the new claim. Increment corroboration count.
          Log: "Claim {claim_id} corroborates existing content {content_id} 
                (now {n} independent sources)"
   → No: First assertion of this content. Store normally.
```

**Corroboration count (used by confidence system):**

```python
def count_independent_sources(content_id: str) -> int:
    """Count claims with same content_id that have independent provenance."""
    claims = db.claims_by_content_id(content_id)
    # Build provenance DAG
    # Two claims are independent if neither appears in the other's provenance chain
    # and they don't share a common ancestor claim
    independent_groups = find_independent_groups(claims)
    return len(independent_groups)
```

**What changes in the schema:**

```
Claim:
  claim_id:    ContentHash     # unique per claim instance
  content_id:  ContentHash     # shared across corroborating claims
  subject:     EntityRef
  predicate:   PredicateRef
  object:      EntityRef | ScalarValue
  confidence:  f64 [0.0, 1.0]
  provenance:  ProvenanceChain
  embedding:   Option<Vec<f32>>
  payload:     Option<TypedPayload>
  timestamp:   i64
```

The content_id index is a hash map: content_id → list[claim_id]. This is the corroboration lookup.

---

## 3. Confidence Computation

### The Problem

The confidence function takes (claim, supporting_claims, contradicting_claims) but there's no definition of how supporting and contradicting claims are identified, when recomputation happens, or what the default weights are.

### The Decision

**Three tiers of confidence, computed differently. Phase 1 ships Tier 1 only.**

### Tier 1: Source-Type Confidence (Phase 1, day 1)

No corroboration, no contradiction, no LLM. Pure function of provenance metadata.

```python
# Locked defaults — configurable per vocabulary
SOURCE_TYPE_WEIGHTS = {
    # Built-in vocabulary
    "observation":           0.70,
    "computation":           0.50,
    "document_extraction":   0.60,
    "llm_inference":         0.30,
    "human_annotation":      0.90,
    # Bio vocabulary (registered by vertical)
    "experimental":          0.85,
    "crystallography":       0.95,
    "mass_spec":             0.80,
    "docking":               0.40,
    "literature":            0.65,
}

def tier1_confidence(claim) -> float:
    """Source-type weight only. No corroboration. No decay."""
    return SOURCE_TYPE_WEIGHTS.get(claim.provenance.source_type, 0.50)
```

**Why start here:** It's deterministic, fast (no queries needed), and provides meaningful signal. Experimental evidence ranks higher than computational predictions. This alone is more useful than uniform confidence. We iterate with eval data — if Tier 1 confidence doesn't correlate with expert assessments, we know we need Tier 2.

**When Tier 1 confidence is computed:** On ingestion. Stored as a field on the claim. Never recomputed (it only depends on source_type, which is immutable).

### Tier 2: Corroboration-Adjusted Confidence (Phase 1, week 2 if Tier 1 is insufficient)

Adds corroboration boost and recency decay. Requires content_id index.

```python
def tier2_confidence(claim, db) -> float:
    """Source weight + corroboration + recency."""
    base = SOURCE_TYPE_WEIGHTS.get(claim.provenance.source_type, 0.50)
    
    # Corroboration: how many independent sources assert the same content?
    n_independent = db.count_independent_sources(claim.content_id)
    # Diminishing returns: 1 source = 1.0x, 2 = 1.3x, 3 = 1.5x, 5+ = 1.7x
    corroboration_boost = 1.0 + 0.3 * math.log2(max(n_independent, 1))
    corroboration_boost = min(corroboration_boost, 1.7)  # cap
    
    # Recency: half-life decay
    age_days = (now() - claim.timestamp) / 86400
    recency = 0.5 ** (age_days / HALF_LIFE_DAYS)  # default HALF_LIFE_DAYS = 365
    
    return min(base * corroboration_boost * recency, 1.0)
```

**When Tier 2 confidence is computed:** Lazy, on read. When a ContextFrame is assembled, confidence is recomputed for all included claims. Results are cached with a dirty flag (invalidated when new claims with matching content_id are ingested).

**"Supporting claims" are now precisely defined:** Claims sharing the same content_id, with independent provenance (no shared ancestors in the provenance DAG).

### Tier 3: Contradiction-Aware Confidence (Phase 2)

Adds explicit contradiction detection. Requires curator to flag contradictions.

```python
def tier3_confidence(claim, db) -> float:
    """Full confidence with contradiction penalties."""
    t2 = tier2_confidence(claim, db)
    
    # Contradictions are explicit: curator writes claims with predicate "contradicts"
    contradicting_claims = db.claims_for(
        entity_id=claim.subject.id,
        predicate_type="contradicts"
    )
    # Filter to contradictions that target THIS claim's content
    relevant = [c for c in contradicting_claims 
                if c.object == claim.content_id or c.subject == claim.content_id]
    
    if not relevant:
        return t2
    
    # Penalty based on strength of contradicting evidence
    contradiction_strength = max(tier2_confidence(c, db) for c in relevant)
    # If contradicting evidence is stronger, pull confidence toward 0.5 (uncertain)
    if contradiction_strength > t2:
        return t2 * 0.5  # significant penalty
    else:
        return t2 * 0.85  # mild penalty — contradiction exists but is weaker
```

**"Contradicting claims" are now precisely defined:** Claims where the curator has written a `contradicts` predicate linking two content_ids. Contradiction is NOT automatically detected from the graph structure — the curator must explicitly flag it. This is correct because contradiction is semantic ("A inhibits B" contradicts "A activates B") and can't be reliably determined from predicates alone.

**`contradicts` is a built-in predicate** with special semantics:
- Subject and object are content_ids (not entity IDs)
- The confidence system checks for contradictions when computing Tier 3
- The ContextFrame includes contradictions in its output

### Retraction (first-class operation)

```python
db.retract(source_id="PMID:12345678", reason="Paper retracted by journal")
```

What this does:
1. Finds all claims with provenance.source_id == "PMID:12345678"
2. Sets their status to "tombstoned" (they remain in the log but are excluded from indexes)
3. Finds all claims whose provenance.chain includes any tombstoned claim
4. Marks those as "provenance_degraded" (still active, but flagged)
5. Triggers confidence recomputation for all provenance_degraded claims
6. Writes a retraction claim for audit:
   ```
   subject: ("PMID:12345678", "document")
   predicate: ("retracted", "retracted")  # built-in
   object: (reason, ScalarValue)
   provenance: {source_type: "human_annotation", source_id: "retraction_system"}
   ```
7. Returns: RetractResult { tombstoned: int, degraded: int, recomputed: int }

`retracted` is a built-in predicate. `provenance_degraded` is a claim status alongside active/archived/tombstoned.

---

## 4. ContextFrame Assembly Algorithm

### The Problem

The schema and query parameters are defined, but there's no algorithm for turning a focal entity + parameters into a coherent, token-budgeted frame.

### The Decision

**A concrete, deterministic algorithm that both engineers can implement and test.**

### Input

```python
frame = db.query(
    focal_entity="trem2",
    depth=2,
    min_confidence=0.5,
    exclude_source_types=["llm_inference"],
    max_claims=500,
    max_tokens=4000
)
```

### Algorithm

```
STEP 1: RESOLVE FOCAL ENTITY
  canonical = db.resolve(normalize("trem2"))
  If not found → EntityNotFoundError
  entity = db.get_entity(canonical)

STEP 2: COLLECT CANDIDATE CLAIMS (graph traversal)
  candidates = []
  visited_entities = {canonical}
  frontier = {canonical}
  
  For hop in 1..depth:
    next_frontier = {}
    For entity_id in frontier:
      claims = db.claims_for(entity_id)
      For claim in claims:
        If claim.confidence < min_confidence → skip
        If claim.provenance.source_type in exclude_source_types → skip
        If claim.status != "active" → skip
        candidates.append((claim, hop))
        other = claim.object if claim.subject == entity_id else claim.subject
        If other not in visited_entities:
          next_frontier.add(other)
          visited_entities.add(other)
    frontier = next_frontier
  
  # Hard cap: if candidates > max_claims, keep top max_claims by score (step 3)

STEP 3: SCORE AND RANK CANDIDATES
  For each (claim, hop) in candidates:
    score = claim.confidence                # base: confidence
           * (1.0 / hop)                    # decay: closer hops score higher
           * source_diversity_bonus(claim)   # bonus: if this claim adds a new source_type
           * recency_bonus(claim)            # bonus: newer claims score higher (7-day half-life)
  
  Sort candidates by score descending.
  Truncate to max_claims.

STEP 4: GROUP BY RELATIONSHIP
  relationships = {}  # key: (subject_canonical, predicate, object_canonical)
  For claim in ranked_candidates:
    key = (claim.subject.canonical_id, claim.predicate.id, claim.object.canonical_id)
    If key not in relationships:
      relationships[key] = Relationship(
        predicate=claim.predicate.id,
        target=entity_summary(other_entity(claim, focal)),
        confidence=claim.confidence,
        n_independent_sources=1,
        source_types=[claim.provenance.source_type],
        latest_claim_timestamp=claim.timestamp,
        payload=claim.payload
      )
    Else:
      rel = relationships[key]
      # Merge: take max confidence, union source types, count independent sources
      rel.confidence = max(rel.confidence, claim.confidence)
      rel.n_independent_sources = count_independent_sources(claim.content_id)
      rel.source_types = list(set(rel.source_types + [claim.provenance.source_type]))
      rel.latest_claim_timestamp = max(rel.latest_claim_timestamp, claim.timestamp)
      If claim.payload and not rel.payload:
        rel.payload = claim.payload

STEP 5: EXTRACT QUANTITATIVE DATA
  quantitative = []
  For claim in ranked_candidates:
    If claim.payload and claim.payload.schema_ref in QUANTITATIVE_SCHEMAS:
      quantitative.append(QuantitativeClaim(
        predicate=claim.predicate.id,
        target=other_entity(claim, focal).id,
        value=claim.payload.data["value"],
        unit=claim.payload.data["unit"],
        metric=claim.payload.data.get("metric", ""),
        source_type=claim.provenance.source_type,
        confidence=claim.confidence
      ))
  # QUANTITATIVE_SCHEMAS: payload schemas whose structure matches 
  # {value: number, unit: string, ...}. Registered on schema creation.

STEP 6: FIND CONTRADICTIONS
  contradictions = []
  For content_id in unique_content_ids(ranked_candidates):
    contra_claims = db.claims_for_content(content_id, predicate_type="contradicts")
    For contra in contra_claims:
      contradictions.append(Contradiction(
        claim_a=contra.subject,  # content_id A
        claim_b=contra.object,   # content_id B
        description=contra.payload.data.get("description", ""),
        evidence_a=count_independent_sources(contra.subject),
        evidence_b=count_independent_sources(contra.object),
        status=infer_status(contra)  # "a_preferred" if evidence_a >> evidence_b, etc.
      ))

STEP 7: TOKEN BUDGET ENFORCEMENT
  # Serialize frame to estimate token count
  # If over budget, remove lowest-scored relationships until under budget
  # Quantitative data and contradictions are never removed (high information density)
  serialized = serialize(frame)
  While token_count(serialized) > max_tokens:
    Remove the lowest-scored relationship from frame.direct_relationships
    serialized = serialize(frame)

STEP 8: ASSEMBLE
  Return ContextFrame(
    focal_entity=entity_summary(entity),
    direct_relationships=sorted(relationships.values(), by=confidence, desc),
    quantitative_data=quantitative,
    contradictions=contradictions,
    knowledge_gaps=[],  # empty unless insight engine provides them (see below)
    provenance_summary=compute_source_distribution(ranked_candidates),
    claim_count=len(ranked_candidates),
    confidence_range=(min_conf, max_conf)
  )
```

### Knowledge Gaps (Optional Overlay)

knowledge_gaps is NOT computed by the query engine. It is an optional enrichment provided by the insight engine:

```python
# Without insight engine: gaps is empty
frame = db.query(focal_entity="trem2", depth=2)
assert frame.knowledge_gaps == []

# With insight engine: gaps are provided as a post-processing step
frame = db.query(focal_entity="trem2", depth=2)
frame.knowledge_gaps = insight_engine.identify_gaps(frame)
```

The insight engine is the only component that writes knowledge_gaps. The query engine never generates them. This removes the dependency: Engineer 2 builds the query engine without any insight engine code.

### Contradiction Status

Inferred automatically from evidence counts:

```python
def infer_status(contradiction) -> str:
    a_count = count_independent_sources(contradiction.subject)
    b_count = count_independent_sources(contradiction.object)
    ratio = max(a_count, b_count) / max(min(a_count, b_count), 1)
    if ratio >= 3:
        return "a_preferred" if a_count > b_count else "b_preferred"
    return "unresolved"
```

No curator or human input needed for initial status. Curator can override by writing a claim with predicate "contradiction_resolved".

---

## 5. Kuzu → Rust Migration

### The Decision

**Claim ID computation is identical in both phases. Migration is a tested, first-class operation.**

Locked on day 1, shared between both engineers:

```python
# This function MUST produce identical output in Python (Phase 1) and Rust (Phase 2)
def compute_claim_id(subject_canonical: str, predicate_id: str, object_canonical: str,
                     source_id: str, source_type: str, timestamp: int) -> str:
    """SHA-256 of canonical inputs. Encoding: UTF-8, pipe-delimited."""
    payload = f"{subject_canonical}|{predicate_id}|{object_canonical}|{source_id}|{source_type}|{timestamp}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def compute_content_id(subject_canonical: str, predicate_id: str, object_canonical: str) -> str:
    """SHA-256 of content only. Same encoding."""
    payload = f"{subject_canonical}|{predicate_id}|{object_canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

**Migration strategy (Engineer 2, week 5-6, not week 7-8):**

1. Export: Kuzu → newline-delimited JSON (one claim per line, all fields)
2. Validate: recompute claim_id and content_id for every exported claim, assert they match
3. Import: JSON → Rust engine claim log (batch mode, deferred index build)
4. Verify: run Phase 1 eval harness on migrated data, assert identical metrics (±0.01)

The migration test is part of the CI suite from the moment the Rust engine accepts its first claim.

---

## 6. Evaluation Harness Validity

### The Decision

**Two eval tracks. Hetionet for automated regression. Omic data for real-world validity.**

### Track A: Hetionet Holdout (automated, runs on every commit)

Purpose: regression detection. If a code change breaks insight engine recall, catch it immediately.

Limitations (acknowledged): Hetionet is clean, curated, and has specific topology. Performance on Hetionet does NOT predict performance on noisy real-world data.

Target: >15% edge recovery in top 1000 predictions. This is a regression threshold, not a success metric.

### Track B: Omic Reality Test (weekly, human-evaluated)

Purpose: actual product validation. Does the system help biologists?

```
Week 1:
  - Seed 2 weeks of historical Omic agent outputs (curator processes them)
  - Biologists define 10 "known but non-obvious" relationships they'd want the system to surface
  - Biologists define what makes a bridge prediction "worth investigating" (actionability criteria):
    1. The prediction connects two entities from different subfields
    2. There is a plausible mechanism (not just statistical co-occurrence)  
    3. A wet-lab experiment could validate or refute it within 3 months
    4. The prediction is not already known to the biologist

Week 2:
  - Insight engine generates top 50 predictions
  - Biologists rate each: "would investigate" / "already known" / "implausible" / "interesting but not actionable"
  - Success metric: ≥10 of top 50 rated "would investigate" (20%)
  - This is the REAL go/no-go metric, not Hetionet recall
```

Track B result overrides Track A. If Hetionet recall is >15% but biologists rate 0/50 predictions as "would investigate," the insight engine has failed regardless.

---

## 7. Engineer Integration

### The Decision

**Day 1 integration smoke test. Shared test fixtures. Escape hatch for Phase 1.**

### Day 1 (first 4 hours)

Both engineers together:

1. Set up shared repo with monorepo structure:
   ```
   attestdb/
     core/          # shared types, normalization function, hash functions
     infrastructure/ # Engineer 2: Kuzu layer, ingestion, query engine
     intelligence/   # Engineer 1: curator, insight engine, eval harness
     tests/
       fixtures/    # shared test data
       integration/ # end-to-end tests
   ```

2. Implement and test the three shared functions:
   - `normalize_entity_id()` — entity normalization
   - `compute_claim_id()` — claim identity
   - `compute_content_id()` — corroboration grouping

3. Create 100 synthetic claims as a test fixture (manually written, covering edge cases: duplicate content_ids, provenance chains, aliases, quantitative payloads)

4. Run end-to-end: ingest 100 claims → query ContextFrame for 3 focal entities → assert frame structure is correct

5. Set up CI: integration tests run on every commit to either engineer's code

### Escape Hatch

Phase 1 includes a raw query method for unblocking:

```python
# When Engineer 1 needs a query pattern not yet in the API
results = db.raw_query("MATCH (a:Entity)-[c:Claim]->(b:Entity) WHERE ...")

# This is Kuzu-specific and intentionally NOT in the API spec.
# It exists only in Phase 1 to unblock development.
# Every raw_query usage is tracked and must be replaced with an API method before Phase 2.
```

### Daily Integration

Every day, both engineers run the integration test suite. If it breaks, that's the priority fix — not whatever feature they're working on.

---

## 8. Complex Biology Claims

### The Problem

"TREM2 expression is upregulated 2.3-fold in microglia from AD patients compared to controls (p=0.001, n=45) as measured by scRNA-seq" doesn't decompose cleanly into a single triple.

### The Decision

**Context-qualified entities + structured payloads. Accept information loss in the triple; preserve detail in the payload.**

The triple captures the core assertion. The payload captures the full context.

```python
db.ingest(
    subject=("trem2", "gene"),
    predicate=("upregulated_in", "upregulated_in"),
    object=("alzheimers_disease", "disease"),
    provenance={
        "source_type": "experimental",
        "source_id": "PMID:98765432",
        "method": "scrna_seq",
        "chain": []
    },
    confidence=0.85,
    payload={
        "schema": "differential_expression",
        "data": {
            "fold_change": 2.3,
            "direction": "up",
            "p_value": 0.001,
            "sample_size": 45,
            "cell_type": "microglia",
            "comparison": "AD_vs_control",
            "method": "single_cell_rna_sequencing"
        }
    }
)
```

### Rules for Triple Decomposition

```
1. Subject: the measured/affected entity (gene, protein, compound)
2. Object: the context entity (disease, tissue, cell type, organism)
3. Predicate: the relationship type (upregulated_in, binds, inhibits)
4. Everything else: payload

If there are multiple context dimensions (cell type AND disease AND tissue),
pick the most specific as the object. Put the rest in payload.

Priority for object selection:
  disease > cell_type > tissue > organism > condition

The curator applies these rules. They are documented in the bio vocabulary plugin.
```

### Context-Qualified Entities (when needed)

Sometimes the same gene in different contexts is genuinely a different biological entity:

```python
# TREM2 in microglia vs TREM2 in macrophages may have different functions
# Use context-qualified entity ID ONLY when the context changes the entity's behavior

# Default: same entity, different context in payload
("trem2", "gene") + payload.cell_type = "microglia"
("trem2", "gene") + payload.cell_type = "macrophage"

# Exception: when the scientific community treats them as distinct
# (e.g., splice variants, tissue-specific isoforms)
("trem2_isoform_1", "protein")  # distinct entity
("trem2_isoform_2", "protein")  # distinct entity
```

The rule: if two PubMed papers would use the same gene symbol to refer to the entity, it's the same entity. If they'd use different symbols, it's a different entity.

### Standard Quantitative Payload Schemas (ship with bio vocabulary)

```python
# These are registered by the bio vocabulary plugin, not ad-hoc

"differential_expression": {
    "fold_change": float, "direction": "up"|"down", "p_value": float,
    "sample_size": int, "cell_type": str?, "comparison": str, "method": str
}

"binding_affinity": {
    "metric": "Kd"|"Ki"|"IC50"|"EC50", "value": float, "unit": "nM"|"uM"|"mM"|"M",
    "conditions": {"temperature": float?, "pH": float?, "buffer": str?}
}

"clinical_outcome": {
    "metric": str, "value": float, "ci_lower": float?, "ci_upper": float?,
    "p_value": float?, "sample_size": int, "population": str
}

"enzyme_activity": {
    "metric": "Km"|"Vmax"|"kcat"|"kcat_over_km", "value": float,
    "unit": str, "attestdb": str, "conditions": {}
}
```

---

## 9. Append-Only + Entity Merge

### The Decision

**Entity alias table is a derived index. Union-find in memory. Resolved at query time.**

```
Data flow:

1. Curator discovers "q9nzc2" and "trem2" are the same entity
2. Curator writes same_as claim (see section 1)
3. On next query, alias index is consulted:
   - db.query(focal_entity="q9nzc2") → resolves to "trem2" → returns frame for "trem2"
   - All claims referencing "q9nzc2" are included in the frame

4. Old claims are NEVER rewritten. The claim log still contains ("q9nzc2", "protein").
   The graph index maps both "q9nzc2" and "trem2" to the same node.

5. The alias index is rebuilt from same_as claims on startup (fast: linear scan of same_as claims).
   It is NOT part of the claim log — it's a derived structure, like the graph index.
```

**Merge semantics:**

```python
# When two entities are merged via same_as:
# - display_name: kept from the target (the one pointed TO by same_as)
# - external_ids: union of both entities' external_ids
# - claims: all claims referencing either canonical_id are included in queries
# - entity_type: must match. same_as between different types is rejected.
#   ("trem2", "gene") same_as ("trem2", "protein") → PredicateConstraintError
#   Use "encodes" predicate instead for cross-type relationships.
```

**Undo mechanism:**

To split incorrectly merged entities, write a new claim:

```python
db.ingest(
    subject=("q9nzc2", "protein"),
    predicate=("not_same_as", "not_same_as"),  # built-in, overrides same_as
    object=("trem2", "protein"),
    provenance={...}
)
```

The alias index resolves conflicts by timestamp: most recent same_as or not_same_as wins.

---

## 10. Retraction

Solved in section 3 (Confidence Computation) under "Retraction (first-class operation)."

Summary: `db.retract(source_id, reason)` tombstones all claims from that source, marks downstream claims as "provenance_degraded", triggers cascading confidence recomputation, and writes an audit claim.

---

## Summary: What Changes in Existing Docs

All changes below have been applied to the architecture, API spec, implementation plan, competitive landscape, and overview documents.

### Claim Schema (02_architecture.md) ✅
- Added `content_id` field for corroboration grouping
- Added `status` field: active | archived | tombstoned | provenance_degraded
- Changed EntityRef.id: normalized canonical string with locked normalization function
- Added `display_name` to EntityRef
- Added `organization` to ProvenanceChain for future federated sharing

### Built-in Vocabulary (02_architecture.md) ✅
- Added built-in predicates: `same_as`, `not_same_as`, `contradicts`, `contradiction_resolved`, `retracted`, `inquiry`
- Documented special engine semantics for each

### Type System (02_architecture.md) ✅
- Added predicate logical properties: `symmetric`, `transitive`, `inverse`
- Phase 1 implements symmetric only; transitive and inverse are Phase 2

### Confidence System (02_architecture.md) ✅
- Replaced underspecified pluggable function with three-tier system
- Added retraction as first-class operation

### Knowledge Topology (02_architecture.md) ✅
- Added full topology section: emergent topic hierarchy, community detection, density mapping
- TopicNode schema, computation algorithm, bio + DevOps examples
- Cross-domain connector identification

### Query Interface (02_architecture.md, 06_api_spec.md) ✅
- ContextFrame updated with: `narrative`, `topic_membership`, `open_inquiries` fields
- Added TopicContextFrame for domain-level queries
- Added `db.query_topic()`, `db.cross_domain()`, `db.topics()`, `db.density_map()`
- Added `db.open_inquiries()` for inquiry tracking

### Ingestion (02_architecture.md, 06_api_spec.md) ✅
- Added text-to-claims pipeline: `db.ingest_text()` with LLM extraction + curator
- Added inquiry ingestion with `inquiry` predicate

### Overview (01_overview.md) ✅
- Reframed from "evidence engine" to "reality model infrastructure"
- Added 4th differentiator: emergent knowledge topology
- Added three worked examples: biology, DevOps, investment research

### Competitive Landscape (04_competitive_landscape.md) ✅
- Updated positioning from "evidence engine" to "reality model engine"
- Added topology and cross-domain discovery as differentiators
- Updated defensibility with federated sharing

### Implementation Plan (05_implementation_plan.md) ✅
- Day 1: shared functions, synthetic fixture, CI
- Week 5-6: topology computation, text ingestion pipeline
- Phase 2 exit criteria: topology computation speed, text extraction, predicate semantics
- Launch checklist: topology demo, DevOps example, demo video
- Week 5-6 (not 7-8): migration tooling
- Track B eval: "would investigate" criteria defined with biologists in week 1
