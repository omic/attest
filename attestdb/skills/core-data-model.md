# Working with Claims and Ingestion

## Quick Start

1. Build a `ClaimInput` with subject, predicate, object, provenance, confidence
2. Call `db.ingest(claim_input)` — validates 13 rules, then persists
3. Use `db.claims_for(entity_id)` to query back

## ClaimInput

```python
from attestdb.core.types import ClaimInput

ci = ClaimInput(
    subject_id="TP53", subject_type="gene",
    predicate_id="causes", predicate_type="causal",
    object_id="apoptosis", object_type="process",
    provenance={"source_type": "paper", "source_id": "pmid:12345"},
    confidence=0.85,
)
result = db.ingest(ci)
```

**Critical:** `ClaimInput` does NOT accept `source_id=` or `source_type=` as kwargs. You must use the `provenance={}` dict. This mistake has been made in 7+ places across the codebase.

## Dual ID System

- `claim_id` = SHA-256(subject + predicate + object + source + timestamp) — unique per observation
- `content_id` = SHA-256(subject + predicate + object) — same across corroborating sources

Two papers both saying "TP53 causes apoptosis" produce different `claim_id`s but the same `content_id`. This powers corroboration counting.

## Ingestion Pipeline

`_validate_and_build()` → `_persist()`. Validation is pure (no side effects). 13 rules checked:
- Required fields present
- Entity types valid
- Predicate in vocabulary
- Confidence in [0.0, 1.0]
- Provenance has source_type + source_id
- No self-referential claims
- Timestamp reasonable

## Batch Ingestion

```python
from attestdb.core.types import ClaimInput

claims = [ClaimInput(...) for row in data]
result = db.ingest_batch(claims)
print(f"Ingested: {result.ingested}, Duplicates: {result.duplicates}")
```

`BatchResult.ingested` and `.duplicates` are **ints** (not lists).

## Querying

```python
# By entity
claims = db.claims_for(entity_id="TP53")

# By predicate
claims = db.claims_for_predicate("causes")

# By source
claims = db.claims_by_source_id("pmid:12345")
```

**RustStore API:** `claims_for(entity_id, predicate_type, source_type, min_confidence)` requires 4 args — not 1. Use `db.claims_for()` (Python wrapper) which fills defaults, or pass `None`/`0.0` for optional filters.

## Testing

- Test ingestion returns valid result with correct counts
- Test duplicate detection via content_id
- Test validation rejects bad claims (missing provenance, confidence > 1.0)
- Test query returns ingested claims
- Mock only external boundaries (LLM APIs, HTTP) — never mock the DB
