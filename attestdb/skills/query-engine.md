# Using the Query Engine

## Core Concepts

The query engine performs BFS traversal of the claim graph, returning `ContextFrame` objects that describe entity neighborhoods.

## Basic Queries

```python
from attestdb.infrastructure.query_engine import QueryEngine

qe = QueryEngine(store)
frame = qe.context("TP53", depth=2, limit=100)

# ContextFrame.focal_entity is an EntitySummary
print(frame.focal_entity.id)

# Use .direct_relationships (not .claims)
for rel in frame.direct_relationships:
    print(rel.predicate, rel.target)
```

## Contradiction Detection

When detecting contradictions, use sorted pair keys to deduplicate:

```python
# Two claims: A→B contradicts and B→A contradicts are the SAME contradiction
key = tuple(sorted([claim_a.id, claim_b.id]))
```

Without sorting, you get duplicate contradictions in opposite directions.

## Performance

- Cap depth-2 expansion to 30 first-hop targets
- Default view: 20 entities (min_claims=100)
- `claims_for()` on high-degree entities is expensive — always pass `limit=N`
- Use `entity_predicate_counts()` for counts without materializing

## Ask Engine (v2)

Entity-first retrieval pipeline (2.8s on 85M claims):
1. Entity extraction from question (100ms)
2. Neighborhood intersection (1.7s)
3. LLM synthesis (1s)

Uses `requests` library for LLM calls, NOT `httpx` — avoids uvicorn deadlock.
