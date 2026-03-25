"""Configurable test data generator for Attest.

Produces vocabulary-compliant ClaimInput objects with deliberate
topological structures: clusters, bridges, causal chains,
corroboration groups, contradictions, provenance chains, and
temporal distribution patterns.

Usage:
    from tests.fixtures.generator import generate, generate_db, GraphSpec

    # Defaults: ~500 claims with interesting topology
    db = generate_db()

    # Custom spec
    spec = GraphSpec(clusters=5, causal_chains=10, scale=2)
    claims = generate(spec)
    db = generate_db(spec)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from attestdb.core.types import ClaimInput
from attestdb.core.vocabulary import (
    BUILT_IN_SOURCE_TYPES,
    CAUSAL_PREDICATES,
    OPPOSITE_PREDICATES,
    STANDARD_PREDICATES,
)

# Timestamp base: 2023-11-14 (matching existing fixture convention)
_TS_BASE = 1_700_000_000_000_000_000
_NS_PER_DAY = 86_400 * 1_000_000_000

# Predicates good for causal composition in predict()
_COMPOSABLE_CAUSAL = ["activates", "inhibits", "upregulates", "downregulates"]

# Weak predicates for bridges and general cluster edges
_BRIDGE_PREDICATES = ["associated_with", "interacts_with"]

# Entity name suffixes — cycle through for variety
_ENTITY_SUFFIXES = ["gene", "protein", "compound"]


@dataclass(frozen=True)
class GraphSpec:
    """Configuration for the data generator."""

    clusters: int = 3
    entities_per_cluster: int = 20
    bridge_count: int = 2
    causal_chain_depth: int = 3
    causal_chains: int = 5
    corroboration_rate: float = 0.15
    contradiction_pairs: int = 3
    single_source_ratio: float = 0.2
    temporal_span_days: int = 90
    source_types: int = 3
    provenance_chains: int = 3
    provenance_chain_depth: int = 3
    seed: int = 42
    scale: int = 1

    @property
    def _clusters(self) -> int:
        return self.clusters * self.scale

    @property
    def _entities_per_cluster(self) -> int:
        return self.entities_per_cluster * self.scale

    @property
    def _bridge_count(self) -> int:
        return self.bridge_count * self.scale

    @property
    def _causal_chains(self) -> int:
        return self.causal_chains * self.scale

    @property
    def _contradiction_pairs(self) -> int:
        return self.contradiction_pairs * self.scale

    @property
    def _provenance_chains(self) -> int:
        return self.provenance_chains * self.scale


def generate(spec: GraphSpec | None = None) -> list[ClaimInput]:
    """Generate vocabulary-compliant ClaimInput objects with deliberate topology.

    Returns a flat list. Deterministic given the seed in spec.
    Provenance chain claims have a '_chain_marker' key in their provenance
    dict — use generate_db() to resolve these into actual claim_id references.
    """
    if spec is None:
        spec = GraphSpec()

    rng = random.Random(spec.seed)
    claims: list[ClaimInput] = []
    ts_counter = [0]  # mutable counter for unique timestamps

    def _next_ts() -> int:
        ts_counter[0] += 1
        return _TS_BASE + ts_counter[0] * 1_000_000_000

    # ── Phase 1: Build pools ─────────────────────────────────────────

    # Source pool
    source_type_list = sorted(BUILT_IN_SOURCE_TYPES)[:spec.source_types]
    sources = []
    for i, st in enumerate(source_type_list):
        sources.append((f"src_{st}_{i}", st))

    # Entity pool: per-cluster lists
    entity_pool: list[list[str]] = []
    all_entities: list[str] = []
    for c in range(spec._clusters):
        cluster_entities = []
        for i in range(spec._entities_per_cluster):
            suffix = _ENTITY_SUFFIXES[i % len(_ENTITY_SUFFIXES)]
            name = f"c{c}_{suffix}_{i}"
            cluster_entities.append(name)
        entity_pool.append(cluster_entities)
        all_entities.extend(cluster_entities)

    # Early return for empty specs
    if not all_entities or not sources:
        return claims

    # ── Phase 2: Cluster internals ───────────────────────────────────

    std_preds = sorted(STANDARD_PREDICATES)

    for c in range(spec._clusters):
        entities = entity_pool[c]
        n = len(entities)
        # Target ~4 edges per entity → ~2*n edges per cluster
        num_edges = min(n * 2, n * (n - 1) // 2)
        pairs_used: set[tuple[str, str]] = set()

        for _ in range(num_edges):
            # Pick a random pair
            for _attempt in range(10):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a != b and (entities[a], entities[b]) not in pairs_used:
                    break
            else:
                continue

            subj, obj = entities[a], entities[b]
            pairs_used.add((subj, obj))
            pred = rng.choice(std_preds)
            src_id, src_type = rng.choice(sources)

            claims.append(ClaimInput(
                subject=(subj, "entity"),
                predicate=(pred, "relates_to"),
                object=(obj, "entity"),
                provenance={"source_type": src_type, "source_id": src_id},
                timestamp=_next_ts(),
            ))

    # ── Phase 3: Bridges ─────────────────────────────────────────────

    bridge_claims: list[ClaimInput] = []
    for i in range(spec._bridge_count):
        if spec._clusters < 2:
            break
        c1, c2 = rng.sample(range(spec._clusters), 2)
        subj = rng.choice(entity_pool[c1])
        obj = rng.choice(entity_pool[c2])
        pred = rng.choice(_BRIDGE_PREDICATES)

        claim = ClaimInput(
            subject=(subj, "entity"),
            predicate=(pred, "relates_to"),
            object=(obj, "entity"),
            provenance={
                "source_type": "computation",
                "source_id": f"bridge_src_{i}",
            },
            timestamp=_next_ts(),
        )
        bridge_claims.append(claim)
        claims.append(claim)

    # ── Phase 4: Causal chains ───────────────────────────────────────

    # Build chains that converge on shared targets for predict()
    # Pick a few shared targets so multiple chains reach them
    causal_chain_entities: list[list[str]] = []
    num_shared_targets = max(1, spec._causal_chains // 3)
    shared_targets = [rng.choice(all_entities) for _ in range(num_shared_targets)]

    for chain_idx in range(spec._causal_chains):
        # Pick a start entity from any cluster
        start = rng.choice(all_entities)
        target = shared_targets[chain_idx % num_shared_targets]

        chain_ents = [start]
        for hop in range(spec.causal_chain_depth - 1):
            if hop == spec.causal_chain_depth - 2:
                # Last hop goes to shared target
                chain_ents.append(target)
            else:
                # Intermediate entity
                mid = f"chain_{chain_idx}_mid_{hop}"
                chain_ents.append(mid)
                if mid not in all_entities:
                    all_entities.append(mid)

        causal_chain_entities.append(chain_ents)

        for hop in range(len(chain_ents) - 1):
            pred = rng.choice(_COMPOSABLE_CAUSAL)
            src_id, src_type = rng.choice(sources)
            claims.append(ClaimInput(
                subject=(chain_ents[hop], "entity"),
                predicate=(pred, "relates_to"),
                object=(chain_ents[hop + 1], "entity"),
                provenance={"source_type": src_type, "source_id": src_id},
                confidence=round(rng.uniform(0.7, 0.95), 2),
                timestamp=_next_ts(),
            ))

    # ── Phase 5: Corroboration ───────────────────────────────────────

    num_to_corroborate = int(len(claims) * spec.corroboration_rate)
    corroboration_indices = rng.sample(range(len(claims)), min(num_to_corroborate, len(claims)))

    for idx in corroboration_indices:
        original = claims[idx]
        num_copies = rng.randint(2, 3)
        for copy_i in range(num_copies):
            # Different source, same triple
            src_id, src_type = rng.choice(sources)
            claims.append(ClaimInput(
                subject=original.subject,
                predicate=original.predicate,
                object=original.object,
                provenance={
                    "source_type": src_type,
                    "source_id": f"{src_id}_corrob_{idx}_{copy_i}",
                },
                timestamp=_next_ts(),
            ))

    # ── Phase 6: Contradictions ──────────────────────────────────────

    # Find existing causal claims that have opposites
    causal_claims = [
        c for c in claims
        if c.predicate[0] in OPPOSITE_PREDICATES
    ]
    contradiction_count = min(spec._contradiction_pairs, len(causal_claims))
    contradiction_sources = rng.sample(causal_claims, contradiction_count) if causal_claims else []

    for i, original in enumerate(contradiction_sources):
        opposite_pred = OPPOSITE_PREDICATES[original.predicate[0]]
        claims.append(ClaimInput(
            subject=original.subject,
            predicate=(opposite_pred, "relates_to"),
            object=original.object,
            provenance={
                "source_type": "llm_inference",
                "source_id": f"contradiction_src_{i}",
            },
            confidence=round(rng.uniform(0.5, 0.7), 2),
            timestamp=_next_ts(),
        ))

    # ── Phase 7: Single-source entities ──────────────────────────────

    # Count how many entities already have all claims from one source
    entity_sources: dict[str, set[str]] = {}
    for c in claims:
        for eid in (c.subject[0], c.object[0]):
            entity_sources.setdefault(eid, set()).add(c.provenance["source_id"])

    already_single = sum(1 for srcs in entity_sources.values() if len(srcs) == 1)
    total_entities = len(entity_sources)
    target_single = int(total_entities * spec.single_source_ratio)
    needed = max(0, target_single - already_single)

    single_src_id, single_src_type = sources[0]
    for i in range(needed):
        ent = f"isolated_{i}"
        num_claims = rng.randint(2, 3)
        for j in range(num_claims):
            other = rng.choice(all_entities)
            claims.append(ClaimInput(
                subject=(ent, "entity"),
                predicate=("associated_with", "relates_to"),
                object=(other, "entity"),
                provenance={
                    "source_type": single_src_type,
                    "source_id": single_src_id,
                },
                timestamp=_next_ts(),
            ))
        all_entities.append(ent)

    # ── Phase 8: Temporal distribution ───────────────────────────────

    regime_shift_point = 0.6  # 60% through the window
    span_ns = max(spec.temporal_span_days, 1) * _NS_PER_DAY

    for claim in claims:
        if rng.random() < 0.3:
            # Cluster near regime shift point
            center = _TS_BASE + int(span_ns * regime_shift_point)
            jitter = int(_NS_PER_DAY * rng.gauss(0, 3))
            new_ts = center + jitter
        else:
            # Spread evenly across the window
            new_ts = _TS_BASE + int(rng.random() * span_ns)

        # Replace timestamp — ClaimInput is not frozen, reassign directly
        object.__setattr__(claim, "timestamp", new_ts)

    # ── Phase 9: Provenance chain markers ────────────────────────────

    # Exclude isolated entities so we don't contaminate single-source tests
    chain_pool = [e for e in all_entities if not e.startswith("isolated_")]

    for chain_idx in range(spec._provenance_chains):
        # Base claim
        subj = rng.choice(chain_pool)
        obj = rng.choice(chain_pool)
        while obj == subj:
            obj = rng.choice(chain_pool)

        base_prov = {
            "source_type": "observation",
            "source_id": f"prov_chain_{chain_idx}_base",
            "_chain_marker": f"chain_{chain_idx:03d}_hop_000",
        }
        claims.append(ClaimInput(
            subject=(subj, "entity"),
            predicate=("associated_with", "relates_to"),
            object=(obj, "entity"),
            provenance=base_prov,
            timestamp=_TS_BASE + int(span_ns * 0.1),
        ))

        prev_subj, prev_obj = subj, obj
        for hop in range(1, spec.provenance_chain_depth):
            # Each derived claim connects from the previous object onward
            new_obj = rng.choice(chain_pool)
            while new_obj == prev_obj:
                new_obj = rng.choice(chain_pool)

            derived_prov = {
                "source_type": "computation",
                "source_id": f"prov_chain_{chain_idx}_hop_{hop}",
                "_chain_marker": f"chain_{chain_idx:03d}_hop_{hop:03d}",
                # chain field will be populated by generate_db()
            }
            claims.append(ClaimInput(
                subject=(prev_obj, "entity"),
                predicate=("associated_with", "derived_from"),
                object=(new_obj, "entity"),
                provenance=derived_prov,
                timestamp=_TS_BASE + int(span_ns * (0.1 + 0.05 * hop)),
            ))
            prev_obj = new_obj

    return claims


def generate_db(
    spec: GraphSpec | None = None,
    db_path: str = ":memory:",
    embedding_dim: int | None = None,
) -> "AttestDB":
    """Generate claims and ingest them into a new AttestDB.

    Handles provenance chain resolution: chain claims are ingested
    sequentially so each hop can reference the previous claim's ID.
    """
    from attestdb.infrastructure.attest_db import AttestDB

    if spec is None:
        spec = GraphSpec()

    all_claims = generate(spec)

    # Partition: bulk claims vs provenance chain claims
    bulk_claims = []
    chain_claims = []
    for c in all_claims:
        if "_chain_marker" in c.provenance:
            chain_claims.append(c)
        else:
            bulk_claims.append(c)

    db = AttestDB(db_path, embedding_dim=embedding_dim)

    # Bulk ingest the non-chain claims
    if bulk_claims:
        db.ingest_batch(bulk_claims)

    # Sequential ingest for provenance chains
    if chain_claims:
        # Sort by marker to ensure correct ordering
        chain_claims.sort(key=lambda c: c.provenance["_chain_marker"])

        # Track claim_ids per chain for building provenance references
        chain_ids: dict[str, list[str]] = {}

        for claim in chain_claims:
            marker = claim.provenance["_chain_marker"]
            # Parse chain index from marker: "chain_001_hop_002"
            parts = marker.split("_")
            chain_key = parts[1]  # "001"
            hop_num = int(parts[3])  # 2

            # Build clean provenance (strip marker)
            clean_prov = {
                k: v for k, v in claim.provenance.items()
                if k != "_chain_marker"
            }

            # Add chain reference to prior hop's claim_id
            if hop_num > 0 and chain_key in chain_ids and chain_ids[chain_key]:
                clean_prov["chain"] = [chain_ids[chain_key][-1]]

            claim_id = db.ingest(
                subject=claim.subject,
                predicate=claim.predicate,
                object=claim.object,
                provenance=clean_prov,
                confidence=claim.confidence,
                timestamp=claim.timestamp,
                namespace=claim.namespace,
            )
            chain_ids.setdefault(chain_key, []).append(claim_id)

    return db
