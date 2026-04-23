"""Multi-hop bridge finder.

Given two disjoint anchor sets A and B, identify entities (or entity
pairs) whose presence creates paths between A and B, ranked by a
salience score that combines predicate weights, path length, and
source diversity.

General-purpose: useful for fraud rings (accounts ↔ accounts via shared
devices), lead-gen (prospect ↔ champion via common employers), drug
discovery (gene ↔ disease via shared pathways), investigations
(person ↔ person via shared associates/entities).

The primitive is ``find_bridges(db, anchors_a, anchors_b, ...)``. It is
distinct from the existing ``InsightEngineV1.find_bridges`` (which does
link prediction by embedding similarity across the full graph). This
one is anchor-set-to-anchor-set multi-hop traversal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from attestdb.core.predicate_salience import (
    DEFAULT_JOURNALISM_REGISTRY,
    PredicateSalienceRegistry,
)
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import Claim


@dataclass
class BridgeFinding:
    bridge_id: str
    bridge_label: str
    paths: list[list[Claim]] = field(default_factory=list)
    path_count: int = 0
    salience: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# Internal graph helpers
# ──────────────────────────────────────────────────────────────────────


def _neighbors(
    db,
    entity_id: str,
    *,
    min_edge_confidence: float,
    max_neighbors_per_entity: int,
    quality_filter: Optional[Callable[[Claim], bool]],
) -> list[tuple[str, Claim]]:
    """Return (neighbor_id, claim) pairs for one-hop expansion.

    Applies confidence threshold, optional user quality filter, and a
    per-entity cap to prevent hub blowups.
    """
    claims = db.claims_for(entity_id)
    out: list[tuple[str, Claim]] = []
    for c in claims:
        if c.confidence < min_edge_confidence:
            continue
        if quality_filter is not None and not quality_filter(c):
            continue
        # Determine the other endpoint relative to entity_id.
        if c.subject.id == entity_id:
            other = c.object.id
        elif c.object.id == entity_id:
            other = c.subject.id
        else:
            # Claim returned from claims_for but neither endpoint matches
            # — skip defensively.
            continue
        if other == entity_id:
            continue
        out.append((other, c))
        if len(out) >= max_neighbors_per_entity:
            break
    return out


def _display_label(db, entity_id: str) -> str:
    """Best-effort display label for an entity id."""
    try:
        ent = db.get_entity(entity_id)
    except Exception:
        return entity_id
    if ent is None:
        return entity_id
    name = getattr(ent, "display_name", "") or getattr(ent, "name", "")
    return name or entity_id


# ──────────────────────────────────────────────────────────────────────
# Public primitive
# ──────────────────────────────────────────────────────────────────────


def find_bridges(
    db,
    anchors_a: Iterable[str],
    anchors_b: Iterable[str],
    *,
    max_hops: int = 3,
    top_k: int = 20,
    quality_filter: Optional[Callable[[Claim], bool]] = None,
    exclude: Optional[Iterable[str]] = None,
    min_edge_confidence: float = 0.3,
    max_neighbors_per_entity: int = 200,
    registry: PredicateSalienceRegistry | None = None,
) -> list[BridgeFinding]:
    """Find entities that bridge anchor set A to anchor set B.

    A bridge is an entity X (not in A ∪ B) such that some path
    A → ... → X → ... → B exists within ``max_hops`` total edges.

    Ranking: sum over paths of (path_predicate_weight / path_length),
    multiplied by (1 + 0.3·log(1+source_diversity-1)) so well-sourced
    bridges outrank single-sourced ones at the same path count.
    """
    if registry is None:
        registry = DEFAULT_JOURNALISM_REGISTRY
    if max_hops < 2:
        raise ValueError("max_hops must be >= 2")

    anchors_a_set = {normalize_entity_id(a) for a in anchors_a}
    anchors_b_set = {normalize_entity_id(b) for b in anchors_b}
    if not anchors_a_set or not anchors_b_set:
        return []
    excluded = {normalize_entity_id(x) for x in (exclude or ())} | anchors_a_set | anchors_b_set

    # Step 1: BFS outward from A up to floor(max_hops/... ) — we need the
    # shortest path from any a∈A to candidate X, bounded by max_hops-1
    # (so there's still room for one edge to some b∈B).
    # We record the best (shortest, highest-weight) path to each node.

    # paths_from[side][node] = list of claim-chains (list[Claim])
    # Only keep shortest-length chains per node to bound memory.
    def _bfs(start_set: set[str], max_depth: int) -> dict[str, list[list[Claim]]]:
        frontier_chains: dict[str, list[list[Claim]]] = {s: [[]] for s in start_set}
        shortest_len: dict[str, int] = {s: 0 for s in start_set}
        all_chains: dict[str, list[list[Claim]]] = {s: [[]] for s in start_set}
        visited_at_depth = {s: 0 for s in start_set}

        current: dict[str, list[list[Claim]]] = {s: [[]] for s in start_set}
        for depth in range(1, max_depth + 1):
            next_layer: dict[str, list[list[Claim]]] = defaultdict(list)
            for node, chains in current.items():
                neigh = _neighbors(
                    db,
                    node,
                    min_edge_confidence=min_edge_confidence,
                    max_neighbors_per_entity=max_neighbors_per_entity,
                    quality_filter=quality_filter,
                )
                for other, claim in neigh:
                    if other in start_set:
                        continue
                    # Only record `other` if we haven't already reached it
                    # at a strictly shorter depth.
                    prev = shortest_len.get(other)
                    if prev is not None and prev < depth:
                        continue
                    for chain in chains:
                        # Avoid cycles within a single chain.
                        if any(
                            (cl.subject.id == other or cl.object.id == other)
                            for cl in chain
                        ):
                            continue
                        new_chain = chain + [claim]
                        next_layer[other].append(new_chain)
                        all_chains.setdefault(other, []).append(new_chain)
                        shortest_len.setdefault(other, depth)
            if not next_layer:
                break
            current = next_layer
        return all_chains

    # Budget depths so A-depth + B-depth + 1 (the bridge itself is one of
    # the nodes; edges span both sides) <= max_hops. We explore up to
    # max_hops-1 edges from each side. That covers 2-hop (1+1) and 3-hop
    # (1+2 or 2+1) cases.
    side_depth = max_hops - 1

    from_a = _bfs(anchors_a_set, side_depth)
    from_b = _bfs(anchors_b_set, side_depth)

    # Step 2: intersect. A candidate bridge is a node reached from both
    # sides whose total edge count <= max_hops.
    common = (set(from_a) & set(from_b)) - excluded

    findings_by_id: dict[str, BridgeFinding] = {}
    for node in common:
        a_chains = from_a.get(node, [])
        b_chains = from_b.get(node, [])
        combined_paths: list[list[Claim]] = []
        for ac in a_chains:
            for bc in b_chains:
                total_edges = len(ac) + len(bc)
                if total_edges == 0 or total_edges > max_hops:
                    continue
                # Reverse b-chain so the full path reads A → ... → X → ... → B.
                full = list(ac) + list(reversed(bc))
                combined_paths.append(full)
        if not combined_paths:
            continue

        # Salience: sum over paths of (avg predicate weight / path length),
        # scaled by source diversity across all claims in all paths.
        source_ids: set[str] = set()
        total = 0.0
        for path in combined_paths:
            if not path:
                continue
            weights = [registry.weight_of(cl.predicate.id) for cl in path]
            avg_w = sum(weights) / len(weights)
            total += avg_w / len(path)
            for cl in path:
                source_ids.add(cl.provenance.source_id)
        import math

        diversity_factor = 1.0 + 0.3 * math.log1p(max(0, len(source_ids) - 1))
        salience = total * diversity_factor

        findings_by_id[node] = BridgeFinding(
            bridge_id=node,
            bridge_label=_display_label(db, node),
            paths=combined_paths,
            path_count=len(combined_paths),
            salience=salience,
        )

    ranked = sorted(
        findings_by_id.values(),
        key=lambda f: (f.salience, f.path_count),
        reverse=True,
    )
    return ranked[:top_k]
