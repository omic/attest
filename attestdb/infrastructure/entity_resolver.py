"""Entity resolution — external ID index, fuzzy matching, and duplicate detection."""

from __future__ import annotations

import logging

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import EntitySummary, entity_summary_from_dict

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolves entity names against existing graph entities.

    Owns a reverse external_id index and a resolution cascade:
    1. Exact match on normalized name (conf=1.0)
    2. External ID match (conf=0.99)
    3. Text search fuzzy match if mode >= "fuzzy" (conf=0.5-0.95)

    Modes:
        "external_ids" — exact name + external ID lookup only
        "fuzzy" — adds text search + token overlap scoring
        "full" — same as fuzzy (reserved for future embedding-based resolution)
    """

    def __init__(self, store, mode: str = "external_ids"):
        self._store = store
        self._mode = mode
        # (namespace, ext_id) -> entity_id
        self._ext_id_index: dict[tuple[str, str], str] = {}
        self._built = False

    def build_index(self) -> None:
        """Scan all entities and build reverse external_id index.

        When multiple entities share the same (namespace, ext_id), all are tracked
        via _ext_id_collisions for duplicate detection.
        """
        self._ext_id_index.clear()
        # Track collisions: (namespace, ext_id) -> set of entity_ids
        self._ext_id_collisions: dict[tuple[str, str], set[str]] = {}
        raw_entities = self._store.list_entities()
        entities = [entity_summary_from_dict(d) for d in raw_entities]
        for entity in entities:
            ext_ids = entity.external_ids or {}
            for namespace, ext_id in ext_ids.items():
                key = (namespace, ext_id)
                if key in self._ext_id_index:
                    # Collision — track both entities
                    self._ext_id_collisions.setdefault(key, set()).add(self._ext_id_index[key])
                    self._ext_id_collisions[key].add(entity.id)
                self._ext_id_index[key] = entity.id
        self._built = True
        logger.info(
            "EntityResolver: indexed %d external_id mappings from %d entities (%d collisions)",
            len(self._ext_id_index), len(entities), len(self._ext_id_collisions),
        )

    def register_external_id(self, entity_id: str, namespace: str, ext_id: str) -> None:
        """Incrementally update index after upsert_entity."""
        self._ext_id_index[(namespace, ext_id)] = entity_id

    def resolve_by_external_id(self, namespace: str, ext_id: str) -> str | None:
        """Reverse lookup: (namespace, ext_id) -> entity_id."""
        return self._ext_id_index.get((namespace, ext_id))

    def resolve(
        self,
        name: str,
        entity_type: str = "",
        external_ids: dict[str, str] | None = None,
    ) -> tuple[str | None, float]:
        """Resolution cascade.

        Returns (existing_entity_id, confidence) or (None, 0.0).
        """
        normalized = normalize_entity_id(name)

        # 1. Exact match on normalized name
        raw = self._store.get_entity(normalized)
        if raw is not None:
            return entity_summary_from_dict(raw).id, 1.0

        # 2. External ID match
        if external_ids:
            for namespace, ext_id in external_ids.items():
                resolved = self.resolve_by_external_id(namespace, ext_id)
                if resolved is not None:
                    return resolved, 0.99

        # 3. Text search fuzzy match (modes: fuzzy, full)
        if self._mode in ("fuzzy", "full") and hasattr(self._store, "search_entities"):
            candidates = [
                entity_summary_from_dict(d)
                for d in self._store.search_entities(normalized, top_k=5)
            ]
            best_id = None
            best_score = 0.0
            for candidate in candidates:
                if candidate.id == normalized:
                    return candidate.id, 1.0
                score = self._score_candidate(normalized, entity_type, candidate)
                if score > best_score:
                    best_score = score
                    best_id = candidate.id
            if best_id is not None and best_score >= 0.5:
                return best_id, best_score

        return None, 0.0

    def find_duplicates(
        self, min_confidence: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Scan existing entities for duplicate pairs.

        Pass 1: External ID collisions (different entity_ids, same ext_id).
        Pass 2: Text similarity via search_entities (if mode >= "fuzzy").
        Returns deduplicated (entity_a, entity_b, confidence) sorted by confidence desc.
        """
        if not self._built:
            self.build_index()

        seen_pairs: set[tuple[str, str]] = set()
        results: list[tuple[str, str, float]] = []

        # Pass 1: External ID collisions from build_index
        collisions = getattr(self, "_ext_id_collisions", {})
        for (_ns, _eid), entity_ids in collisions.items():
            unique_ids = sorted(entity_ids)
            for i in range(len(unique_ids)):
                for j in range(i + 1, len(unique_ids)):
                    pair = (unique_ids[i], unique_ids[j])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        results.append((pair[0], pair[1], 0.99))

        # Pass 2: Text similarity (fuzzy/full mode)
        if self._mode in ("fuzzy", "full") and hasattr(self._store, "search_entities"):
            entities = [entity_summary_from_dict(d) for d in self._store.list_entities()]
            for entity in entities:
                candidates = [
                    entity_summary_from_dict(d)
                    for d in self._store.search_entities(entity.id, top_k=5)
                ]
                for candidate in candidates:
                    if candidate.id == entity.id:
                        continue
                    pair = tuple(sorted([entity.id, candidate.id]))
                    if pair in seen_pairs:
                        continue
                    score = self._score_candidate(entity.id, entity.entity_type, candidate)
                    if score >= min_confidence:
                        seen_pairs.add(pair)
                        results.append((pair[0], pair[1], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    @staticmethod
    def _score_candidate(
        query: str, query_type: str, candidate: EntitySummary,
    ) -> float:
        """Score a candidate entity against a query using Jaccard token overlap.

        Adds a type bonus when entity_types match.
        Returns a score in [0.0, 0.95].
        """
        q_tokens = set(query.lower().split())
        c_tokens = set(candidate.id.lower().split())
        if not q_tokens or not c_tokens:
            return 0.0

        intersection = len(q_tokens & c_tokens)
        union = len(q_tokens | c_tokens)
        jaccard = intersection / union if union > 0 else 0.0

        # Also check display name tokens
        if candidate.name:
            name_tokens = set(candidate.name.lower().split())
            name_intersection = len(q_tokens & name_tokens)
            name_union = len(q_tokens | name_tokens)
            name_jaccard = name_intersection / name_union if name_union > 0 else 0.0
            jaccard = max(jaccard, name_jaccard)

        if jaccard == 0.0:
            return 0.0

        # Scale to 0.5-0.85 range for fuzzy matches
        score = jaccard * 0.85

        # Type bonus: +0.1 if types match
        if query_type and candidate.entity_type and query_type == candidate.entity_type:
            score += 0.1

        return min(0.95, score)
