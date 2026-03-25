"""Entity management subsystem — resolution, merging, search, deduplication."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import EntitySummary, entity_summary_from_dict

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB


class EntityManager:
    """Manages entity operations on behalf of an AttestDB instance."""

    def __init__(self, db: AttestDB):
        self.db = db

    # --- Basic entity access ---

    def get_entity(self, entity_id: str) -> EntitySummary | None:
        canonical = self.db._store.resolve(normalize_entity_id(entity_id))
        raw = self.db._store.get_entity(canonical)
        return entity_summary_from_dict(raw) if raw else None

    def list_entities(
        self,
        entity_type: str | None = None,
        min_claims: int = 0,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[EntitySummary]:
        # Rust uses limit=0 to mean "no limit"; Python uses None.
        rust_limit = 0 if limit is None else limit
        raw = self.db._store.list_entities(entity_type, min_claims, offset, rust_limit)
        return [entity_summary_from_dict(d) for d in raw]

    def count_entities(self, entity_type: str | None = None, min_claims: int = 0) -> int:
        """Return the number of entities without materializing them."""
        return self.db._store.count_entities(entity_type, min_claims)

    def search_entities(self, query: str, top_k: int = 10) -> list[EntitySummary]:
        """Search entities by text matching on id and display_name."""
        return [entity_summary_from_dict(d) for d in self.db._store.search_entities(query, top_k)]

    # --- Entity resolution ---

    def resolve(self, entity_id: str) -> str:
        """Resolve an entity ID to its canonical form via union-find."""
        return self.db._store.resolve(normalize_entity_id(entity_id))

    def enable_entity_resolution(self, mode: str = "external_ids") -> None:
        """Enable entity resolution during ingestion.

        Modes: "external_ids" (exact + ext ID), "fuzzy" (+ text search), "full" (reserved).
        Must be called explicitly -- resolution is opt-in.
        """
        from attestdb.infrastructure.entity_resolver import EntityResolver

        resolver = EntityResolver(self.db._store, mode=mode)
        resolver.build_index()
        self.db._pipeline._resolver = resolver
        self.db._entity_resolver = resolver

    def resolve_entity(
        self,
        name: str,
        entity_type: str = "",
        external_ids: dict[str, str] | None = None,
    ) -> tuple[str | None, float]:
        """Manually resolve an entity name against existing entities.

        Returns (existing_entity_id, confidence) or (None, 0.0).
        """
        if not hasattr(self.db, "_entity_resolver") or self.db._entity_resolver is None:
            from attestdb.infrastructure.entity_resolver import EntityResolver
            resolver = EntityResolver(self.db._store, mode="external_ids")
            resolver.build_index()
            return resolver.resolve(name, entity_type, external_ids)
        return self.db._entity_resolver.resolve(name, entity_type, external_ids)

    # --- Entity merging ---

    def merge_entities(self, entity_a: str, entity_b: str, reason: str = "") -> str:
        """Merge two entities via same_as claim. Returns claim_id.

        The existing union-find handles downstream alias resolution.
        """
        return self.db.ingest(
            subject=(entity_a, "entity"),
            predicate=("same_as", "alias"),
            object=(entity_b, "entity"),
            provenance={
                "source_type": "human_annotation",
                "source_id": "entity_resolution",
            },
            payload={"schema": "", "data": {"reason": reason}} if reason else None,
        )

    # --- Duplicate detection ---

    def find_duplicate_entities(
        self, min_confidence: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Scan for potential duplicate entities.

        Returns list of (entity_a, entity_b, confidence) sorted by confidence desc.
        """
        from attestdb.infrastructure.entity_resolver import EntityResolver

        if hasattr(self.db, "_entity_resolver") and self.db._entity_resolver is not None:
            return self.db._entity_resolver.find_duplicates(min_confidence)
        resolver = EntityResolver(self.db._store, mode="external_ids")
        resolver.build_index()
        return resolver.find_duplicates(min_confidence)

    def auto_merge_duplicates(
        self, min_confidence: float = 0.95,
    ) -> dict:
        """Auto-merge detected duplicate entities above confidence threshold.

        Calls find_duplicate_entities(), ingests same_as claims for each pair.
        Returns dict with merged_count, skipped_count, claim_ids, merged_pairs.
        """
        dupes = self.find_duplicate_entities(min_confidence=min_confidence)
        merged_count = 0
        skipped_count = 0
        claim_ids: list[str] = []
        merged_pairs: list[tuple[str, str, float]] = []

        for entity_a, entity_b, confidence in dupes:
            try:
                claim_id = self.merge_entities(
                    entity_a, entity_b,
                    reason=f"auto-merged (confidence={confidence:.3f})",
                )
                merged_count += 1
                claim_ids.append(claim_id)
                merged_pairs.append((entity_a, entity_b, confidence))
            except Exception:
                skipped_count += 1

        return {
            "merged_count": merged_count,
            "skipped_count": skipped_count,
            "claim_ids": claim_ids,
            "merged_pairs": merged_pairs,
        }
