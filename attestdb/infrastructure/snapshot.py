"""Read-only time-travel view of an AttestDB at a given timestamp."""

from __future__ import annotations

from attestdb.core.types import ContextFrame
from attestdb.infrastructure.query_engine import QueryEngine


class SnapshotQueryEngine(QueryEngine):
    """QueryEngine that filters claims by timestamp, giving a point-in-time view."""

    def __init__(self, store, timestamp: int, claim_converter=None):
        super().__init__(store, claim_converter=claim_converter)
        self._timestamp = timestamp

    def query(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = 500,
        max_tokens: int = 4000,
    ) -> ContextFrame:
        return self._execute_query(
            focal_entity, depth=depth, min_confidence=min_confidence,
            exclude_source_types=exclude_source_types,
            max_claims=max_claims, max_tokens=max_tokens,
            claim_filter=lambda c: c.timestamp <= self._timestamp,
            include_quantitative=False,
            include_contradictions=False,
            include_narrative=False,
        )


class AttestDBSnapshot:
    """Read-only time-travel view. Wraps AttestDB, filtering claims by timestamp."""

    def __init__(self, db, timestamp: int):
        self._db = db
        self._timestamp = timestamp
        self._query_engine = SnapshotQueryEngine(
            db._store, timestamp,
            claim_converter=db._query_engine._convert_claim,
        )

    def query(self, focal_entity: str, **kwargs) -> ContextFrame:
        return self._query_engine.query(focal_entity, **kwargs)

    def stats(self) -> dict:
        """Point-in-time stats: count claims and entities at or before the timestamp."""
        # For Rust backend with claims_in_range
        if hasattr(self._db._store, "claims_in_range"):
            claims = self._db._store.claims_in_range(0, self._timestamp)
            entities: set[str] = set()
            for c in claims:
                subj = c.get("subject")
                obj = c.get("object")
                if isinstance(subj, dict):
                    entities.add(subj.get("id", ""))
                elif isinstance(subj, str):
                    entities.add(subj)
                if isinstance(obj, dict):
                    entities.add(obj.get("id", ""))
                elif isinstance(obj, str):
                    entities.add(obj)
            entities.discard("")
            return {
                "total_claims": len(claims),
                "entity_count": len(entities),
                "timestamp": self._timestamp,
            }
        base_stats = self._db._store.stats()
        return {
            "total_claims": base_stats.get("total_claims", 0),
            "entity_count": base_stats.get("entity_count", 0),
            "timestamp": self._timestamp,
        }
