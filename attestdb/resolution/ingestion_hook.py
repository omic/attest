"""Ingestion hook — auto-resolve entities on create, update, and tombstone events.

Integrates with the resilience module for idempotency and dead-letter
routing, and emits lifecycle events via callbacks.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from attestdb.resolution.entity_index import EntityIndex, UnifiedEntity
from attestdb.resolution.matcher import EntityMatcher, FuzzyNameMatcher
from attestdb.resolution.materialized_views import MaterializedViewManager

log = logging.getLogger(__name__)

# Confidence above which an existing entity match is auto-linked
_AUTO_LINK_THRESHOLD = 0.90

# Confidence below which the match goes to review instead of auto-linking
_REVIEW_THRESHOLD = 0.75

# Fields that trigger a re-match when they change
_REMATCH_FIELDS = frozenset({"name", "duns", "tax_id", "domain"})

# Type alias for event listeners
EventCallback = Callable[[str, dict], None]


class IngestionHook:
    """Hooks into record lifecycle events to keep the entity index in sync.

    Parameters
    ----------
    entity_index:
        The unified entity index.
    matcher:
        Entity matcher for cross-source resolution.
    view_manager:
        Optional materialized view manager for metric updates.
    idempotency_store:
        Optional ``IdempotencyStore`` (from ``attestdb.resilience``).
        When provided, duplicate events are silently skipped.
    dead_letter_queue:
        Optional ``DeadLetterQueue`` (from ``attestdb.resilience``).
        When provided, failed records are routed here instead of raising.
    """

    def __init__(
        self,
        entity_index: EntityIndex,
        matcher: EntityMatcher,
        view_manager: MaterializedViewManager | None = None,
        idempotency_store: object | None = None,
        dead_letter_queue: object | None = None,
    ) -> None:
        self.entity_index = entity_index
        self.matcher = matcher
        self.view_manager = view_manager
        self._idempotency = idempotency_store
        self._dlq = dead_letter_queue
        self._fuzzy = FuzzyNameMatcher()
        self._event_listeners: list[EventCallback] = []

    def on_event(self, callback: EventCallback) -> None:
        """Register an event listener.  Signature: ``callback(event_type, payload)``."""
        self._event_listeners.append(callback)

    def _emit(self, event_type: str, payload: dict) -> None:
        for cb in self._event_listeners:
            try:
                cb(event_type, payload)
            except Exception:
                log.debug("Event listener error for %s", event_type, exc_info=True)

    # ------------------------------------------------------------------
    # Idempotency helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, source_id: str, record_id: str, stage: str) -> bool:
        if self._idempotency is None:
            return False
        event_id = f"{source_id}:{record_id}"
        return self._idempotency.check_and_mark(event_id, stage)  # type: ignore[union-attr]

    def _route_to_dlq(self, source_id: str, record: dict, error: str) -> None:
        if self._dlq is not None:
            self._dlq.enqueue(source_id, record, error)  # type: ignore[union-attr]
        else:
            log.error("Failed record (no DLQ): source=%s error=%s", source_id, error)

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_create(
        self,
        source_id: str,
        record_id: str,
        record: dict,
        entity_type: str = "customer",
    ) -> UnifiedEntity:
        """Handle a new record from a source system.

        Searches the entity index for an existing match by name.
        - High-confidence match: link to existing entity.
        - Low-confidence match: create entity flagged for review.
        - No match: create new entity.

        Idempotent: re-processing the same (source_id, record_id) is a no-op
        that returns the existing entity.
        """
        # Idempotency: skip if already processed
        if self._is_duplicate(source_id, record_id, "create"):
            existing = self.entity_index.resolve(source_id, record_id)
            if existing is not None:
                return existing

        try:
            return self._do_create(source_id, record_id, record, entity_type)
        except Exception as exc:
            self._route_to_dlq(source_id, {"record_id": record_id, **record}, str(exc))
            raise

    def _do_create(
        self,
        source_id: str,
        record_id: str,
        record: dict,
        entity_type: str,
    ) -> UnifiedEntity:
        name = record.get("name", "")
        if not name:
            # Fall back to record_id as name
            name = record_id

        # Search for existing entities with a similar name
        candidates = self.entity_index.search(name, entity_type=entity_type, limit=10)

        best_match: UnifiedEntity | None = None
        best_score = 0.0

        for candidate in candidates:
            score = self._fuzzy._compute_similarity(name, candidate.canonical_name)
            # Also check aliases
            for alias in candidate.aliases:
                alias_score = self._fuzzy._compute_similarity(name, alias)
                score = max(score, alias_score)

            if score > best_score:
                best_score = score
                best_match = candidate

        if best_match is not None and best_score >= _AUTO_LINK_THRESHOLD:
            # High confidence: auto-link to existing entity
            self.entity_index.add_source_record(
                best_match.entity_id,
                source_id,
                record_id,
                alias=name,
            )
            log.info(
                "Auto-linked %s:%s to entity %s (score=%.3f)",
                source_id, record_id, best_match.entity_id, best_score,
            )
            entity = self.entity_index.get(best_match.entity_id)  # type: ignore[return-value]
            self._emit("entity_updated", {
                "entity_id": entity.entity_id,
                "reason": "source_record_linked",
                "source_id": source_id,
                "record_id": record_id,
            })
            return entity

        if best_match is not None and best_score >= _REVIEW_THRESHOLD:
            # Low confidence: create but flag for review
            entity = self.entity_index.create(
                name=name,
                entity_type=entity_type,
                source_id=source_id,
                record_id=record_id,
                confidence=best_score,
                needs_review=True,
            )
            log.info(
                "Created entity %s for %s:%s (needs review, potential match: %s, score=%.3f)",
                entity.entity_id, source_id, record_id, best_match.entity_id, best_score,
            )
            self._emit("entity_created", {
                "entity_id": entity.entity_id,
                "needs_review": True,
                "potential_match": best_match.entity_id,
            })
            return entity

        # No match: create new entity
        entity = self.entity_index.create(
            name=name,
            entity_type=entity_type,
            source_id=source_id,
            record_id=record_id,
            confidence=1.0,
            needs_review=False,
        )
        log.info("Created new entity %s for %s:%s", entity.entity_id, source_id, record_id)
        self._emit("entity_created", {
            "entity_id": entity.entity_id,
            "needs_review": False,
        })
        return entity

    def on_update(
        self,
        source_id: str,
        record_id: str,
        record: dict,
        changed_fields: set[str] | None = None,
    ) -> UnifiedEntity | None:
        """Handle a record update.

        If rematch-relevant fields changed (name, DUNS, domain, tax_id),
        re-run the matcher to check whether this record should move to a
        different entity.  Otherwise just update the canonical name.
        """
        if self._is_duplicate(source_id, record_id, "update"):
            return self.entity_index.resolve(source_id, record_id)

        try:
            return self._do_update(source_id, record_id, record, changed_fields)
        except Exception as exc:
            self._route_to_dlq(source_id, {"record_id": record_id, **record}, str(exc))
            raise

    def _do_update(
        self,
        source_id: str,
        record_id: str,
        record: dict,
        changed_fields: set[str] | None,
    ) -> UnifiedEntity | None:
        entity = self.entity_index.resolve(source_id, record_id)
        if entity is None:
            log.warning("on_update: no entity found for %s:%s", source_id, record_id)
            return None

        # Check if rematch-relevant fields changed
        if changed_fields and changed_fields & _REMATCH_FIELDS:
            # Re-run matcher for this single record against all other entities
            new_name = record.get("name", entity.canonical_name)
            candidates = self.entity_index.search(new_name, entity_type=entity.entity_type, limit=10)

            best_match: UnifiedEntity | None = None
            best_score = 0.0
            for candidate in candidates:
                if candidate.entity_id == entity.entity_id:
                    continue
                score = self._fuzzy._compute_similarity(new_name, candidate.canonical_name)
                for alias in candidate.aliases:
                    score = max(score, self._fuzzy._compute_similarity(new_name, alias))
                if score > best_score:
                    best_score = score
                    best_match = candidate

            if best_match is not None and best_score >= _AUTO_LINK_THRESHOLD:
                # Merge into the better-matched entity
                self.entity_index.merge(best_match.entity_id, entity.entity_id)
                log.info(
                    "Re-match: merged entity %s into %s after field change (score=%.3f)",
                    entity.entity_id, best_match.entity_id, best_score,
                )
                merged = self.entity_index.get(best_match.entity_id)
                self._emit("entity_updated", {
                    "entity_id": best_match.entity_id,
                    "reason": "rematch_merge",
                    "merged_from": entity.entity_id,
                })
                return merged

        new_name = record.get("name")
        if new_name and new_name != entity.canonical_name:
            # Add old name as alias and update canonical
            self.entity_index.update_canonical_name(entity.entity_id, new_name)
            log.info(
                "Updated canonical name for entity %s: '%s' -> '%s'",
                entity.entity_id, entity.canonical_name, new_name,
            )

        updated = self.entity_index.get(entity.entity_id)
        self._emit("entity_updated", {
            "entity_id": entity.entity_id,
            "reason": "record_updated",
            "source_id": source_id,
            "record_id": record_id,
        })
        return updated

    def on_tombstone(
        self,
        source_id: str,
        record_id: str,
    ) -> UnifiedEntity | None:
        """Handle a record deletion (tombstone).

        Remove the source record. If no active source records remain,
        deactivate the entity. Otherwise, re-pick the canonical name
        from remaining records.

        Idempotent: re-processing the same tombstone is a no-op.
        """
        if self._is_duplicate(source_id, record_id, "tombstone"):
            return self.entity_index.resolve(source_id, record_id)

        try:
            return self._do_tombstone(source_id, record_id)
        except Exception as exc:
            self._route_to_dlq(source_id, {"record_id": record_id}, str(exc))
            raise

    def _do_tombstone(
        self,
        source_id: str,
        record_id: str,
    ) -> UnifiedEntity | None:
        entity = self.entity_index.resolve(source_id, record_id)
        if entity is None:
            log.warning("on_tombstone: no entity found for %s:%s", source_id, record_id)
            return None

        # Remove the source record
        remaining_sources = {
            k: v for k, v in entity.source_records.items()
            if not (k == source_id and v == record_id)
        }

        if not remaining_sources:
            # No more source records: deactivate the entity
            self.entity_index.deactivate(entity.entity_id)
            log.info("Deactivated entity %s (last source record removed)", entity.entity_id)
            self._emit("entity_deactivated", {
                "entity_id": entity.entity_id,
                "reason": "last_source_record_removed",
            })
            return self.entity_index.get(entity.entity_id)

        # Update source records in DB (remove the tombstoned one)
        now = time.time()
        self.entity_index._conn.execute(
            """
            UPDATE entities SET source_records = ?, updated_at = ?
            WHERE entity_id = ?
            """,
            (json.dumps(remaining_sources), now, entity.entity_id),
        )
        self.entity_index._conn.commit()

        self._emit("entity_updated", {
            "entity_id": entity.entity_id,
            "reason": "source_record_removed",
            "source_id": source_id,
            "record_id": record_id,
        })

        return self.entity_index.get(entity.entity_id)
