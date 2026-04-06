"""Entity Resolution Engine — cross-source entity matching, unified index, and change detection."""

from __future__ import annotations

from attestdb.resolution.change_detector import ChangeSet, SyncManager, detect_changes
from attestdb.resolution.entity_index import EntityIndex, UnifiedEntity
from attestdb.resolution.ingestion_hook import IngestionHook
from attestdb.resolution.matcher import (
    AIMatcher,
    DomainMatcher,
    EntityMatcher,
    ExactMatcher,
    FuzzyNameMatcher,
    MatchCandidate,
)
from attestdb.resolution.materialized_views import MaterializedViewManager

__all__ = [
    "AIMatcher",
    "ChangeSet",
    "DomainMatcher",
    "EntityIndex",
    "EntityMatcher",
    "ExactMatcher",
    "FuzzyNameMatcher",
    "IngestionHook",
    "MatchCandidate",
    "MaterializedViewManager",
    "SyncManager",
    "UnifiedEntity",
    "detect_changes",
]
