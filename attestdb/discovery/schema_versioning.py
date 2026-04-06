"""Schema version history for data sources.

Maintains an ordered log of :class:`SchemaVersion` snapshots per source,
enabling diff between arbitrary versions and linking claim templates to
the schema version they were generated against.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

from attestdb.discovery.drift_detector import (
    SchemaChange,
    detect_drift,
)
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)


@dataclass
class SchemaVersion:
    """A point-in-time snapshot of a source's schema."""

    version_id: int
    source_id: str
    schema_map: SchemaMap
    created_at: float = field(default_factory=time.time)
    changes_from_previous: list[SchemaChange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "source_id": self.source_id,
            "schema_map": self.schema_map.to_dict(),
            "created_at": self.created_at,
            "changes_from_previous": [
                {
                    "field_name": c.field_name,
                    "change_type": c.change_type,
                    "severity": c.severity,
                    "old_value": c.old_value,
                    "new_value": c.new_value,
                    "details": c.details,
                }
                for c in self.changes_from_previous
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SchemaVersion:
        changes = [
            SchemaChange(
                field_name=c["field_name"],
                change_type=c["change_type"],
                severity=c["severity"],
                old_value=c.get("old_value"),
                new_value=c.get("new_value"),
                details=c["details"],
            )
            for c in d.get("changes_from_previous", [])
        ]
        return cls(
            version_id=d["version_id"],
            source_id=d["source_id"],
            schema_map=SchemaMap.from_dict(d["schema_map"]),
            created_at=d.get("created_at", 0.0),
            changes_from_previous=changes,
        )


class SchemaVersionStore:
    """Persistent, file-backed version history per source.

    Stores each :class:`SchemaVersion` as a JSON file in *store_path*
    using the naming convention ``{source_id}_v{version_id}.json``.
    """

    def __init__(self, store_path: str) -> None:
        self._store_path = store_path
        os.makedirs(self._store_path, exist_ok=True)

    # ── Write ────────────────────────────────────────────────────

    def record_version(
        self,
        source_id: str,
        schema_map: SchemaMap,
        changes: list[SchemaChange] | None = None,
    ) -> SchemaVersion:
        """Persist a new schema version for *source_id*.

        Auto-increments the version_id.  The *schema_map* is stored
        as-is; *changes* records how it differs from the previous version.
        """
        current = self.get_current(source_id)
        next_id = (current.version_id + 1) if current else 1

        version = SchemaVersion(
            version_id=next_id,
            source_id=source_id,
            schema_map=schema_map,
            changes_from_previous=changes or [],
        )
        self._save(version)
        logger.info(
            "Recorded schema version %d for source '%s'",
            next_id,
            source_id,
        )
        return version

    # ── Read ─────────────────────────────────────────────────────

    def get_current(self, source_id: str) -> SchemaVersion | None:
        """Return the latest SchemaVersion for *source_id*, or None."""
        versions = self.get_all(source_id)
        return versions[-1] if versions else None

    def get_version(self, source_id: str, version_id: int) -> SchemaVersion | None:
        """Return a specific version, or None if not found."""
        path = self._version_path(source_id, version_id)
        if not os.path.exists(path):
            return None
        return self._load(path)

    def get_all(self, source_id: str) -> list[SchemaVersion]:
        """Return all versions for *source_id*, ordered by version_id."""
        prefix = f"{source_id}_v"
        versions: list[SchemaVersion] = []
        for fname in os.listdir(self._store_path):
            if fname.startswith(prefix) and fname.endswith(".json"):
                path = os.path.join(self._store_path, fname)
                versions.append(self._load(path))
        versions.sort(key=lambda v: v.version_id)
        return versions

    # ── Diff ─────────────────────────────────────────────────────

    def diff(
        self,
        source_id: str,
        version_a: int,
        version_b: int,
    ) -> list[SchemaChange]:
        """Compute the schema diff between two versions.

        Uses :func:`detect_drift` under the hood — returns the changes
        list from the resulting DriftReport.
        """
        va = self.get_version(source_id, version_a)
        vb = self.get_version(source_id, version_b)
        if va is None:
            raise KeyError(f"Version {version_a} not found for source '{source_id}'")
        if vb is None:
            raise KeyError(f"Version {version_b} not found for source '{source_id}'")

        report = detect_drift(va.schema_map, vb.schema_map)
        return report.changes

    # ── Internal ─────────────────────────────────────────────────

    def _version_path(self, source_id: str, version_id: int) -> str:
        return os.path.join(self._store_path, f"{source_id}_v{version_id}.json")

    def _save(self, version: SchemaVersion) -> None:
        path = self._version_path(version.source_id, version.version_id)
        with open(path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

    def _load(self, path: str) -> SchemaVersion:
        with open(path) as f:
            data = json.load(f)
        return SchemaVersion.from_dict(data)
