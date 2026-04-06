"""Schema drift detection for data sources.

Compares a previously discovered SchemaMap against a fresh sample to
detect structural changes — new fields, removed fields, type changes,
and significant value-distribution shifts.  A DriftMonitor orchestrates
periodic checks and persists versioned SchemaMap history.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

from attestdb.discovery.analyzer import detect_deprecated_fields, infer_semantics
from attestdb.discovery.sampler import FieldProfile, analyze_fields, sample_source
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)

# ── Risk / severity constants ───────────────────────────────────────

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"

# Change type constants
CHANGE_ADDITIVE = "additive"
CHANGE_MODIFIED = "modified"
CHANGE_DESTRUCTIVE = "destructive"
CHANGE_VALUE_DRIFT = "value_drift"

# Legacy aliases (backward compat)
_SEVERITY_LOW = RISK_LOW
_SEVERITY_MEDIUM = RISK_MEDIUM
_SEVERITY_HIGH = RISK_HIGH

_FILL_RATE_THRESHOLD = 0.20  # >20 percentage-point absolute shift
_DRIFT_SAMPLE_SIZE = 200  # smaller re-sample for drift checks


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class SchemaChange:
    """A single detected change between two schema versions."""

    field_name: str
    change_type: str  # "additive" | "modified" | "destructive" | "value_drift"
    severity: str  # "low" | "medium" | "high"
    old_value: str | None
    new_value: str | None
    details: str


@dataclass
class DriftReport:
    """Summary of all schema changes detected for a source."""

    source_id: str
    changes: list[SchemaChange] = field(default_factory=list)
    schema_version_old: int = 0
    schema_version_new: int = 0
    detected_at: float = field(default_factory=time.time)
    requires_review: bool = False

    @property
    def additive_changes(self) -> list[SchemaChange]:
        return [c for c in self.changes if c.change_type == CHANGE_ADDITIVE]

    @property
    def modified_changes(self) -> list[SchemaChange]:
        return [c for c in self.changes if c.change_type == CHANGE_MODIFIED]

    @property
    def destructive_changes(self) -> list[SchemaChange]:
        return [c for c in self.changes if c.change_type == CHANGE_DESTRUCTIVE]

    @property
    def value_drift_changes(self) -> list[SchemaChange]:
        return [c for c in self.changes if c.change_type == CHANGE_VALUE_DRIFT]

    @property
    def changes_by_risk(self) -> dict[str, list[SchemaChange]]:
        """Changes grouped by risk level."""
        result: dict[str, list[SchemaChange]] = {
            RISK_LOW: [],
            RISK_MEDIUM: [],
            RISK_HIGH: [],
        }
        for c in self.changes:
            result[c.severity].append(c)
        return result


# ── Core detection ───────────────────────────────────────────────────


def _dominant_type(type_distribution: dict[str, float]) -> str:
    """Return the type with the highest frequency, ignoring 'null'."""
    best_type = ""
    best_freq = -1.0
    for t, freq in type_distribution.items():
        if t == "null":
            continue
        if freq > best_freq:
            best_freq = freq
            best_type = t
    return best_type


def detect_drift(old_schema: SchemaMap, new_schema: SchemaMap) -> DriftReport:
    """Compare two SchemaMap versions and return a DriftReport.

    Change classification:
    - additive (new field not in old): low risk, auto-trigger analysis
    - modified (field exists but dominant type changed): medium risk, review
    - destructive (field in old but missing from new): high risk, alert
    - value_drift (fill rate shifted >20%): medium risk, review
    """
    old_fields = {fp.field_name: fp for fp in old_schema.field_profiles}
    new_fields = {fp.field_name: fp for fp in new_schema.field_profiles}

    old_names = set(old_fields)
    new_names = set(new_fields)

    changes: list[SchemaChange] = []

    # Additive: new field not in current map
    for name in sorted(new_names - old_names):
        changes.append(
            SchemaChange(
                field_name=name,
                change_type=CHANGE_ADDITIVE,
                severity=RISK_LOW,
                old_value=None,
                new_value=_dominant_type(new_fields[name].type_distribution) or "unknown",
                details=f"New field '{name}' detected",
            )
        )

    # Destructive: field in current map not found in source
    for name in sorted(old_names - new_names):
        changes.append(
            SchemaChange(
                field_name=name,
                change_type=CHANGE_DESTRUCTIVE,
                severity=RISK_HIGH,
                old_value=_dominant_type(old_fields[name].type_distribution) or "unknown",
                new_value=None,
                details=f"Field '{name}' no longer present in source",
            )
        )

    # Fields present in both — check for type changes and value drift
    for name in sorted(old_names & new_names):
        old_fp = old_fields[name]
        new_fp = new_fields[name]

        old_dominant = _dominant_type(old_fp.type_distribution)
        new_dominant = _dominant_type(new_fp.type_distribution)

        if old_dominant and new_dominant and old_dominant != new_dominant:
            changes.append(
                SchemaChange(
                    field_name=name,
                    change_type=CHANGE_MODIFIED,
                    severity=RISK_MEDIUM,
                    old_value=old_dominant,
                    new_value=new_dominant,
                    details=(
                        f"Dominant type changed from '{old_dominant}' to "
                        f"'{new_dominant}'"
                    ),
                )
            )

        fill_delta = abs(new_fp.fill_rate - old_fp.fill_rate)
        if fill_delta > _FILL_RATE_THRESHOLD:
            changes.append(
                SchemaChange(
                    field_name=name,
                    change_type=CHANGE_VALUE_DRIFT,
                    severity=RISK_MEDIUM,
                    old_value=f"{old_fp.fill_rate:.2f}",
                    new_value=f"{new_fp.fill_rate:.2f}",
                    details=(
                        f"Fill rate shifted by {fill_delta:.2f} "
                        f"({old_fp.fill_rate:.2f} -> {new_fp.fill_rate:.2f})"
                    ),
                )
            )

    requires_review = any(
        c.severity in (RISK_MEDIUM, RISK_HIGH) for c in changes
    )

    return DriftReport(
        source_id=old_schema.source_id,
        changes=changes,
        schema_version_old=old_schema.schema_version,
        schema_version_new=old_schema.schema_version + 1 if changes else old_schema.schema_version,
        detected_at=time.time(),
        requires_review=requires_review,
    )


def detect_drift_from_connector(
    connector: object,
    source_id: str,
    current_schema_map: SchemaMap,
    *,
    sample_size: int = _DRIFT_SAMPLE_SIZE,
    tenant_id: str = "default",
) -> DriftReport:
    """Re-sample a source and detect drift against *current_schema_map*.

    Uses a smaller sample (200 records by default) than initial discovery.
    """
    samples = sample_source(connector, sample_size=sample_size, tenant_id=tenant_id)
    profiles = analyze_fields(samples, tenant_id=tenant_id)
    mappings = infer_semantics(profiles, tenant_id=tenant_id)
    deprecated = detect_deprecated_fields(profiles)

    new_schema = SchemaMap(
        source_id=source_id,
        source_type=current_schema_map.source_type,
        tenant_id=tenant_id,
        field_profiles=profiles,
        semantic_mappings=mappings,
        deprecated_fields=deprecated,
        schema_version=current_schema_map.schema_version,
    )

    return detect_drift(current_schema_map, new_schema)


# ── Monitor ──────────────────────────────────────────────────────────


class DriftMonitor:
    """Orchestrates schema drift checks with versioned persistence.

    Stores SchemaMap JSON files in *schema_store_path* using the naming
    convention ``{source_id}_v{version}.json``.
    """

    def __init__(self, schema_store_path: str) -> None:
        self._store_path = schema_store_path
        os.makedirs(self._store_path, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────

    def check_source(
        self,
        source_id: str,
        connector: object,
        tenant_id: str = "default",
    ) -> DriftReport | None:
        """Compare the current source schema against the last known version.

        Returns a :class:`DriftReport` when changes are detected, or
        ``None`` when the schema is unchanged.  Saves a new versioned
        SchemaMap on drift.
        """
        old_schema = self._load_latest(source_id)

        # Sample and profile the source
        samples = sample_source(connector, tenant_id=tenant_id)
        profiles = analyze_fields(samples, tenant_id=tenant_id)
        mappings = infer_semantics(profiles, tenant_id=tenant_id)
        deprecated = detect_deprecated_fields(profiles)

        if old_schema is None:
            # First observation — persist as version 1
            new_schema = SchemaMap(
                source_id=source_id,
                source_type=type(connector).__name__,
                tenant_id=tenant_id,
                field_profiles=profiles,
                semantic_mappings=mappings,
                deprecated_fields=deprecated,
                schema_version=1,
            )
            self._save(new_schema)
            logger.info("First schema recorded for source '%s'", source_id)
            return None

        new_schema = SchemaMap(
            source_id=source_id,
            source_type=old_schema.source_type,
            tenant_id=tenant_id,
            field_profiles=profiles,
            semantic_mappings=mappings,
            deprecated_fields=deprecated,
            schema_version=old_schema.schema_version,
        )

        report = detect_drift(old_schema, new_schema)

        if not report.changes:
            return None

        # Persist updated schema with incremented version
        new_schema.schema_version = report.schema_version_new
        self._save(new_schema)
        logger.info(
            "Drift detected for '%s': %d change(s), new version %d",
            source_id,
            len(report.changes),
            new_schema.schema_version,
        )
        return report

    def check_all(
        self,
        connectors: dict[str, object],
    ) -> list[DriftReport]:
        """Check all sources and return reports for those with changes."""
        reports: list[DriftReport] = []
        for source_id, connector in connectors.items():
            report = self.check_source(source_id, connector)
            if report is not None:
                reports.append(report)
        return reports

    def get_history(self, source_id: str) -> list[SchemaMap]:
        """Return all stored SchemaMap versions for *source_id*, ordered by version."""
        schemas: list[SchemaMap] = []
        prefix = f"{source_id}_v"
        for fname in os.listdir(self._store_path):
            if fname.startswith(prefix) and fname.endswith(".json"):
                path = os.path.join(self._store_path, fname)
                schemas.append(SchemaMap.load(path))
        schemas.sort(key=lambda s: s.schema_version)
        return schemas

    # ── Internal helpers ─────────────────────────────────────────

    def _load_latest(self, source_id: str) -> SchemaMap | None:
        """Load the highest-versioned SchemaMap for *source_id*, or None."""
        history = self.get_history(source_id)
        return history[-1] if history else None

    def _save(self, schema: SchemaMap) -> None:
        """Persist a SchemaMap to the store directory."""
        fname = f"{schema.source_id}_v{schema.schema_version}.json"
        path = os.path.join(self._store_path, fname)
        schema.save(path)
