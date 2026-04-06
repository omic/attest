"""Template migration in response to schema drift.

Given a :class:`DriftReport` and the current claim templates, decides
how to handle each change type:

- **additive**: generate new claim templates for new fields (runs them
  through the analyzer + template generation pipeline).
- **modified**: propose template updates and route to review queue.
- **destructive**: mark affected templates as ``suspended`` and stop
  extraction for those fields.

Existing claims are *never* deleted or modified — only future extraction
is affected.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

from attestdb.discovery.claim_templates import ClaimTemplate
from attestdb.discovery.drift_detector import (
    CHANGE_ADDITIVE,
    CHANGE_DESTRUCTIVE,
    CHANGE_MODIFIED,
    CHANGE_VALUE_DRIFT,
    DriftReport,
    SchemaChange,
)
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)

# Template status constants
STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "suspended"
STATUS_PENDING_REVIEW = "pending_review"


@dataclass
class TemplateMigrationAction:
    """A single migration action proposed for a claim template."""

    template: ClaimTemplate
    action: str  # "create" | "update" | "suspend"
    reason: str
    change: SchemaChange
    requires_review: bool


@dataclass
class TemplateMigrationResult:
    """Result of migrating templates against a drift report."""

    new_templates: list[ClaimTemplate] = field(default_factory=list)
    updated_templates: list[ClaimTemplate] = field(default_factory=list)
    suspended_templates: list[ClaimTemplate] = field(default_factory=list)
    review_items: list[TemplateMigrationAction] = field(default_factory=list)


def migrate_templates(
    drift_report: DriftReport,
    current_templates: list[ClaimTemplate],
    new_schema: SchemaMap | None = None,
) -> TemplateMigrationResult:
    """Propose template changes based on a drift report.

    Parameters
    ----------
    drift_report:
        The detected schema changes.
    current_templates:
        Active claim templates to evaluate against drift.
    new_schema:
        If provided, used to generate new templates for additive fields.

    Returns
    -------
    TemplateMigrationResult with categorised actions.
    """
    result = TemplateMigrationResult()

    # Index templates by the fields they reference
    templates_by_field: dict[str, list[ClaimTemplate]] = {}
    for t in current_templates:
        templates_by_field.setdefault(t.value_field, []).append(t)
        templates_by_field.setdefault(t.entity_key_field, []).append(t)
        for cfg in t.source_configs.values():
            path = cfg.get("field_path", "")
            leaf = path.rsplit(".", 1)[-1] if "." in path else path
            if leaf:
                templates_by_field.setdefault(leaf, []).append(t)

    for change in drift_report.changes:
        if change.change_type == CHANGE_ADDITIVE:
            _handle_additive(change, new_schema, result)

        elif change.change_type == CHANGE_MODIFIED:
            _handle_modified(change, templates_by_field, result)

        elif change.change_type == CHANGE_DESTRUCTIVE:
            _handle_destructive(change, templates_by_field, result)

        elif change.change_type == CHANGE_VALUE_DRIFT:
            _handle_value_drift(change, templates_by_field, result)

    return result


def validate_templates(
    templates: list[ClaimTemplate],
    schema_map: SchemaMap,
) -> list[dict]:
    """Check all templates are compatible with *schema_map*.

    Returns a list of issue dicts for templates referencing missing fields.
    Each dict has keys: ``template_claim_type``, ``field``, ``issue``.
    """
    field_names = {fp.field_name for fp in schema_map.field_profiles}
    issues: list[dict] = []

    for t in templates:
        # Check value_field
        if t.value_field and t.value_field not in field_names and t.value_field != "unknown_key":
            issues.append({
                "template_claim_type": t.claim_type,
                "field": t.value_field,
                "issue": "value_field not found in schema",
            })

        # Check entity_key_field
        if (
            t.entity_key_field
            and t.entity_key_field != "unknown_key"
            and t.entity_key_field not in field_names
        ):
            issues.append({
                "template_claim_type": t.claim_type,
                "field": t.entity_key_field,
                "issue": "entity_key_field not found in schema",
            })

        # Check source_config field paths
        for source_id, cfg in t.source_configs.items():
            path = cfg.get("field_path", "")
            leaf = path.rsplit(".", 1)[-1] if "." in path else path
            if leaf and leaf not in field_names:
                issues.append({
                    "template_claim_type": t.claim_type,
                    "field": leaf,
                    "issue": f"source_config field '{path}' (source={source_id}) not in schema",
                })

    return issues


# ── Internal handlers ────────────────────────────────────────────────


def _handle_additive(
    change: SchemaChange,
    new_schema: SchemaMap | None,
    result: TemplateMigrationResult,
) -> None:
    """New field → generate a stub template for review."""
    stub = ClaimTemplate(
        claim_type=f"auto.{change.field_name}",
        entity_key_field="unknown_key",
        value_field=change.field_name,
        source_configs={},
        confidence=0.0,
    )
    result.new_templates.append(stub)
    result.review_items.append(
        TemplateMigrationAction(
            template=stub,
            action="create",
            reason=f"New field '{change.field_name}' discovered ({change.new_value})",
            change=change,
            requires_review=False,
        )
    )


def _handle_modified(
    change: SchemaChange,
    templates_by_field: dict[str, list[ClaimTemplate]],
    result: TemplateMigrationResult,
) -> None:
    """Type/format changed → propose update, route to review."""
    affected = templates_by_field.get(change.field_name, [])
    seen: set[str] = set()
    for t in affected:
        key = t.claim_type + t.value_field
        if key in seen:
            continue
        seen.add(key)
        result.updated_templates.append(t)
        result.review_items.append(
            TemplateMigrationAction(
                template=t,
                action="update",
                reason=(
                    f"Field '{change.field_name}' type changed: "
                    f"{change.old_value} -> {change.new_value}"
                ),
                change=change,
                requires_review=True,
            )
        )


def _handle_destructive(
    change: SchemaChange,
    templates_by_field: dict[str, list[ClaimTemplate]],
    result: TemplateMigrationResult,
) -> None:
    """Field removed → suspend affected templates."""
    affected = templates_by_field.get(change.field_name, [])
    seen: set[str] = set()
    for t in affected:
        key = t.claim_type + t.value_field
        if key in seen:
            continue
        seen.add(key)
        result.suspended_templates.append(t)
        result.review_items.append(
            TemplateMigrationAction(
                template=t,
                action="suspend",
                reason=f"Field '{change.field_name}' no longer present in source",
                change=change,
                requires_review=True,
            )
        )


def _handle_value_drift(
    change: SchemaChange,
    templates_by_field: dict[str, list[ClaimTemplate]],
    result: TemplateMigrationResult,
) -> None:
    """Value distribution shifted → route to review."""
    affected = templates_by_field.get(change.field_name, [])
    seen: set[str] = set()
    for t in affected:
        key = t.claim_type + t.value_field
        if key in seen:
            continue
        seen.add(key)
        result.review_items.append(
            TemplateMigrationAction(
                template=t,
                action="update",
                reason=(
                    f"Fill rate for '{change.field_name}' shifted: "
                    f"{change.old_value} -> {change.new_value}"
                ),
                change=change,
                requires_review=True,
            )
        )
