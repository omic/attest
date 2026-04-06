"""Claim template generation from aligned schemas.

Given a :class:`UnifiedSchema` and the original :class:`SchemaMap` instances,
generates :class:`ClaimTemplate` objects that describe how to extract
structured claims from each source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from attestdb.discovery.aligner import AlignedFieldGroup, UnifiedSchema
from attestdb.discovery.sampler import FieldProfile
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)

# ── Patterns for detecting entity key fields ─────────────────────────

_ID_FIELD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|[_\s])id$", re.IGNORECASE),
    re.compile(r"_id$", re.IGNORECASE),
    re.compile(r"Id$"),
    re.compile(r"(?:account|customer|contact|user|org|company)(?:_?id)?$", re.IGNORECASE),
]

# Minimum confidence for emitting a template
_MIN_TEMPLATE_CONFIDENCE = 0.3


# ── Dataclass ────────────────────────────────────────────────────────


@dataclass
class ClaimTemplate:
    """A recipe for extracting claims from aligned source fields."""

    claim_type: str
    entity_key_field: str
    value_field: str
    source_configs: dict[str, dict] = field(default_factory=dict)
    confidence: float = 0.0
    tenant_id: str = "default"


# ── Public API ───────────────────────────────────────────────────────


def generate_claim_templates(
    unified_schema: UnifiedSchema,
    schema_maps: list[SchemaMap],
    tenant_id: str = "default",
    llm_client: object | None = None,
) -> list[ClaimTemplate]:
    """Generate claim templates from aligned field groups.

    For each :class:`AlignedFieldGroup` in the unified schema, this function:

    1. Sets ``claim_type`` from the group's ``semantic_type``.
    2. Detects the ``entity_key_field`` — a field in the same
       ``parent_object`` that looks like a primary/foreign key.
    3. Builds ``source_configs`` mapping each source to its extraction path.
    4. Computes ``confidence`` as ``alignment_confidence * avg_fill_rate``.

    Templates with confidence below 0.3 are discarded.  The result is
    sorted by confidence descending.
    """
    # Build lookup: source_id -> field_name -> FieldProfile
    profile_index: dict[str, dict[str, FieldProfile]] = {}
    for sm in schema_maps:
        p_map: dict[str, FieldProfile] = {}
        for fp in sm.field_profiles:
            p_map[fp.field_name] = fp
        profile_index[sm.source_id] = p_map

    # Build lookup: source_id -> parent_object -> list[FieldProfile]
    parent_index: dict[str, dict[str, list[FieldProfile]]] = {}
    for sm in schema_maps:
        by_parent: dict[str, list[FieldProfile]] = {}
        for fp in sm.field_profiles:
            by_parent.setdefault(fp.parent_object, []).append(fp)
        parent_index[sm.source_id] = by_parent

    # Optionally ask LLM to suggest a claim_type taxonomy for all groups
    llm_taxonomy = _suggest_taxonomy_via_llm(unified_schema.aligned_fields, llm_client)

    templates: list[ClaimTemplate] = []

    for group in unified_schema.aligned_fields:
        if group.semantic_type == "unknown" or group.semantic_type == "identifier":
            continue

        # Compute average fill rate across sources for this value field
        fill_rates: list[float] = []
        for source_id, field_name in group.source_mappings.items():
            prof = profile_index.get(source_id, {}).get(field_name)
            if prof:
                fill_rates.append(prof.fill_rate)

        avg_fill = sum(fill_rates) / len(fill_rates) if fill_rates else 0.5
        confidence = group.alignment_confidence * avg_fill

        if confidence < _MIN_TEMPLATE_CONFIDENCE:
            continue

        # Detect entity key field: look for *Id / *_id in the same parent_object
        entity_key = _detect_entity_key(group, profile_index, parent_index)

        # Build source configs
        source_configs: dict[str, dict] = {}
        for source_id, field_name in group.source_mappings.items():
            prof = profile_index.get(source_id, {}).get(field_name)
            parent = prof.parent_object if prof else ""
            field_path = f"{parent}.{field_name}" if parent else field_name
            source_configs[source_id] = {
                "field_path": field_path,
                "transform": None,
            }

        claim_type = llm_taxonomy.get(group.canonical_name, group.semantic_type)

        template = ClaimTemplate(
            claim_type=claim_type,
            entity_key_field=entity_key,
            value_field=group.canonical_name,
            source_configs=source_configs,
            confidence=round(confidence, 4),
            tenant_id=tenant_id,
        )
        templates.append(template)

    # Sort by confidence descending
    templates.sort(key=lambda t: t.confidence, reverse=True)
    return templates


def templates_to_report(templates: list[ClaimTemplate]) -> str:
    """Generate a human-readable report of claim templates.

    For each template, shows claim type, entity key, contributing sources,
    and confidence score.
    """
    if not templates:
        return "No claim templates generated."

    lines = [
        "Claim Template Report",
        "=" * 60,
        f"Total templates: {len(templates)}",
        "",
    ]

    for i, t in enumerate(templates, 1):
        sources = ", ".join(sorted(t.source_configs.keys()))
        lines.append(f"{i}. {t.claim_type}")
        lines.append(f"   Entity key:  {t.entity_key_field}")
        lines.append(f"   Value field: {t.value_field}")
        lines.append(f"   Sources:     {sources}")
        lines.append(f"   Confidence:  {t.confidence:.2f}")

        # Show per-source field paths
        for sid, cfg in sorted(t.source_configs.items()):
            path = cfg.get("field_path", "?")
            transform = cfg.get("transform")
            suffix = f" (transform: {transform})" if transform else ""
            lines.append(f"     {sid}: {path}{suffix}")

        lines.append("")

    return "\n".join(lines)


# ── LLM taxonomy suggestion ─────────────────────────────────────────


def _suggest_taxonomy_via_llm(
    groups: list[AlignedFieldGroup],
    llm_client: object | None,
) -> dict[str, str]:
    """Ask the LLM to suggest hierarchical claim_type strings for aligned fields.

    Returns ``{canonical_name: claim_type}`` mapping.  Falls back to an empty
    dict (caller uses ``semantic_type`` as default) when no LLM is available
    or the call fails.
    """
    if llm_client is None or not groups:
        return {}

    import json as _json

    field_list = "\n".join(
        f"- {g.canonical_name} (semantic_type={g.semantic_type}, "
        f"sources={list(g.source_mappings.keys())})"
        for g in groups
        if g.semantic_type not in ("unknown", "identifier")
    )

    if not field_list:
        return {}

    prompt = (
        "You are a data modeling expert. Given the following aligned fields "
        "discovered from enterprise data sources, suggest a hierarchical "
        "claim_type string for each (e.g., 'customer.revenue.arr', "
        "'customer.satisfaction.nps', 'pipeline.deal.stage').\n\n"
        f"Fields:\n{field_list}\n\n"
        "Respond with JSON only: {\"<canonical_name>\": \"<claim_type>\", ...}"
    )

    try:
        response = llm_client.chat.completions.create(
            model=getattr(llm_client, "_model", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return _json.loads(text)
    except Exception as exc:
        logger.debug("LLM taxonomy suggestion failed: %s", exc)
        return {}


# ── Internal helpers ─────────────────────────────────────────────────


def _detect_entity_key(
    group: AlignedFieldGroup,
    profile_index: dict[str, dict[str, FieldProfile]],
    parent_index: dict[str, dict[str, list[FieldProfile]]],
) -> str:
    """Find the best entity key field for a value field group.

    Looks for fields in the same ``parent_object`` that match ID-like
    patterns (high cardinality, string/int type, names containing
    "account", "customer", "id", etc.).

    Returns the best candidate name, or ``"unknown_key"`` if none found.
    """
    candidates: dict[str, int] = {}  # field_name -> vote count

    for source_id, field_name in group.source_mappings.items():
        prof = profile_index.get(source_id, {}).get(field_name)
        if not prof:
            continue

        parent = prof.parent_object
        siblings = parent_index.get(source_id, {}).get(parent, [])

        for sibling in siblings:
            if sibling.field_name == field_name:
                continue  # Skip the value field itself
            if _is_entity_key(sibling):
                candidates[sibling.field_name] = candidates.get(sibling.field_name, 0) + 1

    if not candidates:
        return "unknown_key"

    # Return the most-voted candidate
    return max(candidates, key=candidates.get)


def _is_entity_key(profile: FieldProfile) -> bool:
    """Check whether a field profile looks like an entity key.

    Criteria:
    - Name matches an ID-like pattern, OR
    - High cardinality + string/int dominant type + high fill rate
    """
    # Name-based check
    for pattern in _ID_FIELD_PATTERNS:
        if pattern.search(profile.field_name):
            return True

    # Heuristic: high cardinality, string/int, well-populated
    dominant_types = {"string", "int"}
    actual_types = set(profile.type_distribution.keys()) - {"null"}
    return bool(
        actual_types
        and actual_types.issubset(dominant_types)
        and profile.cardinality > 10
        and profile.fill_rate > 0.80
    )
