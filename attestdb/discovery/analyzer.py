"""Semantic field analysis for auto-connector discovery.

Given field profiles from the sampler, uses LLM inference (or heuristic
fallback) to classify each field into a semantic taxonomy and detect
deprecated fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from attestdb.core.providers import (
    EXTRACTION_FALLBACK_CHAIN,
    PROVIDERS,
    load_env_file as _load_env_file,
)
from attestdb.discovery.sampler import FieldProfile

logger = logging.getLogger(__name__)

# ── Confidence thresholds (cold start defaults) ──────────────────────

_AUTO_MAP_THRESHOLD = 0.95
_NEEDS_REVIEW_THRESHOLD = 0.60

# ── Heuristic mapping rules ──────────────────────────────────────────

_HEURISTIC_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"revenue|arr|mrr|bookings", re.IGNORECASE), "revenue"),
    (re.compile(r"churn", re.IGNORECASE), "churn"),
    (re.compile(r"nps|csat|satisfaction", re.IGNORECASE), "satisfaction"),
    (re.compile(r"health[_\s]?score", re.IGNORECASE), "satisfaction.health"),
    (re.compile(r"renewal", re.IGNORECASE), "renewal"),
    (re.compile(r"contract[_\s]?value|deal[_\s]?size|acv|tcv", re.IGNORECASE), "contract.value"),
    (re.compile(r"stage|pipeline", re.IGNORECASE), "pipeline.stage"),
    (re.compile(r"owner|rep|account[_\s]?manager", re.IGNORECASE), "ownership"),
    (re.compile(r"industry|sector|vertical", re.IGNORECASE), "classification.industry"),
    (re.compile(r"segment|tier", re.IGNORECASE), "classification.segment"),
    (re.compile(r"created|create[_\s]?date", re.IGNORECASE), "timestamp.created"),
    (re.compile(r"updated|modified|last[_\s]?modified", re.IGNORECASE), "timestamp.updated"),
    (re.compile(r"closed|close[_\s]?date", re.IGNORECASE), "timestamp.closed"),
    (re.compile(r"status", re.IGNORECASE), "status"),
    (re.compile(r"country|region|geo", re.IGNORECASE), "geography"),
    (re.compile(r"employee|headcount|size", re.IGNORECASE), "company.size"),
    (re.compile(r"email", re.IGNORECASE), "contact.email"),
    (re.compile(r"phone", re.IGNORECASE), "contact.phone"),
    (re.compile(r"name", re.IGNORECASE), "entity.name"),
    (re.compile(r"id$|_id$|Id$", re.IGNORECASE), "identifier"),
    (re.compile(r"url|website|link", re.IGNORECASE), "reference.url"),
    (re.compile(r"description|notes|comment", re.IGNORECASE), "text.description"),
    (re.compile(r"type|kind|category", re.IGNORECASE), "classification.type"),
    (re.compile(r"count|total|quantity|num", re.IGNORECASE), "metric.count"),
    (re.compile(r"amount|price|cost|fee", re.IGNORECASE), "financial.amount"),
    (re.compile(r"score|rating", re.IGNORECASE), "metric.score"),
]

# ── Deprecated field patterns ─────────────────────────────────────────

_DEPRECATED_NAME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^old_", re.IGNORECASE),
    re.compile(r"^deprecated_", re.IGNORECASE),
    re.compile(r"^do_not_use_", re.IGNORECASE),
    re.compile(r"^unused_", re.IGNORECASE),
    re.compile(r"^legacy_", re.IGNORECASE),
    re.compile(r"_old$", re.IGNORECASE),
    re.compile(r"_deprecated$", re.IGNORECASE),
]


# ── Dataclass ─────────────────────────────────────────────────────────


@dataclass
class SemanticMapping:
    """Maps a source field to a semantic type in the taxonomy."""

    field_name: str
    semantic_type: str
    confidence: float
    reasoning: str
    review_status: str  # "auto_mapped" | "needs_review" | "unmapped"
    tenant_id: str = "default"


# ── Public API ────────────────────────────────────────────────────────


def infer_semantics(
    field_profiles: list[FieldProfile],
    threshold_engine: object | None = None,
    tenant_id: str = "default",
) -> list[SemanticMapping]:
    """Classify each field profile into a semantic type.

    Attempts LLM-based classification first (using the provider fallback
    chain from ``attestdb.core.providers``).  Falls back to heuristic
    matching when no LLM is available.

    If *threshold_engine* is provided, calls
    ``threshold_engine.get_thresholds(source_id, "schema_mapping")`` to
    determine review status.  Otherwise uses cold-start defaults
    (auto_mapped >= 0.95, needs_review >= 0.60, unmapped below 0.60).
    """
    client, model = _get_llm_client()
    mappings: list[SemanticMapping] = []

    for profile in field_profiles:
        if client is not None:
            mapping = _infer_with_llm(client, model, profile, tenant_id)
        else:
            mapping = _infer_heuristic(profile, tenant_id)

        # Determine review_status
        if threshold_engine is not None:
            try:
                thresholds = threshold_engine.get_thresholds(
                    profile.parent_object, "schema_mapping"
                )
                auto_thresh = thresholds.get("auto_map", _AUTO_MAP_THRESHOLD)
                review_thresh = thresholds.get("needs_review", _NEEDS_REVIEW_THRESHOLD)
            except Exception:
                auto_thresh = _AUTO_MAP_THRESHOLD
                review_thresh = _NEEDS_REVIEW_THRESHOLD
        else:
            auto_thresh = _AUTO_MAP_THRESHOLD
            review_thresh = _NEEDS_REVIEW_THRESHOLD

        if mapping.confidence >= auto_thresh:
            mapping.review_status = "auto_mapped"
        elif mapping.confidence >= review_thresh:
            mapping.review_status = "needs_review"
        else:
            mapping.review_status = "unmapped"

        mappings.append(mapping)

    return mappings


def detect_deprecated_fields(field_profiles: list[FieldProfile]) -> list[str]:
    """Identify fields that appear deprecated or unused.

    Flags fields that have:
    - fill_rate < 0.05 (less than 5% populated)
    - Zero variance (cardinality == 1 and not boolean-typed)
    - Names matching deprecated patterns (old_*, deprecated_*, etc.)
    """
    deprecated: list[str] = []
    seen: set[str] = set()

    for profile in field_profiles:
        reasons: list[str] = []

        # Low fill rate
        if profile.fill_rate < 0.05:
            reasons.append("low_fill_rate")

        # Zero variance (cardinality 1, non-boolean)
        is_boolean = profile.type_distribution.get("boolean", 0.0) > 0.5
        if profile.cardinality == 1 and not is_boolean:
            reasons.append("zero_variance")

        # Name patterns
        for pattern in _DEPRECATED_NAME_PATTERNS:
            if pattern.search(profile.field_name):
                reasons.append("deprecated_name")
                break

        if reasons and profile.field_name not in seen:
            deprecated.append(profile.field_name)
            seen.add(profile.field_name)
            logger.debug(
                "Deprecated field: %s (%s)", profile.field_name, ", ".join(reasons)
            )

    return deprecated


# ── LLM helpers ───────────────────────────────────────────────────────

_CLASSIFY_SYSTEM_PROMPT = """\
You are a data schema analyst. Given a field description from an enterprise \
data source, classify it into a semantic type.

Respond with EXACTLY one JSON object:
{"semantic_type": "category.subcategory", "confidence": 0.85, "reasoning": "brief explanation"}

Common semantic types include:
- revenue.arr, revenue.mrr, revenue.bookings
- satisfaction.nps, satisfaction.csat, satisfaction.health
- churn.rate, churn.risk
- pipeline.stage, pipeline.value
- contract.value, contract.term
- classification.industry, classification.segment, classification.type
- timestamp.created, timestamp.updated, timestamp.closed
- contact.email, contact.phone
- entity.name, identifier
- geography, ownership, status
- metric.count, metric.score
- financial.amount
- text.description
- reference.url

Use your best judgment for fields that don't fit neatly. Be specific \
(prefer "revenue.arr" over just "revenue").\
"""


def _get_llm_client() -> tuple[object | None, str]:
    """Walk the extraction fallback chain and return the first available client.

    Returns (client, model_name) or (None, "") if no provider is available.
    """
    for provider_name in EXTRACTION_FALLBACK_CHAIN:
        provider = PROVIDERS.get(provider_name)
        if not provider:
            continue

        api_key = os.environ.get(provider["env_key"])
        if not api_key:
            # Try .env file
            env_vars = _load_env_file(".env")
            api_key = env_vars.get(provider["env_key"])

        if not api_key:
            continue

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=provider["base_url"],
            )
            model = provider["default_model"]
            logger.info(
                "Schema analyzer using provider=%s, model=%s", provider_name, model
            )
            return client, model
        except ImportError:
            logger.debug("openai package not installed")
            return None, ""
        except Exception as exc:
            logger.warning("Failed to init %s client: %s", provider_name, exc)
            continue

    logger.info("No LLM provider available; using heuristic classification")
    return None, ""


def _infer_with_llm(
    client: object,
    model: str,
    profile: FieldProfile,
    tenant_id: str,
) -> SemanticMapping:
    """Classify a single field using an LLM call."""
    # Build a concise description of the field
    sample_display = ", ".join(profile.value_samples[:5]) if profile.value_samples else "(no samples)"
    prompt = (
        f"Field: {profile.field_name}\n"
        f"Object: {profile.parent_object}\n"
        f"Types: {profile.type_distribution}\n"
        f"Fill rate: {profile.fill_rate}\n"
        f"Cardinality: {profile.cardinality}\n"
        f"Samples: {sample_display}\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=256,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""

        # Extract JSON from response (handle markdown fences)
        # Strip markdown code fences first
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        # Find outermost braces
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            result = json.loads(content[start : end + 1])
            return SemanticMapping(
                field_name=profile.field_name,
                semantic_type=result.get("semantic_type", "unknown"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                review_status="",  # Set by caller
                tenant_id=tenant_id,
            )
    except Exception as exc:
        logger.warning(
            "LLM classification failed for field '%s': %s; falling back to heuristic",
            profile.field_name,
            exc,
        )

    # Fallback to heuristic on LLM failure
    return _infer_heuristic(profile, tenant_id)


def _infer_heuristic(profile: FieldProfile, tenant_id: str) -> SemanticMapping:
    """Classify a field using rule-based heuristics."""
    for pattern, semantic_type in _HEURISTIC_RULES:
        if pattern.search(profile.field_name):
            return SemanticMapping(
                field_name=profile.field_name,
                semantic_type=semantic_type,
                confidence=0.70,
                reasoning=f"Heuristic: field name matches '{pattern.pattern}'",
                review_status="",  # Set by caller
                tenant_id=tenant_id,
            )

    return SemanticMapping(
        field_name=profile.field_name,
        semantic_type="unknown",
        confidence=0.0,
        reasoning="No heuristic rule matched",
        review_status="",  # Set by caller
        tenant_id=tenant_id,
    )
