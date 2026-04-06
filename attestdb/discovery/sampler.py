"""Schema sampling and field profiling for auto-connector discovery.

Given a connector, samples records and computes per-field statistics
including fill rate, type distribution, cardinality, PII detection,
and statistical summaries.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── PII detection patterns ────────────────────────────────────────────

_PII_FIELD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|[_\s])email(?:$|[_\s])", re.IGNORECASE),
    re.compile(r"(?:^|[_\s])phone(?:$|[_\s])", re.IGNORECASE),
    re.compile(r"(?:^|[_\s])ssn(?:$|[_\s])", re.IGNORECASE),
    re.compile(r"social_security", re.IGNORECASE),
    re.compile(r"(?:^|[_\s])address(?:$|[_\s])", re.IGNORECASE),
    re.compile(r"(?:first|last|full)[_\s]?name", re.IGNORECASE),
    re.compile(r"(?:^|[_\s])dob(?:$|[_\s])", re.IGNORECASE),
    re.compile(r"date_of_birth", re.IGNORECASE),
    re.compile(r"passport", re.IGNORECASE),
    re.compile(r"driver_license", re.IGNORECASE),
    re.compile(r"credit_card", re.IGNORECASE),
]

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
_PHONE_RE = re.compile(r"^[\+]?[(]?\d{1,4}[)]?[\s.\-]\d{1,4}[\s.\-]\d{4,}$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_CREDIT_CARD_RE = re.compile(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$")

# ── Date detection ────────────────────────────────────────────────────

_DATE_PATTERNS: list[re.Pattern[str]] = [
    # ISO 8601
    re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?"),
    # US: MM/DD/YYYY or MM-DD-YYYY
    re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$"),
    # EU: DD.MM.YYYY
    re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}$"),
]


# ── Dataclass ─────────────────────────────────────────────────────────


@dataclass
class FieldProfile:
    """Statistical profile of a single field across sampled records."""

    field_name: str
    parent_object: str
    fill_rate: float
    type_distribution: dict[str, float] = field(default_factory=dict)
    cardinality: int = 0
    value_samples: list[str] = field(default_factory=list)
    statistical_summary: dict = field(default_factory=dict)
    is_pii: bool = False
    tenant_id: str = "default"


# ── Public API ────────────────────────────────────────────────────────


def sample_source(
    connector: object,
    sample_size: int = 1000,
    tenant_id: str = "default",
) -> list[dict]:
    """Pull a sample of records from *connector*.

    Tries ``connector.sample(sample_size)`` first. If the connector does
    not expose a ``sample`` method, falls back to ``connector.fetch()``
    and collects up to *sample_size* records.

    Returns a list of raw record dicts.
    """
    if hasattr(connector, "sample") and callable(connector.sample):
        logger.info("Using connector.sample(%d)", sample_size)
        records = list(connector.sample(sample_size))
    elif hasattr(connector, "fetch") and callable(connector.fetch):
        logger.info("Falling back to connector.fetch() with limit %d", sample_size)
        records: list[dict] = []
        for record in connector.fetch():
            records.append(record)
            if len(records) >= sample_size:
                break
    else:
        raise TypeError(
            f"Connector {type(connector).__name__} has neither sample() nor fetch()"
        )

    logger.info("Sampled %d records (tenant=%s)", len(records), tenant_id)
    return records


def analyze_fields(
    samples: list[dict],
    parent_object: str = "",
    tenant_id: str = "default",
) -> list[FieldProfile]:
    """Compute a :class:`FieldProfile` for each field across *samples*.

    Handles flat dicts — nested objects are serialised to their string
    representation for profiling purposes.
    """
    if not samples:
        return []

    # Collect all field names across all samples
    all_fields: set[str] = set()
    for record in samples:
        all_fields.update(record.keys())

    total = len(samples)
    profiles: list[FieldProfile] = []

    for fname in sorted(all_fields):
        values: list[object] = []
        non_null_count = 0

        for record in samples:
            val = record.get(fname)
            values.append(val)
            if val is not None:
                non_null_count += 1

        fill_rate = non_null_count / total if total > 0 else 0.0
        non_null_values = [v for v in values if v is not None]

        type_dist = _compute_type_distribution(non_null_values, total)
        str_values = [str(v) for v in non_null_values]
        unique_values = set(str_values)
        cardinality = len(unique_values)

        # Representative samples (up to 10, anonymised if PII)
        sample_vals = _select_samples(str_values, max_samples=10)
        is_pii = detect_pii(fname, sample_vals)
        if is_pii:
            sample_vals = [_anonymize(v) for v in sample_vals]

        stats = _compute_statistics(non_null_values, type_dist)

        profiles.append(
            FieldProfile(
                field_name=fname,
                parent_object=parent_object,
                fill_rate=round(fill_rate, 4),
                type_distribution=type_dist,
                cardinality=cardinality,
                value_samples=sample_vals,
                statistical_summary=stats,
                is_pii=is_pii,
                tenant_id=tenant_id,
            )
        )

    return profiles


def detect_pii(field_name: str, value_samples: list[str]) -> bool:
    """Check whether a field likely contains personally identifiable information.

    Uses both field-name heuristics and value-pattern matching.
    """
    # Check field name
    for pattern in _PII_FIELD_PATTERNS:
        if pattern.search(field_name):
            return True

    # Check value patterns
    for val in value_samples:
        val = val.strip()
        if not val:
            continue
        if _EMAIL_RE.match(val):
            return True
        if _PHONE_RE.match(val):
            return True
        if _SSN_RE.match(val):
            return True
        if _CREDIT_CARD_RE.match(val):
            return True

    return False


# ── Internal helpers ──────────────────────────────────────────────────


def _detect_value_type(value: object) -> str:
    """Classify a single non-null value into a type string."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        s = value.strip()
        # Try boolean strings
        if s.lower() in ("true", "false"):
            return "boolean"
        # Try int
        try:
            int(s)
            return "int"
        except (ValueError, OverflowError):
            pass
        # Try float
        try:
            float(s)
            return "float"
        except (ValueError, OverflowError):
            pass
        # Try date
        for pattern in _DATE_PATTERNS:
            if pattern.match(s):
                return "date"
        return "string"
    # Lists, dicts, etc.
    return type(value).__name__


def _compute_type_distribution(
    non_null_values: list[object], total: int
) -> dict[str, float]:
    """Return detected types mapped to their frequency (0-1)."""
    if total == 0:
        return {}

    type_counts: dict[str, int] = {}
    null_count = total - len(non_null_values)

    for val in non_null_values:
        t = _detect_value_type(val)
        type_counts[t] = type_counts.get(t, 0) + 1

    if null_count > 0:
        type_counts["null"] = null_count

    return {t: round(c / total, 4) for t, c in sorted(type_counts.items())}


def _compute_statistics(
    non_null_values: list[object], type_dist: dict[str, float]
) -> dict:
    """Compute summary statistics appropriate for the dominant type."""
    if not non_null_values:
        return {}

    # Determine if field is primarily numeric
    numeric_frac = type_dist.get("int", 0.0) + type_dist.get("float", 0.0)

    if numeric_frac > 0.5:
        return _numeric_stats(non_null_values)

    # String statistics
    string_frac = type_dist.get("string", 0.0) + type_dist.get("date", 0.0)
    if string_frac > 0.5:
        return _string_stats(non_null_values)

    return {}


def _numeric_stats(values: list[object]) -> dict:
    """Min, max, mean, stddev for values coercible to float."""
    nums: list[float] = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue

    if not nums:
        return {}

    n = len(nums)
    mean = sum(nums) / n
    variance = sum((x - mean) ** 2 for x in nums) / n if n > 1 else 0.0

    return {
        "min": min(nums),
        "max": max(nums),
        "mean": round(mean, 4),
        "stddev": round(math.sqrt(variance), 4),
    }


def _string_stats(values: list[object]) -> dict:
    """Length statistics for string values."""
    lengths = [len(str(v)) for v in values]
    if not lengths:
        return {}

    n = len(lengths)
    mean_len = sum(lengths) / n

    return {
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": round(mean_len, 4),
    }


def _select_samples(str_values: list[str], max_samples: int = 10) -> list[str]:
    """Pick up to *max_samples* representative values.

    Prefers unique values, evenly spaced through the sorted set.
    """
    unique = sorted(set(str_values))
    if len(unique) <= max_samples:
        return unique

    # Evenly spaced selection
    step = len(unique) / max_samples
    return [unique[int(i * step)] for i in range(max_samples)]


def _anonymize(value: str) -> str:
    """Mask a PII value, keeping only structure hints."""
    if _EMAIL_RE.match(value):
        parts = value.split("@")
        return f"{parts[0][0]}***@{parts[1]}"
    if _SSN_RE.match(value):
        return "***-**-" + value[-4:]
    if _CREDIT_CARD_RE.match(value.replace(" ", "").replace("-", "")):
        return "****-****-****-" + value.replace(" ", "").replace("-", "")[-4:]
    # Generic: keep first and last character
    if len(value) > 2:
        return value[0] + "*" * (len(value) - 2) + value[-1]
    return "***"
