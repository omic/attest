"""Confidence computation — Tier 1: source-type weights, Tier 2: corroboration + recency."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

SOURCE_TYPE_WEIGHTS: dict[str, float] = {
    # Built-in vocabulary
    "observation": 0.70,
    "computation": 0.50,
    "document_extraction": 0.60,
    "llm_inference": 0.30,
    "human_annotation": 0.90,
    # Bio vocabulary
    "experimental": 0.85,
    "crystallography": 0.95,
    "mass_spec": 0.80,
    "docking": 0.40,
    "literature": 0.65,
    # Bio-specific source types
    "experimental_crystallography": 1.0,
    "experimental_cryo_em": 0.95,
    "experimental_spr": 0.9,
    "experimental_assay": 0.85,
    "experimental_mass_spec": 0.85,
    "computational_alphafold": 0.6,
    "computational_docking": 0.4,
    "computational_md": 0.5,
    "computational_ml_prediction": 0.35,
    "literature_extraction": 0.50,
    "database_import": 0.70,
    "expert_annotation": 0.80,
    # Multi-source loaders
    "pathway_database": 0.70,
}

DEFAULT_WEIGHT = 0.50

# Source reliability — rates individual data sources by curation quality.
# Applied as a weight multiplier when aggregating evidence across sources.
# Higher = more reliable directional claims. Used by directional_confidence.
SOURCE_RELIABILITY: dict[str, float] = {
    # Expert-curated databases (directional claims are reviewed)
    "reactome": 1.0,        # manually curated pathways
    "drugbank": 0.95,       # expert-reviewed drug-target
    "ctd": 0.85,            # curated chemical-gene interactions
    "pharmgkb": 0.90,       # pharmacogenomic expert curation
    "dgidb": 0.85,          # drug-gene interaction aggregator
    "clinvar": 0.90,        # clinical variant significance
    # Computationally derived (high-throughput, less curation)
    "string_ppi": 0.70,     # text-mining + experimental combined score
    "biogrid": 0.75,        # curated interactions but no directionality
    "intact": 0.75,         # molecular interaction data
    "hetionet": 0.70,       # integrated hetnet (mixed quality)
    "primekg": 0.70,        # composite KG
    "pharmebinet": 0.65,    # composite KG
    # NLP-extracted (directional accuracy ~70%)
    "semmeddb": 0.50,       # SemRep NLP extraction from PubMed
    "kg2c": 0.55,           # RTX-KG2 (mixed NLP + curated)
    # Low-confidence
    "monarch_kg": 0.60,     # mixed provenance
}


def source_reliability(source_id: str) -> float:
    """Get reliability weight for a source.

    Extracts the database name from source_id (e.g., "semmeddb:12345" → "semmeddb")
    and looks up its reliability score.
    """
    db_name = source_id.split(":")[0] if ":" in source_id else source_id
    return SOURCE_RELIABILITY.get(db_name, DEFAULT_WEIGHT)

# Tier 2 parameters
HALF_LIFE_DAYS = 365
CORROBORATION_CAP = 1.7

# Default per-predicate half-lives (days). Predicates not listed use default.
DEFAULT_PREDICATE_HALF_LIVES: dict[str, int] = {
    # Status/operational — changes rapidly
    "has_status": 30,
    "has_warning": 90,
    "has_vulnerability": 90,
    "has_issue": 90,
    # Durable scientific knowledge — decays slowly
    "interacts_with": 730,
    "binds": 730,
    "phosphorylates": 730,
    "inhibits": 730,
    "activates": 730,
    "catalyzes": 730,
    "encodes": 1460,
    "transcribes": 1460,
    # General relationships — default pace
    "associates_with": 365,
    "relates_to": 365,
    "causes": 365,
    "treats": 365,
}


@dataclass
class DecayConfig:
    """Configuration for time-weighted confidence decay.

    Decay is query-time only — stored claims are never modified.
    """

    default_half_life_days: int = 365
    predicate_half_lives: dict[str, int] = field(default_factory=dict)
    enabled: bool = True

    def half_life_for(self, predicate_id: str) -> int:
        """Resolve half-life for a predicate: specific override > default."""
        return self.predicate_half_lives.get(
            predicate_id,
            DEFAULT_PREDICATE_HALF_LIVES.get(predicate_id, self.default_half_life_days),
        )


def tier1_confidence(source_type: str) -> float:
    """Source-type weight only. No corroboration. No decay."""
    return SOURCE_TYPE_WEIGHTS.get(source_type, DEFAULT_WEIGHT)


def count_independent_sources(claims: list) -> int:
    """Count claims with independent provenance (no shared ancestors).

    Two claims are independent if their provenance chains and source_ids
    don't overlap. Claims sharing a common provenance ancestor are grouped.

    Args:
        claims: list of Claim objects sharing the same content_id.

    Returns:
        Number of independent provenance groups.
    """
    if not claims:
        return 0
    if len(claims) == 1:
        return 1

    # Build provenance sets for each claim
    claim_chains: list[set[str]] = []
    for c in claims:
        chain_set = set(c.provenance.chain) if c.provenance.chain else set()
        chain_set.add(c.provenance.source_id)
        claim_chains.append(chain_set)

    # Greedy grouping: two claims are independent if their chain sets don't overlap
    groups: list[set[str]] = []
    for chain in claim_chains:
        merged = False
        for group in groups:
            if chain & group:
                group.update(chain)
                merged = True
                break
        if not merged:
            groups.append(set(chain))

    return len(groups)


def corroboration_boost(n_independent: int) -> float:
    """Compute corroboration multiplier.

    1 source = 1.0x, 2 sources = 1.3x, 4 sources = 1.6x, capped at 1.7x.
    Formula: 1.0 + 0.3 * log2(max(n, 1))
    """
    if n_independent <= 1:
        return 1.0
    boost = 1.0 + 0.3 * math.log2(n_independent)
    return min(boost, CORROBORATION_CAP)


def recency_factor(claim_timestamp_ns: int, half_life_days: int = HALF_LIFE_DAYS) -> float:
    """Compute recency decay factor.

    Recent claims get factor close to 1.0.
    Claims older than half_life_days get factor ~0.5.
    """
    now_ns = time.time_ns()
    age_seconds = (now_ns - claim_timestamp_ns) / 1_000_000_000
    if age_seconds <= 0:
        return 1.0
    age_days = age_seconds / 86400
    if half_life_days <= 0:
        return 0.0
    return 0.5 ** (age_days / half_life_days)


def tier2_confidence(
    claim,
    corroborating_claims: list | None = None,
    half_life_days: int = HALF_LIFE_DAYS,
) -> float:
    """Source weight * corroboration boost * recency decay.

    Args:
        claim: The Claim object to score.
        corroborating_claims: All claims sharing the same content_id.
            If None, falls back to tier1 with recency.
        half_life_days: Half-life for recency decay.

    Returns:
        Confidence score in [0.0, 1.0].
    """
    base = tier1_confidence(claim.provenance.source_type)

    if corroborating_claims:
        n_independent = count_independent_sources(corroborating_claims)
        boost = corroboration_boost(n_independent)
    else:
        boost = 1.0

    recency = recency_factor(claim.timestamp, half_life_days)

    return min(base * boost * recency, 1.0)


def effective_confidence(
    claim,
    decay_config: DecayConfig,
    corroborating_claims: list | None = None,
) -> float:
    """Tier 2 confidence with predicate-aware decay half-life.

    Resolves the correct half-life for the claim's predicate via
    *decay_config*, then delegates to :func:`tier2_confidence`.
    When decay is disabled, falls back to the default 365-day half-life.
    """
    if not decay_config.enabled:
        return tier2_confidence(claim, corroborating_claims)
    hl = decay_config.half_life_for(claim.predicate.id)
    return tier2_confidence(claim, corroborating_claims, half_life_days=hl)


# ---------------------------------------------------------------------------
# LLM confidence calibration
# ---------------------------------------------------------------------------

LLM_CONFIDENCE_MAP: dict[str, float] = {
    "high": 0.85,
    "very high": 0.90,
    "medium": 0.60,
    "moderate": 0.60,
    "low": 0.35,
    "very low": 0.20,
    "uncertain": 0.30,
}


def calibrate_llm_confidence(
    value: str | float | int,
    source_type: str = "llm_inference",
) -> float:
    """Convert LLM confidence to calibrated numeric score.

    Handles:
    - Categorical: "high" → 0.85, "medium" → 0.60, "low" → 0.35
    - Numeric strings: "0.9" → 0.72 (compressed toward center)
    - Already numeric: 0.9 → 0.72 (compressed toward center)

    LLMs are overconfident — they cluster at 0.85-0.95.  Compression
    maps the LLM's [0.5, 1.0] range to a more useful [0.35, 0.85] range.

    Args:
        value: Raw confidence from an LLM (string label, numeric string,
            or float/int).
        source_type: Unused today but reserved for per-provider calibration.

    Returns:
        Calibrated confidence in [0.1, 0.95].
    """
    # Categorical lookup (case-insensitive, stripped)
    if isinstance(value, str):
        key = value.strip().lower()
        if key in LLM_CONFIDENCE_MAP:
            return LLM_CONFIDENCE_MAP[key]
        # Try parsing as a number
        try:
            value = float(key)
        except ValueError:
            # Unknown label — return a neutral default
            return 0.5

    raw = float(value)

    # Compress: map [0.5, 1.0] → [0.35, 0.85]
    calibrated = 0.35 + (raw - 0.5) * (0.85 - 0.35) / (1.0 - 0.5)

    # Clamp to [0.1, 0.95]
    return max(0.1, min(0.95, calibrated))
