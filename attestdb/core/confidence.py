"""Confidence computation — Tier 1: source-type weights, Tier 2: corroboration + recency."""

from __future__ import annotations

import math
import time

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

# Tier 2 parameters
HALF_LIFE_DAYS = 365
CORROBORATION_CAP = 1.7


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
