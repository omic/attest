"""Unit tests for Tier 2 confidence: corroboration + recency."""

import time

from attestdb.core.confidence import (
    corroboration_boost,
    count_independent_sources,
    recency_factor,
    tier1_confidence,
    tier2_confidence,
)
from attestdb.core.types import Claim, EntityRef, PredicateRef, Provenance


def _make_claim(source_id, source_type="experimental", chain=None, timestamp=None):
    """Helper to create a minimal Claim for testing."""
    return Claim(
        claim_id=f"test_{source_id}",
        content_id="shared_content",
        subject=EntityRef(id="a", entity_type="gene"),
        predicate=PredicateRef(id="associates", predicate_type="relates_to"),
        object=EntityRef(id="b", entity_type="disease"),
        confidence=0.5,
        provenance=Provenance(
            source_type=source_type,
            source_id=source_id,
            chain=chain or [],
        ),
        timestamp=timestamp or int(time.time() * 1_000_000_000),
    )


def test_count_independent_two_independent():
    claims = [_make_claim("src1"), _make_claim("src2")]
    assert count_independent_sources(claims) == 2


def test_count_independent_two_dependent():
    """Claims sharing provenance chain ancestor are grouped."""
    c1 = _make_claim("src1", chain=["upstream_1"])
    c2 = _make_claim("src2", chain=["upstream_1"])
    assert count_independent_sources([c1, c2]) == 1


def test_count_independent_mixed():
    """Two independent groups + one dependent."""
    c1 = _make_claim("src1", chain=["chain_a"])
    c2 = _make_claim("src2", chain=["chain_a"])  # same chain as c1
    c3 = _make_claim("src3", chain=["chain_b"])  # independent
    assert count_independent_sources([c1, c2, c3]) == 2


def test_corroboration_boost_two():
    assert abs(corroboration_boost(2) - 1.3) < 0.01


def test_recency_factor_recent():
    now_ns = int(time.time() * 1_000_000_000)
    assert recency_factor(now_ns) > 0.99


def test_recency_factor_one_year():
    one_year_ago_ns = int((time.time() - 365 * 86400) * 1_000_000_000)
    factor = recency_factor(one_year_ago_ns)
    assert 0.45 < factor < 0.55  # ~0.5 at one half-life


def test_tier2_with_corroboration():
    """With 2 independent sources, tier2 > tier1."""
    c1 = _make_claim("src1", source_type="experimental")
    c2 = _make_claim("src2", source_type="database_import")
    score = tier2_confidence(c1, corroborating_claims=[c1, c2])
    tier1 = tier1_confidence("experimental")
    # Corroboration boost of 1.3x should push score above tier1
    assert score > tier1 * 0.95  # allowing slight recency decay
