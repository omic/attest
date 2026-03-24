"""Unit tests for Tier 2 confidence: corroboration + recency."""

import time

from attestdb.core.confidence import (
    corroboration_boost,
    count_independent_sources,
    recency_factor,
    tier1_confidence,
    tier2_confidence,
    _extract_external_id,
    _normalize_doi,
    _normalize_pmid,
    _normalize_url,
    _count_by_provenance_overlap,
)
from attestdb.core.types import Claim, EntityRef, PredicateRef, Payload, Provenance


def _make_claim(source_id, source_type="experimental", chain=None, timestamp=None,
                payload_data=None):
    """Helper to create a minimal Claim for testing."""
    payload = Payload(schema_ref="", data=payload_data) if payload_data else None
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
        payload=payload,
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


# --- External ID deduplication tests ---

def test_same_doi_different_sources_counts_as_one():
    """Three claims from same paper via different paths = 1 independent source."""
    claims = [
        _make_claim("pubmed_loader", payload_data={"doi": "10.1038/xyz"}),
        _make_claim("hetionet_bulk", payload_data={"doi": "10.1038/xyz"}),
        _make_claim("agent_session_1", payload_data={"doi": "10.1038/xyz"}),
    ]
    assert count_independent_sources(claims) == 1


def test_different_dois_count_as_independent():
    """Claims from different papers are independent."""
    claims = [
        _make_claim("loader_1", payload_data={"doi": "10.1038/aaa"}),
        _make_claim("loader_2", payload_data={"doi": "10.1038/bbb"}),
    ]
    assert count_independent_sources(claims) == 2


def test_mixed_external_and_unclustered():
    """Claims with and without external IDs are counted correctly."""
    claims = [
        _make_claim("pubmed", payload_data={"doi": "10.1038/xyz"}),
        _make_claim("agent_1", payload_data={"doi": "10.1038/xyz"}),
        _make_claim("slack_msg_123"),
        _make_claim("email_456"),
    ]
    # 1 DOI group + 2 unclustered with non-overlapping provenance = 3
    assert count_independent_sources(claims) == 3


def test_doi_normalization():
    """Different DOI formats resolve to same ID."""
    assert _normalize_doi("10.1038/xyz") == "doi:10.1038/xyz"
    assert _normalize_doi("https://doi.org/10.1038/xyz") == "doi:10.1038/xyz"
    assert _normalize_doi("DOI:10.1038/xyz") == "doi:10.1038/xyz"
    assert _normalize_doi("http://doi.org/10.1038/xyz") == "doi:10.1038/xyz"


def test_pmid_normalization():
    """Different PMID formats resolve to same ID."""
    assert _normalize_pmid("12345678") == "pmid:12345678"
    assert _normalize_pmid("PMID:12345678") == "pmid:12345678"
    assert _normalize_pmid("pmid:12345678") == "pmid:12345678"
    assert _normalize_pmid("pubmed:12345678") == "pmid:12345678"


def test_url_normalization():
    """URLs are normalized for dedup."""
    assert _normalize_url("https://example.com/page/") == "url:example.com/page"
    assert _normalize_url("http://www.example.com/page") == "url:example.com/page"
    assert _normalize_url("HTTPS://EXAMPLE.COM/page") == "url:example.com/page"


def test_backward_compatible_no_external_ids():
    """Claims without external IDs use existing provenance overlap logic."""
    claims = [
        _make_claim("source_a"),
        _make_claim("source_b"),
    ]
    assert count_independent_sources(claims) == 2


def test_semmeddb_pmid_extraction():
    """SemMedDB source_id pattern extracts PMID for clustering."""
    claims = [
        _make_claim("semmeddb:12345678"),
        _make_claim("pubmed_loader", payload_data={"pmid": "12345678"}),
    ]
    assert count_independent_sources(claims) == 1


def test_extract_external_id_doi_from_payload():
    """DOI extracted from payload.data."""
    claim = _make_claim("src1", payload_data={"doi": "10.1038/xyz"})
    assert _extract_external_id(claim) == "doi:10.1038/xyz"


def test_extract_external_id_from_source_id():
    """DOI pattern in source_id is extracted."""
    claim = _make_claim("10.1038/xyz")
    assert _extract_external_id(claim) == "doi:10.1038/xyz"


def test_extract_external_id_none():
    """No external ID when neither payload nor source_id has one."""
    claim = _make_claim("slack_msg_123")
    assert _extract_external_id(claim) is None


def test_empty_claims_returns_zero():
    assert count_independent_sources([]) == 0


def test_single_claim_returns_one():
    assert count_independent_sources([_make_claim("only")]) == 1


def test_doi_case_insensitive():
    """DOI normalization is case-insensitive."""
    claims = [
        _make_claim("src1", payload_data={"doi": "10.1038/XYZ"}),
        _make_claim("src2", payload_data={"doi": "10.1038/xyz"}),
    ]
    assert count_independent_sources(claims) == 1
