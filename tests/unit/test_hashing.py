"""Tests for claim and content ID computation — locked functions."""

import hashlib

from attestdb.core.hashing import compute_claim_id, compute_content_id


def test_claim_id_pipe_delimited():
    """Verify the exact hash matches pipe-delimited SHA-256."""
    expected_payload = "trem2|binds|apoe|PMID:123|experimental|1000"
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    actual = compute_claim_id("trem2", "binds", "apoe", "PMID:123", "experimental", 1000)
    assert actual == expected


def test_claim_id_different_source_differs():
    id1 = compute_claim_id("trem2", "binds", "apoe", "PMID:123", "experimental", 1000)
    id2 = compute_claim_id("trem2", "binds", "apoe", "PMID:456", "experimental", 1000)
    assert id1 != id2


def test_claim_id_different_timestamp_differs():
    id1 = compute_claim_id("trem2", "binds", "apoe", "PMID:123", "experimental", 1000)
    id2 = compute_claim_id("trem2", "binds", "apoe", "PMID:123", "experimental", 2000)
    assert id1 != id2


def test_content_id_deterministic():
    id1 = compute_content_id("trem2", "binds", "apoe")
    id2 = compute_content_id("trem2", "binds", "apoe")
    assert id1 == id2


def test_content_id_pipe_delimited():
    expected_payload = "trem2|binds|apoe"
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    actual = compute_content_id("trem2", "binds", "apoe")
    assert actual == expected


def test_same_content_different_source_shares_content_id():
    """Claims about the same fact from different sources share content_id but differ in claim_id."""
    # Two claims with same subject/predicate/object but different sources
    claim1 = compute_claim_id("trem2", "binds", "apoe", "PMID:123", "experimental", 1000)
    claim2 = compute_claim_id("trem2", "binds", "apoe", "PMID:456", "computational", 2000)
    # claim_ids must differ (different source/timestamp)
    assert claim1 != claim2

    # But content_ids must be identical (same subject/predicate/object)
    cid1 = compute_content_id("trem2", "binds", "apoe")
    cid2 = compute_content_id("trem2", "binds", "apoe")
    assert cid1 == cid2
    # Verify content_id is independent of source (by construction — only takes s/p/o)


def test_content_id_differs_for_different_relationships():
    id1 = compute_content_id("trem2", "binds", "apoe")
    id2 = compute_content_id("trem2", "inhibits", "apoe")
    assert id1 != id2


# ── Vocabulary invariants (relocated from test_vocabulary.py) ──────


def test_opposite_predicates_symmetric():
    """If A opposes B, then B must oppose A."""
    from attestdb.core.vocabulary import OPPOSITE_PREDICATES

    for a, b in OPPOSITE_PREDICATES.items():
        assert b in OPPOSITE_PREDICATES, f"{b!r} not in OPPOSITE_PREDICATES"
        assert OPPOSITE_PREDICATES[b] == a, (
            f"OPPOSITE_PREDICATES[{b!r}] = {OPPOSITE_PREDICATES[b]!r}, expected {a!r}"
        )


def test_no_overlap_symmetric_and_opposites():
    """Symmetric predicates should not appear in opposite pairs."""
    from attestdb.core.vocabulary import OPPOSITE_PREDICATES, SYMMETRIC_PREDICATES

    for pred in SYMMETRIC_PREDICATES:
        assert pred not in OPPOSITE_PREDICATES, (
            f"{pred!r} is both symmetric and has an opposite"
        )
