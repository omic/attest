"""Tests for entity normalization — locked function."""

from attestdb.core.normalization import normalize_entity_id


def test_case_insensitive():
    assert normalize_entity_id("TREM2") == "trem2"
    assert normalize_entity_id("trem2") == "trem2"
    assert normalize_entity_id("Trem2") == "trem2"


def test_whitespace_collapse():
    assert normalize_entity_id("  foo   bar  ") == "foo bar"
    assert normalize_entity_id("foo\tbar") == "foo bar"
    assert normalize_entity_id("foo\nbar") == "foo bar"


def test_greek_letters():
    assert normalize_entity_id("β-amyloid") == "beta-amyloid"
    assert normalize_entity_id("α-synuclein") == "alpha-synuclein"
    assert normalize_entity_id("τ protein") == "tau protein"
    assert normalize_entity_id("μ-opioid") == "mu-opioid"


def test_nfkd_decomposition():
    # NFKD decomposes compatibility characters
    assert normalize_entity_id("ﬁbronectin") == "fibronectin"


def test_multiple_greek():
    assert normalize_entity_id("αβγ") == "alphabetagamma"


def test_idempotent():
    raw = "β-Amyloid  Precursor"
    first = normalize_entity_id(raw)
    second = normalize_entity_id(first)
    assert first == second
