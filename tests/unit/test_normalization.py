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


def test_zero_width_space_stripped():
    """U+200B zero-width space must not create invisible duplicates."""
    assert normalize_entity_id("TREM2\u200b") == "trem2"
    assert normalize_entity_id("TREM2") == normalize_entity_id("TREM2\u200b")


def test_zero_width_non_joiner_stripped():
    """U+200C zero-width non-joiner stripped."""
    assert normalize_entity_id("foo\u200Cbar") == "foobar"


def test_zero_width_joiner_stripped():
    """U+200D zero-width joiner stripped."""
    assert normalize_entity_id("foo\u200Dbar") == "foobar"


def test_bom_stripped():
    """U+FEFF BOM / zero-width no-break space stripped."""
    assert normalize_entity_id("\uFEFFprotein") == "protein"


def test_directional_marks_stripped():
    """U+200E LRM and U+200F RLM stripped."""
    assert normalize_entity_id("hello\u200Eworld") == "helloworld"
    assert normalize_entity_id("hello\u200Fworld") == "helloworld"


def test_multiple_cf_chars_stripped():
    """Multiple invisible format chars in sequence are all removed."""
    assert normalize_entity_id("\u200B\u200C\u200DTREM2\uFEFF") == "trem2"


def test_soft_hyphen_stripped():
    """U+00AD soft hyphen (Cf) stripped."""
    assert normalize_entity_id("fibro\u00ADnectin") == "fibronectin"
