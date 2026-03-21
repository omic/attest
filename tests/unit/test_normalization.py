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


# --- Predicate normalization tests ---

def test_predicate_normalization_standard():
    """Standard predicates pass through unchanged."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("inhibits") == "inhibits"
    assert normalize_predicate("activates") == "activates"
    assert normalize_predicate("associated_with") == "associated_with"


def test_predicate_normalization_verb_stems():
    """Verb stems normalized to 3rd person singular."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("activate") == "activates"
    assert normalize_predicate("inhibit") == "inhibits"
    assert normalize_predicate("reduce") == "reduces"
    assert normalize_predicate("treat") == "treats"
    assert normalize_predicate("upregulate") == "upregulates"


def test_predicate_normalization_llm_phrasings():
    """Wordy LLM phrasings normalized to standard predicates."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("is_associated_with") == "associated_with"
    assert normalize_predicate("is_linked_to") == "associated_with"
    assert normalize_predicate("is_correlated_with") == "correlates_with"
    assert normalize_predicate("is_a_biomarker_for") == "biomarker_for"
    assert normalize_predicate("facilitate_the_spread_of") == "promotes"


def test_predicate_normalization_prefix_stripping():
    """Common prefixes stripped before matching."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("may_inhibits") == "inhibits"
    assert normalize_predicate("potentially_activates") == "activates"
    assert normalize_predicate("directly_inhibits") == "inhibits"


def test_predicate_normalization_long_fallback():
    """Very long unknown predicates fall back to associated_with."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("is_a_very_long_and_specific_unusual_predicate_type") == "associated_with"


def test_predicate_normalization_spaces_to_underscores():
    """Spaces converted to underscores before matching."""
    from attestdb.core.vocabulary import normalize_predicate
    assert normalize_predicate("is associated with") == "associated_with"
    assert normalize_predicate("interacts with") == "interacts_with"


def test_predicate_normalization_at_ingestion():
    """Predicate normalization applied during ingestion."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.ingest(
        subject=("TP53", "gene"),
        predicate=("is_associated_with", "relation"),
        object=("BRCA1", "gene"),
        provenance={"source_type": "observation", "source_id": "test"},
        confidence=0.8,
    )
    claims = db.claims_for("TP53")
    assert len(claims) == 1
    assert claims[0].predicate.id == "associated_with"
    db.close()


# --- Entity alias normalization tests ---

def test_entity_alias_builtin_p53():
    """Built-in alias: p53 -> tp53."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("p53") == "tp53"
    assert normalize_entity_name("P53") == "tp53"
    assert normalize_entity_name("tumour protein p53") == "tp53"
    assert normalize_entity_name("tumor protein p53") == "tp53"


def test_entity_alias_builtin_pdl1():
    """Built-in alias: PD-L1 variants -> cd274."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("PD-L1") == "cd274"
    assert normalize_entity_name("PDL1") == "cd274"
    assert normalize_entity_name("PD L1") == "cd274"
    assert normalize_entity_name("programmed death-ligand 1") == "cd274"


def test_entity_alias_builtin_amyloid():
    """Built-in alias: Abeta variants -> amyloid beta."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("Abeta") == "amyloid beta"
    assert normalize_entity_name("A-beta") == "amyloid beta"
    assert normalize_entity_name("amyloid-beta") == "amyloid beta"
    assert normalize_entity_name("beta-amyloid") == "amyloid beta"
    # Greek letter variant — normalize_entity_id converts β to "beta"
    assert normalize_entity_name("Aβ") == "amyloid beta"


def test_entity_alias_builtin_tnf():
    """Built-in alias: TNF-alpha variants -> tnf."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("TNF-alpha") == "tnf"
    assert normalize_entity_name("TNFalpha") == "tnf"
    assert normalize_entity_name("tumor necrosis factor alpha") == "tnf"
    assert normalize_entity_name("TNF-α") == "tnf"


def test_entity_alias_builtin_disease():
    """Built-in alias: disease name variants."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("Alzheimer's disease") == "alzheimer disease"
    assert normalize_entity_name("Alzheimers") == "alzheimer disease"
    assert normalize_entity_name("ALS") == "amyotrophic lateral sclerosis"


def test_entity_alias_passthrough():
    """Unknown entities pass through unchanged (just normalized)."""
    from attestdb.core.vocabulary import normalize_entity_name
    assert normalize_entity_name("TREM2") == "trem2"
    assert normalize_entity_name("some unknown entity") == "some unknown entity"


def test_entity_alias_extra_aliases_override():
    """Runtime aliases take precedence over built-in map."""
    from attestdb.core.vocabulary import normalize_entity_name
    extras = {"p53": "custom_p53_id"}
    assert normalize_entity_name("p53", extra_aliases=extras) == "custom_p53_id"
    # Without extras, uses built-in
    assert normalize_entity_name("p53") == "tp53"


def test_entity_alias_at_ingestion():
    """Entity aliases applied during ingestion pipeline."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.ingest(
        subject=("p53", "gene"),
        predicate=("inhibits", "relation"),
        object=("BRCA1", "gene"),
        provenance={"source_type": "observation", "source_id": "test"},
        confidence=0.8,
    )
    # p53 should have been resolved to tp53
    claims = db.claims_for("tp53")
    assert len(claims) == 1
    assert claims[0].subject.id == "tp53"
    # No claims under "p53"
    assert len(db.claims_for("p53")) == 0
    db.close()


def test_entity_alias_at_ingestion_object():
    """Entity aliases applied to object entities too."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.ingest(
        subject=("TREM2", "gene"),
        predicate=("activates", "relation"),
        object=("TNF-alpha", "protein"),
        provenance={"source_type": "observation", "source_id": "test"},
        confidence=0.8,
    )
    claims = db.claims_for("trem2")
    assert len(claims) == 1
    assert claims[0].object.id == "tnf"
    db.close()


def test_add_entity_alias_runtime():
    """db.add_entity_alias() registers runtime aliases."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.add_entity_alias("csf abeta42", "csf amyloid beta 42")
    db.ingest(
        subject=("CSF Abeta42", "biomarker"),
        predicate=("biomarker_for", "relation"),
        object=("Alzheimer's disease", "disease"),
        provenance={"source_type": "observation", "source_id": "test"},
        confidence=0.9,
    )
    claims = db.claims_for("csf amyloid beta 42")
    assert len(claims) == 1
    assert claims[0].subject.id == "csf amyloid beta 42"
    assert claims[0].object.id == "alzheimer disease"
    db.close()


def test_remove_entity_alias():
    """db.remove_entity_alias() removes a runtime alias."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.add_entity_alias("my_alias", "my_canonical")
    assert db.remove_entity_alias("my_alias") is True
    assert db.remove_entity_alias("my_alias") is False
    assert db.get_entity_aliases() == {}
    db.close()


def test_get_entity_aliases():
    """db.get_entity_aliases() returns the runtime alias map."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    db.add_entity_alias("alias1", "canonical1")
    db.add_entity_alias("alias2", "canonical2")
    aliases = db.get_entity_aliases()
    assert aliases == {"alias1": "canonical1", "alias2": "canonical2"}
    db.close()


def test_entity_alias_sidecar_persistence(tmp_path):
    """Entity aliases persist to .aliases.json sidecar file."""
    db_path = str(tmp_path / "test.attest")
    from attestdb import AttestDB
    db = AttestDB(db_path, embedding_dim=None)
    db.add_entity_alias("my_gene", "canonical_gene")
    db.close()

    # Reopen — alias should be loaded from sidecar
    db2 = AttestDB(db_path, embedding_dim=None)
    assert db2.get_entity_aliases() == {"my_gene": "canonical_gene"}
    db2.close()


def test_entity_alias_corroboration():
    """Claims from different alias forms corroborate each other."""
    from attestdb import AttestDB
    db = AttestDB(":memory:", embedding_dim=None)
    prov = {"source_type": "observation", "source_id": "test1"}
    prov2 = {"source_type": "observation", "source_id": "test2"}
    db.ingest(
        subject=("p53", "gene"),
        predicate=("inhibits", "relation"),
        object=("MDM2", "gene"),
        provenance=prov,
        confidence=0.7,
    )
    db.ingest(
        subject=("tumor protein p53", "gene"),
        predicate=("inhibits", "relation"),
        object=("MDM2", "gene"),
        provenance=prov2,
        confidence=0.8,
    )
    # Both should resolve to tp53 — same content_id, corroborated
    claims = db.claims_for("tp53")
    assert len(claims) == 2
    assert all(c.subject.id == "tp53" for c in claims)
    db.close()


# --- Directional confidence tests ---

def test_directional_confidence_strong():
    """High evidence count + high agreement = strong."""
    from attestdb.core.vocabulary import directional_confidence
    conf, verdict = directional_confidence(124, 10)
    assert verdict == "strong"
    assert conf > 0.7


def test_directional_confidence_insufficient():
    """Below min_evidence = insufficient."""
    from attestdb.core.vocabulary import directional_confidence
    conf, verdict = directional_confidence(2, 0)
    assert verdict == "insufficient"
    assert conf == 0.0


def test_directional_confidence_moderate():
    """Moderate evidence count."""
    from attestdb.core.vocabulary import directional_confidence
    conf, verdict = directional_confidence(15, 3)
    assert verdict in ("moderate", "strong")
    assert conf > 0.5


def test_directional_confidence_contested():
    """Nearly equal for/against = weak or insufficient."""
    from attestdb.core.vocabulary import directional_confidence
    conf, verdict = directional_confidence(5, 4)
    assert verdict in ("weak", "insufficient")
    assert conf < 0.6
