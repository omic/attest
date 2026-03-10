//! Entity normalization — locked, never changes.
//!
//! Deterministic normalization function identical in Phase 1 (Python) and Phase 2 (Rust).

use unicode_normalization::UnicodeNormalization;

/// Greek letter → spelled-out mapping. Order matches Python implementation.
const GREEK_MAP: &[(&str, &str)] = &[
    ("\u{03B1}", "alpha"),   // α
    ("\u{03B2}", "beta"),    // β
    ("\u{03B3}", "gamma"),   // γ
    ("\u{03B4}", "delta"),   // δ
    ("\u{03B5}", "epsilon"), // ε
    ("\u{03BA}", "kappa"),   // κ
    ("\u{03BB}", "lambda"),  // λ
    ("\u{03BC}", "mu"),      // μ
    ("\u{03C4}", "tau"),     // τ
    ("\u{03C9}", "omega"),   // ω
];

/// Returns true if a character is in Unicode General Category Cf (Format).
/// These are invisible control characters like zero-width space, BOM, etc.
fn is_unicode_format_cf(c: char) -> bool {
    matches!(c,
        '\u{00AD}'           // soft hyphen
        | '\u{0600}'..='\u{0605}' // Arabic number signs
        | '\u{061C}'         // Arabic letter mark
        | '\u{06DD}'         // Arabic end of ayah
        | '\u{070F}'         // Syriac abbreviation mark
        | '\u{0890}'..='\u{0891}' // Arabic pound/piastre marks
        | '\u{08E2}'         // Arabic disputed end of ayah
        | '\u{180E}'         // Mongolian vowel separator
        | '\u{200B}'..='\u{200F}' // zero-width space, ZWNJ, ZWJ, LRM, RLM
        | '\u{202A}'..='\u{202E}' // bidi embedding/override
        | '\u{2060}'..='\u{2064}' // word joiner, invisible times/separator/plus, invisible plus
        | '\u{2066}'..='\u{206F}' // bidi isolates, deprecated formatting
        | '\u{FEFF}'         // BOM / zero-width no-break space
        | '\u{FFF9}'..='\u{FFFB}' // interlinear annotations
        | '\u{110BD}'        // Kaithi number sign
        | '\u{110CD}'        // Kaithi number sign above
        | '\u{13430}'..='\u{1343F}' // Egyptian hieroglyph format controls
        | '\u{1BCA0}'..='\u{1BCA3}' // shorthand format controls
        | '\u{1D173}'..='\u{1D17A}' // musical symbol formatting
        | '\u{E0001}'        // language tag
        | '\u{E0020}'..='\u{E007F}' // tag components
    )
}

/// Deterministic entity ID normalization. Once shipped, never changes.
///
/// Steps:
/// 1. Unicode normalize (NFKD decomposition)
/// 2. Strip zero-width / invisible format characters (Unicode category Cf)
/// 3. Lowercase
/// 4. Collapse whitespace (multiple spaces/tabs/newlines → single space)
/// 5. Strip leading/trailing whitespace
/// 6. Replace common Greek letters with spelled-out forms
pub fn normalize_entity_id(raw: &str) -> String {
    // 1. NFKD decomposition
    let s: String = raw.nfkd().collect();

    // 2. Strip zero-width / invisible format characters (Unicode category Cf)
    let s: String = s.chars().filter(|c| !is_unicode_format_cf(*c)).collect();

    // 3. Lowercase
    let s = s.to_lowercase();

    // 4. Collapse whitespace + 5. Strip
    let mut s = s.split_whitespace().collect::<Vec<&str>>().join(" ");

    // 6. Replace Greek letters with spelled-out forms
    for &(greek, spelled) in GREEK_MAP {
        s = s.replace(greek, spelled);
    }

    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_normalization() {
        assert_eq!(normalize_entity_id("Hello World"), "hello world");
    }

    #[test]
    fn test_whitespace_collapse() {
        assert_eq!(normalize_entity_id("  hello   world  "), "hello world");
    }

    #[test]
    fn test_tab_and_newline() {
        assert_eq!(normalize_entity_id("hello\t\nworld"), "hello world");
    }

    #[test]
    fn test_greek_alpha() {
        assert_eq!(normalize_entity_id("TNF-α"), "tnf-alpha");
    }

    #[test]
    fn test_greek_beta() {
        assert_eq!(normalize_entity_id("β-catenin"), "beta-catenin");
    }

    #[test]
    fn test_greek_gamma() {
        assert_eq!(normalize_entity_id("IFN-γ"), "ifn-gamma");
    }

    #[test]
    fn test_greek_mu() {
        assert_eq!(normalize_entity_id("μ-opioid"), "mu-opioid");
    }

    #[test]
    fn test_multiple_greek() {
        assert_eq!(normalize_entity_id("α-β complex"), "alpha-beta complex");
    }

    #[test]
    fn test_empty_string() {
        assert_eq!(normalize_entity_id(""), "");
    }

    #[test]
    fn test_unicode_nfkd() {
        // ﬁ (U+FB01) should decompose to "fi"
        assert_eq!(normalize_entity_id("ﬁbronectin"), "fibronectin");
    }

    #[test]
    fn test_already_normalized() {
        assert_eq!(normalize_entity_id("brca1"), "brca1");
    }

    #[test]
    fn test_uppercase_greek() {
        // Uppercase Greek alpha (U+0391) → NFKD → lowercase → α (U+03B1) → "alpha"
        assert_eq!(normalize_entity_id("\u{0391}-synuclein"), "alpha-synuclein");
    }

    #[test]
    fn test_non_mapped_greek() {
        // σ (sigma), φ (phi), ψ (psi) are NOT in GREEK_MAP — left as-is after lowercasing
        assert_eq!(normalize_entity_id("σ-factor"), "\u{03C3}-factor");
        assert_eq!(normalize_entity_id("φ-value"), "\u{03C6}-value");
        assert_eq!(normalize_entity_id("ψ-angle"), "\u{03C8}-angle");
    }

    #[test]
    fn test_mixed_greek_latin() {
        assert_eq!(normalize_entity_id("TNFα receptor"), "tnfalpha receptor");
    }

    #[test]
    fn test_unicode_non_breaking_space() {
        // Non-breaking space (U+00A0) should be collapsed like regular whitespace
        assert_eq!(normalize_entity_id("hello\u{00A0}world"), "hello world");
    }

    #[test]
    fn test_all_whitespace_input() {
        // Input of only whitespace should normalize to empty string
        assert_eq!(normalize_entity_id("   \t\n  "), "");
    }

    #[test]
    fn test_zero_width_space_stripped() {
        // U+200B zero-width space must not create invisible duplicates
        assert_eq!(normalize_entity_id("TREM2\u{200B}"), "trem2");
        assert_eq!(
            normalize_entity_id("TREM2"),
            normalize_entity_id("TREM2\u{200B}")
        );
    }

    #[test]
    fn test_zero_width_non_joiner_stripped() {
        assert_eq!(normalize_entity_id("foo\u{200C}bar"), "foobar");
    }

    #[test]
    fn test_zero_width_joiner_stripped() {
        assert_eq!(normalize_entity_id("foo\u{200D}bar"), "foobar");
    }

    #[test]
    fn test_bom_stripped() {
        assert_eq!(normalize_entity_id("\u{FEFF}protein"), "protein");
    }

    #[test]
    fn test_directional_marks_stripped() {
        assert_eq!(normalize_entity_id("hello\u{200E}world"), "helloworld");
        assert_eq!(normalize_entity_id("hello\u{200F}world"), "helloworld");
    }

    #[test]
    fn test_multiple_cf_chars_stripped() {
        assert_eq!(
            normalize_entity_id("\u{200B}\u{200C}\u{200D}TREM2\u{FEFF}"),
            "trem2"
        );
    }

    #[test]
    fn test_soft_hyphen_stripped() {
        assert_eq!(normalize_entity_id("fibro\u{00AD}nectin"), "fibronectin");
    }
}
