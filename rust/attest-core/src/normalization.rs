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

/// Deterministic entity ID normalization. Once shipped, never changes.
///
/// Steps:
/// 1. Unicode normalize (NFKD decomposition)
/// 2. Lowercase
/// 3. Collapse whitespace (multiple spaces/tabs/newlines → single space)
/// 4. Strip leading/trailing whitespace
/// 5. Replace common Greek letters with spelled-out forms
pub fn normalize_entity_id(raw: &str) -> String {
    // 1. NFKD decomposition
    let s: String = raw.nfkd().collect();

    // 2. Lowercase
    let s = s.to_lowercase();

    // 3. Collapse whitespace + 4. Strip
    let mut s = s.split_whitespace().collect::<Vec<&str>>().join(" ");

    // 5. Replace Greek letters with spelled-out forms
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
}
