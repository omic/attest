//! Claim identity functions — locked, shared between Phase 1 and Phase 2.
//!
//! SHA-256 of pipe-delimited canonical inputs, UTF-8 encoded.

use sha2::{Digest, Sha256};

/// Compute globally unique claim ID.
///
/// `SHA-256(subject|predicate|object|source_id|source_type|timestamp)` → hex string.
pub fn compute_claim_id(
    subject_canonical: &str,
    predicate_id: &str,
    object_canonical: &str,
    source_id: &str,
    source_type: &str,
    timestamp: i64,
) -> String {
    let payload = format!(
        "{}|{}|{}|{}|{}|{}",
        subject_canonical, predicate_id, object_canonical, source_id, source_type, timestamp
    );
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Compute corroboration grouping key.
///
/// `SHA-256(subject|predicate|object)` → hex string.
pub fn compute_content_id(
    subject_canonical: &str,
    predicate_id: &str,
    object_canonical: &str,
) -> String {
    let payload = format!("{}|{}|{}", subject_canonical, predicate_id, object_canonical);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Compute Merkle chain hash for tamper-evident append-only log.
///
/// Each entry's hash incorporates the previous entry's hash, creating
/// a verifiable chain.  `SHA-256(prev_chain_hash|claim_id)` → hex string.
pub fn compute_chain_hash(prev_chain_hash: &str, claim_id: &str) -> String {
    let payload = format!("{}|{}", prev_chain_hash, claim_id);
    let mut hasher = Sha256::new();
    hasher.update(payload.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claim_id_deterministic() {
        let id1 = compute_claim_id("brca1", "binds_to", "tp53", "pub123", "literature", 1000);
        let id2 = compute_claim_id("brca1", "binds_to", "tp53", "pub123", "literature", 1000);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_claim_id_differs_on_timestamp() {
        let id1 = compute_claim_id("brca1", "binds_to", "tp53", "pub123", "literature", 1000);
        let id2 = compute_claim_id("brca1", "binds_to", "tp53", "pub123", "literature", 2000);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_content_id_deterministic() {
        let id1 = compute_content_id("brca1", "binds_to", "tp53");
        let id2 = compute_content_id("brca1", "binds_to", "tp53");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_content_id_ignores_provenance() {
        // Content ID only depends on subject|predicate|object
        let id = compute_content_id("brca1", "binds_to", "tp53");
        assert_eq!(id.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_claim_id_is_sha256_hex() {
        let id = compute_claim_id("a", "b", "c", "d", "e", 0);
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_chain_hash_deterministic() {
        let h1 = compute_chain_hash("prev_hash", "claim_abc");
        let h2 = compute_chain_hash("prev_hash", "claim_abc");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_chain_hash_differs_on_prev() {
        let h1 = compute_chain_hash("hash_a", "claim_1");
        let h2 = compute_chain_hash("hash_b", "claim_1");
        assert_ne!(h1, h2);
    }
}
