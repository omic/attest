//! attest-core: Locked invariants for Attest.
//!
//! This crate contains the deterministic, never-changing functions shared between
//! the Python (Phase 1) and Rust (Phase 2) implementations:
//!
//! - **normalization** — Entity ID normalization (NFKD, lowercase, Greek expansion)
//! - **hashing** — Claim ID and content ID computation (SHA-256)
//! - **types** — Shared data structures (Claim, EntityRef, etc.)
//! - **errors** — Error hierarchy
//! - **confidence** — Tier 1 source-type weights
//! - **vocabulary** — Built-in vocabulary constants

pub mod confidence;
pub mod errors;
pub mod hashing;
pub mod normalization;
pub mod types;
pub mod vocabulary;

// Re-exports for convenience
pub use confidence::tier1_confidence;
pub use errors::{AttestError, Result};
pub use hashing::{compute_chain_hash, compute_claim_id, compute_content_id};
pub use normalization::normalize_entity_id;
