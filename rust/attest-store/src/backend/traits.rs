//! Shared trait for storage backends.
//!
//! Both [`MemoryBackend`] and [`RedbBackend`] implement this trait,
//! allowing [`RustStore`] to dispatch through the [`Backend`] enum
//! without per-method match arms.

use std::collections::{HashMap, HashSet};

use attest_core::errors::AttestError;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};

use crate::metadata::Vocabulary;
use crate::store::StoreStats;

/// Unified interface for claim storage backends.
///
/// Every method here has an identical signature on both `MemoryBackend`
/// and `RedbBackend`. The `Backend` enum implements this trait by
/// delegating to whichever variant it holds.
pub trait StorageBackend {
    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    fn close(&mut self) -> Result<(), AttestError>;

    /// Write a full checkpoint without releasing the lock.
    fn checkpoint(&mut self) -> Result<(), AttestError>;

    /// Compact the database file, reclaiming free pages.
    /// Returns `true` if compaction freed any space.
    /// Default: returns `Ok(false)` (no-op).
    fn compact(&mut self) -> Result<bool, AttestError> {
        Ok(false)
    }

    // ── Metadata ───────────────────────────────────────────────────────

    fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary);

    fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value);

    fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value);

    fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary>;

    fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value>;

    fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value>;

    // ── Alias resolution ───────────────────────────────────────────────

    /// Resolve entity ID through alias chain.
    fn resolve(&mut self, entity_id: &str) -> String;

    /// Get all entity IDs that resolve to the same canonical.
    fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String>;

    // ── Cache management ───────────────────────────────────────────────

    /// Warm caches for bulk operations.
    /// Default: no-op (everything already in memory).
    fn warm_caches(&self) {}

    // ── Entity CRUD ────────────────────────────────────────────────────

    fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    );

    /// Batch-upsert entities. Default implementation loops over `upsert_entity`.
    fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) {
        for (id, etype, display, ext_ids) in entities {
            self.upsert_entity(id, etype, display, Some(ext_ids), timestamp);
        }
    }

    fn get_entity(&self, entity_id: &str) -> Option<EntitySummary>;

    fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
        offset: usize,
        limit: usize,
    ) -> Vec<EntitySummary>;

    /// Count entities matching the given filter without materializing results.
    fn count_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> usize;

    /// Count total claims.
    fn count_claims(&self) -> usize;

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim. Returns `false` if the claim_id already exists.
    fn insert_claim(&mut self, claim: Claim, checkpoint_interval: u64) -> bool;

    /// Insert a batch of claims. Returns the number of newly inserted claims.
    fn insert_claims_batch(&mut self, claims: Vec<Claim>, checkpoint_interval: u64) -> usize;

    fn claim_exists(&self, claim_id: &str) -> bool;

    fn get_claim(&self, claim_id: &str) -> Option<Claim>;

    fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim>;

    /// Get all claims with optional pagination.
    fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim>;

    fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim>;

    fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim>;

    fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim>;

    /// Get the provenance chain for a claim.
    fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String>;

    // ── Graph traversal ────────────────────────────────────────────────

    /// BFS traversal collecting claims at each hop depth.
    fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)>;

    /// Check if a path exists between two entities within max_depth hops.
    fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool;

    /// Get the bidirectional adjacency list.
    fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>>;

    // ── Temporal queries ───────────────────────────────────────────────

    /// Get claims within a timestamp range [min_ts, max_ts] (inclusive).
    fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim>;

    /// Get the most recent N claims by timestamp.
    fn most_recent_claims(&mut self, n: usize) -> Vec<Claim>;

    // ── Text search ────────────────────────────────────────────────────

    /// Search entities by text query.
    fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary>;

    // ── Stats ──────────────────────────────────────────────────────────

    fn stats(&self) -> StoreStats;

    // ── Retraction / status overlay ─────────────────────────────────────

    /// Update the status of a claim. Returns false if claim doesn't exist.
    fn update_claim_status(&mut self, claim_id: &str, status: ClaimStatus) -> Result<bool, AttestError>;

    /// Batch-update claim statuses. Returns number of claims actually updated.
    fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError>;

    /// Set whether query methods include retracted (Tombstoned) claims. Default: false.
    fn set_include_retracted(&mut self, include: bool);

    // ── Namespace filtering ─────────────────────────────────────────────

    /// Set the namespace filter. Empty vec = no filter (all namespaces visible).
    fn set_namespace_filter(&mut self, namespaces: Vec<String>);

    /// Get the current namespace filter.
    fn get_namespace_filter(&self) -> &[String];
}
