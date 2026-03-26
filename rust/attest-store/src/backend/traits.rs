//! Shared trait for storage backends.
//!
//! Both [`MemoryBackend`] and [`LmdbBackend`] implement this trait,
//! allowing [`RustStore`] to dispatch through the [`Backend`] enum.

use std::collections::{HashMap, HashSet};

use attest_core::errors::AttestError;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};

use crate::metadata::Vocabulary;
use crate::store::StoreStats;

/// Unified interface for claim storage backends.
///
/// Every method here has an identical signature on both `MemoryBackend`
/// and `LmdbBackend`. The `Backend` enum implements this trait by
/// delegating to whichever variant it holds.
pub trait StorageBackend {
    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    fn close(&mut self) -> Result<(), AttestError>;

    /// Write a full checkpoint without releasing the lock.
    fn checkpoint(&mut self) -> Result<(), AttestError>;

    /// Rebuild the causal_adj index from existing claims.
    fn rebuild_causal_adj(&mut self) -> Result<usize, AttestError> {
        Ok(0)
    }

    /// Reset the union-find alias table.
    fn clear_aliases(&mut self) -> Result<(), AttestError> {
        Ok(())
    }

    /// Physically delete all claims from a source_id.
    /// Default: returns `Ok(0)` (no-op).
    fn purge_source(&mut self, _source_id: &str) -> Result<usize, AttestError> {
        Ok(0)
    }

    /// Get outgoing causal edges (lightweight — no full claim deserialization on fast path).
    fn outgoing_causal_edges(
        &self,
        _entity_id: &str,
        _causal_predicates: &std::collections::HashSet<String>,
    ) -> Vec<(String, String, f64)> {
        Vec::new()
    }

    /// Get claims for an entity, optionally including inverse-derived claims.
    /// When `include_inverse` is true, for claims where the entity is the OBJECT,
    /// derive a synthetic claim with subject/object swapped and predicate replaced
    /// by its inverse (e.g., A→causes→B becomes B→caused_by→A when querying B).
    fn claims_for_with_inverse(
        &mut self,
        _entity_id: &str,
        _predicate_type: Option<&str>,
        _source_type: Option<&str>,
        _min_confidence: f64,
        _include_inverse: bool,
    ) -> Vec<attest_core::types::Claim> {
        Vec::new()
    }

    /// Compute transitive closure over causal predicates from an entity.
    /// Returns Vec of (target_entity_id, composed_predicate, depth, composed_confidence).
    /// Follows outgoing causal edges up to max_depth hops, composing predicates via
    /// the PREDICATE_COMPOSITION table.
    fn transitive_closure(
        &mut self,
        _entity_id: &str,
        _predicates: &std::collections::HashSet<String>,
        _max_depth: usize,
    ) -> Vec<(String, String, usize, f64)> {
        Vec::new()
    }

    /// Get the corroboration count for a content_id (number of independent sources).
    /// Default returns 1 (no corroboration tracking).
    fn corroboration_count(&self, _content_id: &str) -> u32 {
        1
    }

    /// Batch lookup of corroboration counts.
    fn corroboration_counts_batch(&self, _content_ids: &[&str]) -> std::collections::HashMap<String, u32> {
        std::collections::HashMap::new()
    }

    /// Compact the database file, reclaiming free pages.
    /// Returns `true` if compaction freed any space.
    /// Default: returns `Ok(false)` (no-op).
    fn compact(&mut self) -> Result<bool, AttestError> {
        Ok(false)
    }

    /// Return the schema version of this database.
    /// In-memory stores always return the current version.
    fn schema_version(&self) -> u32 {
        // Default: return current version (in-memory stores don't need migration)
        3
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
    ) -> Result<(), AttestError> {
        for (id, etype, display, ext_ids) in entities {
            self.upsert_entity(id, etype, display, Some(ext_ids), timestamp);
        }
        Ok(())
    }

    /// Update display_name for an existing entity. Returns true if updated.
    fn update_display_name(&mut self, entity_id: &str, new_display: &str) -> bool;

    fn get_entity(&self, entity_id: &str) -> Option<EntitySummary>;

    /// Batch-lookup entity metadata (type, display_name) for multiple IDs.
    /// Returns a map from entity_id → (entity_type, display_name) for found entities.
    /// Default implementation loops over `get_entity()`.
    fn get_entities_batch(&self, entity_ids: &[&str]) -> HashMap<String, (String, String)> {
        let mut result = HashMap::with_capacity(entity_ids.len());
        for id in entity_ids {
            if let Some(es) = self.get_entity(id) {
                result.insert(es.id.clone(), (es.entity_type.clone(), es.name.clone()));
            }
        }
        result
    }

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
        limit: usize,
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

    // ── Analytics (Rust-native aggregation) ───────────────────────────

    /// Count claims per predicate_id. Returns HashMap<predicate_id, count>.
    fn predicate_counts(&self) -> HashMap<String, u64>;

    /// Backfill pred_id_counts for existing databases. Returns number of unique predicate IDs.
    fn backfill_pred_id_counts(&mut self) -> Result<usize, AttestError> {
        Ok(0) // Default: no-op for backends without persistent counter tables
    }

    /// Backfill claim_summaries table for existing databases. Returns number of summaries written.
    fn backfill_claim_summaries(&mut self) -> Result<u64, AttestError> {
        Ok(0) // Default: no-op for in-memory backend
    }

    /// Backfill analytics counter tables (entity_pred_counts, entity_src_counts, pred_pair_counts).
    /// Returns number of claims processed.
    fn backfill_analytics_counters(&mut self) -> Result<u64, AttestError> {
        Ok(0) // Default: no-op for in-memory backend
    }

    /// Find contradictions: (subject, object) pairs under both pred_a and pred_b.
    /// Returns Vec<(subject_id, object_id, count_a, count_b, avg_conf_a, avg_conf_b)>.
    fn find_contradictions(
        &self,
        pred_a: &str,
        pred_b: &str,
    ) -> Vec<(String, String, u64, u64, f64, f64)>;

    /// Gap analysis: entities of a type missing expected predicates.
    /// Returns Vec<(entity_id, Vec<missing_predicate_ids>)>.
    fn gap_analysis(
        &self,
        entity_type: &str,
        expected_predicates: &[&str],
        limit: usize,
    ) -> Vec<(String, Vec<String>)>;

    /// Get predicate_id → count for a single entity.
    fn entity_predicate_counts(&self, entity_id: &str) -> Vec<(String, u64)>;

    /// Get source_id → (count, avg_confidence) for a single entity.
    fn entity_source_counts(&self, entity_id: &str) -> Vec<(String, u64, f64)>;

    /// Find entities that have claims from exactly one source.
    /// Returns entity IDs where total claim count >= min_claims and distinct sources == 1.
    fn find_single_source_entities(&self, min_claims: u64) -> Vec<String>;

    // ── Bulk load mode ───────────────────────────────────────────────

    /// Enable or disable bulk load mode. In bulk mode, analytics counters
    /// are skipped during insert for ~2.5× throughput.
    fn set_bulk_load_mode(&mut self, _enabled: bool) {}

    /// Rebuild all analytics counters from the claims table.
    /// Call after bulk loading/merging is complete.
    fn rebuild_counters(&mut self) -> Result<(), AttestError> {
        Ok(()) // No-op for in-memory backend
    }

    /// Merge all claims and entities from another LMDB database.
    /// Returns the number of newly merged claims.
    fn merge_from(&mut self, _source_path: &str) -> Result<usize, AttestError> {
        Err(AttestError::Provenance(
            "merge_from is only supported on LMDB backends".to_string()
        ))
    }

    /// Count claims per source_id by iterating the source index.
    /// Returns HashMap<source_id, count>. Default: empty map.
    fn source_id_counts(&self) -> HashMap<String, u64> {
        HashMap::new()
    }
}
