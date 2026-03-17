//! RustStore — claim-native store with LMDB backend.
//!
//! Thin dispatcher over storage backends. All business logic
//! lives in the backend modules (`backend::memory`, `backend::lmdb`).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};

use crate::backend::memory::DEFAULT_CHECKPOINT_INTERVAL;
use crate::backend::{Backend, LmdbBackend, MemoryBackend, StorageBackend};
use crate::metadata::Vocabulary;

/// Store statistics.
#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    pub total_claims: usize,
    pub entity_count: usize,
    pub entity_types: HashMap<String, usize>,
    pub predicate_types: HashMap<String, usize>,
    pub source_types: HashMap<String, usize>,
}

/// Claim-native store with pluggable backend.
///
/// Concurrency model: single-writer via advisory file lock.
/// The lock is acquired on `new()` and released on `close()`.
pub struct RustStore {
    backend: Backend,
    /// Auto-checkpoint after this many WAL entries. 0 = disabled.
    checkpoint_interval: u64,
    /// When true, all write operations return an error.
    read_only: bool,
    /// Temp file path for read-only copies (deleted on close).
    temp_path: Option<PathBuf>,
}

impl RustStore {
    /// Open or create a store at the given path.
    ///
    /// Acquires an exclusive file lock. If the database file exists but
    /// has a checksum mismatch (crash during write), the store starts
    /// empty with a warning — the corrupted file is not trusted.
    pub fn new(db_path: &str) -> Result<Self, AttestError> {
        let path = std::path::Path::new(db_path);

        // Existing LMDB directory → open
        if path.is_dir() && path.join("data.mdb").exists() {
            let backend = LmdbBackend::open(db_path)?;
            return Ok(Self {
                backend: Backend::Lmdb(Box::new(backend)),
                checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
                read_only: false,
                temp_path: None,
            });
        }

        // Existing file (not a directory) → unsupported legacy format
        if path.exists() && path.is_file() {
            return Err(AttestError::Provenance(format!(
                "Legacy database format detected at {db_path}. \
                 Only LMDB databases are supported. Please re-create the database."
            )));
        }

        // New database → create LMDB
        let backend = LmdbBackend::open(db_path)?;
        Ok(Self {
            backend: Backend::Lmdb(Box::new(backend)),
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            read_only: false,
            temp_path: None,
        })
    }

    /// Create a purely in-memory store (no file path, no persistence, no lock, no WAL).
    pub fn in_memory() -> Self {
        Self {
            backend: Backend::InMemory(Box::new(MemoryBackend::new_empty())),
            checkpoint_interval: 0,
            read_only: false,
            temp_path: None,
        }
    }

    /// Open a database in read-only mode.
    ///
    /// Uses LMDB's MVCC: concurrent readers coexist with a single writer.
    /// Read-only opens do NOT acquire the write lock.
    pub fn open_read_only(db_path: &str) -> Result<Self, AttestError> {
        let backend = LmdbBackend::open_read_only(db_path)?;
        Ok(Self {
            backend: Backend::Lmdb(Box::new(backend)),
            checkpoint_interval: 0,
            read_only: true,
            temp_path: None,
        })
    }

    /// Returns true if this store was opened in read-only mode.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    /// For read-only stores, deletes the temp file.
    pub fn close(&mut self) -> Result<(), AttestError> {
        let result = self.backend.close();
        // Clean up temp file for read-only copies
        if let Some(ref temp) = self.temp_path {
            let _ = std::fs::remove_file(temp);
            self.temp_path = None;
        }
        result
    }

    /// Write a full checkpoint without releasing the lock.
    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        self.backend.checkpoint()
    }

    /// Compact the database file, reclaiming free pages.
    /// Returns `true` if compaction freed any space.
    /// No-op (returns `false`) for in-memory backends.
    pub fn compact(&mut self) -> Result<bool, AttestError> {
        self.backend.compact()
    }

    /// Return the schema version of this database.
    pub fn schema_version(&self) -> u32 {
        self.backend.schema_version()
    }

    // ── Metadata ───────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        if self.read_only { return; }
        self.backend.register_vocabulary(namespace, vocab);
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        if self.read_only { return; }
        self.backend.register_predicate(predicate_id, constraints);
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        if self.read_only { return; }
        self.backend.register_payload_schema(schema_id, schema);
    }

    pub fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        self.backend.get_registered_vocabularies()
    }

    pub fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        self.backend.get_predicate_constraints()
    }

    pub fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.backend.get_payload_schemas()
    }

    // ── Alias resolution ───────────────────────────────────────────────

    /// Resolve entity ID through alias chain.
    pub fn resolve(&mut self, entity_id: &str) -> String {
        self.backend.resolve(entity_id)
    }

    /// Get all entity IDs that resolve to the same canonical.
    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        self.backend.get_alias_group(entity_id)
    }

    // ── Cache management ───────────────────────────────────────────────

    /// No-op in Rust — everything is already in memory.
    pub fn warm_caches(&self) {
        self.backend.warm_caches();
    }

    // ── Entity CRUD ────────────────────────────────────────────────────

    pub fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        if self.read_only { return; }
        self.backend.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp);
    }

    /// Update display_name for an existing entity. Returns true if updated.
    pub fn update_display_name(&mut self, entity_id: &str, new_display: &str) -> bool {
        if self.read_only { return false; }
        self.backend.update_display_name(entity_id, new_display)
    }

    /// Batch-upsert entities in a single transaction (LMDB) or loop (in-memory).
    pub fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) -> Result<(), AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        self.backend.upsert_entities_batch(entities, timestamp)
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        self.backend.get_entity(entity_id)
    }

    /// Batch-lookup entity metadata (type, display_name) for multiple IDs.
    pub fn get_entities_batch(&self, entity_ids: &[&str]) -> HashMap<String, (String, String)> {
        self.backend.get_entities_batch(entity_ids)
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
        offset: usize,
        limit: usize,
    ) -> Vec<EntitySummary> {
        self.backend.list_entities(entity_type, min_claims, offset, limit)
    }

    /// Count entities matching the given filter without materializing results.
    pub fn count_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> usize {
        self.backend.count_entities(entity_type, min_claims)
    }

    /// Count total claims without materializing.
    pub fn count_claims(&self) -> usize {
        self.backend.count_claims()
    }

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim. Caller must ensure entities exist.
    ///
    /// The claim is appended to the WAL and fsynced BEFORE being applied
    /// to in-memory state. Returns `false` if the claim_id already exists
    /// or if the store is read-only.
    pub fn insert_claim(&mut self, claim: Claim) -> bool {
        if self.read_only { return false; }
        let interval = self.checkpoint_interval;
        self.backend.insert_claim(claim, interval)
    }

    /// Insert a batch of claims with a single WAL sync at the end.
    /// Returns 0 if the store is read-only.
    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>) -> usize {
        if self.read_only { return 0; }
        let interval = self.checkpoint_interval;
        self.backend.insert_claims_batch(claims, interval)
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        self.backend.claim_exists(claim_id)
    }

    /// Get a single claim by ID. O(1) via index lookup.
    pub fn get_claim(&self, claim_id: &str) -> Option<Claim> {
        self.backend.get_claim(claim_id)
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        self.backend.claims_by_content_id(content_id)
    }

    /// Get all claims with optional pagination.
    pub fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        self.backend.all_claims(offset, limit)
    }

    /// Get all claims matching a source_id.
    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        self.backend.claims_by_source_id(source_id)
    }

    /// Get all claims matching a predicate_id.
    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        self.backend.claims_by_predicate_id(predicate_id)
    }

    pub fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim> {
        self.backend.claims_for(entity_id, predicate_type, source_type, min_confidence)
    }

    /// Get the provenance chain for a claim.
    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        self.backend.get_claim_provenance_chain(claim_id)
    }

    // ── Graph traversal ────────────────────────────────────────────────

    /// BFS traversal collecting claims at each hop depth.
    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        self.backend.bfs_claims(entity_id, max_depth)
    }

    /// Check if a path exists between two entities within max_depth hops.
    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        self.backend.path_exists(entity_a, entity_b, max_depth)
    }

    /// Get the bidirectional adjacency list.
    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        self.backend.get_adjacency_list()
    }

    /// Get claims within a timestamp range [min_ts, max_ts] (inclusive).
    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        self.backend.claims_in_range(min_ts, max_ts)
    }

    /// Get the most recent N claims by timestamp.
    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        self.backend.most_recent_claims(n)
    }

    /// Search entities by text query.
    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        self.backend.search_entities(query, top_k)
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        self.backend.stats()
    }

    // ── Retraction / status overlay ─────────────────────────────────────

    /// Retract a single claim (set status to Tombstoned).
    /// Returns false if the claim doesn't exist.
    pub fn retract_claim(&mut self, claim_id: &str) -> Result<bool, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        self.backend.update_claim_status(claim_id, ClaimStatus::Tombstoned)
    }

    /// Retract all claims from a source. Returns the list of claim IDs that were retracted.
    pub fn retract_source(&mut self, source_id: &str) -> Result<Vec<String>, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        // Temporarily include retracted claims to see all source claims
        self.backend.set_include_retracted(true);
        let claims = self.backend.claims_by_source_id(source_id);
        self.backend.set_include_retracted(false);
        let ids: Vec<String> = claims.iter()
            .filter(|c| c.status != ClaimStatus::Tombstoned)
            .map(|c| c.claim_id.clone())
            .collect();
        if ids.is_empty() {
            return Ok(ids);
        }
        let updates: Vec<(String, ClaimStatus)> = ids.iter()
            .map(|id| (id.clone(), ClaimStatus::Tombstoned))
            .collect();
        self.backend.update_claim_status_batch(&updates)?;
        Ok(ids)
    }

    /// Update the status of a claim by string name.
    /// Accepted values: "active", "archived", "tombstoned"/"retracted", "provenance_degraded"/"degraded".
    pub fn update_claim_status(&mut self, claim_id: &str, status: &str) -> Result<bool, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        let cs = parse_status(status)?;
        self.backend.update_claim_status(claim_id, cs)
    }

    /// Batch-update claim statuses. Returns number of claims actually updated.
    pub fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        self.backend.update_claim_status_batch(updates)
    }

    /// Set whether query methods include retracted (Tombstoned) claims.
    pub fn set_include_retracted(&mut self, include: bool) {
        self.backend.set_include_retracted(include);
    }

    pub fn set_namespace_filter(&mut self, namespaces: Vec<String>) {
        self.backend.set_namespace_filter(namespaces);
    }

    pub fn get_namespace_filter(&self) -> &[String] {
        self.backend.get_namespace_filter()
    }

    // ── Analytics (Rust-native aggregation) ───────────────────────────

    /// Count claims per predicate_id. Returns HashMap<predicate_id, count>.
    pub fn predicate_counts(&self) -> HashMap<String, u64> {
        self.backend.predicate_counts()
    }

    /// Backfill pred_id_counts for existing databases.
    pub fn backfill_pred_id_counts(&mut self) -> Result<usize, AttestError> {
        self.backend.backfill_pred_id_counts()
    }

    /// Backfill claim_summaries table for existing databases.
    pub fn backfill_claim_summaries(&mut self) -> Result<u64, AttestError> {
        self.backend.backfill_claim_summaries()
    }

    /// Backfill analytics counter tables for sub-second analytics.
    pub fn backfill_analytics_counters(&mut self) -> Result<u64, AttestError> {
        self.backend.backfill_analytics_counters()
    }

    /// Find contradictions: (subject, object) pairs under both predicates.
    pub fn find_contradictions(
        &self,
        pred_a: &str,
        pred_b: &str,
    ) -> Vec<(String, String, u64, u64, f64, f64)> {
        self.backend.find_contradictions(pred_a, pred_b)
    }

    /// Gap analysis: entities of a type missing expected predicates.
    pub fn gap_analysis(
        &self,
        entity_type: &str,
        expected_predicates: &[&str],
        limit: usize,
    ) -> Vec<(String, Vec<String>)> {
        self.backend.gap_analysis(entity_type, expected_predicates, limit)
    }

    /// Get predicate_id → count for a single entity.
    pub fn entity_predicate_counts(&self, entity_id: &str) -> Vec<(String, u64)> {
        self.backend.entity_predicate_counts(entity_id)
    }

    /// Get source_id → (count, avg_confidence) for a single entity.
    pub fn entity_source_counts(&self, entity_id: &str) -> Vec<(String, u64, f64)> {
        self.backend.entity_source_counts(entity_id)
    }

    /// Find entities with exactly one distinct source and >= min_claims total claims.
    pub fn find_single_source_entities(&self, min_claims: u64) -> Vec<String> {
        self.backend.find_single_source_entities(min_claims)
    }

    // ── Bulk load mode ───────────────────────────────────────────────

    /// Enable or disable bulk load mode. Skips analytics counters during insert
    /// for ~2.5× throughput. Call `rebuild_counters()` after loading is complete.
    pub fn set_bulk_load_mode(&mut self, enabled: bool) {
        self.backend.set_bulk_load_mode(enabled);
    }

    /// Rebuild all analytics counters from the claims table.
    pub fn rebuild_counters(&mut self) -> Result<(), AttestError> {
        self.backend.rebuild_counters()
    }

    /// Count claims per source_id. Returns HashMap<source_id, count>.
    pub fn source_id_counts(&self) -> HashMap<String, u64> {
        self.backend.source_id_counts()
    }

    /// Merge all claims and entities from another LMDB database.
    /// Returns the number of newly merged claims.
    pub fn merge_from(&mut self, source_path: &str) -> Result<usize, AttestError> {
        if self.read_only {
            return Err(AttestError::Provenance(
                "cannot merge into a read-only store".to_string()
            ));
        }
        self.backend.merge_from(source_path)
    }
}

/// Parse a status string into a ClaimStatus.
fn parse_status(s: &str) -> Result<ClaimStatus, AttestError> {
    match s.to_lowercase().as_str() {
        "active" => Ok(ClaimStatus::Active),
        "archived" => Ok(ClaimStatus::Archived),
        "tombstoned" | "retracted" => Ok(ClaimStatus::Tombstoned),
        "provenance_degraded" | "degraded" => Ok(ClaimStatus::ProvenanceDegraded),
        _ => Err(AttestError::Validation(format!("Unknown claim status: {}", s))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

    /// Clean up a test database path (handles both files and LMDB directories).
    fn cleanup_db(path: &std::path::Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir_all(path);
        let _ = std::fs::remove_file(path.with_extension("attest.lock"));
        crate::wal::Wal::remove(path);
    }

    fn make_claim(
        claim_id: &str,
        subj: &str,
        pred: &str,
        obj: &str,
        source_type: &str,
    ) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: attest_core::compute_content_id(subj, pred, obj),
            subject: EntityRef {
                id: subj.to_string(),
                entity_type: "entity".to_string(),
                display_name: subj.to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: pred.to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: obj.to_string(),
                entity_type: "entity".to_string(),
                display_name: obj.to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.7,
            provenance: Provenance {
                source_type: source_type.to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        }
    }

    fn setup_store() -> RustStore {
        let mut store = RustStore::in_memory();

        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "experimental"));
        store.insert_claim(make_claim("c2", "tp53", "inhibits", "aspirin", "literature"));

        store
    }

    #[test]
    fn test_entity_upsert_and_get() {
        let store = setup_store();
        let entity = store.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");
        assert_eq!(entity.entity_type, "gene");
        assert_eq!(entity.claim_count, 1); // brca1 is in c1
    }

    #[test]
    fn test_claim_exists() {
        let store = setup_store();
        assert!(store.claim_exists("c1"));
        assert!(!store.claim_exists("c99"));
    }

    #[test]
    fn test_claims_by_content_id() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("c1", "a", "rel", "b", "obs");
        let content_id = c1.content_id.clone();
        store.insert_claim(c1);

        let mut c2 = make_claim("c2", "a", "rel", "b", "exp");
        c2.content_id.clone_from(&content_id);
        store.insert_claim(c2);

        assert_eq!(store.claims_by_content_id(&content_id).len(), 2);
    }

    #[test]
    fn test_claims_for() {
        let mut store = setup_store();
        let claims = store.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        // tp53 is in both claims
        let claims = store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_claims_for_with_filters() {
        let mut store = setup_store();
        // Filter by source_type
        let claims = store.claims_for("tp53", None, Some("experimental"), 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");
    }

    #[test]
    fn test_list_entities() {
        let store = setup_store();
        assert_eq!(store.list_entities(Some("gene"), 0, 0, 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0, 0, 0).len(), 1);
        assert_eq!(store.list_entities(None, 0, 0, 0).len(), 3);
    }

    #[test]
    fn test_list_entities_min_claims() {
        let store = setup_store();
        // tp53 has 2 claims, brca1 and aspirin have 1 each
        let high = store.list_entities(None, 2, 0, 0);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "tp53");
    }

    #[test]
    fn test_alias_resolution() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("gene_a", "gene", "A", None, 0);
        store.upsert_entity("gene_a_alias", "gene", "A alias", None, 0);
        store.upsert_entity("gene_b", "gene", "B", None, 0);

        // Create same_as claim
        let alias_claim = Claim {
            claim_id: "alias1".to_string(),
            content_id: "alias_content".to_string(),
            subject: EntityRef {
                id: "gene_a".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A".to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: "same_as".to_string(),
                predicate_type: "same_as".to_string(),
            },
            object: EntityRef {
                id: "gene_a_alias".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A alias".to_string(),
                external_ids: Default::default(),
            },
            confidence: 1.0,
            provenance: Provenance {
                source_type: "human_annotation".to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        };
        store.insert_claim(alias_claim);

        // Both should resolve to the same root
        let r1 = store.resolve("gene_a");
        let r2 = store.resolve("gene_a_alias");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_bfs_claims() {
        let mut store = setup_store();
        // From brca1, depth 1: should find c1 (brca1->tp53)
        let results = store.bfs_claims("brca1", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.claim_id, "c1");
        assert_eq!(results[0].1, 1); // hop 1

        // From brca1, depth 2: hop 1 finds c1, hop 2 re-finds c1 (via tp53) + c2
        // No cross-hop dedup — matches KuzuStore behavior
        let results = store.bfs_claims("brca1", 2);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_path_exists() {
        let mut store = setup_store();
        assert!(store.path_exists("brca1", "tp53", 1));
        assert!(store.path_exists("brca1", "aspirin", 2));
        assert!(!store.path_exists("brca1", "aspirin", 1)); // too far in 1 hop
    }

    #[test]
    fn test_adjacency_list() {
        let store = setup_store();
        let adj = store.get_adjacency_list();
        assert!(adj["brca1"].contains("tp53"));
        assert!(adj["tp53"].contains("brca1")); // bidirectional
        assert!(adj["tp53"].contains("aspirin"));
    }

    #[test]
    fn test_stats() {
        let store = setup_store();
        let stats = store.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.entity_types["gene"], 2);
        assert_eq!(stats.entity_types["compound"], 1);
    }

    #[test]
    fn test_get_claim_provenance_chain() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut claim = make_claim("c1", "a", "rel", "b", "obs");
        claim.provenance.chain = vec!["upstream1".to_string(), "upstream2".to_string()];
        store.insert_claim(claim);

        let chain = store.get_claim_provenance_chain("c1");
        assert_eq!(chain, vec!["upstream1", "upstream2"]);

        // Non-existent claim returns empty
        assert!(store.get_claim_provenance_chain("nonexistent").is_empty());
    }

    #[test]
    fn test_warm_caches_noop() {
        let store = setup_store();
        store.warm_caches(); // Should not panic
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = std::env::temp_dir().join("attest_store_test.attest");
        cleanup_db(&dir);

        // Create and populate
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            store.upsert_entity("tp53", "gene", "TP53", None, 0);
            store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "exp"));
            store.close().unwrap();
        }

        // Reopen and verify
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert!(store.claim_exists("c1"));
            assert_eq!(store.get_entity("brca1").unwrap().name, "BRCA1");
            assert_eq!(store.stats().total_claims, 1);
        }

        cleanup_db(&dir);
    }


    #[test]
    fn test_atomic_write_leaves_no_tmp() {
        let dir = std::env::temp_dir().join("attest_atomic_test.attest");
        let tmp = dir.with_extension("attest.tmp");
        cleanup_db(&dir);
        let _ = std::fs::remove_file(&tmp);

        let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.close().unwrap();

        // Temp file should not exist after successful close
        assert!(!tmp.exists(), "Temp file should be cleaned up after close");
        // Main file should exist
        assert!(dir.exists(), "Main file should exist after close");

        cleanup_db(&dir);
    }

    #[test]
    fn test_claims_in_range() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.timestamp = 1000;
        let mut c2 = make_claim("c2", "a", "rel", "b", "obs");
        c2.claim_id = "c2".to_string();
        c2.timestamp = 2000;
        let mut c3 = make_claim("c3", "a", "rel", "b", "obs");
        c3.claim_id = "c3".to_string();
        c3.timestamp = 3000;

        store.insert_claim(c1);
        store.insert_claim(c2);
        store.insert_claim(c3);

        assert_eq!(store.claims_in_range(1500, 2500).len(), 1);
        assert_eq!(store.claims_in_range(1000, 3000).len(), 3);
        assert_eq!(store.claims_in_range(4000, 5000).len(), 0);
    }

    #[test]
    fn test_most_recent_claims() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        for i in 0..5 {
            let mut c = make_claim(&format!("c{i}"), "a", "rel", "b", "obs");
            c.timestamp = (i + 1) * 1000;
            store.insert_claim(c);
        }

        let recent = store.most_recent_claims(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[0].timestamp >= recent[1].timestamp);
    }

    #[test]
    fn test_get_claim() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "gene", "A", None, 0);
        store.upsert_entity("b", "gene", "B", None, 0);
        let c = make_claim("c1", "a", "interacts_with", "b", "obs");
        store.insert_claim(c.clone());

        let found = store.get_claim("c1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().claim_id, "c1");

        assert!(store.get_claim("nonexistent").is_none());
    }

    #[test]
    fn test_search_entities() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("brca1", "gene", "BRCA1 DNA repair", None, 0);
        store.upsert_entity("tp53", "gene", "TP53 tumor protein", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        // Search for "BRCA" — should find brca1
        let results = store.search_entities("brca1", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        // Search for "protein" — should find tp53
        let results = store.search_entities("protein", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "tp53");

        // Search for "gene" — should find entities with "gene" in ID
        let results = store.search_entities("gene", 10);
        // "gene" appears as token in "brca1" → no, "gene" only in type not display_name
        // Actually EntityStore tokenizes entity_id and display_name
        // "brca1" tokens: ["brca1"], "BRCA1 DNA repair" tokens: ["brca1", "dna", "repair"]
        // "gene" doesn't appear in any of those, so 0 results
        assert_eq!(results.len(), 0);

        // Search for "repair" — should find brca1
        let results = store.search_entities("repair", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        // Search for "aspirin"
        let results = store.search_entities("aspirin", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "aspirin");
    }

    #[test]
    fn test_adjacency_maintained() {
        // Verify adjacency is maintained on insert, not rebuilt
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
        let adj = store.get_adjacency_list();
        assert!(adj["a"].contains("b"));
        assert!(adj["b"].contains("a"));
        assert_eq!(adj.len(), 2); // only a and b

        store.insert_claim(make_claim("c2", "b", "rel", "c", "obs"));
        let adj = store.get_adjacency_list();
        assert!(adj["b"].contains("c"));
        assert!(adj["c"].contains("b"));
        assert_eq!(adj.len(), 3); // a, b, c
    }

    #[test]
    fn test_metadata_registration() {
        let mut store = RustStore::in_memory();

        store.register_vocabulary(
            "bio",
            Vocabulary {
                entity_types: vec!["gene".to_string(), "protein".to_string()],
                predicate_types: vec!["binds_to".to_string()],
                source_types: vec!["experimental".to_string()],
            },
        );

        let vocabs = store.get_registered_vocabularies();
        assert!(vocabs.contains_key("bio"));
        assert!(vocabs["bio"].entity_types.contains(&"gene".to_string()));

        store.register_predicate(
            "binds_to",
            serde_json::json!({
                "subject_types": ["protein"],
                "object_types": ["protein", "compound"]
            }),
        );
        assert!(store.get_predicate_constraints().contains_key("binds_to"));
    }

    #[test]
    fn test_wal_crash_recovery() {
        // Simulate: write claims → crash (no close) → reopen → data recovered
        let dir = std::env::temp_dir().join("attest_wal_recovery_test.attest");
        cleanup_db(&dir);

        // Phase 1: write claims, then DROP without close (simulates crash)
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            store.upsert_entity("tp53", "gene", "TP53", None, 0);
            store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "exp"));
            store.insert_claim(make_claim("c2", "tp53", "inhibits", "brca1", "lit"));
            // Intentionally no close() — Drop will attempt best-effort checkpoint
            // but even if that fails, WAL entries are durable
        }

        // Phase 2: reopen — should recover via WAL replay
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 2, "Should recover 2 claims from WAL/checkpoint");
            assert!(store.claim_exists("c1"));
            assert!(store.claim_exists("c2"));
            let mut store = store;
            store.close().unwrap();
        }

        cleanup_db(&dir);
    }

    #[test]
    fn test_checkpoint_without_close() {
        let dir = std::env::temp_dir().join("attest_checkpoint_test.attest");
        cleanup_db(&dir);

        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));

            // Checkpoint should persist without closing
            store.checkpoint().unwrap();

            // Insert more after checkpoint
            store.insert_claim(make_claim("c2", "b", "rel", "a", "obs"));
            store.close().unwrap();
        }

        // Reopen — should have both claims
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 2);
            let mut store = store;
            store.close().unwrap();
        }

        cleanup_db(&dir);
    }

    #[test]
    fn test_auto_checkpoint() {
        let dir = std::env::temp_dir().join("attest_auto_cp_test.attest");
        cleanup_db(&dir);

        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            // Set very low interval for testing
            store.checkpoint_interval = 3;

            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);

            // Insert 5 claims — should trigger auto-checkpoint at claim 3
            for i in 0..5 {
                let mut c = make_claim(&format!("c{i}"), "a", "rel", "b", "obs");
                c.timestamp = (i + 1) * 1000;
                store.insert_claim(c);
            }

            // The .attest file should exist (auto-checkpoint wrote it)
            assert!(dir.exists(), "Auto-checkpoint should have written .attest file");

            store.close().unwrap();
        }

        // Verify all 5 claims
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 5);
            let mut store = store;
            store.close().unwrap();
        }

        cleanup_db(&dir);
    }

    #[test]
    fn test_claims_in_range_boundary() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.timestamp = 1000;
        let mut c2 = make_claim("c2", "a", "rel", "b", "obs");
        c2.claim_id = "c2".to_string();
        c2.timestamp = 2000;
        let mut c3 = make_claim("c3", "a", "rel", "b", "obs");
        c3.claim_id = "c3".to_string();
        c3.timestamp = 3000;

        store.insert_claim(c1);
        store.insert_claim(c2);
        store.insert_claim(c3);

        // Exact boundary: single timestamp range should be inclusive
        let results = store.claims_in_range(1000, 1000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, 1000);

        // Range [2000, 3000] should return 2 claims (inclusive boundaries)
        let results = store.claims_in_range(2000, 3000);
        assert_eq!(results.len(), 2);

        // Empty store returns empty
        let empty_store = RustStore::in_memory();
        let mut empty_store = empty_store;
        assert!(empty_store.claims_in_range(0, 9999).is_empty());
    }

    #[test]
    fn test_most_recent_claims_edge_cases() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        for i in 0..3 {
            let mut c = make_claim(&format!("mr{i}"), "a", "rel", "b", "obs");
            c.timestamp = (i + 1) * 1000;
            store.insert_claim(c);
        }

        // n=0 should return empty
        assert!(store.most_recent_claims(0).is_empty());

        // n > total should return all
        let all = store.most_recent_claims(100);
        assert_eq!(all.len(), 3);

        // Empty store returns empty
        let mut empty_store = RustStore::in_memory();
        assert!(empty_store.most_recent_claims(5).is_empty());
    }

    #[test]
    fn test_bfs_depth_zero() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
        store.insert_claim(make_claim("c2", "b", "rel", "c", "obs"));

        // depth=0 should return no claims (no traversal)
        let results = store.bfs_claims("a", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bfs_cyclic_graph() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        // Create A→B→C→A cycle
        store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
        store.insert_claim(make_claim("c2", "b", "rel", "c", "obs"));
        store.insert_claim(make_claim("c3", "c", "rel", "a", "obs"));

        // BFS with depth=3 should not infinite loop and should return claims
        let results = store.bfs_claims("a", 3);
        // Should find all 3 claims (no infinite loop)
        let claim_ids: Vec<&str> = results.iter().map(|(c, _)| c.claim_id.as_str()).collect();
        assert!(claim_ids.contains(&"c1"));
        assert!(claim_ids.contains(&"c2"));
        assert!(claim_ids.contains(&"c3"));
    }

    #[test]
    fn test_path_exists_same_entity() {
        let mut store = setup_store();
        // Path from entity to itself should handle gracefully
        let result = store.path_exists("brca1", "brca1", 1);
        // Should return true (entity is trivially reachable from itself)
        assert!(result);
    }

    #[test]
    fn test_path_exists_nonexistent() {
        let mut store = setup_store();
        // Path with nonexistent entity should return false, not panic
        assert!(!store.path_exists("nonexistent", "brca1", 5));
        assert!(!store.path_exists("brca1", "nonexistent", 5));
        assert!(!store.path_exists("nonexistent", "also_nonexistent", 5));
    }

    #[test]
    fn test_search_entities_empty_query() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);

        // Empty query should return empty or all — just should not panic
        let results = store.search_entities("", 10);
        // Implementation-dependent: empty query typically returns nothing
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_search_entities_top_k_zero() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);

        // top_k=0 should return empty
        let results = store.search_entities("brca1", 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_bm25_ranking() {
        // BM25 should rank "brca1" (short doc, high term density) higher than
        // "brca1_pathway" (longer doc, lower term density) when searching for "brca1"
        let mut store = RustStore::in_memory();
        // Short doc: "brca1" appears in a 2-token document (high TF density)
        store.upsert_entity("brca1_gene", "gene", "brca1 gene", None, 0);
        // Longer doc: "brca1" appears once among many tokens (lower TF density)
        store.upsert_entity(
            "brca1_related_pathway",
            "pathway",
            "brca1 related dna repair signaling pathway mechanism",
            None,
            0,
        );
        // Unrelated entity that should not appear
        store.upsert_entity("tp53", "gene", "TP53 tumor protein", None, 0);

        let results = store.search_entities("brca1", 10);
        assert_eq!(results.len(), 2, "should match both brca1 entities");
        assert_eq!(
            results[0].id, "brca1_gene",
            "shorter doc with higher BM25 term density should rank first"
        );
        assert_eq!(
            results[1].id, "brca1_related_pathway",
            "longer doc should rank second"
        );
    }

    #[test]
    fn test_bm25_idf_boost() {
        // A rare term should boost an entity above one matching only common terms
        let mut store = RustStore::in_memory();
        // "alpha" is common across all 3 entities; "zyxin" is rare (only in entity_a)
        store.upsert_entity("entity_a", "protein", "alpha zyxin binding", None, 0);
        store.upsert_entity("entity_b", "protein", "alpha beta gamma", None, 0);
        store.upsert_entity("entity_c", "protein", "alpha delta epsilon", None, 0);

        // Searching "alpha zyxin": entity_a matches both (one rare), others match only "alpha"
        let results = store.search_entities("alpha zyxin", 10);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].id, "entity_a",
            "entity matching rare term 'zyxin' should rank first"
        );
    }

    #[test]
    fn test_batch_dedup_within_batch() {
        // Verifies that duplicate claim_ids within a single batch are deduplicated
        let dir = std::env::temp_dir().join("attest_batch_dedup_test.attest");
        cleanup_db(&dir);

        let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("dup1", "a", "rel", "b", "exp");
        let c1_copy = make_claim("dup1", "a", "rel", "b", "exp");
        let c2 = make_claim("unique1", "a", "rel", "b", "obs");

        let inserted = store.insert_claims_batch(vec![c1, c1_copy, c2]);
        assert_eq!(inserted, 2, "Should insert 2 unique claims, not 3");
        assert_eq!(store.stats().total_claims, 2);

        // Second batch with one already-existing and one new
        let c1_again = make_claim("dup1", "a", "rel", "b", "exp");
        let c3 = make_claim("unique2", "b", "rel", "a", "lit");
        let inserted2 = store.insert_claims_batch(vec![c1_again, c3]);
        assert_eq!(inserted2, 1, "Should only insert the new claim");
        assert_eq!(store.stats().total_claims, 3);

        store.close().unwrap();
        cleanup_db(&dir);
    }

    // ── Retraction tests ────────────────────────────────────────────────

    #[test]
    fn test_retract_claim() {
        let mut store = setup_store();
        // Before retraction
        assert_eq!(store.all_claims(0, 0).len(), 2);

        // Retract c1
        assert!(store.retract_claim("c1").unwrap());

        // get_claim should show Tombstoned status
        let c1 = store.get_claim("c1").unwrap();
        assert_eq!(c1.status, ClaimStatus::Tombstoned);

        // all_claims should exclude c1
        let all = store.all_claims(0, 0);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].claim_id, "c2");

        // claims_for should exclude c1
        let claims = store.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 0);

        // claims_for tp53 should only have c2
        let claims = store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c2");
    }

    #[test]
    fn test_retract_source() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.provenance.source_id = "src1".to_string();
        let mut c2 = make_claim("c2", "b", "rel", "c", "obs");
        c2.provenance.source_id = "src1".to_string();
        let mut c3 = make_claim("c3", "a", "rel", "c", "exp");
        c3.provenance.source_id = "src2".to_string();

        store.insert_claim(c1);
        store.insert_claim(c2);
        store.insert_claim(c3);

        // Retract source "src1"
        let retracted = store.retract_source("src1").unwrap();
        assert_eq!(retracted.len(), 2);
        assert!(retracted.contains(&"c1".to_string()));
        assert!(retracted.contains(&"c2".to_string()));

        // Only c3 should remain
        let all = store.all_claims(0, 0);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].claim_id, "c3");
    }

    #[test]
    fn test_include_retracted_flag() {
        let mut store = setup_store();

        store.retract_claim("c1").unwrap();
        assert_eq!(store.all_claims(0, 0).len(), 1);

        // Enable include_retracted
        store.set_include_retracted(true);
        assert_eq!(store.all_claims(0, 0).len(), 2);

        // Disable again
        store.set_include_retracted(false);
        assert_eq!(store.all_claims(0, 0).len(), 1);
    }

    #[test]
    fn test_retract_nonexistent() {
        let mut store = setup_store();
        let result = store.retract_claim("nonexistent").unwrap();
        assert!(!result, "Retracting nonexistent claim should return false");
    }

    #[test]
    fn test_retraction_persists_memory() {
        let dir = std::env::temp_dir().join("attest_retract_persist_mem.attest");
        cleanup_db(&dir);

        // Create, insert, retract, close
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
            store.insert_claim(make_claim("c2", "a", "rel", "b", "exp"));
            store.retract_claim("c1").unwrap();
            store.close().unwrap();
        }

        // Reopen and verify
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            let c1 = store.get_claim("c1").unwrap();
            assert_eq!(c1.status, ClaimStatus::Tombstoned, "Retraction should persist");
            assert_eq!(store.all_claims(0, 0).len(), 1, "Retracted claim should be filtered");
            store.close().unwrap();
        }

        cleanup_db(&dir);
    }

    #[test]
    fn test_retraction_persists_lmdb() {
        let dir = std::env::temp_dir().join("attest_retract_persist_lmdb.attest");
        cleanup_db(&dir);

        // Create, insert, retract, close
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
            store.insert_claim(make_claim("c2", "a", "rel", "b", "exp"));
            store.retract_claim("c1").unwrap();
            store.close().unwrap();
        }

        // Reopen and verify
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            let c1 = store.get_claim("c1").unwrap();
            assert_eq!(c1.status, ClaimStatus::Tombstoned, "Retraction should persist in LMDB");
            assert_eq!(store.all_claims(0, 0).len(), 1, "Retracted claim should be filtered after reopen");
            store.close().unwrap();
        }

        cleanup_db(&dir);
    }

    #[test]
    fn test_update_claim_status_string() {
        let mut store = setup_store();

        // Archive a claim
        assert!(store.update_claim_status("c1", "archived").unwrap());
        let c1 = store.get_claim("c1").unwrap();
        assert_eq!(c1.status, ClaimStatus::Archived);

        // Archived claims should still appear in queries (only Tombstoned are filtered)
        assert_eq!(store.all_claims(0, 0).len(), 2);

        // Now tombstone it
        assert!(store.update_claim_status("c1", "retracted").unwrap());
        assert_eq!(store.all_claims(0, 0).len(), 1);

        // Un-retract (set back to active)
        assert!(store.update_claim_status("c1", "active").unwrap());
        assert_eq!(store.all_claims(0, 0).len(), 2);
        let c1 = store.get_claim("c1").unwrap();
        assert_eq!(c1.status, ClaimStatus::Active);
    }

    #[test]
    fn test_retraction_filters_bfs() {
        let mut store = setup_store();
        store.retract_claim("c1").unwrap();

        // BFS from brca1 depth 2: c1 is retracted, so should not appear
        let results = store.bfs_claims("brca1", 2);
        assert!(results.is_empty(), "BFS should not return retracted claims");
    }

    #[test]
    fn test_retraction_filters_temporal() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.timestamp = 1000;
        let mut c2 = make_claim("c2", "a", "rel", "b", "obs");
        c2.timestamp = 2000;
        store.insert_claim(c1);
        store.insert_claim(c2);

        store.retract_claim("c1").unwrap();

        // claims_in_range should filter retracted
        assert_eq!(store.claims_in_range(500, 2500).len(), 1);

        // most_recent should filter retracted
        let recent = store.most_recent_claims(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].claim_id, "c2");
    }

    // ── Namespace tests ─────────────────────────────────────────────────

    fn make_ns_claim(
        claim_id: &str,
        subj: &str,
        pred: &str,
        obj: &str,
        namespace: &str,
    ) -> Claim {
        let mut c = make_claim(claim_id, subj, pred, obj, "experimental");
        c.namespace = namespace.to_string();
        c
    }

    #[test]
    fn test_namespace_filter_basic() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        store.insert_claim(make_ns_claim("c1", "a", "rel", "b", "team_alpha"));
        store.insert_claim(make_ns_claim("c2", "a", "rel", "b", "team_beta"));

        // No filter: both visible
        assert_eq!(store.all_claims(0, 0).len(), 2);

        // Filter to team_alpha
        store.set_namespace_filter(vec!["team_alpha".to_string()]);
        let claims = store.all_claims(0, 0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        // Filter to team_beta
        store.set_namespace_filter(vec!["team_beta".to_string()]);
        let claims = store.all_claims(0, 0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c2");
    }

    #[test]
    fn test_namespace_filter_empty_means_all() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        store.insert_claim(make_ns_claim("c1", "a", "rel", "b", "ns1"));
        store.insert_claim(make_ns_claim("c2", "a", "rel", "b", "ns2"));

        // Empty filter = all visible
        store.set_namespace_filter(vec![]);
        assert_eq!(store.all_claims(0, 0).len(), 2);

        // Verify getter
        assert!(store.get_namespace_filter().is_empty());
    }

    #[test]
    fn test_namespace_default_empty() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        // Claim without explicit namespace → namespace is ""
        store.insert_claim(make_claim("c1", "a", "rel", "b", "experimental"));
        let claim = store.get_claim("c1").unwrap();
        assert_eq!(claim.namespace, "");

        // Filter to empty string namespace should include it
        store.set_namespace_filter(vec!["".to_string()]);
        assert_eq!(store.all_claims(0, 0).len(), 1);

        // Filter to non-empty namespace should exclude it
        store.set_namespace_filter(vec!["other".to_string()]);
        assert_eq!(store.all_claims(0, 0).len(), 0);
    }

    #[test]
    fn test_namespace_filter_multiple() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        store.insert_claim(make_ns_claim("c1", "a", "rel", "b", "ns1"));
        store.insert_claim(make_ns_claim("c2", "a", "rel", "b", "ns2"));
        store.insert_claim(make_ns_claim("c3", "a", "rel", "b", "ns3"));

        // Filter to ns1 + ns3
        store.set_namespace_filter(vec!["ns1".to_string(), "ns3".to_string()]);
        let claims = store.all_claims(0, 0);
        assert_eq!(claims.len(), 2);
        let ids: Vec<&str> = claims.iter().map(|c| c.claim_id.as_str()).collect();
        assert!(ids.contains(&"c1"));
        assert!(ids.contains(&"c3"));
        assert!(!ids.contains(&"c2"));
    }

    #[test]
    fn test_namespace_filter_claims_for() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        store.insert_claim(make_ns_claim("c1", "a", "rel", "b", "ns1"));
        store.insert_claim(make_ns_claim("c2", "a", "rel", "b", "ns2"));

        store.set_namespace_filter(vec!["ns1".to_string()]);
        let claims = store.claims_for("a", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");
    }

    #[test]
    fn test_namespace_filter_bfs() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        store.insert_claim(make_ns_claim("c1", "a", "rel", "b", "ns1"));
        store.insert_claim(make_ns_claim("c2", "a", "rel", "b", "ns2"));

        store.set_namespace_filter(vec!["ns2".to_string()]);
        let results = store.bfs_claims("a", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.claim_id, "c2");
    }

    #[test]
    fn test_namespace_filter_temporal() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_ns_claim("c1", "a", "rel", "b", "ns1");
        c1.timestamp = 100;
        let mut c2 = make_ns_claim("c2", "a", "rel", "b", "ns2");
        c2.timestamp = 200;

        store.insert_claim(c1);
        store.insert_claim(c2);

        store.set_namespace_filter(vec!["ns1".to_string()]);
        let claims = store.claims_in_range(0, 300);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        let recent = store.most_recent_claims(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].claim_id, "c1");
    }

    #[test]
    fn test_get_entities_batch_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);

        let result = store.get_entities_batch(&["brca1", "tp53", "missing"]);
        assert_eq!(result.len(), 2);
        assert_eq!(result["brca1"], ("gene".into(), "BRCA1".into()));
        assert!(!result.contains_key("missing"));
    }

    #[test]
    fn test_insert_claims_batch_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("c1", "a", "rel", "b", "obs");
        let c2 = make_claim("c2", "a", "inhibits", "b", "exp");
        let c3 = make_claim("c3", "b", "activates", "a", "lit");

        let inserted = store.insert_claims_batch(vec![c1, c2, c3]);
        assert_eq!(inserted, 3);
        assert_eq!(store.stats().total_claims, 3);

        // All claims retrievable
        assert!(store.claim_exists("c1"));
        assert!(store.claim_exists("c2"));
        assert!(store.claim_exists("c3"));

        // Adjacency updated
        let claims = store.claims_for("a", None, None, 0.0);
        assert_eq!(claims.len(), 3);
    }

    #[test]
    fn test_insert_claims_batch_dedup_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("dup", "a", "rel", "b", "obs");
        let c1_copy = make_claim("dup", "a", "rel", "b", "obs");
        let c2 = make_claim("unique", "a", "rel", "b", "exp");

        let inserted = store.insert_claims_batch(vec![c1, c1_copy, c2]);
        assert_eq!(inserted, 2, "Duplicate claim_id should be skipped");
        assert_eq!(store.stats().total_claims, 2);
    }

    #[test]
    fn test_insert_claims_batch_skips_existing_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        // Insert one claim normally
        store.insert_claim(make_claim("existing", "a", "rel", "b", "obs"));
        assert_eq!(store.stats().total_claims, 1);

        // Batch includes the existing claim + a new one
        let batch = vec![
            make_claim("existing", "a", "rel", "b", "obs"),
            make_claim("new_one", "a", "inhibits", "b", "exp"),
        ];
        let inserted = store.insert_claims_batch(batch);
        assert_eq!(inserted, 1, "Only new claim should be inserted");
        assert_eq!(store.stats().total_claims, 2);
    }

    #[test]
    fn test_update_display_name_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("gene_1956", "gene", "Gene_1956", None, 0);

        // Verify original name
        let entity = store.get_entity("gene_1956").unwrap();
        assert_eq!(entity.name, "Gene_1956");

        // Update display name
        let updated = store.update_display_name("gene_1956", "EGFR");
        assert!(updated);

        // Verify new name
        let entity = store.get_entity("gene_1956").unwrap();
        assert_eq!(entity.name, "EGFR");
    }

    #[test]
    fn test_update_display_name_nonexistent_memory() {
        let mut store = RustStore::in_memory();

        // Updating a nonexistent entity returns false
        let updated = store.update_display_name("missing", "NewName");
        assert!(!updated);
    }

    #[test]
    fn test_update_display_name_search_reindex_memory() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("gene_1956", "gene", "Gene_1956", None, 0);

        // Should be searchable by old name
        let results = store.search_entities("Gene_1956", 10);
        assert_eq!(results.len(), 1);

        // Update
        store.update_display_name("gene_1956", "EGFR epidermal growth factor receptor");

        // Should be searchable by new name
        let results = store.search_entities("EGFR", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "gene_1956");
        assert_eq!(results[0].name, "EGFR epidermal growth factor receptor");
    }
}

// ── LMDB backend tests ────────────────────────────────────────────────
// Mirrors the in-memory tests above but uses RustStore::new() (LmdbBackend).

#[cfg(test)]
mod lmdb_tests {
    use super::*;
    use attest_core::types::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Generate a unique temp path per test to avoid conflicts.
    fn temp_db() -> (std::path::PathBuf, TempGuard) {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let path = std::env::temp_dir().join(format!("attest_lmdb_{tid:?}_{n}.attest"));
        // Clean up both files and directories (LMDB creates dirs)
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(&path);
        (path.clone(), TempGuard(path))
    }

    /// RAII cleanup for temp files or directories.
    struct TempGuard(std::path::PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn make_claim(
        claim_id: &str,
        subj: &str,
        pred: &str,
        obj: &str,
        source_type: &str,
    ) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: attest_core::compute_content_id(subj, pred, obj),
            subject: EntityRef {
                id: subj.to_string(),
                entity_type: "entity".to_string(),
                display_name: subj.to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: pred.to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: obj.to_string(),
                entity_type: "entity".to_string(),
                display_name: obj.to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.7,
            provenance: Provenance {
                source_type: source_type.to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        }
    }

    fn setup_store(path: &str) -> RustStore {
        let mut store = RustStore::new(path).unwrap();
        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);
        store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "experimental"));
        store.insert_claim(make_claim("c2", "tp53", "inhibits", "aspirin", "literature"));
        store
    }

    #[test]
    fn test_lmdb_entity_upsert_and_get() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let entity = store.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");
        assert_eq!(entity.entity_type, "gene");
        assert_eq!(entity.claim_count, 1);
    }

    #[test]
    fn test_lmdb_claim_exists() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        assert!(store.claim_exists("c1"));
        assert!(!store.claim_exists("c99"));
    }

    #[test]
    fn test_lmdb_get_claim() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());

        let found = store.get_claim("c1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().claim_id, "c1");
        assert!(store.get_claim("missing").is_none());
    }

    #[test]
    fn test_lmdb_claims_by_content_id() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("c1", "a", "rel", "b", "obs");
        let content_id = c1.content_id.clone();
        store.insert_claim(c1);

        let mut c2 = make_claim("c2", "a", "rel", "b", "exp");
        c2.content_id.clone_from(&content_id);
        store.insert_claim(c2);

        assert_eq!(store.claims_by_content_id(&content_id).len(), 2);
    }

    #[test]
    fn test_lmdb_claims_for() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        let claims = store.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        let claims = store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_lmdb_claims_for_with_filters() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        let claims = store.claims_for("tp53", None, Some("experimental"), 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");
    }

    #[test]
    fn test_lmdb_list_entities() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        assert_eq!(store.list_entities(Some("gene"), 0, 0, 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0, 0, 0).len(), 1);
        assert_eq!(store.list_entities(None, 0, 0, 0).len(), 3);
    }

    #[test]
    fn test_lmdb_list_entities_min_claims() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let high = store.list_entities(None, 2, 0, 0);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "tp53");
    }

    #[test]
    fn test_lmdb_alias_resolution() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("gene_a", "gene", "A", None, 0);
        store.upsert_entity("gene_a_alias", "gene", "A alias", None, 0);

        let alias_claim = Claim {
            claim_id: "alias1".to_string(),
            content_id: "alias_content".to_string(),
            subject: EntityRef {
                id: "gene_a".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A".to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: "same_as".to_string(),
                predicate_type: "same_as".to_string(),
            },
            object: EntityRef {
                id: "gene_a_alias".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A alias".to_string(),
                external_ids: Default::default(),
            },
            confidence: 1.0,
            provenance: Provenance {
                source_type: "human_annotation".to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        };
        store.insert_claim(alias_claim);

        let r1 = store.resolve("gene_a");
        let r2 = store.resolve("gene_a_alias");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_lmdb_bfs_claims() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        let results = store.bfs_claims("brca1", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.claim_id, "c1");
        assert_eq!(results[0].1, 1);

        let results = store.bfs_claims("brca1", 2);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_lmdb_path_exists() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        assert!(store.path_exists("brca1", "tp53", 1));
        assert!(store.path_exists("brca1", "aspirin", 2));
        assert!(!store.path_exists("brca1", "aspirin", 1));
    }

    #[test]
    fn test_lmdb_adjacency_list() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let adj = store.get_adjacency_list();
        assert!(adj["brca1"].contains("tp53"));
        assert!(adj["tp53"].contains("brca1"));
        assert!(adj["tp53"].contains("aspirin"));
    }

    #[test]
    fn test_lmdb_stats() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let stats = store.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.entity_types["gene"], 2);
        assert_eq!(stats.entity_types["compound"], 1);
    }

    #[test]
    fn test_lmdb_claims_in_range() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.timestamp = 1000;
        let mut c2 = make_claim("c2", "a", "rel", "b", "obs");
        c2.claim_id = "c2".to_string();
        c2.timestamp = 2000;
        let mut c3 = make_claim("c3", "a", "rel", "b", "obs");
        c3.claim_id = "c3".to_string();
        c3.timestamp = 3000;

        store.insert_claim(c1);
        store.insert_claim(c2);
        store.insert_claim(c3);

        assert_eq!(store.claims_in_range(1500, 2500).len(), 1);
        assert_eq!(store.claims_in_range(1000, 3000).len(), 3);
        assert_eq!(store.claims_in_range(4000, 5000).len(), 0);
    }

    #[test]
    fn test_lmdb_most_recent_claims() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        for i in 0..5 {
            let mut c = make_claim(&format!("c{i}"), "a", "rel", "b", "obs");
            c.timestamp = (i + 1) * 1000;
            store.insert_claim(c);
        }

        let recent = store.most_recent_claims(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[0].timestamp >= recent[1].timestamp);
    }

    #[test]
    fn test_lmdb_search_entities() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("brca1", "gene", "BRCA1 DNA repair", None, 0);
        store.upsert_entity("tp53", "gene", "TP53 tumor protein", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        let results = store.search_entities("brca1", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        let results = store.search_entities("protein", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "tp53");

        let results = store.search_entities("aspirin", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "aspirin");
    }

    #[test]
    fn test_lmdb_bm25_ranking() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("brca1_gene", "gene", "brca1 gene", None, 0);
        store.upsert_entity(
            "brca1_related_pathway",
            "pathway",
            "brca1 related dna repair signaling pathway mechanism",
            None,
            0,
        );
        store.upsert_entity("tp53", "gene", "TP53 tumor protein", None, 0);

        let results = store.search_entities("brca1", 10);
        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].id, "brca1_gene",
            "LMDB: shorter doc should rank first"
        );
        assert_eq!(
            results[1].id, "brca1_related_pathway",
            "LMDB: longer doc should rank second"
        );
    }

    #[test]
    fn test_lmdb_persistence_roundtrip() {
        let (path, _guard) = temp_db();

        // Create, populate, close
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
            store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            store.upsert_entity("tp53", "gene", "TP53", None, 0);
            store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "exp"));
            store.close().unwrap();
        }

        // Reopen, verify
        {
            let store = RustStore::new(path.to_str().unwrap()).unwrap();
            assert!(store.claim_exists("c1"));
            assert_eq!(store.get_entity("brca1").unwrap().name, "BRCA1");
            assert_eq!(store.stats().total_claims, 1);
        }
    }

    #[test]
    fn test_lmdb_batch_insert() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("dup1", "a", "rel", "b", "exp");
        let c1_copy = make_claim("dup1", "a", "rel", "b", "exp");
        let c2 = make_claim("unique1", "a", "rel", "b", "obs");

        let inserted = store.insert_claims_batch(vec![c1, c1_copy, c2]);
        assert_eq!(inserted, 2, "Should insert 2 unique claims, not 3");
        assert_eq!(store.stats().total_claims, 2);
    }

    #[test]
    fn test_lmdb_all_claims() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let all = store.all_claims(0, 0);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_lmdb_provenance_chain() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut claim = make_claim("c1", "a", "rel", "b", "obs");
        claim.provenance.chain = vec!["upstream1".to_string(), "upstream2".to_string()];
        store.insert_claim(claim);

        let chain = store.get_claim_provenance_chain("c1");
        assert_eq!(chain, vec!["upstream1", "upstream2"]);
        assert!(store.get_claim_provenance_chain("nonexistent").is_empty());
    }

    #[test]
    fn test_lmdb_metadata_persists() {
        let (path, _guard) = temp_db();

        // Create store with metadata, close
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
            store.register_vocabulary(
                "bio",
                Vocabulary {
                    entity_types: vec!["gene".to_string()],
                    predicate_types: vec!["binds_to".to_string()],
                    source_types: vec!["experimental".to_string()],
                },
            );
            store.close().unwrap();
        }

        // Reopen, verify metadata survived
        {
            let store = RustStore::new(path.to_str().unwrap()).unwrap();
            let vocabs = store.get_registered_vocabularies();
            assert!(vocabs.contains_key("bio"));
            assert!(vocabs["bio"].entity_types.contains(&"gene".to_string()));
        }
    }

    #[test]
    fn test_lmdb_alias_persists() {
        let (path, _guard) = temp_db();

        // Create store with alias, close
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
            store.upsert_entity("gene_a", "gene", "A", None, 0);
            store.upsert_entity("gene_a_alias", "gene", "A alias", None, 0);

            let alias_claim = Claim {
                claim_id: "alias1".to_string(),
                content_id: "alias_content".to_string(),
                subject: EntityRef {
                    id: "gene_a".to_string(),
                    entity_type: "gene".to_string(),
                    display_name: "A".to_string(),
                    external_ids: Default::default(),
                },
                predicate: PredicateRef {
                    id: "same_as".to_string(),
                    predicate_type: "same_as".to_string(),
                },
                object: EntityRef {
                    id: "gene_a_alias".to_string(),
                    entity_type: "gene".to_string(),
                    display_name: "A alias".to_string(),
                    external_ids: Default::default(),
                },
                confidence: 1.0,
                provenance: Provenance {
                    source_type: "human_annotation".to_string(),
                    source_id: "test".to_string(),
                    method: None,
                    chain: vec![],
                    model_version: None,
                    organization: None,
                },
                embedding: None,
                payload: None,
                timestamp: 1000,
                status: ClaimStatus::Active,
                namespace: String::new(),
            expires_at: 0,
            };
            store.insert_claim(alias_claim);
            store.close().unwrap();
        }

        // Reopen, verify alias survived
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
            let r1 = store.resolve("gene_a");
            let r2 = store.resolve("gene_a_alias");
            assert_eq!(r1, r2, "Alias should persist across close/reopen");
        }
    }

    #[test]
    fn test_namespace_persists_lmdb() {
        let (path, _guard) = temp_db();

        // Create store with namespaced claims, close
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);

            let mut c1 = make_claim("c1", "a", "rel", "b", "experimental");
            c1.namespace = "team_alpha".to_string();
            let mut c2 = make_claim("c2", "a", "rel", "b", "experimental");
            c2.namespace = "team_beta".to_string();

            store.insert_claim(c1);
            store.insert_claim(c2);
            store.close().unwrap();
        }

        // Reopen, verify namespaces survived
        {
            let mut store = RustStore::new(path.to_str().unwrap()).unwrap();

            // Without filter: both visible
            assert_eq!(store.all_claims(0, 0).len(), 2);

            // Filter to team_alpha
            store.set_namespace_filter(vec!["team_alpha".to_string()]);
            let claims = store.all_claims(0, 0);
            assert_eq!(claims.len(), 1);
            assert_eq!(claims[0].claim_id, "c1");
            assert_eq!(claims[0].namespace, "team_alpha");

            // Verify claims_for also filters
            let claims = store.claims_for("a", None, None, 0.0);
            assert_eq!(claims.len(), 1);
            assert_eq!(claims[0].claim_id, "c1");

            store.close().unwrap();
        }
    }

    #[test]
    fn test_lmdb_read_only_mode() {
        let (path, _guard) = temp_db();
        let path_str = path.to_str().unwrap();

        // Write some data with a normal store
        {
            let mut store = setup_store(path_str);
            store.close().unwrap();
        }

        // Open read-only — should not require exclusive lock
        let mut ro_store = RustStore::open_read_only(path_str).unwrap();
        assert!(ro_store.is_read_only());

        // Reads work
        let e = ro_store.get_entity("brca1");
        assert!(e.is_some());
        assert_eq!(e.unwrap().name, "BRCA1");

        let claims = ro_store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 2); // tp53 appears in both c1 and c2

        let results = ro_store.search_entities("BRCA1", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "brca1");

        let stats = ro_store.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 3);
    }

    #[test]
    fn test_lmdb_read_only_multiple_readers() {
        let (path, _guard) = temp_db();
        let path_str = path.to_str().unwrap();

        // Write some data, then close the writer
        {
            let mut store = setup_store(path_str);
            store.close().unwrap();
        }

        // Open two concurrent readers (both use shared file locks)
        let mut ro1 = RustStore::open_read_only(path_str).unwrap();
        let mut ro2 = RustStore::open_read_only(path_str).unwrap();

        // Both can read simultaneously
        let e1 = ro1.get_entity("brca1");
        assert!(e1.is_some());

        let e2 = ro2.get_entity("tp53");
        assert!(e2.is_some());

        let claims1 = ro1.claims_for("tp53", None, None, 0.0);
        let claims2 = ro2.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims1.len(), 2);
        assert_eq!(claims2.len(), 1);
    }

    #[test]
    fn test_lmdb_get_entities_batch() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();

        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        // Batch lookup — should return all 3 + skip unknown
        let ids = vec!["brca1", "tp53", "aspirin", "unknown_entity"];
        let result = store.get_entities_batch(&ids);
        assert_eq!(result.len(), 3);
        assert_eq!(result["brca1"], ("gene".into(), "BRCA1".into()));
        assert_eq!(result["tp53"], ("gene".into(), "TP53".into()));
        assert_eq!(result["aspirin"], ("compound".into(), "Aspirin".into()));
        assert!(!result.contains_key("unknown_entity"));
    }

    #[test]
    fn test_lmdb_two_phase_entity_resolution() {
        // Simulate two-phase bulk loading:
        // 1. Register entities via upsert_entities_batch
        // 2. Insert claims — entity_map is empty, Rust resolves from store
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();

        // Phase 1: register entities
        let entities = vec![
            ("brca1".to_string(), "gene".to_string(), "BRCA1".to_string(), HashMap::new()),
            ("tp53".to_string(), "gene".to_string(), "TP53".to_string(), HashMap::new()),
        ];
        store.upsert_entities_batch(&entities, 1000).unwrap();

        // Verify entities are readable
        let batch = store.get_entities_batch(&["brca1", "tp53"]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch["brca1"].1, "BRCA1");

        // Phase 2: insert claims (display names come from store)
        let claim = Claim {
            claim_id: "test_claim".to_string(),
            content_id: attest_core::compute_content_id("brca1", "binds_to", "tp53"),
            subject: EntityRef {
                id: "brca1".to_string(),
                entity_type: "gene".to_string(),
                display_name: "BRCA1".to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: "binds_to".to_string(),
                predicate_type: "relation".to_string(),
            },
            object: EntityRef {
                id: "tp53".to_string(),
                entity_type: "gene".to_string(),
                display_name: "TP53".to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.9,
            provenance: Provenance {
                source_type: "database_import".to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        };
        let inserted = store.insert_claims_batch(vec![claim]);
        assert_eq!(inserted, 1);

        // Verify claim has correct entity metadata
        let retrieved = store.get_claim("test_claim").unwrap();
        assert_eq!(retrieved.subject.display_name, "BRCA1");
        assert_eq!(retrieved.object.display_name, "TP53");
    }

    #[test]
    fn test_lmdb_insert_claims_batch_adjacency() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        let claims = vec![
            make_claim("c1", "a", "rel", "b", "obs"),
            make_claim("c2", "b", "inhibits", "c", "exp"),
            make_claim("c3", "a", "activates", "c", "lit"),
        ];

        let inserted = store.insert_claims_batch(claims);
        assert_eq!(inserted, 3);
        assert_eq!(store.stats().total_claims, 3);

        // Adjacency works for batch-inserted claims
        let claims_a = store.claims_for("a", None, None, 0.0);
        assert_eq!(claims_a.len(), 2); // c1 + c3

        let claims_b = store.claims_for("b", None, None, 0.0);
        assert_eq!(claims_b.len(), 2); // c1 + c2

        // BFS traversal works
        assert!(store.path_exists("a", "c", 1)); // direct via c3
        assert!(store.path_exists("a", "c", 2)); // also via b
    }

    #[test]
    fn test_lmdb_insert_claims_batch_skips_existing() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        // Insert one claim normally
        store.insert_claim(make_claim("existing", "a", "rel", "b", "obs"));
        assert_eq!(store.stats().total_claims, 1);

        // Batch includes the existing claim + a new one
        let batch = vec![
            make_claim("existing", "a", "rel", "b", "obs"),
            make_claim("new_one", "a", "inhibits", "b", "exp"),
        ];
        let inserted = store.insert_claims_batch(batch);
        assert_eq!(inserted, 1, "Only new claim should be inserted");
        assert_eq!(store.stats().total_claims, 2);
    }

    #[test]
    fn test_lmdb_insert_claims_batch_persists() {
        let (path, _guard) = temp_db();
        let path_str = path.to_str().unwrap();

        // Batch insert, then close
        {
            let mut store = RustStore::new(path_str).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);

            let claims = vec![
                make_claim("c1", "a", "rel", "b", "obs"),
                make_claim("c2", "a", "inhibits", "b", "exp"),
            ];
            let inserted = store.insert_claims_batch(claims);
            assert_eq!(inserted, 2);
            store.close().unwrap();
        }

        // Reopen, verify data survived
        {
            let store = RustStore::new(path_str).unwrap();
            assert!(store.claim_exists("c1"));
            assert!(store.claim_exists("c2"));
            assert_eq!(store.stats().total_claims, 2);
        }
    }

    #[test]
    fn test_lmdb_update_display_name() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("gene_1956", "gene", "Gene_1956", None, 0);

        // Verify original name
        let entity = store.get_entity("gene_1956").unwrap();
        assert_eq!(entity.name, "Gene_1956");

        // Update display name
        let updated = store.update_display_name("gene_1956", "EGFR");
        assert!(updated);

        // Verify new name
        let entity = store.get_entity("gene_1956").unwrap();
        assert_eq!(entity.name, "EGFR");
    }

    #[test]
    fn test_lmdb_update_display_name_nonexistent() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();

        let updated = store.update_display_name("missing", "NewName");
        assert!(!updated);
    }

    #[test]
    fn test_lmdb_update_display_name_search_reindex() {
        let (path, _guard) = temp_db();
        let mut store = RustStore::new(path.to_str().unwrap()).unwrap();
        store.upsert_entity("gene_1956", "gene", "Gene_1956", None, 0);

        // Searchable by old name
        let results = store.search_entities("Gene_1956", 10);
        assert_eq!(results.len(), 1);

        // Update
        store.update_display_name("gene_1956", "EGFR epidermal growth factor receptor");

        // Searchable by new name
        let results = store.search_entities("EGFR", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "gene_1956");
        assert_eq!(results[0].name, "EGFR epidermal growth factor receptor");
    }

    #[test]
    fn test_lmdb_update_display_name_persists() {
        let (path, _guard) = temp_db();
        let path_str = path.to_str().unwrap();

        // Update, close
        {
            let mut store = RustStore::new(path_str).unwrap();
            store.upsert_entity("gene_1956", "gene", "Gene_1956", None, 0);
            store.update_display_name("gene_1956", "EGFR");
            store.close().unwrap();
        }

        // Reopen, verify
        {
            let store = RustStore::new(path_str).unwrap();
            let entity = store.get_entity("gene_1956").unwrap();
            assert_eq!(entity.name, "EGFR");

            // Search index also persisted
            let results = store.search_entities("EGFR", 10);
            assert_eq!(results.len(), 1);
        }
    }
}
