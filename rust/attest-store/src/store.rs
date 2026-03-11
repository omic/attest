//! RustStore — drop-in replacement for KuzuStore.
//!
//! Thin dispatcher over storage backends. All business logic
//! lives in the backend modules (`backend::memory`, `backend::redb`).

use std::collections::{HashMap, HashSet};

use attest_core::errors::AttestError;
use attest_core::types::{Claim, EntitySummary};

use crate::backend::memory::DEFAULT_CHECKPOINT_INTERVAL;
use crate::backend::migration;
use crate::backend::{Backend, MemoryBackend, RedbBackend};
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
}

impl RustStore {
    /// Open or create a store at the given path.
    ///
    /// Acquires an exclusive file lock. If the database file exists but
    /// has a checksum mismatch (crash during write), the store starts
    /// empty with a warning — the corrupted file is not trusted.
    pub fn new(db_path: &str) -> Result<Self, AttestError> {
        let path = std::path::Path::new(db_path);

        // If file exists with old SUBSTRT\0 format → auto-migrate to redb
        if migration::needs_migration(path) {
            log::info!("Detected old format database, migrating to redb...");
            let backend = migration::migrate_to_redb(db_path)?;
            return Ok(Self {
                backend: Backend::File(Box::new(backend)),
                checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            });
        }

        // Otherwise → use RedbBackend (handles non-existent, valid redb, and corrupted files)
        let backend = RedbBackend::open(db_path)?;
        Ok(Self {
            backend: Backend::File(Box::new(backend)),
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
        })
    }

    /// Open or create a store using the legacy in-memory backend.
    /// This is used for `:memory:` stores and for opening old SUBSTRT\0 format files.
    #[cfg(test)]
    pub fn new_memory_backend(db_path: &str) -> Result<Self, AttestError> {
        let backend = MemoryBackend::open(db_path)?;
        Ok(Self {
            backend: Backend::InMemory(Box::new(backend)),
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
        })
    }

    /// Create a purely in-memory store (no file path, no persistence, no lock, no WAL).
    pub fn in_memory() -> Self {
        Self {
            backend: Backend::InMemory(Box::new(MemoryBackend::new_empty())),
            checkpoint_interval: 0,
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    pub fn close(&mut self) -> Result<(), AttestError> {
        match &mut self.backend {
            Backend::InMemory(m) => m.close(),
            Backend::File(r) => r.close(),
        }
    }

    /// Write a full checkpoint without releasing the lock.
    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        match &mut self.backend {
            Backend::InMemory(m) => m.checkpoint(),
            Backend::File(r) => r.checkpoint(),
        }
    }

    // ── Metadata ───────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        match &mut self.backend {
            Backend::InMemory(m) => m.register_vocabulary(namespace, vocab),
            Backend::File(r) => r.register_vocabulary(namespace, vocab),
        }
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        match &mut self.backend {
            Backend::InMemory(m) => m.register_predicate(predicate_id, constraints),
            Backend::File(r) => r.register_predicate(predicate_id, constraints),
        }
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        match &mut self.backend {
            Backend::InMemory(m) => m.register_payload_schema(schema_id, schema),
            Backend::File(r) => r.register_payload_schema(schema_id, schema),
        }
    }

    pub fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        match &self.backend {
            Backend::InMemory(m) => m.get_registered_vocabularies(),
            Backend::File(r) => r.get_registered_vocabularies(),
        }
    }

    pub fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        match &self.backend {
            Backend::InMemory(m) => m.get_predicate_constraints(),
            Backend::File(r) => r.get_predicate_constraints(),
        }
    }

    pub fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        match &self.backend {
            Backend::InMemory(m) => m.get_payload_schemas(),
            Backend::File(r) => r.get_payload_schemas(),
        }
    }

    // ── Alias resolution ───────────────────────────────────────────────

    /// Resolve entity ID through alias chain.
    pub fn resolve(&mut self, entity_id: &str) -> String {
        match &mut self.backend {
            Backend::InMemory(m) => m.resolve(entity_id),
            Backend::File(r) => r.resolve(entity_id),
        }
    }

    /// Get all entity IDs that resolve to the same canonical.
    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        match &mut self.backend {
            Backend::InMemory(m) => m.get_alias_group(entity_id),
            Backend::File(r) => r.get_alias_group(entity_id),
        }
    }

    // ── Cache management ───────────────────────────────────────────────

    /// No-op in Rust — everything is already in memory.
    pub fn warm_caches(&self) {
        match &self.backend {
            Backend::InMemory(m) => m.warm_caches(),
            Backend::File(r) => r.warm_caches(),
        }
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
        match &mut self.backend {
            Backend::InMemory(m) => {
                m.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp)
            }
            Backend::File(r) => {
                r.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp)
            }
        }
    }

    /// Batch-upsert entities in a single transaction (redb) or loop (in-memory).
    pub fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) {
        match &mut self.backend {
            Backend::InMemory(m) => {
                for (id, etype, display, ext_ids) in entities {
                    m.upsert_entity(id, etype, display, Some(ext_ids), timestamp);
                }
            }
            Backend::File(r) => r.upsert_entities_batch(entities, timestamp),
        }
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        match &self.backend {
            Backend::InMemory(m) => m.get_entity(entity_id),
            Backend::File(r) => r.get_entity(entity_id),
        }
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> Vec<EntitySummary> {
        match &self.backend {
            Backend::InMemory(m) => m.list_entities(entity_type, min_claims),
            Backend::File(r) => r.list_entities(entity_type, min_claims),
        }
    }

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim. Caller must ensure entities exist.
    ///
    /// The claim is appended to the WAL and fsynced BEFORE being applied
    /// to in-memory state. Returns `false` if the claim_id already exists.
    pub fn insert_claim(&mut self, claim: Claim) -> bool {
        let interval = self.checkpoint_interval;
        match &mut self.backend {
            Backend::InMemory(m) => m.insert_claim(claim, interval),
            Backend::File(r) => r.insert_claim(claim, interval),
        }
    }

    /// Insert a batch of claims with a single WAL sync at the end.
    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>) -> usize {
        let interval = self.checkpoint_interval;
        match &mut self.backend {
            Backend::InMemory(m) => m.insert_claims_batch(claims, interval),
            Backend::File(r) => r.insert_claims_batch(claims, interval),
        }
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        match &self.backend {
            Backend::InMemory(m) => m.claim_exists(claim_id),
            Backend::File(r) => r.claim_exists(claim_id),
        }
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        match &self.backend {
            Backend::InMemory(m) => m.claims_by_content_id(content_id),
            Backend::File(r) => r.claims_by_content_id(content_id),
        }
    }

    /// Get all claims.
    pub fn all_claims(&self) -> Vec<Claim> {
        match &self.backend {
            Backend::InMemory(m) => m.all_claims(),
            Backend::File(r) => r.all_claims(),
        }
    }

    /// Get all claims matching a source_id.
    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        match &self.backend {
            Backend::InMemory(m) => m.claims_by_source_id(source_id),
            Backend::File(r) => r.claims_by_source_id(source_id),
        }
    }

    /// Get all claims matching a predicate_id.
    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        match &self.backend {
            Backend::InMemory(m) => m.claims_by_predicate_id(predicate_id),
            Backend::File(r) => r.claims_by_predicate_id(predicate_id),
        }
    }

    pub fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim> {
        match &mut self.backend {
            Backend::InMemory(m) => {
                m.claims_for(entity_id, predicate_type, source_type, min_confidence)
            }
            Backend::File(r) => {
                r.claims_for(entity_id, predicate_type, source_type, min_confidence)
            }
        }
    }

    /// Get the provenance chain for a claim.
    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        match &self.backend {
            Backend::InMemory(m) => m.get_claim_provenance_chain(claim_id),
            Backend::File(r) => r.get_claim_provenance_chain(claim_id),
        }
    }

    // ── Graph traversal ────────────────────────────────────────────────

    /// BFS traversal collecting claims at each hop depth.
    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        match &mut self.backend {
            Backend::InMemory(m) => m.bfs_claims(entity_id, max_depth),
            Backend::File(r) => r.bfs_claims(entity_id, max_depth),
        }
    }

    /// Check if a path exists between two entities within max_depth hops.
    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        match &mut self.backend {
            Backend::InMemory(m) => m.path_exists(entity_a, entity_b, max_depth),
            Backend::File(r) => r.path_exists(entity_a, entity_b, max_depth),
        }
    }

    /// Get the bidirectional adjacency list.
    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        match &self.backend {
            Backend::InMemory(m) => m.get_adjacency_list(),
            Backend::File(r) => r.get_adjacency_list(),
        }
    }

    /// Get claims within a timestamp range [min_ts, max_ts] (inclusive).
    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        match &mut self.backend {
            Backend::InMemory(m) => m.claims_in_range(min_ts, max_ts),
            Backend::File(r) => r.claims_in_range(min_ts, max_ts),
        }
    }

    /// Get the most recent N claims by timestamp.
    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        match &mut self.backend {
            Backend::InMemory(m) => m.most_recent_claims(n),
            Backend::File(r) => r.most_recent_claims(n),
        }
    }

    /// Search entities by text query.
    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        match &self.backend {
            Backend::InMemory(m) => m.search_entities(query, top_k),
            Backend::File(r) => r.search_entities(query, top_k),
        }
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        match &self.backend {
            Backend::InMemory(m) => m.stats(),
            Backend::File(r) => r.stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

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
        assert_eq!(store.list_entities(Some("gene"), 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0).len(), 1);
        assert_eq!(store.list_entities(None, 0).len(), 3);
    }

    #[test]
    fn test_list_entities_min_claims() {
        let store = setup_store();
        // tp53 has 2 claims, brca1 and aspirin have 1 each
        let high = store.list_entities(None, 2);
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
        let _ = std::fs::remove_file(&dir);

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

        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn test_file_locking() {
        let dir = std::env::temp_dir().join("attest_lock_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        // First open succeeds
        let mut store1 = RustStore::new(dir.to_str().unwrap()).unwrap();

        // Second open should fail with lock error
        let result = RustStore::new(dir.to_str().unwrap());
        assert!(result.is_err());
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(err_msg.contains("lock"), "Error should mention lock: {err_msg}");
            }
            Ok(_) => panic!("Expected lock error"),
        }

        // After close, lock is released
        store1.close().unwrap();
        let mut store2 = RustStore::new(dir.to_str().unwrap()).unwrap();
        store2.close().unwrap();

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
    }

    #[test]
    fn test_crash_recovery_corrupted_file() {
        let dir = std::env::temp_dir().join("attest_crash_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        // Write a valid store
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "a", "obs"));
            store.close().unwrap();
        }

        // Corrupt the file by overwriting the magic bytes (detected by both backends)
        {
            let mut data = std::fs::read(&dir).unwrap();
            // Overwrite first 16 bytes — triggers magic number check in both
            // the old SUBSTRT\0 format and redb's file header
            for i in 0..std::cmp::min(16, data.len()) {
                data[i] = 0xFF;
            }
            std::fs::write(&dir, &data).unwrap();
        }

        // Opening should succeed (crash recovery) but start empty
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 0, "Corrupted file should start empty");
            assert_eq!(store.stats().entity_count, 0);
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp() {
        let dir = std::env::temp_dir().join("attest_atomic_test.attest");
        let tmp = dir.with_extension("attest.tmp");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.close().unwrap();

        // Temp file should not exist after successful close
        assert!(!tmp.exists(), "Temp file should be cleaned up after close");
        // Main file should exist
        assert!(dir.exists(), "Main file should exist after close");

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
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
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

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

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }

    #[test]
    fn test_checkpoint_without_close() {
        let dir = std::env::temp_dir().join("attest_checkpoint_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

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

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }

    #[test]
    fn test_auto_checkpoint() {
        let dir = std::env::temp_dir().join("attest_auto_cp_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

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

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
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
    fn test_batch_dedup_within_batch() {
        // Verifies that duplicate claim_ids within a single batch are deduplicated
        let dir = std::env::temp_dir().join("attest_batch_dedup_test.attest");
        let _ = std::fs::remove_file(&dir);

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
        let _ = std::fs::remove_file(&dir);
    }
}

// ── Redb backend tests ─────────────────────────────────────────────────
// Mirrors the in-memory tests above but uses RustStore::new() (RedbBackend).

#[cfg(test)]
mod redb_tests {
    use super::*;
    use attest_core::types::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    /// Generate a unique temp path per test to avoid conflicts.
    fn temp_db() -> (std::path::PathBuf, TempGuard) {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tid = std::thread::current().id();
        let path = std::env::temp_dir().join(format!("attest_redb_{tid:?}_{n}.attest"));
        let _ = std::fs::remove_file(&path);
        (path.clone(), TempGuard(path))
    }

    /// RAII cleanup for temp files.
    struct TempGuard(std::path::PathBuf);
    impl Drop for TempGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
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
    fn test_redb_entity_upsert_and_get() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let entity = store.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");
        assert_eq!(entity.entity_type, "gene");
        assert_eq!(entity.claim_count, 1);
    }

    #[test]
    fn test_redb_claim_exists() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        assert!(store.claim_exists("c1"));
        assert!(!store.claim_exists("c99"));
    }

    #[test]
    fn test_redb_claims_by_content_id() {
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
    fn test_redb_claims_for() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        let claims = store.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        let claims = store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_redb_claims_for_with_filters() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        let claims = store.claims_for("tp53", None, Some("experimental"), 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");
    }

    #[test]
    fn test_redb_list_entities() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        assert_eq!(store.list_entities(Some("gene"), 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0).len(), 1);
        assert_eq!(store.list_entities(None, 0).len(), 3);
    }

    #[test]
    fn test_redb_list_entities_min_claims() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let high = store.list_entities(None, 2);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "tp53");
    }

    #[test]
    fn test_redb_alias_resolution() {
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
        };
        store.insert_claim(alias_claim);

        let r1 = store.resolve("gene_a");
        let r2 = store.resolve("gene_a_alias");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_redb_bfs_claims() {
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
    fn test_redb_path_exists() {
        let (path, _guard) = temp_db();
        let mut store = setup_store(path.to_str().unwrap());
        assert!(store.path_exists("brca1", "tp53", 1));
        assert!(store.path_exists("brca1", "aspirin", 2));
        assert!(!store.path_exists("brca1", "aspirin", 1));
    }

    #[test]
    fn test_redb_adjacency_list() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let adj = store.get_adjacency_list();
        assert!(adj["brca1"].contains("tp53"));
        assert!(adj["tp53"].contains("brca1"));
        assert!(adj["tp53"].contains("aspirin"));
    }

    #[test]
    fn test_redb_stats() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let stats = store.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.entity_types["gene"], 2);
        assert_eq!(stats.entity_types["compound"], 1);
    }

    #[test]
    fn test_redb_claims_in_range() {
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
    fn test_redb_most_recent_claims() {
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
    fn test_redb_search_entities() {
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
    fn test_redb_persistence_roundtrip() {
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
    fn test_redb_batch_insert() {
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
    fn test_redb_all_claims() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let all = store.all_claims();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_redb_provenance_chain() {
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
    fn test_redb_metadata_persists() {
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
    fn test_redb_alias_persists() {
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
}
