//! RustStore — drop-in replacement for KuzuStore.
//!
//! Thin dispatcher over storage backends. All business logic
//! lives in the backend modules (`backend::memory`, `backend::redb`).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};

use crate::backend::memory::DEFAULT_CHECKPOINT_INTERVAL;
use crate::backend::migration;
use crate::backend::{Backend, MemoryBackend, RedbBackend, StorageBackend};
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

        // If file exists with old SUBSTRT\0 format → auto-migrate to redb
        if migration::needs_migration(path) {
            log::info!("Detected old format database, migrating to redb...");
            let backend = migration::migrate_to_redb(db_path)?;
            return Ok(Self {
                backend: Backend::File(Box::new(backend)),
                checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
                read_only: false,
                temp_path: None,
            });
        }

        // Otherwise → use RedbBackend (handles non-existent, valid redb, and corrupted files)
        let backend = RedbBackend::open(db_path)?;
        Ok(Self {
            backend: Backend::File(Box::new(backend)),
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            read_only: false,
            temp_path: None,
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

    /// Open a database in read-only mode by copying it to a temp file.
    ///
    /// This avoids acquiring the exclusive write lock on the original file,
    /// allowing MCP hooks and other readers to inspect the database while
    /// a writer holds the lock on the real file.
    ///
    /// All write operations will return an error. On `close()`, the temp
    /// file is deleted.
    pub fn open_read_only(db_path: &str) -> Result<Self, AttestError> {
        let src = std::path::Path::new(db_path);
        if !src.exists() {
            return Err(AttestError::Provenance(format!(
                "database not found: {db_path}"
            )));
        }

        // Copy to temp file so we don't contend for the lock
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join(format!(
            "attest_ro_{}_{}.attest",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::copy(src, &temp_file).map_err(|e| {
            AttestError::Provenance(format!("failed to copy for read-only: {e}"))
        })?;

        let temp_path_str = temp_file.to_str().ok_or_else(|| {
            AttestError::Provenance("invalid temp path".to_string())
        })?;
        let backend = RedbBackend::open(temp_path_str)?;

        Ok(Self {
            backend: Backend::File(Box::new(backend)),
            checkpoint_interval: 0,
            read_only: true,
            temp_path: Some(temp_file),
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

    /// Batch-upsert entities in a single transaction (redb) or loop (in-memory).
    pub fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) {
        if self.read_only { return; }
        self.backend.upsert_entities_batch(entities, timestamp);
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        self.backend.get_entity(entity_id)
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
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

        // Create, insert, retract, close
        {
            let mut store = RustStore::new_memory_backend(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
            store.insert_claim(make_claim("c2", "a", "rel", "b", "exp"));
            store.retract_claim("c1").unwrap();
            store.close().unwrap();
        }

        // Reopen and verify
        {
            let mut store = RustStore::new_memory_backend(dir.to_str().unwrap()).unwrap();
            let c1 = store.get_claim("c1").unwrap();
            assert_eq!(c1.status, ClaimStatus::Tombstoned, "Retraction should persist");
            assert_eq!(store.all_claims(0, 0).len(), 1, "Retracted claim should be filtered");
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }

    #[test]
    fn test_retraction_persists_redb() {
        let dir = std::env::temp_dir().join("attest_retract_persist_redb.attest");
        let _ = std::fs::remove_file(&dir);

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
            assert_eq!(c1.status, ClaimStatus::Tombstoned, "Retraction should persist in redb");
            assert_eq!(store.all_claims(0, 0).len(), 1, "Retracted claim should be filtered after reopen");
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
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
    fn test_redb_get_claim() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());

        let found = store.get_claim("c1");
        assert!(found.is_some());
        assert_eq!(found.unwrap().claim_id, "c1");
        assert!(store.get_claim("missing").is_none());
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
        assert_eq!(store.list_entities(Some("gene"), 0, 0, 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0, 0, 0).len(), 1);
        assert_eq!(store.list_entities(None, 0, 0, 0).len(), 3);
    }

    #[test]
    fn test_redb_list_entities_min_claims() {
        let (path, _guard) = temp_db();
        let store = setup_store(path.to_str().unwrap());
        let high = store.list_entities(None, 2, 0, 0);
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
            namespace: String::new(),
            expires_at: 0,
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
    fn test_redb_bm25_ranking() {
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
            "redb: shorter doc should rank first"
        );
        assert_eq!(
            results[1].id, "brca1_related_pathway",
            "redb: longer doc should rank second"
        );
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
        let all = store.all_claims(0, 0);
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
    fn test_namespace_persists_redb() {
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
}
