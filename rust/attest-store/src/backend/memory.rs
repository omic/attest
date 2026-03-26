//! In-memory storage backend with optional file persistence.
//!
//! This is the original RustStore implementation extracted into a backend.
//! Used for `:memory:` databases.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::normalization::normalize_entity_id;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};
use attest_core::vocabulary::ALIAS_PREDICATES;

use crate::claim_log::ClaimLog;
use crate::entity_store::EntityStore;
use crate::file_format;
use crate::metadata::{MetadataStore, Vocabulary};
use crate::store::StoreStats;
use crate::union_find::UnionFind;
use crate::wal::{Wal, WalEntry};

/// Serializable snapshot of the entire store state (owned, for deserialization).
#[derive(Debug, Clone, serde::Deserialize)]
struct StoreState {
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
    #[serde(default)]
    status_overrides: HashMap<String, u8>,
}

/// Borrowed view of store state for serialization (avoids cloning).
#[derive(Debug, serde::Serialize)]
struct StoreStateRef<'a> {
    claims: &'a ClaimLog,
    entities: &'a EntityStore,
    metadata: &'a MetadataStore,
    aliases: &'a UnionFind,
    status_overrides: &'a HashMap<String, u8>,
}

/// Default: auto-checkpoint after this many WAL entries.
pub(crate) const DEFAULT_CHECKPOINT_INTERVAL: u64 = 1000;

/// In-memory storage backend with optional file persistence via checkpoint + WAL.
pub struct MemoryBackend {
    db_path: PathBuf,
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
    lock_file: Option<std::fs::File>,
    wal: Option<Wal>,
    /// Status overrides: claim_id → ClaimStatus. Persisted in checkpoint + WAL.
    status_overrides: HashMap<String, ClaimStatus>,
    /// In-memory set of tombstoned IDs for O(1) filtering.
    retracted_ids: HashSet<String>,
    /// Whether query methods should include retracted claims.
    include_retracted: bool,
    /// Namespace filter: empty = all namespaces visible.
    namespace_filter: Vec<String>,
    /// Corroboration counts: content_id → number of active claims.
    content_id_counts: HashMap<String, u32>,
}

impl MemoryBackend {
    /// Create a purely in-memory backend (no file path, no persistence).
    pub fn new_empty() -> Self {
        Self {
            db_path: PathBuf::new(),
            claims: ClaimLog::new(),
            entities: EntityStore::new(),
            metadata: MetadataStore::new(),
            aliases: UnionFind::new(),
            lock_file: None,
            wal: None,
            status_overrides: HashMap::new(),
            retracted_ids: HashSet::new(),
            include_retracted: false,
            namespace_filter: Vec::new(),
            content_id_counts: HashMap::new(),
        }
    }

    /// Open or create a file-backed memory backend.
    ///
    /// Acquires an exclusive file lock. If the database file exists but
    /// has a checksum mismatch (crash during write), the store starts
    /// empty with a warning — the corrupted file is not trusted.
    pub fn open(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);

        // Acquire exclusive lock
        let lock_file = file_format::acquire_lock(&path)
            .map_err(|e| AttestError::Provenance(format!("failed to lock database: {e}")))?;

        let mut backend = if path.exists() && file_format::is_attest_file(&path) {
            match file_format::read_store::<StoreState>(&path) {
                Ok(state) => {
                    let mut claims = state.claims;
                    let mut entities = state.entities;
                    let mut aliases = state.aliases;
                    // Rebuild indexes that may be empty from older file formats
                    claims.rebuild_derived_indexes();
                    entities.rebuild_text_index();
                    aliases.rebuild_groups();
                    // Rebuild status overrides and retracted_ids from persisted map
                    let status_overrides: HashMap<String, ClaimStatus> = state.status_overrides
                        .into_iter()
                        .map(|(k, v)| (k, Self::u8_to_status(v)))
                        .collect();
                    let retracted_ids: HashSet<String> = status_overrides.iter()
                        .filter(|(_, s)| **s == ClaimStatus::Tombstoned)
                        .map(|(k, _)| k.clone())
                        .collect();
                    Self {
                        db_path: path.clone(),
                        claims,
                        entities,
                        metadata: state.metadata,
                        aliases,
                        lock_file: Some(lock_file),
                        wal: None,
                        status_overrides,
                        retracted_ids,
                        include_retracted: false,
                        namespace_filter: Vec::new(),
            content_id_counts: HashMap::new(),
                    }
                }
                Err(file_format::FileError::ChecksumMismatch { expected, actual }) => {
                    // Crash recovery: checkpoint is corrupted, WAL replay below
                    // may still recover data
                    log::warn!(
                        "Database file corrupted (checksum {actual:#010x} != {expected:#010x}). \
                         Starting with empty store, will attempt WAL replay."
                    );
                    Self {
                        db_path: path.clone(),
                        claims: ClaimLog::new(),
                        entities: EntityStore::new(),
                        metadata: MetadataStore::new(),
                        aliases: UnionFind::new(),
                        lock_file: Some(lock_file),
                        wal: None,
                        status_overrides: HashMap::new(),
                        retracted_ids: HashSet::new(),
                        include_retracted: false,
                        namespace_filter: Vec::new(),
            content_id_counts: HashMap::new(),
                    }
                }
                Err(e) => {
                    file_format::release_lock(&lock_file);
                    return Err(AttestError::Provenance(
                        format!("failed to open database: {e}"),
                    ));
                }
            }
        } else {
            Self {
                db_path: path.clone(),
                claims: ClaimLog::new(),
                entities: EntityStore::new(),
                metadata: MetadataStore::new(),
                aliases: UnionFind::new(),
                lock_file: Some(lock_file),
                wal: None,
                status_overrides: HashMap::new(),
                retracted_ids: HashSet::new(),
                include_retracted: false,
                namespace_filter: Vec::new(),
            content_id_counts: HashMap::new(),
            }
        };

        // Replay WAL entries (claims recorded after last checkpoint)
        match Wal::read_entries(&path) {
            Ok(entries) => {
                let count = entries.len();
                for entry in entries {
                    match entry {
                        WalEntry::InsertClaim(claim) => {
                            backend.insert_claim_no_wal(claim);
                        }
                        WalEntry::UpdateClaimStatus(claim_id, status_u8) => {
                            let status = Self::u8_to_status(status_u8);
                            if status == ClaimStatus::Tombstoned {
                                backend.retracted_ids.insert(claim_id.clone());
                            } else {
                                backend.retracted_ids.remove(&claim_id);
                            }
                            backend.status_overrides.insert(claim_id, status);
                        }
                    }
                }
                if count > 0 {
                    log::info!("Replayed {count} WAL entries");
                }
            }
            Err(e) => {
                log::warn!("Failed to read WAL: {e}");
            }
        }

        // Open WAL for future writes
        backend.wal = Some(
            Wal::open(&path)
                .map_err(|e| AttestError::Provenance(format!("failed to open WAL: {e}")))?
        );

        Ok(backend)
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    pub fn close(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self.lock_file.is_none() {
            return Ok(()); // in-memory or already closed
        }
        let status_overrides_u8: HashMap<String, u8> = self.status_overrides.iter()
            .map(|(k, v)| (k.clone(), Self::status_to_u8(v)))
            .collect();
        let state = StoreStateRef {
            claims: &self.claims,
            entities: &self.entities,
            metadata: &self.metadata,
            aliases: &self.aliases,
            status_overrides: &status_overrides_u8,
        };
        let write_result = file_format::write_store(&self.db_path, &state);

        // Truncate WAL after successful checkpoint
        if write_result.is_ok() {
            if let Some(ref mut wal) = self.wal {
                if let Err(e) = wal.truncate() {
                    log::warn!("Failed to truncate WAL after checkpoint: {e}");
                }
            }
        }

        // Always release the lock, even if write failed — holding it wedges the DB
        if let Some(lock) = self.lock_file.take() {
            file_format::release_lock(&lock);
        }
        self.wal = None;

        write_result
            .map_err(|e| AttestError::Provenance(format!("failed to save database: {e}")))
    }

    /// Write a full checkpoint without releasing the lock.
    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self.lock_file.is_none() {
            return Ok(());
        }
        let status_overrides_u8: HashMap<String, u8> = self.status_overrides.iter()
            .map(|(k, v)| (k.clone(), Self::status_to_u8(v)))
            .collect();
        let state = StoreStateRef {
            claims: &self.claims,
            entities: &self.entities,
            metadata: &self.metadata,
            aliases: &self.aliases,
            status_overrides: &status_overrides_u8,
        };
        file_format::write_store(&self.db_path, &state)
            .map_err(|e| AttestError::Provenance(format!("checkpoint failed: {e}")))?;

        if let Some(ref mut wal) = self.wal {
            wal.truncate()
                .map_err(|e| AttestError::Provenance(format!("WAL truncate failed: {e}")))?;
        }
        Ok(())
    }

    // ── Metadata ───────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        self.metadata.register_vocabulary(namespace, vocab);
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        self.metadata.register_predicate(predicate_id, constraints);
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        self.metadata.register_payload_schema(schema_id, schema);
    }

    pub fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        self.metadata.vocabularies()
    }

    pub fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.predicate_constraints()
    }

    pub fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.payload_schemas()
    }

    // ── Alias resolution ───────────────────────────────────────────────

    /// Resolve entity ID through alias chain.
    pub fn resolve(&mut self, entity_id: &str) -> String {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.find(&canonical)
    }

    /// Get all entity IDs that resolve to the same canonical.
    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.get_group(&canonical)
    }

    // ── Cache management ───────────────────────────────────────────────

    /// No-op in memory backend — everything is already in memory.
    pub fn warm_caches(&self) {}

    // ── Entity CRUD ────────────────────────────────────────────────────

    pub fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        self.entities
            .upsert(entity_id, entity_type, display_name, external_ids, timestamp);
    }

    pub fn update_display_name(&mut self, entity_id: &str, new_display: &str) -> bool {
        self.entities.update_display_name(entity_id, new_display)
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        let claim_count = self.claims.count_for_entity(entity_id);
        self.entities.get_summary(entity_id, claim_count)
    }

    /// Batch-lookup entity metadata (type, display_name) for multiple IDs.
    pub fn get_entities_batch(&self, entity_ids: &[&str]) -> HashMap<String, (String, String)> {
        let mut result = HashMap::with_capacity(entity_ids.len());
        for id in entity_ids {
            if let Some(es) = self.get_entity(id) {
                result.insert(es.id.clone(), (es.entity_type.clone(), es.name.clone()));
            }
        }
        result
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
        offset: usize,
        limit: usize,
    ) -> Vec<EntitySummary> {
        let filtered = self.entities
            .list(entity_type)
            .into_iter()
            .filter_map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                if count >= min_claims {
                    Some(e.to_summary(count))
                } else {
                    None
                }
            });
        if limit == 0 {
            filtered.skip(offset).collect()
        } else {
            filtered.skip(offset).take(limit).collect()
        }
    }

    /// Count entities matching the given filter without materializing results.
    pub fn count_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> usize {
        if min_claims == 0 {
            match entity_type {
                Some(_) => self.entities.list(entity_type).len(),
                None => self.entities.len(),
            }
        } else {
            self.entities
                .list(entity_type)
                .into_iter()
                .filter(|e| self.claims.count_for_entity(&e.id) >= min_claims)
                .count()
        }
    }

    /// Count total claims.
    pub fn count_claims(&self) -> usize {
        self.claims.len()
    }

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim with WAL write.
    /// Returns `false` if the claim_id already exists.
    pub fn insert_claim(&mut self, claim: Claim, checkpoint_interval: u64) -> bool {
        if self.claims.contains(&claim.claim_id) {
            return false;
        }

        // Write to WAL first (durability before visibility)
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.append_claim(&claim) {
                log::error!("WAL write failed for claim {}: {e}", claim.claim_id);
                return false;
            }
        }

        self.apply_claim(claim);

        // Auto-checkpoint if interval reached
        if checkpoint_interval > 0 {
            if let Some(ref wal) = self.wal {
                if wal.entry_count() >= checkpoint_interval {
                    if let Err(e) = self.checkpoint() {
                        log::warn!("Auto-checkpoint failed: {e}");
                    }
                }
            }
        }

        true
    }

    /// Batch-insert claims with a single WAL sync.
    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>, checkpoint_interval: u64) -> usize {
        let mut inserted = 0usize;
        for claim in claims {
            if self.claims.contains(&claim.claim_id) {
                continue;
            }

            // Write to WAL without fsync
            if let Some(ref mut wal) = self.wal {
                if let Err(e) = wal.append_claim_no_sync(&claim) {
                    log::error!("WAL write failed for claim {}: {e}", claim.claim_id);
                    continue; // Skip claim — not durable
                }
            }

            self.apply_claim(claim);
            inserted += 1;
        }

        // Single sync + checkpoint at end of batch
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.sync() {
                log::error!("WAL sync failed after batch insert: {e}");
            }
        }
        if checkpoint_interval > 0 {
            if let Some(ref wal) = self.wal {
                if wal.entry_count() >= checkpoint_interval {
                    if let Err(e) = self.checkpoint() {
                        log::warn!("Auto-checkpoint failed after batch: {e}");
                    }
                }
            }
        }

        inserted
    }

    /// Insert a claim without WAL write (used during WAL replay).
    fn insert_claim_no_wal(&mut self, claim: Claim) -> bool {
        if self.claims.contains(&claim.claim_id) {
            return false;
        }
        self.apply_claim(claim);
        true
    }

    /// Apply a claim to in-memory state (indexes + aliases).
    fn apply_claim(&mut self, claim: Claim) {
        let pred_id = claim.predicate.id.clone();
        let subj_id = claim.subject.id.clone();
        let obj_id = claim.object.id.clone();
        let content_id = claim.content_id.clone();

        self.claims.append(claim);

        // Increment corroboration counter
        *self.content_id_counts.entry(content_id).or_insert(0) += 1;

        if ALIAS_PREDICATES.contains(pred_id.as_str()) {
            if pred_id == "same_as" {
                self.aliases.union(&subj_id, &obj_id);
            } else if pred_id == "not_same_as" {
                self.aliases.split(&subj_id);
            }
        }
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        self.claims.contains(claim_id)
    }

    // ── Retraction / status overlay helpers ─────────────────────────────

    /// Whether a claim should be included in query results.
    fn should_include_claim(&self, claim_id: &str) -> bool {
        self.include_retracted || !self.retracted_ids.contains(claim_id)
    }

    /// Apply status overlay from in-memory overrides map.
    fn apply_status_overlay(overrides: &HashMap<String, ClaimStatus>, claim: &mut Claim) {
        if let Some(status) = overrides.get(&claim.claim_id) {
            claim.status = status.clone();
        }
    }

    fn status_to_u8(status: &ClaimStatus) -> u8 {
        match status {
            ClaimStatus::Active => 0,
            ClaimStatus::Archived => 1,
            ClaimStatus::Tombstoned => 2,
            ClaimStatus::ProvenanceDegraded => 3,
            ClaimStatus::Verified => 4,
            ClaimStatus::VerificationFailed => 5,
            ClaimStatus::Disputed => 6,
        }
    }

    fn u8_to_status(v: u8) -> ClaimStatus {
        match v {
            0 => ClaimStatus::Active,
            1 => ClaimStatus::Archived,
            2 => ClaimStatus::Tombstoned,
            3 => ClaimStatus::ProvenanceDegraded,
            4 => ClaimStatus::Verified,
            5 => ClaimStatus::VerificationFailed,
            6 => ClaimStatus::Disputed,
            _ => ClaimStatus::Active,
        }
    }

    pub fn update_claim_status(&mut self, claim_id: &str, status: ClaimStatus) -> Result<bool, AttestError> {
        if !self.claims.contains(claim_id) {
            return Ok(false);
        }

        // WAL write
        if let Some(ref mut wal) = self.wal {
            let status_u8 = Self::status_to_u8(&status);
            if let Err(e) = wal.append_status_update(claim_id, status_u8) {
                log::error!("WAL write failed for status update {claim_id}: {e}");
                return Err(AttestError::Provenance(format!("WAL write failed: {e}")));
            }
        }

        // Update corroboration counter
        if let Some(claim) = self.claims.get(claim_id) {
            let cid = claim.content_id.clone();
            if status == ClaimStatus::Tombstoned {
                if let Some(count) = self.content_id_counts.get_mut(&cid) {
                    *count = count.saturating_sub(1);
                }
            } else if self.retracted_ids.contains(claim_id) {
                // Reinstating a previously retracted claim
                *self.content_id_counts.entry(cid).or_insert(0) += 1;
            }
        }

        if status == ClaimStatus::Tombstoned {
            self.retracted_ids.insert(claim_id.to_string());
        } else {
            self.retracted_ids.remove(claim_id);
        }
        self.status_overrides.insert(claim_id.to_string(), status);
        Ok(true)
    }

    pub fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError> {
        let mut count = 0usize;
        for (claim_id, status) in updates {
            if !self.claims.contains(claim_id) {
                continue;
            }

            // WAL write (no sync per entry)
            if let Some(ref mut wal) = self.wal {
                let status_u8 = Self::status_to_u8(status);
                if let Err(e) = wal.append_entry(&WalEntry::UpdateClaimStatus(claim_id.clone(), status_u8)) {
                    log::error!("WAL write failed for batch status update {claim_id}: {e}");
                    continue;
                }
            }

            if *status == ClaimStatus::Tombstoned {
                self.retracted_ids.insert(claim_id.clone());
            } else {
                self.retracted_ids.remove(claim_id);
            }
            self.status_overrides.insert(claim_id.clone(), status.clone());
            count += 1;
        }

        // Sync WAL after batch
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.sync() {
                log::error!("WAL sync failed after batch status update: {e}");
            }
        }

        Ok(count)
    }

    pub fn set_include_retracted(&mut self, include: bool) {
        self.include_retracted = include;
    }

    /// Check if a claim's namespace passes the current filter.
    fn should_include_namespace(&self, namespace: &str) -> bool {
        self.namespace_filter.is_empty() || self.namespace_filter.iter().any(|ns| ns == namespace)
    }

    pub fn set_namespace_filter(&mut self, namespaces: Vec<String>) {
        self.namespace_filter = namespaces;
    }

    pub fn get_namespace_filter(&self) -> &[String] {
        &self.namespace_filter
    }

    pub fn get_claim(&self, claim_id: &str) -> Option<Claim> {
        let mut claim = self.claims.get(claim_id)?.clone();
        // Apply status overlay
        if let Some(status) = self.status_overrides.get(claim_id) {
            claim.status = status.clone();
        }
        Some(claim)
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        self.claims
            .by_content_id(content_id)
            .into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c })
            .collect()
    }

    pub fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        let all = self.claims.all_claims();
        let filtered = all.iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c });
        if limit == 0 {
            filtered.skip(offset).collect()
        } else {
            filtered.skip(offset).take(limit).collect()
        }
    }

    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        self.claims
            .by_source_id(source_id)
            .into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c })
            .collect()
    }

    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        self.claims
            .by_predicate_id(predicate_id)
            .into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c })
            .collect()
    }

    pub fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
        limit: usize,
    ) -> Vec<Claim> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);
        let iter = self.claims
            .for_entity_filtered(&aliases, predicate_type, source_type, min_confidence)
            .into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c });
        if limit > 0 {
            iter.take(limit).collect()
        } else {
            iter.collect()
        }
    }

    /// Get outgoing causal edges for an entity — lightweight, returns
    /// list of (target_entity_id, predicate_id, confidence) tuples.
    /// Only returns claims where entity_id is the SUBJECT and predicate is causal.
    pub fn outgoing_causal_edges(
        &self,
        entity_id: &str,
        causal_predicates: &HashSet<String>,
    ) -> Vec<(String, String, f64)> {
        let normalized = normalize_entity_id(entity_id);
        let mut results = Vec::new();
        for claim in self.claims.for_entity(&normalized) {
            if claim.subject.id != normalized {
                continue;
            }
            if !causal_predicates.contains(&claim.predicate.id) {
                continue;
            }
            if !self.should_include_claim(&claim.claim_id) {
                continue;
            }
            if !self.should_include_namespace(&claim.namespace) {
                continue;
            }
            results.push((claim.object.id.clone(), claim.predicate.id.clone(), claim.confidence));
        }
        results
    }

    /// Get claims for an entity, optionally including inverse-derived claims.
    pub fn claims_for_with_inverse(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
        include_inverse: bool,
    ) -> Vec<Claim> {
        let mut results = self.claims_for(entity_id, predicate_type, source_type, min_confidence, 0);
        if !include_inverse {
            return results;
        }
        let resolved = self.resolve(entity_id);
        // Track existing (subject, predicate, object) triples to avoid duplicates
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        for c in &results {
            seen.insert((c.subject.id.clone(), c.predicate.id.clone(), c.object.id.clone()));
        }
        // For claims where entity is the OBJECT, derive inverse claims
        let mut inverse_claims = Vec::new();
        for c in &results {
            if c.object.id == resolved {
                if let Some(&inv_pred) = attest_core::vocabulary::INVERSE_PREDICATES.get(c.predicate.id.as_str()) {
                    let key = (resolved.clone(), inv_pred.to_string(), c.subject.id.clone());
                    if !seen.contains(&key) {
                        seen.insert(key);
                        let mut derived = c.clone();
                        // Swap subject and object
                        let tmp = derived.subject.clone();
                        derived.subject = derived.object.clone();
                        derived.object = tmp;
                        derived.predicate.id = inv_pred.to_string();
                        derived.provenance.method = Some("inverse_derived".to_string());
                        derived.claim_id = format!("inv:{}", c.claim_id);
                        inverse_claims.push(derived);
                    }
                }
            }
        }
        results.extend(inverse_claims);
        results
    }

    /// Compute transitive closure over causal predicates from an entity.
    pub fn transitive_closure(
        &mut self,
        entity_id: &str,
        predicates: &HashSet<String>,
        max_depth: usize,
    ) -> Vec<(String, String, usize, f64)> {
        use attest_core::vocabulary::compose_predicates;
        use std::collections::VecDeque;

        let max_depth = max_depth.min(5); // hard cap
        let resolved = self.resolve(entity_id);

        // BFS: (current_entity, accumulated_predicate, depth, accumulated_confidence)
        let mut queue: VecDeque<(String, String, usize, f64)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(resolved.clone());

        // Seed with direct causal neighbors
        let direct_edges = self.outgoing_causal_edges(&resolved, predicates);
        for (target, pred, conf) in direct_edges {
            if !visited.contains(&target) {
                queue.push_back((target, pred, 1, conf));
            }
        }

        let mut results: Vec<(String, String, usize, f64)> = Vec::new();

        while let Some((current, acc_pred, depth, acc_conf)) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            // At depth >= 2, this is a composed result (not a direct neighbor)
            if depth >= 2 {
                results.push((current.clone(), acc_pred.clone(), depth, acc_conf));
            }

            // Continue BFS if not at max depth
            if depth < max_depth {
                let edges = self.outgoing_causal_edges(&current, predicates);
                for (next_target, edge_pred, edge_conf) in edges {
                    if visited.contains(&next_target) {
                        continue;
                    }
                    let composed = compose_predicates(&acc_pred, &edge_pred).to_string();
                    if composed == "associated_with" {
                        continue; // weak composition, skip
                    }
                    let new_conf = acc_conf * edge_conf;
                    queue.push_back((next_target, composed, depth + 1, new_conf));
                }
            }
        }

        results
    }

    /// Get corroboration count for a content_id.
    pub fn corroboration_count(&self, content_id: &str) -> u32 {
        *self.content_id_counts.get(content_id).unwrap_or(&0)
    }

    /// Get the provenance chain for a claim.
    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        self.claims
            .get(claim_id)
            .map(|c| c.provenance.chain.clone())
            .unwrap_or_default()
    }

    // ── Graph traversal ────────────────────────────────────────────────

    /// BFS traversal collecting claims at each hop depth.
    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);
        let mut visited: HashSet<String> = aliases.clone();
        let mut frontier: HashSet<String> = aliases;
        let mut results: Vec<(Claim, usize)> = Vec::new();

        for hop in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }

            let claims = self.claims.for_entities(&frontier);
            let mut next_frontier = HashSet::new();

            for claim in claims {
                if !self.should_include_claim(&claim.claim_id) || !self.should_include_namespace(&claim.namespace) {
                    continue;
                }
                let mut c = claim.clone();
                Self::apply_status_overlay(&self.status_overrides, &mut c);
                results.push((c, hop));
                for eid in [&claim.subject.id, &claim.object.id] {
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
            }

            frontier = next_frontier;
        }

        results
    }

    /// Check if a path exists between two entities within max_depth hops.
    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        let ra = self.resolve(entity_a);
        let rb = self.resolve(entity_b);

        if ra == rb {
            return true;
        }

        // BFS from ra
        let mut visited = HashSet::new();
        visited.insert(ra.clone());
        let mut frontier = HashSet::new();
        frontier.insert(ra);

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let claims = self.claims.for_entities(&frontier);
            let mut next_frontier = HashSet::new();

            for claim in claims {
                for eid in [&claim.subject.id, &claim.object.id] {
                    if *eid == rb {
                        return true;
                    }
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
            }

            frontier = next_frontier;
        }

        false
    }

    /// Get the bidirectional adjacency list.
    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        self.claims.adjacency_list().clone()
    }

    /// Get claims within a timestamp range [min_ts, max_ts] (inclusive).
    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        let include_retracted = self.include_retracted;
        let retracted_ids = &self.retracted_ids;
        let ns_filter = &self.namespace_filter;
        self.claims
            .in_time_range(min_ts, max_ts)
            .into_iter()
            .filter(|c| (include_retracted || !retracted_ids.contains(&c.claim_id))
                && (ns_filter.is_empty() || ns_filter.iter().any(|ns| ns == &c.namespace)))
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c })
            .collect()
    }

    /// Get the most recent N claims by timestamp.
    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        let extra = self.retracted_ids.len();
        let include_retracted = self.include_retracted;
        let retracted_ids = &self.retracted_ids;
        let ns_filter = &self.namespace_filter;
        self.claims.most_recent(n + extra)
            .into_iter()
            .filter(|c| (include_retracted || !retracted_ids.contains(&c.claim_id))
                && (ns_filter.is_empty() || ns_filter.iter().any(|ns| ns == &c.namespace)))
            .take(n)
            .cloned()
            .map(|mut c| { Self::apply_status_overlay(&self.status_overrides, &mut c); c })
            .collect()
    }

    /// Search entities by text query.
    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        self.entities
            .search(query, top_k)
            .into_iter()
            .map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                e.to_summary(count)
            })
            .collect()
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            total_claims: self.claims.len(),
            entity_count: self.entities.len(),
            entity_types: self.entities.count_by_type(),
            predicate_types: self.claims.count_by_predicate_type(),
            source_types: self.claims.count_by_source_type(),
        }
    }

    // ── Analytics (Rust-native aggregation) ───────────────────────────

    pub fn predicate_counts(&self) -> HashMap<String, u64> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for claim in self.claims.all_claims() {
            if !self.should_include_claim(&claim.claim_id) || !self.should_include_namespace(&claim.namespace) {
                continue;
            }
            *counts.entry(claim.predicate.id.clone()).or_insert(0) += 1;
        }
        counts
    }

    pub fn find_contradictions(
        &self,
        pred_a: &str,
        pred_b: &str,
    ) -> Vec<(String, String, u64, u64, f64, f64)> {
        let claims_a = self.claims_by_predicate_id(pred_a);
        let mut pairs_a: HashMap<(String, String), (u64, f64)> = HashMap::new();
        for c in &claims_a {
            let key = (c.subject.id.clone(), c.object.id.clone());
            let entry = pairs_a.entry(key).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += c.confidence;
        }

        let claims_b = self.claims_by_predicate_id(pred_b);
        let mut results = Vec::new();
        let mut pairs_b: HashMap<(String, String), (u64, f64)> = HashMap::new();
        for c in &claims_b {
            let key = (c.subject.id.clone(), c.object.id.clone());
            if pairs_a.contains_key(&key) {
                let entry = pairs_b.entry(key).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += c.confidence;
            }
        }
        for (pair, (count_b, sum_conf_b)) in &pairs_b {
            if let Some((count_a, sum_conf_a)) = pairs_a.get(pair) {
                results.push((
                    pair.0.clone(),
                    pair.1.clone(),
                    *count_a,
                    *count_b,
                    sum_conf_a / *count_a as f64,
                    sum_conf_b / *count_b as f64,
                ));
            }
        }
        results.sort_by(|a, b| (b.2 + b.3).cmp(&(a.2 + a.3)));
        results
    }

    pub fn gap_analysis(
        &self,
        entity_type: &str,
        expected_predicates: &[&str],
        limit: usize,
    ) -> Vec<(String, Vec<String>)> {
        let mut entity_preds: HashMap<String, HashSet<String>> = HashMap::new();
        for pred_id in expected_predicates {
            for claim in self.claims_by_predicate_id(pred_id) {
                if claim.subject.entity_type == entity_type {
                    entity_preds.entry(claim.subject.id.clone())
                        .or_default()
                        .insert(pred_id.to_string());
                }
                if claim.object.entity_type == entity_type {
                    entity_preds.entry(claim.object.id.clone())
                        .or_default()
                        .insert(pred_id.to_string());
                }
            }
        }
        let expected_set: HashSet<&str> = expected_predicates.iter().copied().collect();
        let mut gaps: Vec<(String, Vec<String>)> = Vec::new();
        for (entity_id, preds) in &entity_preds {
            let missing: Vec<String> = expected_set.iter()
                .filter(|p| !preds.contains(**p))
                .map(|p| p.to_string())
                .collect();
            if !missing.is_empty() {
                gaps.push((entity_id.clone(), missing));
            }
        }
        gaps.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        if limit > 0 && gaps.len() > limit {
            gaps.truncate(limit);
        }
        gaps
    }

    pub fn entity_predicate_counts(&self, entity_id: &str) -> Vec<(String, u64)> {
        let mut counts: HashMap<String, u64> = HashMap::new();
        for claim in self.claims.all_claims() {
            if (claim.subject.id == entity_id || claim.object.id == entity_id)
                && self.should_include_claim(&claim.claim_id)
                && self.should_include_namespace(&claim.namespace) {
                *counts.entry(claim.predicate.id.clone()).or_insert(0) += 1;
            }
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    pub fn entity_source_counts(&self, entity_id: &str) -> Vec<(String, u64, f64)> {
        let mut sources: HashMap<String, (u64, f64)> = HashMap::new();
        for claim in self.claims.all_claims() {
            if (claim.subject.id == entity_id || claim.object.id == entity_id)
                && self.should_include_claim(&claim.claim_id)
                && self.should_include_namespace(&claim.namespace) {
                let entry = sources.entry(claim.provenance.source_id.clone()).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += claim.confidence;
            }
        }
        let mut result: Vec<_> = sources.into_iter()
            .map(|(src, (cnt, sum_conf))| (src, cnt, sum_conf / cnt as f64))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    pub fn find_single_source_entities(&self, min_claims: u64) -> Vec<String> {
        // Build per-entity source sets from adjacency
        let mut entity_stats: HashMap<String, HashMap<String, u64>> = HashMap::new();
        for claim in self.claims.all_claims() {
            if !self.should_include_claim(&claim.claim_id)
                || !self.should_include_namespace(&claim.namespace) {
                continue;
            }
            let src = &claim.provenance.source_id;
            for eid in [&claim.subject.id, &claim.object.id] {
                *entity_stats.entry(eid.clone()).or_default()
                    .entry(src.clone()).or_insert(0) += 1;
            }
        }
        entity_stats.into_iter()
            .filter(|(_, sources)| {
                sources.len() == 1 && sources.values().sum::<u64>() >= min_claims
            })
            .map(|(id, _)| id)
            .collect()
    }
}

impl Drop for MemoryBackend {
    fn drop(&mut self) {
        if self.lock_file.is_some() {
            // close() was never called — but WAL has durable entries.
            // Best-effort: try to checkpoint before releasing lock.
            log::warn!(
                "RustStore dropped without calling close() — \
                 attempting best-effort checkpoint."
            );
            if let Err(e) = self.checkpoint() {
                log::error!(
                    "Best-effort checkpoint in Drop failed: {e}. \
                     WAL entries are still durable and will replay on next open."
                );
            }
            self.wal = None;
            if let Some(lock) = self.lock_file.take() {
                file_format::release_lock(&lock);
            }
        }
    }
}
