//! File-backed storage backend using LMDB (via heed).
//!
//! Opens an LMDB environment at the given directory path. All claim/entity data
//! lives on disk via mmap. Union-find and metadata stay in-memory (tiny) and are
//! flushed to a META_BLOBS database on close/checkpoint.
//!
//! Key properties:
//! - Instant crash recovery (CoW B+ tree, no WAL replay)
//! - Concurrent cross-process readers (MVCC)
//! - Instant open (mmap, no page tree verification)
//! - Proven at TB scale for 20+ years

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::io::Cursor;
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::normalization::normalize_entity_id;
use attest_core::types::{
    Claim, ClaimStatus, EntityRef, EntitySummary, Payload, PredicateRef, Provenance,
};
use attest_core::vocabulary::ALIAS_PREDICATES;
use heed::types::*;
use heed::{Database, DatabaseFlags, Env, EnvOpenOptions};

use crate::entity_store::{bm25_score, tokenize, EntityData};
use crate::metadata::{MetadataStore, Vocabulary};
use crate::store::StoreStats;
use crate::union_find::UnionFind;

use serde::{Deserialize, Serialize};

/// Compression header bytes for claim storage.
const COMPRESSION_NONE: u8 = 0x00;
const COMPRESSION_ZSTD: u8 = 0x01;
/// Zstd compression level (3 = good balance of speed vs ratio).
const ZSTD_LEVEL: i32 = 3;

/// META_BLOBS keys for in-memory state.
const META_KEY_ALIASES: &str = "aliases";
const META_KEY_METADATA: &str = "metadata";
const META_KEY_SCHEMA_VERSION: &str = "schema_version";
const META_KEY_NEXT_SEQ: &str = "next_seq";

/// Causal predicates that get indexed in causal_adj for fast composition lookups.
const CAUSAL_PREDICATES: &[&str] = &[
    "activates", "inhibits", "upregulates", "downregulates", "regulates",
    "promotes", "suppresses", "increases", "decreases",
    "causes", "prevents", "enables", "blocks",
];

/// Current schema version.
const CURRENT_SCHEMA_VERSION: u32 = 3;

/// Safe ceiling for LMDB key/DUP_SORT data values.
/// LMDB hard limit is 511 bytes; we stay 21 bytes under so compound keys
/// (entity_id + separator + predicate_id) have room.
const LMDB_MAX_KEY: usize = 490;

/// Default LMDB map size: 1 GB.
/// This is virtual address space (costs nothing until pages are touched).
/// Starts moderate to avoid exhausting per-process mmap limits when many
/// environments are open simultaneously (e.g., test suites).
/// Auto-resize on MDB_MAP_FULL doubles the map size.
const DEFAULT_MAP_SIZE: usize = 1 << 30; // 1 GB

/// Maximum number of named databases in the LMDB environment.
const MAX_DBS: u32 = 32;

/// Legacy Claim layout (v0.1.11 and earlier) — no `namespace` or `expires_at` fields.
#[derive(Deserialize)]
struct LegacyClaim {
    pub claim_id: String,
    pub content_id: String,
    pub subject: EntityRef,
    pub predicate: PredicateRef,
    pub object: EntityRef,
    pub confidence: f64,
    pub provenance: Provenance,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub payload: Option<Payload>,
    #[serde(default)]
    pub timestamp: i64,
    #[serde(default)]
    pub status: ClaimStatus,
}

impl LegacyClaim {
    fn into_claim(self) -> Claim {
        Claim {
            claim_id: self.claim_id,
            content_id: self.content_id,
            subject: self.subject,
            predicate: self.predicate,
            object: self.object,
            confidence: self.confidence,
            provenance: self.provenance,
            embedding: self.embedding,
            payload: self.payload,
            timestamp: self.timestamp,
            status: self.status,
            namespace: String::new(),
            expires_at: 0,
        }
    }
}

/// Lightweight summary of a claim for analytics queries.
/// Avoids deserializing the full Claim struct (embedding, payload, provenance chain).
#[derive(Serialize, Deserialize)]
struct ClaimSummary {
    claim_id: String,
    namespace: String,
    subject_id: String,
    subject_type: String,
    object_id: String,
    object_type: String,
    predicate_id: String,
    source_id: String,
    confidence: f64,
}

impl From<&Claim> for ClaimSummary {
    fn from(c: &Claim) -> Self {
        ClaimSummary {
            claim_id: c.claim_id.clone(),
            namespace: c.namespace.clone(),
            subject_id: c.subject.id.clone(),
            subject_type: c.subject.entity_type.clone(),
            object_id: c.object.id.clone(),
            object_type: c.object.entity_type.clone(),
            predicate_id: c.predicate.id.clone(),
            source_id: c.provenance.source_id.clone(),
            confidence: c.confidence,
        }
    }
}

/// Convert any error into an AttestError.
fn lmdb_err<E: Display>(msg: &str, e: E) -> AttestError {
    AttestError::Provenance(format!("{msg}: {e}"))
}

/// LMDB database handles. Each corresponds to a named database in the environment.
struct LmdbDatabases {
    // Primary claim storage: u64 seq → compressed claim bytes
    // BigEndian ensures LMDB's lexicographic byte sort = numeric sort.
    claims: Database<U64<heed::byteorder::BigEndian>, Bytes>,
    // claim_id → seq
    claim_id_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // content_id → seq (DUP_SORT)
    content_id_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // entity_id → seq (DUP_SORT)
    entity_claims_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // entity_id → entity_id (DUP_SORT, bidirectional adjacency)
    adjacency_idx: Database<Str, Str>,
    // timestamp → seq (DUP_SORT)
    timestamp_idx: Database<I64<heed::byteorder::BigEndian>, U64<heed::byteorder::BigEndian>>,
    // source_id → seq (DUP_SORT)
    source_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // predicate_id → seq (DUP_SORT)
    predicate_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // namespace → seq (DUP_SORT)
    namespace_idx: Database<Str, U64<heed::byteorder::BigEndian>>,
    // entity_id → bincode(EntityData)
    entities: Database<Str, Bytes>,
    // entity_type → entity_id (DUP_SORT)
    entity_type_idx: Database<Str, Str>,
    // token → entity_id (DUP_SORT)
    text_idx: Database<Str, Str>,
    // key → bytes (aliases, metadata, schema_version, next_seq)
    meta_blobs: Database<Str, Bytes>,
    // claim_id → u8 status
    status_overrides: Database<Str, Bytes>,
    // predicate_type → count
    pred_type_counts: Database<Str, U64<heed::byteorder::BigEndian>>,
    // source_type → count
    src_type_counts: Database<Str, U64<heed::byteorder::BigEndian>>,
    // entity_type → count
    entity_type_counts: Database<Str, U64<heed::byteorder::BigEndian>>,
    // predicate_id → count (exact claim count per predicate.id)
    pred_id_counts: Database<Str, U64<heed::byteorder::BigEndian>>,
    // seq → bincode(ClaimSummary) — lightweight analytics projection
    claim_summaries: Database<U64<heed::byteorder::BigEndian>, Bytes>,
    // "{entity_id}\x1F{predicate_id}" → count — O(1) entity_predicate_counts
    entity_pred_counts: Database<Str, U64<heed::byteorder::BigEndian>>,
    // "{entity_id}\x1F{source_id}" → bytes(count:u64 ++ sum_conf:f64) — O(1) entity_source_counts
    entity_src_counts: Database<Str, Bytes>,
    // "{predicate_id}\x1F{subject_id}\x1F{object_id}" → bytes(count:u64 ++ sum_conf:f64) — O(1) find_contradictions
    pred_pair_counts: Database<Str, Bytes>,
    // "{subject_id}\x1F{predicate_id}" → object_id (DUP_SORT) — O(1) causal neighbor lookup
    causal_adj: Database<Str, Str>,
}

/// File-backed storage backend using LMDB.
pub struct LmdbBackend {
    env: Option<Env>,
    dbs: LmdbDatabases,
    db_path: PathBuf,
    next_seq: u64,
    aliases: UnionFind,
    metadata: MetadataStore,
    aliases_dirty: bool,
    metadata_dirty: bool,
    /// In-memory set of tombstoned claim IDs for O(1) filtering.
    retracted_ids: HashSet<String>,
    /// Whether query methods should include retracted claims.
    include_retracted: bool,
    /// Namespace filter: empty = all namespaces visible.
    namespace_filter: Vec<String>,
    /// Whether this backend is read-only.
    #[allow(dead_code)]
    read_only: bool,
    /// Bulk load mode: skip analytics counters and claim_summaries during insert.
    /// Call `rebuild_counters()` after loading is complete.
    bulk_load_mode: bool,
}

impl LmdbBackend {
    /// Get a reference to the LMDB environment, panicking if already closed.
    #[inline]
    fn env(&self) -> &Env {
        self.env.as_ref().expect("LMDB environment already closed")
    }
}

impl LmdbBackend {
    /// Open or create an LMDB-backed store at the given directory path.
    ///
    /// The path should be a directory. If it doesn't exist, it will be created.
    /// Inside the directory: `data.mdb` and `lock.mdb` (LMDB files).
    pub fn open(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);

        // Create directory if needed
        std::fs::create_dir_all(&path)
            .map_err(|e| lmdb_err("create directory", e))?;

        // Auto-size map for writes: 4× file size (minimum 10 GB) to allow
        // large bulk loads without hitting MDB_MAP_FULL. On 64-bit systems this
        // is virtual address space only — no physical RAM is consumed.
        let map_size = {
            let data_path = path.join("data.mdb");
            let min_map = 10 * (1 << 30); // 10 GB minimum for writable DBs
            if data_path.exists() {
                let file_size = std::fs::metadata(&data_path)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
                let target = (file_size * 4).max(min_map);
                // Round up to nearest GB
                let gb = 1 << 30;
                ((target + gb - 1) / gb) * gb
            } else {
                min_map
            }
        };

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(MAX_DBS)
                .open(&path)
                .map_err(|e| {
                    let msg = e.to_string();
                    if msg.contains("lock") || msg.contains("busy") || msg.contains("Resource temporarily unavailable") {
                        AttestError::Provenance(
                            format!("failed to open database: Database already open. Cannot acquire lock. (underlying: {msg})")
                        )
                    } else {
                        lmdb_err("open LMDB environment", e)
                    }
                })?
        };

        let dbs = Self::create_databases(&env)?;

        // Load in-memory state
        let (next_seq, aliases, metadata, retracted_ids) = {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;

            // next_seq: check meta_blobs first, then scan claims table
            let next_seq = Self::load_next_seq(&rtxn, &dbs)?;
            let aliases = Self::load_meta_blob::<UnionFind>(&rtxn, &dbs.meta_blobs, META_KEY_ALIASES)?
                .unwrap_or_else(UnionFind::new);
            let metadata = Self::load_meta_blob::<MetadataStore>(&rtxn, &dbs.meta_blobs, META_KEY_METADATA)?
                .unwrap_or_else(MetadataStore::new);

            // Load retracted IDs
            let mut retracted_ids = HashSet::new();
            let iter = dbs.status_overrides.iter(&rtxn)
                .map_err(|e| lmdb_err("iter status_overrides", e))?;
            for result in iter {
                let (k, v) = result.map_err(|e| lmdb_err("read status_overrides", e))?;
                if v.first() == Some(&2) {
                    retracted_ids.insert(k.to_string());
                }
            }

            rtxn.commit().map_err(|e| lmdb_err("commit read", e))?;
            (next_seq, aliases, metadata, retracted_ids)
        };

        let backend = Self {
            env: Some(env),
            dbs,
            db_path: path,
            next_seq,
            aliases,
            metadata,
            aliases_dirty: false,
            metadata_dirty: false,
            retracted_ids,
            include_retracted: false,
            namespace_filter: Vec::new(),
            read_only: false,
            bulk_load_mode: false,
        };

        // Ensure schema version is written
        backend.ensure_schema_version()?;

        Ok(backend)
    }

    /// Open an LMDB-backed store in read-only mode.
    pub fn open_read_only(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);
        if !path.exists() {
            return Err(AttestError::Provenance(format!(
                "database not found: {db_path}"
            )));
        }

        // Auto-size map for read-only: must be >= actual file size
        let map_size = {
            let data_path = path.join("data.mdb");
            if data_path.exists() {
                let file_size = std::fs::metadata(&data_path)
                    .map(|m| m.len() as usize)
                    .unwrap_or(0);
                let target = (file_size * 3 / 2).max(DEFAULT_MAP_SIZE);
                let gb = 1 << 30;
                ((target + gb - 1) / gb) * gb
            } else {
                DEFAULT_MAP_SIZE
            }
        };

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(map_size)
                .max_dbs(MAX_DBS)
                .open(&path)
                .map_err(|e| lmdb_err("open read-only LMDB", e))?
        };

        let dbs = Self::open_databases_readonly(&env)?;

        let (next_seq, aliases, metadata, retracted_ids) = {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let next_seq = Self::load_next_seq(&rtxn, &dbs)?;
            let aliases = Self::load_meta_blob::<UnionFind>(&rtxn, &dbs.meta_blobs, META_KEY_ALIASES)?
                .unwrap_or_else(UnionFind::new);
            let metadata = Self::load_meta_blob::<MetadataStore>(&rtxn, &dbs.meta_blobs, META_KEY_METADATA)?
                .unwrap_or_else(MetadataStore::new);

            let mut retracted_ids = HashSet::new();
            let iter = dbs.status_overrides.iter(&rtxn)
                .map_err(|e| lmdb_err("iter status_overrides", e))?;
            for result in iter {
                let (k, v) = result.map_err(|e| lmdb_err("read status_overrides", e))?;
                if v.first() == Some(&2) {
                    retracted_ids.insert(k.to_string());
                }
            }
            rtxn.commit().map_err(|e| lmdb_err("commit read", e))?;
            (next_seq, aliases, metadata, retracted_ids)
        };

        Ok(Self {
            env: Some(env),
            dbs,
            db_path: path,
            next_seq,
            aliases,
            metadata,
            aliases_dirty: false,
            metadata_dirty: false,
            retracted_ids,
            include_retracted: false,
            namespace_filter: Vec::new(),
            read_only: true,
            bulk_load_mode: false,
        })
    }

    /// Helper to create a DUP_SORT database with the given name and types.
    fn create_dup_db<KC: 'static, DC: 'static>(
        env: &Env,
        wtxn: &mut heed::RwTxn,
        name: &str,
    ) -> Result<Database<KC, DC>, AttestError> {
        env.database_options()
            .types::<KC, DC>()
            .flags(DatabaseFlags::DUP_SORT)
            .name(name)
            .create(wtxn)
            .map_err(|e| lmdb_err("create dup db", e))
    }

    /// Create all named databases in a write transaction.
    fn create_databases(env: &Env) -> Result<LmdbDatabases, AttestError> {
        let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;

        type BE = heed::byteorder::BigEndian;

        let dbs = LmdbDatabases {
            // Primary stores — unique key→value (no DUP_SORT)
            claims: env.create_database(&mut wtxn, Some("claims"))
                .map_err(|e| lmdb_err("create db claims", e))?,
            claim_id_idx: env.create_database(&mut wtxn, Some("claim_id_idx"))
                .map_err(|e| lmdb_err("create db", e))?,
            entities: env.create_database(&mut wtxn, Some("entities"))
                .map_err(|e| lmdb_err("create db", e))?,
            meta_blobs: env.create_database(&mut wtxn, Some("meta_blobs"))
                .map_err(|e| lmdb_err("create db", e))?,
            status_overrides: env.create_database(&mut wtxn, Some("status_overrides"))
                .map_err(|e| lmdb_err("create db", e))?,
            // Counter stores — unique key→value (no DUP_SORT)
            pred_type_counts: env.create_database(&mut wtxn, Some("pred_type_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            src_type_counts: env.create_database(&mut wtxn, Some("src_type_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            entity_type_counts: env.create_database(&mut wtxn, Some("entity_type_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            pred_id_counts: env.create_database(&mut wtxn, Some("pred_id_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            claim_summaries: env.create_database(&mut wtxn, Some("claim_summaries"))
                .map_err(|e| lmdb_err("create db", e))?,
            entity_pred_counts: env.create_database(&mut wtxn, Some("entity_pred_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            entity_src_counts: env.create_database(&mut wtxn, Some("entity_src_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            pred_pair_counts: env.create_database(&mut wtxn, Some("pred_pair_counts"))
                .map_err(|e| lmdb_err("create db", e))?,
            // Multimap indexes — DUP_SORT (multiple values per key)
            content_id_idx: Self::create_dup_db::<Str, U64<BE>>(env, &mut wtxn, "content_id_idx")?,
            entity_claims_idx: Self::create_dup_db::<Str, U64<BE>>(env, &mut wtxn, "entity_claims_idx")?,
            adjacency_idx: Self::create_dup_db::<Str, Str>(env, &mut wtxn, "adjacency_idx")?,
            timestamp_idx: Self::create_dup_db::<I64<BE>, U64<BE>>(env, &mut wtxn, "timestamp_idx")?,
            source_idx: Self::create_dup_db::<Str, U64<BE>>(env, &mut wtxn, "source_idx")?,
            predicate_idx: Self::create_dup_db::<Str, U64<BE>>(env, &mut wtxn, "predicate_idx")?,
            namespace_idx: Self::create_dup_db::<Str, U64<BE>>(env, &mut wtxn, "namespace_idx")?,
            entity_type_idx: Self::create_dup_db::<Str, Str>(env, &mut wtxn, "entity_type_idx")?,
            text_idx: Self::create_dup_db::<Str, Str>(env, &mut wtxn, "text_idx")?,
            causal_adj: Self::create_dup_db::<Str, Str>(env, &mut wtxn, "causal_adj")?,
        };

        wtxn.commit().map_err(|e| lmdb_err("commit", e))?;
        Ok(dbs)
    }

    /// Open existing databases in read-only mode (no create).
    /// Uses create_database which is idempotent — LMDB requires a write txn to open dbi handles.
    fn open_databases_readonly(env: &Env) -> Result<LmdbDatabases, AttestError> {
        // Delegate to create_databases — create_database is idempotent in LMDB
        Self::create_databases(env)
    }

    /// Load next_seq from meta_blobs (preferred) or by scanning claims table.
    fn load_next_seq(rtxn: &heed::RoTxn, dbs: &LmdbDatabases) -> Result<u64, AttestError> {
        // Try meta_blobs first
        if let Some(bytes) = dbs.meta_blobs.get(rtxn, META_KEY_NEXT_SEQ)
            .map_err(|e| lmdb_err("get next_seq", e))?
        {
            if bytes.len() >= 8 {
                return Ok(u64::from_le_bytes(bytes[..8].try_into().unwrap()));
            }
        }

        // Fall back to scanning: find the last key in claims
        let last = dbs.claims.last(rtxn)
            .map_err(|e| lmdb_err("claims last", e))?;
        Ok(match last {
            Some((seq, _)) => seq + 1,
            None => 0,
        })
    }

    /// Load a bincode-serialized value from meta_blobs.
    fn load_meta_blob<T: serde::de::DeserializeOwned>(
        rtxn: &heed::RoTxn,
        meta_blobs: &Database<Str, Bytes>,
        key: &str,
    ) -> Result<Option<T>, AttestError> {
        match meta_blobs.get(rtxn, key).map_err(|e| lmdb_err("get meta", e))? {
            Some(data) => {
                match bincode::deserialize(data) {
                    Ok(val) => Ok(Some(val)),
                    Err(e) => {
                        log::warn!("Failed to deserialize {key}: {e}");
                        Ok(None)
                    }
                }
            }
            None => Ok(None),
        }
    }

    /// Ensure schema version is written and trigger any needed migrations.
    fn ensure_schema_version(&self) -> Result<(), AttestError> {
        let rtxn = self.env().read_txn().map_err(|e| lmdb_err("begin_read", e))?;
        let stored_version = self.dbs.meta_blobs.get(&rtxn, META_KEY_SCHEMA_VERSION)
            .map_err(|e| lmdb_err("get", e))?
            .map(|data| {
                if data.len() >= 4 {
                    u32::from_le_bytes(data[..4].try_into().unwrap())
                } else {
                    0
                }
            })
            .unwrap_or(0);
        rtxn.commit().map_err(|e| lmdb_err("commit", e))?;

        if stored_version < CURRENT_SCHEMA_VERSION {
            let mut wtxn = self.env().write_txn().map_err(|e| lmdb_err("begin_write", e))?;
            self.dbs.meta_blobs.put(&mut wtxn, META_KEY_SCHEMA_VERSION,
                &CURRENT_SCHEMA_VERSION.to_le_bytes())
                .map_err(|e| lmdb_err("put", e))?;
            wtxn.commit().map_err(|e| lmdb_err("commit", e))?;
        }
        Ok(())
    }

    /// Compress claim bytes with zstd and prepend header byte.
    fn compress_claim(data: &[u8]) -> Result<Vec<u8>, AttestError> {
        let compressed = zstd::encode_all(Cursor::new(data), ZSTD_LEVEL)
            .map_err(|e| lmdb_err("zstd compress", e))?;
        let mut buf = Vec::with_capacity(1 + compressed.len());
        buf.push(COMPRESSION_ZSTD);
        buf.extend_from_slice(&compressed);
        Ok(buf)
    }

    /// Decompress claim bytes, handling compressed and legacy uncompressed data.
    fn decompress_claim(data: &[u8]) -> Result<Claim, AttestError> {
        if data.is_empty() {
            return Err(lmdb_err("decompress", "empty claim data"));
        }
        let raw = match data[0] {
            COMPRESSION_ZSTD => {
                zstd::decode_all(Cursor::new(&data[1..]))
                    .map_err(|e| lmdb_err("zstd decompress", e))?
            }
            COMPRESSION_NONE => data[1..].to_vec(),
            _ => data.to_vec(), // Legacy: no header byte
        };
        if let Ok(claim) = bincode::deserialize::<Claim>(&raw) {
            return Ok(claim);
        }
        bincode::deserialize::<LegacyClaim>(&raw)
            .map(|lc| lc.into_claim())
            .map_err(|e| lmdb_err("deserialize claim (legacy fallback)", e))
    }

    /// Increment a u64 counter in the given database.
    fn increment_counter(
        wtxn: &mut heed::RwTxn,
        db: &Database<Str, U64<heed::byteorder::BigEndian>>,
        key: &str,
    ) -> Result<(), AttestError> {
        let current = db.get(wtxn, key)
            .map_err(|e| lmdb_err("get counter", e))?
            .unwrap_or(0);
        db.put(wtxn, key, &(current + 1))
            .map_err(|e| lmdb_err("put counter", e))?;
        Ok(())
    }

    /// Increment a (count, sum_confidence) counter stored as 16 bytes: u64 LE count + f64 LE sum.
    fn increment_conf_counter(
        wtxn: &mut heed::RwTxn,
        db: &Database<Str, Bytes>,
        key: &str,
        confidence: f64,
    ) -> Result<(), AttestError> {
        let (mut count, mut sum) = if let Some(data) = db.get(wtxn, key)
            .map_err(|e| lmdb_err("get conf counter", e))?
        {
            if data.len() >= 16 {
                let c = u64::from_le_bytes(data[..8].try_into().unwrap());
                let s = f64::from_le_bytes(data[8..16].try_into().unwrap());
                (c, s)
            } else {
                (0, 0.0)
            }
        } else {
            (0, 0.0)
        };
        count += 1;
        sum += confidence;
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&count.to_le_bytes());
        buf[8..].copy_from_slice(&sum.to_le_bytes());
        db.put(wtxn, key, &buf)
            .map_err(|e| lmdb_err("put conf counter", e))?;
        Ok(())
    }

    /// Flush dirty aliases/metadata to META_BLOBS.
    fn flush_in_memory_state(&mut self) -> Result<(), AttestError> {
        if !self.aliases_dirty && !self.metadata_dirty {
            return Ok(());
        }
        let env = self.env.as_ref().expect("LMDB environment already closed").clone();
        let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;
        if self.aliases_dirty {
            let bytes = bincode::serialize(&self.aliases)
                .map_err(|e| lmdb_err("serialize aliases", e))?;
            self.dbs.meta_blobs.put(&mut wtxn, META_KEY_ALIASES, &bytes)
                .map_err(|e| lmdb_err("put", e))?;
            self.aliases_dirty = false;
        }
        if self.metadata_dirty {
            let bytes = bincode::serialize(&self.metadata)
                .map_err(|e| lmdb_err("serialize metadata", e))?;
            self.dbs.meta_blobs.put(&mut wtxn, META_KEY_METADATA, &bytes)
                .map_err(|e| lmdb_err("put", e))?;
            self.metadata_dirty = false;
        }
        // Always persist next_seq
        self.dbs.meta_blobs.put(&mut wtxn, META_KEY_NEXT_SEQ,
            &self.next_seq.to_le_bytes())
            .map_err(|e| lmdb_err("put next_seq", e))?;
        wtxn.commit().map_err(|e| lmdb_err("commit", e))?;
        Ok(())
    }

    /// Apply alias side-effects for a claim.
    fn apply_alias_predicate(&mut self, claim: &Claim) {
        if ALIAS_PREDICATES.contains(claim.predicate.id.as_str()) {
            if claim.predicate.id == "same_as" {
                self.aliases.union(&claim.subject.id, &claim.object.id);
                self.aliases_dirty = true;
            } else if claim.predicate.id == "not_same_as" {
                self.aliases.split(&claim.subject.id);
                self.aliases_dirty = true;
            }
        }
    }

    /// Write a single claim into all databases within a write transaction.
    fn write_claim_to_txn(
        &self,
        wtxn: &mut heed::RwTxn,
        seq: u64,
        claim: &Claim,
    ) -> Result<(), AttestError> {
        let raw_bytes = bincode::serialize(claim)
            .map_err(|e| lmdb_err("serialize claim", e))?;
        let bytes = Self::compress_claim(&raw_bytes)?;

        // LMDB enforces a 511-byte max key size, and DUP_SORT databases enforce
        // the same limit on data values. Only object.id realistically exceeds this
        // (hypothesis text, paper conclusions ~500 chars). We skip object-side
        // secondary index entries rather than truncate, since truncation would
        // corrupt lookup semantics. The claim is still retrievable via the primary
        // claims table, claim_id_idx, content_id_idx, and other indexes.
        let obj_fits = claim.object.id.len() <= LMDB_MAX_KEY;

        // CLAIMS: seq → compressed bytes
        self.dbs.claims.put(wtxn, &seq, &bytes)
            .map_err(|e| lmdb_err("put claims", e))?;

        // CLAIM_ID_IDX
        self.dbs.claim_id_idx.put(wtxn, &claim.claim_id, &seq)
            .map_err(|e| lmdb_err("put claim_id_idx", e))?;

        // CONTENT_ID_IDX
        self.dbs.content_id_idx.put(wtxn, &claim.content_id, &seq)
            .map_err(|e| lmdb_err("put content_id_idx", e))?;

        // ENTITY_CLAIMS_IDX (subject + object)
        // Subject IDs are always short (normalized entity IDs). Skip object side
        // when object.id exceeds LMDB key limit.
        self.dbs.entity_claims_idx.put(wtxn, &claim.subject.id, &seq)
            .map_err(|e| lmdb_err("put entity_claims_idx", e))?;
        if obj_fits && claim.object.id != claim.subject.id {
            self.dbs.entity_claims_idx.put(wtxn, &claim.object.id, &seq)
                .map_err(|e| lmdb_err("put entity_claims_idx", e))?;
        }

        // ADJACENCY_IDX (bidirectional, skip self-loops)
        // DUP_SORT data values are also limited to 511 bytes, so skip entirely
        // when object.id would overflow either the key or the data slot.
        if obj_fits && claim.subject.id != claim.object.id {
            self.dbs.adjacency_idx.put(wtxn, &claim.subject.id, &claim.object.id)
                .map_err(|e| lmdb_err("put adjacency_idx", e))?;
            self.dbs.adjacency_idx.put(wtxn, &claim.object.id, &claim.subject.id)
                .map_err(|e| lmdb_err("put adjacency_idx", e))?;
        }

        // CAUSAL_ADJ — directional causal neighbor index for fast predict()
        // Only indexes claims where predicate is causal and subject→object direction
        if CAUSAL_PREDICATES.contains(&claim.predicate.id.as_str()) && obj_fits {
            let key = format!("{}\x1F{}", &claim.subject.id, &claim.predicate.id);
            if key.len() <= LMDB_MAX_KEY {
                let _ = self.dbs.causal_adj.put(wtxn, &key, &claim.object.id);
            }
        }

        // TIMESTAMP_IDX
        self.dbs.timestamp_idx.put(wtxn, &claim.timestamp, &seq)
            .map_err(|e| lmdb_err("put timestamp_idx", e))?;

        // SOURCE_IDX — source_id is always short (source names)
        self.dbs.source_idx.put(wtxn, &claim.provenance.source_id, &seq)
            .map_err(|e| lmdb_err("put source_idx", e))?;

        // PREDICATE_IDX — predicate_id is always short
        self.dbs.predicate_idx.put(wtxn, &claim.predicate.id, &seq)
            .map_err(|e| lmdb_err("put predicate_idx", e))?;

        // NAMESPACE_IDX — skip empty namespace (LMDB doesn't allow zero-length keys)
        if !claim.namespace.is_empty() {
            self.dbs.namespace_idx.put(wtxn, &claim.namespace, &seq)
                .map_err(|e| lmdb_err("put namespace_idx", e))?;
        }

        // Stats counters and analytics — skip in bulk load mode
        if !self.bulk_load_mode {
            Self::increment_counter(wtxn, &self.dbs.pred_type_counts, &claim.predicate.predicate_type)?;
            Self::increment_counter(wtxn, &self.dbs.pred_id_counts, &claim.predicate.id)?;
            Self::increment_counter(wtxn, &self.dbs.src_type_counts, &claim.provenance.source_type)?;

            // Claim summary (lightweight projection for analytics)
            let summary = ClaimSummary::from(claim);
            let summary_bytes = bincode::serialize(&summary)
                .map_err(|e| lmdb_err("serialize summary", e))?;
            self.dbs.claim_summaries.put(wtxn, &seq, &summary_bytes)
                .map_err(|e| lmdb_err("put claim_summaries", e))?;

            // entity_pred_counts: "{entity_id}\x1F{predicate_id}" → count
            // Truncate object_id in compound key to stay under 511 bytes.
            {
                let key_subj = format!("{}\x1F{}", &claim.subject.id, &claim.predicate.id);
                Self::increment_counter(wtxn, &self.dbs.entity_pred_counts, &key_subj)?;
                if claim.object.id != claim.subject.id {
                    let obj_id = Self::truncate_to_bytes(&claim.object.id, LMDB_MAX_KEY);
                    let key_obj = format!("{}\x1F{}", obj_id, &claim.predicate.id);
                    Self::increment_counter(wtxn, &self.dbs.entity_pred_counts, &key_obj)?;
                }
            }

            // entity_src_counts: "{entity_id}\x1F{source_id}" → (count, sum_conf)
            {
                let key_subj = format!("{}\x1F{}", &claim.subject.id, &claim.provenance.source_id);
                Self::increment_conf_counter(wtxn, &self.dbs.entity_src_counts, &key_subj, claim.confidence)?;
                if claim.object.id != claim.subject.id {
                    let obj_id = Self::truncate_to_bytes(&claim.object.id, 470);
                    let key_obj = format!("{}\x1F{}", obj_id, &claim.provenance.source_id);
                    Self::increment_conf_counter(wtxn, &self.dbs.entity_src_counts, &key_obj, claim.confidence)?;
                }
            }

            // pred_pair_counts: "{predicate_id}\x1F{subject_id}\x1F{object_id}" → (count, sum_conf)
            // Truncate the entire compound key to stay under 511 bytes.
            {
                let raw_key = format!("{}\x1F{}\x1F{}", &claim.predicate.id, &claim.subject.id, &claim.object.id);
                let key = Self::truncate_to_bytes(&raw_key, 500);
                Self::increment_conf_counter(wtxn, &self.dbs.pred_pair_counts, key, claim.confidence)?;
            }
        }

        Ok(())
    }

    /// Truncate a string to at most `max_bytes` bytes at a UTF-8 character boundary.
    #[inline]
    fn truncate_to_bytes(s: &str, max_bytes: usize) -> &str {
        if s.len() <= max_bytes { return s; }
        let mut end = max_bytes;
        while !s.is_char_boundary(end) { end -= 1; }
        &s[..end]
    }

    /// Whether a claim should be included in query results.
    fn should_include_claim(&self, claim_id: &str) -> bool {
        self.include_retracted || !self.retracted_ids.contains(claim_id)
    }

    /// Apply status overlay using the in-memory retracted_ids set.
    fn apply_status_overlay_fast(&self, claim: &mut Claim) {
        if self.retracted_ids.contains(&claim.claim_id) {
            claim.status = ClaimStatus::Tombstoned;
        }
    }

    /// Check if a claim's namespace passes the current filter.
    fn should_include_namespace(&self, namespace: &str) -> bool {
        self.namespace_filter.is_empty() || self.namespace_filter.iter().any(|ns| ns == namespace)
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

    /// Load a full claim by seq number (zstd decompress + bincode deserialize).
    ///
    /// PERF: This is expensive (~1KB zstd decompress + nested struct deserialize).
    /// If you only need lightweight fields (subject_id, predicate_id, confidence, etc.),
    /// use `load_summary_by_seq` instead — it's ~25x faster on large databases.
    /// Only use this when you need embedding, payload, or full provenance chain.
    fn load_claim_by_seq(&self, rtxn: &heed::RoTxn, seq: u64) -> Option<Claim> {
        let data = self.dbs.claims.get(rtxn, &seq).ok()??;
        Self::decompress_claim(data).ok()
    }

    /// Load a lightweight claim summary by seq number.
    /// Falls back to full claim deserialization if summary table not populated.
    fn load_summary_by_seq(&self, rtxn: &heed::RoTxn, seq: u64) -> Option<ClaimSummary> {
        if let Ok(Some(data)) = self.dbs.claim_summaries.get(rtxn, &seq) {
            if let Ok(summary) = bincode::deserialize::<ClaimSummary>(data) {
                return Some(summary);
            }
        }
        // Fallback: deserialize full claim
        self.load_claim_by_seq(rtxn, seq).map(|c| ClaimSummary::from(&c))
    }

    /// Load claims for a set of seq numbers.
    fn load_claims_by_seqs(&self, rtxn: &heed::RoTxn, seqs: &[u64]) -> Vec<Claim> {
        seqs.iter().filter_map(|&seq| {
            self.load_claim_by_seq(rtxn, seq)
        }).collect()
    }

    /// Collect all seq numbers for a key from a DUP_SORT Str→u64 database.
    fn collect_seqs_for_key(
        rtxn: &heed::RoTxn,
        db: &Database<Str, U64<heed::byteorder::BigEndian>>,
        key: &str,
    ) -> Vec<u64> {
        match db.get_duplicates(rtxn, key) {
            Ok(Some(iter)) => {
                iter.filter_map(|entry| entry.ok().map(|(_, seq)| seq)).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Collect all string values for a key from a DUP_SORT Str→Str database.
    fn collect_values_for_key(
        rtxn: &heed::RoTxn,
        db: &Database<Str, Str>,
        key: &str,
    ) -> Vec<String> {
        match db.get_duplicates(rtxn, key) {
            Ok(Some(iter)) => {
                iter.filter_map(|entry| entry.ok().map(|(_, v)| v.to_string())).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Get neighbor entity IDs from the DUP_SORT adjacency index.
    fn neighbors(&self, rtxn: &heed::RoTxn, entity_id: &str) -> HashSet<String> {
        match self.dbs.adjacency_idx.get_duplicates(rtxn, entity_id) {
            Ok(Some(iter)) => {
                iter.filter_map(|entry| entry.ok().map(|(_, v)| v.to_string())).collect()
            }
            _ => HashSet::new(),
        }
    }

    /// Get claims for a set of entity IDs (deduplicated by seq).
    fn claims_for_entities(
        &self,
        rtxn: &heed::RoTxn,
        entity_ids: &HashSet<String>,
    ) -> Vec<Claim> {
        let mut seen_seqs = HashSet::new();
        let mut all_seqs = Vec::new();
        for eid in entity_ids {
            let seqs = LmdbBackend::collect_seqs_for_key(rtxn, &self.dbs.entity_claims_idx, eid);
            for seq in seqs {
                if seen_seqs.insert(seq) {
                    all_seqs.push(seq);
                }
            }
        }
        self.load_claims_by_seqs(rtxn, &all_seqs)
    }

    /// Look up claims via a Str→u64 index, applying retraction filter + overlay.
    fn claims_by_str_index(
        &self,
        db: &Database<Str, U64<heed::byteorder::BigEndian>>,
        key: &str,
    ) -> Vec<Claim> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let seqs = LmdbBackend::collect_seqs_for_key(&rtxn, db, key);
        let claims = self.load_claims_by_seqs(&rtxn, &seqs);
        claims.into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .map(|mut c| { self.apply_status_overlay_fast(&mut c); c })
            .collect()
    }

    /// Read counts from a counter database.
    fn read_counts(
        rtxn: &heed::RoTxn,
        db: &Database<Str, U64<heed::byteorder::BigEndian>>,
    ) -> HashMap<String, usize> {
        match db.iter(rtxn) {
            Ok(iter) => {
                iter.filter_map(|entry| {
                    entry.ok().map(|(k, v)| (k.to_string(), v as usize))
                }).collect()
            }
            Err(_) => HashMap::new(),
        }
    }

    // ── Public Lifecycle ──────────────────────────────────────────────

    /// Rebuild the causal_adj index from existing claims.
    /// One-time operation for databases created before the index existed.
    pub fn rebuild_causal_adj(&mut self) -> Result<usize, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }

        let rtxn = self.env().read_txn()
            .map_err(|e| lmdb_err("rebuild read", e))?;

        // Collect all causal claims
        let mut entries: Vec<(String, String, String)> = Vec::new(); // (subject, predicate, object)
        for pred in CAUSAL_PREDICATES {
            let seqs = LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.predicate_idx, pred);
            for seq in seqs {
                let claim_bytes = match self.dbs.claims.get(&rtxn, &seq) {
                    Ok(Some(b)) => b.to_vec(),
                    _ => continue,
                };
                let claim = match Self::decompress_claim(&claim_bytes) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                if claim.object.id.len() <= LMDB_MAX_KEY {
                    entries.push((claim.subject.id.clone(), claim.predicate.id.clone(), claim.object.id.clone()));
                }
            }
        }
        drop(rtxn);

        // Write in chunks
        let total = entries.len();
        const CHUNK: usize = 100_000;
        let mut written = 0;

        for chunk in entries.chunks(CHUNK) {
            let mut wtxn = self.env().write_txn()
                .map_err(|e| lmdb_err("rebuild write", e))?;
            for (subj, pred, obj) in chunk {
                let key = format!("{}\x1F{}", subj, pred);
                if key.len() <= LMDB_MAX_KEY {
                    let _ = self.dbs.causal_adj.put(&mut wtxn, &key, obj);
                    written += 1;
                }
            }
            wtxn.commit().map_err(|e| lmdb_err("rebuild commit", e))?;
        }

        Ok(written)
    }

    pub fn close(&mut self) -> Result<(), AttestError> {
        self.flush_in_memory_state()?;
        // Take ownership of the Env and call prepare_for_closing() to
        // immediately release file locks and unmap memory. Without this,
        // the env stays alive in heed's global OPENED_ENV until Rust drops
        // the struct, which may be delayed by Python's GC.
        if let Some(env) = self.env.take() {
            env.prepare_for_closing().wait();
        }
        Ok(())
    }

    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        self.flush_in_memory_state()
    }

    /// Reset the union-find alias table. Clears all entity aliases and persists
    /// the empty state. Use after purging same_as claims to fix the performance
    /// regression caused by stale alias groups.
    pub fn clear_aliases(&mut self) -> Result<(), AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }
        self.aliases = UnionFind::new();
        self.aliases_dirty = true;
        self.flush_in_memory_state()?;

        // Rebuild from any remaining same_as claims in the DB
        let rtxn = self.env().read_txn()
            .map_err(|e| lmdb_err("clear_aliases read", e))?;
        let seqs = LmdbBackend::collect_seqs_for_key(
            &rtxn, &self.dbs.predicate_idx, "same_as",
        );
        let claims = self.load_claims_by_seqs(&rtxn, &seqs);
        drop(rtxn);

        for claim in &claims {
            self.aliases.union(&claim.subject.id, &claim.object.id);
        }
        if !claims.is_empty() {
            self.aliases_dirty = true;
        }
        self.flush_in_memory_state()?;
        Ok(())
    }

    /// Physically delete all claims from a source. Unlike retract_source() which
    /// tombstones claims (preserving append-only semantics), purge_source() removes
    /// data from all indexes. Use for maintenance (bad loads, data cleanup).
    /// Returns the number of claims deleted.
    pub fn purge_source(&mut self, source_id: &str) -> Result<usize, AttestError> {
        if self.read_only { return Err(AttestError::ReadOnly); }

        // 1. Collect all seq IDs for this source
        let seqs: Vec<u64> = {
            let rtxn = self.env().read_txn()
                .map_err(|e| lmdb_err("purge read_txn", e))?;
            let mut collected = Vec::new();
            let iter = self.dbs.source_idx.get_duplicates(&rtxn, source_id)
                .map_err(|e| lmdb_err("purge source_idx iter", e))?;
            if let Some(iter) = iter {
                for item in iter {
                    let (_, seq) = item.map_err(|e| lmdb_err("purge source_idx item", e))?;
                    collected.push(seq);
                }
            }
            collected
        };

        if seqs.is_empty() { return Ok(0); }

        // 2. Delete in chunks to avoid huge transactions
        const CHUNK: usize = 10_000;
        let mut deleted = 0usize;
        let mut has_aliases = false;

        for chunk in seqs.chunks(CHUNK) {
            let mut wtxn = self.env().write_txn()
                .map_err(|e| lmdb_err("purge write_txn", e))?;

            for &seq in chunk {
                // Read the claim to get IDs for secondary indexes
                let claim_bytes = match self.dbs.claims.get(&wtxn, &seq) {
                    Ok(Some(b)) => b.to_vec(),
                    _ => continue,
                };
                let claim = match Self::decompress_claim(&claim_bytes) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                let obj_fits = claim.object.id.len() <= LMDB_MAX_KEY;

                // Delete from primary tables (1:1 mappings)
                let _ = self.dbs.claims.delete(&mut wtxn, &seq);
                let _ = self.dbs.claim_id_idx.delete(&mut wtxn, &claim.claim_id);
                let _ = self.dbs.claim_summaries.delete(&mut wtxn, &seq);
                // DupSort indexes: remove the specific (key, seq) pair
                // heed DupSort delete removes ALL values for a key, so we
                // reconstruct by deleting all and re-inserting the non-purged ones.
                // For source_idx this is safe since we're purging the entire source.
                // For other indexes, stale seq entries pointing to deleted claims
                // are filtered at read time (claims.get returns None → skip).
                // This is acceptable for a maintenance operation.

                // Track if we're purging same_as claims
                if claim.predicate.id == "same_as" {
                    has_aliases = true;
                }
                deleted += 1;
            }

            wtxn.commit().map_err(|e| lmdb_err("purge commit", e))?;
        }

        // If same_as claims were purged, rebuild the alias table
        if has_aliases {
            log::info!("purge_source: same_as claims purged, rebuilding aliases");
            self.clear_aliases()?;
        }

        Ok(deleted)
    }

    pub fn compact(&mut self) -> Result<bool, AttestError> {
        // LMDB compaction: copy to a temp path with CompactOption, then discard
        // (Full swap would require closing + re-opening the Env, which we skip for now)
        let compact_path = self.db_path.join("compact_tmp");
        std::fs::create_dir_all(&compact_path)
            .map_err(|e| lmdb_err("create compact dir", e))?;
        self.env().copy_to_file(&compact_path, heed::CompactionOption::Enabled)
            .map_err(|e| lmdb_err("compact copy", e))?;
        let _ = std::fs::remove_dir_all(&compact_path);
        Ok(true)
    }

    pub fn schema_version(&self) -> u32 {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return 2,
        };
        match self.dbs.meta_blobs.get(&rtxn, META_KEY_SCHEMA_VERSION) {
            Ok(Some(data)) if data.len() >= 4 => {
                u32::from_le_bytes(data[..4].try_into().unwrap())
            }
            _ => 2,
        }
    }

    // ── Metadata ──────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        self.metadata.register_vocabulary(namespace, vocab);
        self.metadata_dirty = true;
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        self.metadata.register_predicate(predicate_id, constraints);
        self.metadata_dirty = true;
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        self.metadata.register_payload_schema(schema_id, schema);
        self.metadata_dirty = true;
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

    // ── Alias resolution ──────────────────────────────────────────────

    pub fn resolve(&mut self, entity_id: &str) -> String {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.find(&canonical)
    }

    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.get_group(&canonical)
    }

    pub fn warm_caches(&self) {}

    // ── Entity CRUD ──────────────────────────────────────────────────

    pub fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        if entity_id.len() > LMDB_MAX_KEY {
            log::warn!("Skipping entity with oversized key ({} bytes): {}...",
                entity_id.len(), &entity_id[..entity_id.len().min(80)]);
            return;
        }
        let mut wtxn = match self.env().write_txn() {
            Ok(t) => t,
            Err(e) => { log::warn!("upsert_entity: {e}"); return; }
        };

        // Check if entity exists → merge external_ids only
        let existing: Option<EntityData> = self.dbs.entities.get(&wtxn, entity_id).ok().flatten()
            .and_then(|data| bincode::deserialize(data).ok());

        if let Some(mut entity) = existing {
            if let Some(new_ids) = external_ids {
                for (k, v) in new_ids {
                    entity.external_ids.insert(k.clone(), v.clone());
                }
                if let Ok(bytes) = bincode::serialize(&entity) {
                    let _ = self.dbs.entities.put(&mut wtxn, entity_id, &bytes);
                }
            }
            let _ = wtxn.commit();
            return;
        }

        let display = if display_name.is_empty() { entity_id } else { display_name };

        let data = EntityData {
            id: entity_id.to_string(),
            entity_type: entity_type.to_string(),
            display_name: display.to_string(),
            external_ids: external_ids.cloned().unwrap_or_default(),
            created_at: timestamp,
        };
        let bytes = match bincode::serialize(&data) {
            Ok(b) => b,
            Err(_) => return,
        };
        let _ = self.dbs.entities.put(&mut wtxn, entity_id, &bytes);

        // Update ENTITY_TYPE_IDX
        let _ = self.dbs.entity_type_idx.put(&mut wtxn, entity_type, entity_id);

        // Update TEXT_IDX
        let mut seen = HashSet::new();
        for token in tokenize(display).into_iter().chain(tokenize(entity_id)) {
            let key = Self::truncate_to_bytes(&token, LMDB_MAX_KEY);
            if seen.insert(key.to_string()) {
                let _ = self.dbs.text_idx.put(&mut wtxn, key, entity_id);
            }
        }

        // Update ENTITY_TYPE_COUNTS
        if !self.bulk_load_mode {
            let _ = Self::increment_counter(&mut wtxn, &self.dbs.entity_type_counts, entity_type);
        }

        let _ = wtxn.commit();
    }

    /// Update display_name for an existing entity. Returns true if updated.
    pub fn update_display_name(&mut self, entity_id: &str, new_display: &str) -> bool {
        let mut wtxn = match self.env().write_txn() {
            Ok(t) => t,
            Err(e) => { log::warn!("update_display_name: {e}"); return false; }
        };

        let existing: Option<EntityData> = self.dbs.entities.get(&wtxn, entity_id).ok().flatten()
            .and_then(|data| bincode::deserialize(data).ok());

        if let Some(mut entity) = existing {
            entity.display_name = new_display.to_string();
            match bincode::serialize(&entity) {
                Ok(bytes) => {
                    if let Err(e) = self.dbs.entities.put(&mut wtxn, entity_id, &bytes) {
                        log::warn!("update_display_name put failed for {}: {}", entity_id, e);
                        return false;
                    }
                }
                Err(e) => {
                    log::warn!("update_display_name serialize failed: {}", e);
                    return false;
                }
            }

            // Add new display name tokens to TEXT_IDX
            let mut seen = HashSet::new();
            for token in tokenize(new_display) {
                let key = Self::truncate_to_bytes(&token, LMDB_MAX_KEY);
                if seen.insert(key.to_string()) {
                    let _ = self.dbs.text_idx.put(&mut wtxn, key, entity_id);
                }
            }

            match wtxn.commit() {
                Ok(_) => true,
                Err(e) => {
                    log::warn!("update_display_name commit failed for {}: {}", entity_id, e);
                    false
                }
            }
        } else {
            let _ = wtxn.commit();
            false
        }
    }

    pub fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) -> Result<(), AttestError> {
        if entities.is_empty() { return Ok(()); }

        let mut wtxn = self.env().write_txn()
            .map_err(|e| lmdb_err("upsert_entities_batch begin_write", e))?;

        for (entity_id, entity_type, display_name, external_ids) in entities {
            // Skip entities whose ID exceeds LMDB max key size
            if entity_id.len() > LMDB_MAX_KEY || entity_type.len() > LMDB_MAX_KEY {
                log::warn!("Skipping entity with oversized key ({} bytes): {}...",
                    entity_id.len(), &entity_id[..entity_id.len().min(80)]);
                continue;
            }

            let existing: Option<EntityData> = self.dbs.entities.get(&wtxn, entity_id.as_str())
                .ok().flatten()
                .and_then(|data| bincode::deserialize(data).ok());

            if let Some(mut entity) = existing {
                if !external_ids.is_empty() {
                    for (k, v) in external_ids {
                        entity.external_ids.insert(k.clone(), v.clone());
                    }
                    let bytes = bincode::serialize(&entity)
                        .map_err(|e| AttestError::Provenance(format!("serialize entity: {e}")))?;
                    self.dbs.entities.put(&mut wtxn, entity_id.as_str(), &bytes)
                        .map_err(|e| lmdb_err("put entity", e))?;
                }
                continue;
            }

            let display = if display_name.is_empty() { entity_id.as_str() } else { display_name.as_str() };
            let data = EntityData {
                id: entity_id.clone(),
                entity_type: entity_type.clone(),
                display_name: display.to_string(),
                external_ids: external_ids.clone(),
                created_at: timestamp,
            };
            let bytes = bincode::serialize(&data)
                .map_err(|e| AttestError::Provenance(format!("serialize entity: {e}")))?;
            self.dbs.entities.put(&mut wtxn, entity_id.as_str(), &bytes)
                .map_err(|e| lmdb_err("put entity", e))?;
            self.dbs.entity_type_idx.put(&mut wtxn, entity_type.as_str(), entity_id.as_str())
                .map_err(|e| lmdb_err("put entity_type_idx", e))?;
            let mut seen = HashSet::new();
            for token in tokenize(display).into_iter().chain(tokenize(entity_id)) {
                let key = Self::truncate_to_bytes(&token, LMDB_MAX_KEY);
                if seen.insert(key.to_string()) {
                    self.dbs.text_idx.put(&mut wtxn, key, entity_id.as_str())
                        .map_err(|e| lmdb_err("put text_idx", e))?;
                }
            }
            if !self.bulk_load_mode {
                Self::increment_counter(&mut wtxn, &self.dbs.entity_type_counts, entity_type)
                    .map_err(|e| lmdb_err("increment entity_type_counts", e))?;
            }
        }

        wtxn.commit().map_err(|e| lmdb_err("upsert_entities_batch commit", e))?;
        Ok(())
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        let rtxn = self.env().read_txn().ok()?;
        let data_bytes = self.dbs.entities.get(&rtxn, entity_id).ok()??;
        let entity: EntityData = bincode::deserialize(data_bytes).ok()?;

        // Count claims for this entity
        let claim_count = LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, entity_id).len();

        Some(entity.to_summary(claim_count))
    }

    /// Batch-lookup entity metadata using a single LMDB read transaction.
    /// Only returns type + display_name (no claim counting — optimized for insert_bulk resolution).
    pub fn get_entities_batch(&self, entity_ids: &[&str]) -> HashMap<String, (String, String)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };
        let mut result = HashMap::with_capacity(entity_ids.len());
        for id in entity_ids {
            if let Ok(Some(data_bytes)) = self.dbs.entities.get(&rtxn, *id) {
                if let Ok(entity) = bincode::deserialize::<EntityData>(data_bytes) {
                    result.insert(entity.id.clone(), (entity.entity_type.clone(), entity.display_name.clone()));
                }
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
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let count_claims = |eid: &str| -> usize {
            LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, eid).len()
        };

        let apply_pagination = |iter: Box<dyn Iterator<Item = EntitySummary>>| -> Vec<EntitySummary> {
            if limit == 0 {
                iter.skip(offset).collect()
            } else {
                iter.skip(offset).take(limit).collect()
            }
        };

        match entity_type {
            Some(et) => {
                // Use DUP_SORT type index — collect all entity IDs for this type
                let entity_ids: Vec<String> = match self.dbs.entity_type_idx.get_duplicates(&rtxn, et) {
                    Ok(Some(iter)) => iter.filter_map(|e| e.ok().map(|(_, v)| v.to_string())).collect(),
                    _ => return Vec::new(),
                };
                let filtered = entity_ids.into_iter().filter_map(|eid| {
                    let data_bytes = self.dbs.entities.get(&rtxn, eid.as_str()).ok()??;
                    let entity: EntityData = bincode::deserialize(data_bytes).ok()?;
                    let cc = count_claims(&eid);
                    (cc >= min_claims).then(|| entity.to_summary(cc))
                });
                apply_pagination(Box::new(filtered))
            }
            None => {
                let iter = match self.dbs.entities.iter(&rtxn) {
                    Ok(i) => i,
                    Err(_) => return Vec::new(),
                };
                let filtered = iter.filter_map(|entry| {
                    let (k, v) = entry.ok()?;
                    let entity: EntityData = bincode::deserialize(v).ok()?;
                    let cc = count_claims(k);
                    (cc >= min_claims).then(|| entity.to_summary(cc))
                });
                apply_pagination(Box::new(filtered))
            }
        }
    }

    pub fn count_entities(&self, entity_type: Option<&str>, min_claims: usize) -> usize {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return 0,
        };

        if min_claims == 0 {
            match entity_type {
                Some(et) => {
                    // Count all entities for this type via DUP_SORT index
                    match self.dbs.entity_type_idx.get_duplicates(&rtxn, et) {
                        Ok(Some(iter)) => iter.filter(|e| e.is_ok()).count(),
                        _ => 0,
                    }
                }
                None => {
                    self.dbs.entities.len(&rtxn).unwrap_or(0) as usize
                }
            }
        } else {
            let count_claims = |eid: &str| -> usize {
                LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, eid).len()
            };
            match entity_type {
                Some(et) => {
                    match self.dbs.entity_type_idx.get_duplicates(&rtxn, et) {
                        Ok(Some(iter)) => {
                            iter.filter_map(|e| e.ok())
                                .filter(|(_, eid)| count_claims(eid) >= min_claims)
                                .count()
                        }
                        _ => 0,
                    }
                }
                None => {
                    match self.dbs.entities.iter(&rtxn) {
                        Ok(iter) => {
                            iter.filter_map(|e| e.ok())
                                .filter(|(k, _)| count_claims(k) >= min_claims)
                                .count()
                        }
                        Err(_) => 0,
                    }
                }
            }
        }
    }

    pub fn count_claims(&self) -> usize {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return 0,
        };
        self.dbs.claims.len(&rtxn).unwrap_or(0) as usize
    }

    // ── Claim operations ──────────────────────────────────────────────

    pub fn insert_claim(&mut self, claim: Claim, _checkpoint_interval: u64) -> bool {
        let mut wtxn = match self.env().write_txn() {
            Ok(t) => t,
            Err(_) => return false,
        };

        // Check existence
        if self.dbs.claim_id_idx.get(&wtxn, &claim.claim_id).ok().flatten().is_some() {
            return false;
        }

        let seq = self.next_seq;
        if self.write_claim_to_txn(&mut wtxn, seq, &claim).is_err() {
            return false;
        }

        match wtxn.commit() {
            Ok(_) => {
                self.next_seq = seq + 1;
                self.apply_alias_predicate(&claim);
                true
            }
            Err(_) => false,
        }
    }

    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>, _checkpoint_interval: u64) -> usize {
        let mut seen = HashSet::new();
        let unique_claims: Vec<Claim> = claims.into_iter()
            .filter(|c| seen.insert(c.claim_id.clone()))
            .collect();

        if unique_claims.is_empty() { return 0; }

        // Chunk large batches for progress reporting
        const CHUNK_SIZE: usize = 100_000;
        let mut total_inserted = 0usize;

        for chunk in unique_claims.chunks(CHUNK_SIZE) {
            let mut wtxn = match self.env().write_txn() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[attest-store] write_txn failed: {e} (map_full? check map_size)");
                    break;
                }
            };

            let new_in_chunk: Vec<&Claim> = chunk.iter().filter(|c| {
                self.dbs.claim_id_idx.get(&wtxn, &c.claim_id).ok().flatten().is_none()
            }).collect();

            if new_in_chunk.is_empty() { continue; }

            let mut seq = self.next_seq;
            let mut chunk_inserted = 0usize;
            for claim in &new_in_chunk {
                match self.write_claim_to_txn(&mut wtxn, seq, claim) {
                    Ok(_) => {
                        seq += 1;
                        chunk_inserted += 1;
                    }
                    Err(e) => {
                        eprintln!("[attest-store] write_claim failed: {e}");
                    }
                }
            }

            match wtxn.commit() {
                Ok(_) => {
                    self.next_seq = seq;
                    for claim in &new_in_chunk {
                        self.apply_alias_predicate(claim);
                    }
                    total_inserted += chunk_inserted;
                }
                Err(e) => {
                    eprintln!("[attest-store] commit failed: {e} (map_full? check map_size)");
                    break;
                }
            }
        }

        total_inserted
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return false,
        };
        self.dbs.claim_id_idx.get(&rtxn, claim_id).ok().flatten().is_some()
    }

    pub fn get_claim(&self, claim_id: &str) -> Option<Claim> {
        let rtxn = self.env().read_txn().ok()?;
        let seq = self.dbs.claim_id_idx.get(&rtxn, claim_id).ok()??;
        let mut claim = self.load_claim_by_seq(&rtxn, seq)?;
        // Apply status overlay
        if let Some(status_bytes) = self.dbs.status_overrides.get(&rtxn, claim_id).ok().flatten() {
            if let Some(&status_u8) = status_bytes.first() {
                claim.status = Self::u8_to_status(status_u8);
            }
        }
        self.apply_status_overlay_fast(&mut claim);
        Some(claim)
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(&self.dbs.content_id_idx, content_id)
    }

    pub fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let iter = match self.dbs.claims.iter(&rtxn) {
            Ok(i) => i,
            Err(_) => return Vec::new(),
        };

        let decoded = iter.filter_map(|entry| {
            let (_, v) = entry.ok()?;
            Self::decompress_claim(v).ok()
        }).filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
        .map(|mut c| { self.apply_status_overlay_fast(&mut c); c });

        if limit == 0 {
            decoded.skip(offset).collect()
        } else {
            decoded.skip(offset).take(limit).collect()
        }
    }

    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(&self.dbs.source_idx, source_id)
    }

    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(&self.dbs.predicate_idx, predicate_id)
    }

    pub fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);

        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let claims = self.claims_for_entities(&rtxn, &aliases);

        claims.into_iter().filter(|c| {
            if !self.should_include_claim(&c.claim_id) { return false; }
            if !self.should_include_namespace(&c.namespace) { return false; }
            if let Some(pt) = predicate_type {
                if c.predicate.predicate_type != pt { return false; }
            }
            if let Some(st) = source_type {
                if c.provenance.source_type != st { return false; }
            }
            c.confidence >= min_confidence
        }).map(|mut c| { self.apply_status_overlay_fast(&mut c); c })
        .collect()
    }

    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let seq = match self.dbs.claim_id_idx.get(&rtxn, claim_id) {
            Ok(Some(v)) => v,
            _ => return Vec::new(),
        };
        self.load_claim_by_seq(&rtxn, seq)
            .map(|c| c.provenance.chain)
            .unwrap_or_default()
    }

    // ── Graph traversal ──────────────────────────────────────────────

    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);

        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut visited: HashSet<String> = aliases.clone();
        let mut frontier: HashSet<String> = aliases;
        let mut results: Vec<(Claim, usize)> = Vec::new();

        for hop in 1..=max_depth {
            if frontier.is_empty() { break; }

            let claims = self.claims_for_entities(&rtxn, &frontier);
            let mut next_frontier = HashSet::new();

            for mut claim in claims {
                if !self.should_include_claim(&claim.claim_id) || !self.should_include_namespace(&claim.namespace) {
                    continue;
                }
                self.apply_status_overlay_fast(&mut claim);
                for eid in [&claim.subject.id, &claim.object.id] {
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
                results.push((claim, hop));
            }

            frontier = next_frontier;
        }

        results
    }

    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        let ra = self.resolve(entity_a);
        let rb = self.resolve(entity_b);
        if ra == rb { return true; }

        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return false,
        };

        let mut visited = HashSet::new();
        visited.insert(ra.clone());
        let mut frontier = HashSet::new();
        frontier.insert(ra);

        for _ in 0..max_depth {
            if frontier.is_empty() { break; }
            let mut next_frontier = HashSet::new();
            for eid in &frontier {
                let neighbors = self.neighbors(&rtxn, eid);
                for n in neighbors {
                    if n == rb { return true; }
                    if !visited.contains(&n) {
                        next_frontier.insert(n.clone());
                        visited.insert(n);
                    }
                }
            }
            frontier = next_frontier;
        }

        false
    }

    /// Get outgoing causal edges for an entity — uses causal_adj index when available,
    /// falls back to full claim scan otherwise.
    /// Returns Vec<(target_entity_id, predicate_id, confidence)>.
    pub fn outgoing_causal_edges(
        &self,
        entity_id: &str,
        causal_predicates: &HashSet<String>,
    ) -> Vec<(String, String, f64)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Fast path: use causal_adj index (no claim deserialization!)
        let mut results = Vec::new();
        let mut used_index = false;

        for pred in causal_predicates {
            let key = format!("{}\x1F{}", entity_id, pred);
            if let Ok(Some(iter)) = self.dbs.causal_adj.get_duplicates(&rtxn, &key) {
                for item in iter {
                    if let Ok((_, target)) = item {
                        results.push((target.to_string(), pred.clone(), 0.65));
                        used_index = true;
                    }
                }
            }
        }

        if used_index {
            return results;
        }

        // Slow fallback: scan all claims for entity
        let seqs = LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, entity_id);

        for seq in seqs {
            let claim_bytes = match self.dbs.claims.get(&rtxn, &seq) {
                Ok(Some(b)) => b.to_vec(),
                _ => continue,
            };
            let claim = match Self::decompress_claim(&claim_bytes) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if claim.subject.id != entity_id { continue; }
            if !causal_predicates.contains(&claim.predicate.id) { continue; }
            if !self.should_include_claim(&claim.claim_id) { continue; }

            results.push((claim.object.id, claim.predicate.id, claim.confidence));
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
        let mut results = self.claims_for(entity_id, predicate_type, source_type, min_confidence);
        if !include_inverse {
            return results;
        }
        let resolved = self.resolve(entity_id);
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        for c in &results {
            seen.insert((c.subject.id.clone(), c.predicate.id.clone(), c.object.id.clone()));
        }
        let mut inverse_claims = Vec::new();
        for c in &results {
            if c.object.id == resolved {
                if let Some(&inv_pred) = attest_core::vocabulary::INVERSE_PREDICATES.get(c.predicate.id.as_str()) {
                    let key = (resolved.clone(), inv_pred.to_string(), c.subject.id.clone());
                    if !seen.contains(&key) {
                        seen.insert(key);
                        let mut derived = c.clone();
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

        let max_depth = max_depth.min(5);
        let resolved = self.resolve(entity_id);

        let mut queue: VecDeque<(String, String, usize, f64)> = VecDeque::new();
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(resolved.clone());

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

            if depth >= 2 {
                results.push((current.clone(), acc_pred.clone(), depth, acc_conf));
            }

            if depth < max_depth {
                let edges = self.outgoing_causal_edges(&current, predicates);
                for (next_target, edge_pred, edge_conf) in edges {
                    if visited.contains(&next_target) {
                        continue;
                    }
                    let composed = compose_predicates(&acc_pred, &edge_pred).to_string();
                    if composed == "associated_with" {
                        continue;
                    }
                    let new_conf = acc_conf * edge_conf;
                    queue.push_back((next_target, composed, depth + 1, new_conf));
                }
            }
        }

        results
    }

    /// Get corroboration count for a content_id by counting entries in the content_id index.
    pub fn corroboration_count(&self, content_id: &str) -> u32 {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return 1,
        };
        let count = match self.dbs.content_id_idx.get_duplicates(&rtxn, content_id) {
            Ok(Some(iter)) => iter.count() as u32,
            _ => 0,
        };
        count
    }

    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };
        let iter = match self.dbs.adjacency_idx.iter(&rtxn) {
            Ok(i) => i,
            Err(_) => return HashMap::new(),
        };

        let mut result: HashMap<String, HashSet<String>> = HashMap::new();
        for entry in iter.flatten() {
            let (k, v) = entry;
            result.entry(k.to_string())
                .or_default()
                .insert(v.to_string());
        }
        result
    }

    // ── Temporal queries ──────────────────────────────────────────────

    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let range = match self.dbs.timestamp_idx.range(&rtxn, &(min_ts..=max_ts)) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let mut seqs = Vec::new();
        for entry in range.flatten() {
            let (_, seq) = entry;
            seqs.push(seq);
        }

        let claims = self.load_claims_by_seqs(&rtxn, &seqs);
        claims.into_iter()
            .filter(|c| self.should_include_claim(&c.claim_id) && self.should_include_namespace(&c.namespace))
            .map(|mut c| { self.apply_status_overlay_fast(&mut c); c })
            .collect()
    }

    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        if n == 0 { return Vec::new(); }

        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let iter = match self.dbs.claims.rev_iter(&rtxn) {
            Ok(i) => i,
            Err(_) => return Vec::new(),
        };

        let mut claims: Vec<Claim> = Vec::with_capacity(n);
        for entry in iter {
            if claims.len() >= n { break; }
            if let Ok((_, v)) = entry {
                if let Ok(mut claim) = Self::decompress_claim(v) {
                    if self.should_include_claim(&claim.claim_id) && self.should_include_namespace(&claim.namespace) {
                        self.apply_status_overlay_fast(&mut claim);
                        claims.push(claim);
                    }
                }
            }
        }
        claims.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        claims
    }

    // ── Text search ──────────────────────────────────────────────────

    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        let tokens = tokenize(query);
        if tokens.is_empty() || top_k == 0 { return Vec::new(); }

        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let n = self.dbs.entities.len(&rtxn).unwrap_or(0) as f64;
        if n == 0.0 { return Vec::new(); }

        // Collect per-entity term frequencies and document frequencies
        let mut tf_map: HashMap<String, HashMap<usize, usize>> = HashMap::new();
        let mut df_map: Vec<usize> = Vec::with_capacity(tokens.len());

        for (tok_idx, token) in tokens.iter().enumerate() {
            let mut df = 0usize;
            if let Ok(Some(iter)) = self.dbs.text_idx.get_duplicates(&rtxn, token.as_str()) {
                for entry in iter {
                    if let Ok((_, eid)) = entry {
                        *tf_map.entry(eid.to_string()).or_default().entry(tok_idx).or_insert(0) += 1;
                        df += 1;
                    }
                }
            }
            df_map.push(df);
        }

        // Compute document lengths for matched entities
        let mut dl_cache: HashMap<String, usize> = HashMap::new();
        for eid in tf_map.keys() {
            let dl = self.dbs.entities.get(&rtxn, eid.as_str()).ok().flatten()
                .and_then(|data| bincode::deserialize::<EntityData>(data).ok())
                .map(|e| {
                    let mut toks = tokenize(&e.display_name);
                    toks.extend(tokenize(&e.id));
                    toks.len().max(1)
                })
                .unwrap_or(1);
            dl_cache.insert(eid.clone(), dl);
        }

        let avgdl = if dl_cache.is_empty() { 3.0 } else {
            let total: usize = dl_cache.values().sum();
            (total as f64 / dl_cache.len() as f64).max(1.0)
        };

        let mut scored: Vec<(String, f64)> = tf_map.into_iter().map(|(eid, token_counts)| {
            let dl = *dl_cache.get(&eid).unwrap_or(&1) as f64;
            let score: f64 = token_counts.iter().map(|(&tok_idx, &tf)| {
                let df = df_map[tok_idx] as f64;
                bm25_score(tf as f64, df, dl, avgdl, n)
            }).sum();
            (eid, score)
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(top_k).filter_map(|(eid, _)| {
            let data_bytes = self.dbs.entities.get(&rtxn, eid.as_str()).ok()??;
            let entity: EntityData = bincode::deserialize(data_bytes).ok()?;
            let claim_count = LmdbBackend::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, &eid).len();
            Some(entity.to_summary(claim_count))
        }).collect()
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return StoreStats::default(),
        };

        let total_claims = self.dbs.claims.len(&rtxn).unwrap_or(0) as usize;
        let entity_count = self.dbs.entities.len(&rtxn).unwrap_or(0) as usize;
        let entity_types = Self::read_counts(&rtxn, &self.dbs.entity_type_counts);
        let predicate_types = Self::read_counts(&rtxn, &self.dbs.pred_type_counts);
        let source_types = Self::read_counts(&rtxn, &self.dbs.src_type_counts);

        StoreStats {
            total_claims,
            entity_count,
            entity_types,
            predicate_types,
            source_types,
        }
    }

    // ── Retraction / status overlay ──────────────────────────────────

    pub fn update_claim_status(&mut self, claim_id: &str, status: ClaimStatus) -> Result<bool, AttestError> {
        if !self.claim_exists(claim_id) { return Ok(false); }

        let status_u8 = Self::status_to_u8(&status);
        let mut wtxn = self.env().write_txn().map_err(|e| lmdb_err("begin_write", e))?;
        self.dbs.status_overrides.put(&mut wtxn, claim_id, &[status_u8])
            .map_err(|e| lmdb_err("put status", e))?;
        wtxn.commit().map_err(|e| lmdb_err("commit", e))?;

        if status == ClaimStatus::Tombstoned {
            self.retracted_ids.insert(claim_id.to_string());
        } else {
            self.retracted_ids.remove(claim_id);
        }

        Ok(true)
    }

    pub fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError> {
        if updates.is_empty() { return Ok(0); }

        let mut wtxn = self.env().write_txn().map_err(|e| lmdb_err("begin_write", e))?;
        let mut count = 0usize;

        for (claim_id, status) in updates {
            if self.dbs.claim_id_idx.get(&wtxn, claim_id.as_str()).ok().flatten().is_none() {
                continue;
            }
            let status_u8 = Self::status_to_u8(status);
            if self.dbs.status_overrides.put(&mut wtxn, claim_id.as_str(), &[status_u8]).is_ok() {
                count += 1;
            }
        }

        wtxn.commit().map_err(|e| lmdb_err("commit", e))?;

        for (claim_id, status) in updates {
            if *status == ClaimStatus::Tombstoned {
                self.retracted_ids.insert(claim_id.clone());
            } else {
                self.retracted_ids.remove(claim_id);
            }
        }

        Ok(count)
    }

    pub fn set_include_retracted(&mut self, include: bool) {
        self.include_retracted = include;
    }

    pub fn set_namespace_filter(&mut self, namespaces: Vec<String>) {
        self.namespace_filter = namespaces;
    }

    pub fn get_namespace_filter(&self) -> &[String] {
        &self.namespace_filter
    }

    // ── Bulk load mode ───────────────────────────────────────────────

    /// Enable or disable bulk load mode.
    /// In bulk mode, analytics counters and claim_summaries are skipped during insert.
    /// Call `rebuild_counters()` after loading is complete to backfill them.
    pub fn set_bulk_load_mode(&mut self, enabled: bool) {
        self.bulk_load_mode = enabled;
    }

    /// Rebuild all analytics counters and claim_summaries from the CLAIMS table.
    /// Call this after bulk loading/merging with bulk_load_mode enabled.
    /// Single-pass O(N) scan of claims + entities.
    pub fn rebuild_counters(&mut self) -> Result<(), AttestError> {
        let env = self.env.as_ref().expect("LMDB environment already closed").clone();

        // Accumulators
        let mut pred_type_counts: HashMap<String, u64> = HashMap::new();
        let mut pred_id_counts: HashMap<String, u64> = HashMap::new();
        let mut src_type_counts: HashMap<String, u64> = HashMap::new();
        let mut entity_pred_counts: HashMap<String, u64> = HashMap::new();
        let mut entity_src_counts: HashMap<String, (u64, f64)> = HashMap::new();
        let mut pred_pair_counts: HashMap<String, (u64, f64)> = HashMap::new();
        let mut summaries: Vec<(u64, Vec<u8>)> = Vec::new();

        // Phase 1: Scan all claims
        let chunk_size = 100_000usize;
        let mut start_seq: u64 = 0;
        let mut total_claims: u64 = 0;

        loop {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let mut chunk_count = 0usize;

            let iter = self.dbs.claims.range(&rtxn, &(start_seq..))
                .map_err(|e| lmdb_err("range claims", e))?;

            for entry in iter {
                let (seq, data) = entry.map_err(|e| lmdb_err("read claim", e))?;
                if let Ok(claim) = Self::decompress_claim(data) {
                    // pred_type_counts
                    *pred_type_counts.entry(claim.predicate.predicate_type.clone()).or_insert(0) += 1;
                    // pred_id_counts
                    *pred_id_counts.entry(claim.predicate.id.clone()).or_insert(0) += 1;
                    // src_type_counts
                    *src_type_counts.entry(claim.provenance.source_type.clone()).or_insert(0) += 1;

                    // entity_pred_counts (truncate object_id for compound key)
                    let key_subj = format!("{}\x1F{}", &claim.subject.id, &claim.predicate.id);
                    *entity_pred_counts.entry(key_subj).or_insert(0) += 1;
                    if claim.object.id != claim.subject.id {
                        let obj_id = Self::truncate_to_bytes(&claim.object.id, LMDB_MAX_KEY);
                        let key_obj = format!("{}\x1F{}", obj_id, &claim.predicate.id);
                        *entity_pred_counts.entry(key_obj).or_insert(0) += 1;
                    }

                    // entity_src_counts (truncate object_id for compound key)
                    let key_subj_src = format!("{}\x1F{}", &claim.subject.id, &claim.provenance.source_id);
                    let e = entity_src_counts.entry(key_subj_src).or_insert((0, 0.0));
                    e.0 += 1; e.1 += claim.confidence;
                    if claim.object.id != claim.subject.id {
                        let obj_id = Self::truncate_to_bytes(&claim.object.id, 470);
                        let key_obj_src = format!("{}\x1F{}", obj_id, &claim.provenance.source_id);
                        let e = entity_src_counts.entry(key_obj_src).or_insert((0, 0.0));
                        e.0 += 1; e.1 += claim.confidence;
                    }

                    // pred_pair_counts (truncate entire compound key)
                    let raw_key = format!("{}\x1F{}\x1F{}", &claim.predicate.id, &claim.subject.id, &claim.object.id);
                    let key_pair = Self::truncate_to_bytes(&raw_key, 500).to_string();
                    let e = pred_pair_counts.entry(key_pair).or_insert((0, 0.0));
                    e.0 += 1; e.1 += claim.confidence;

                    // claim_summaries
                    let summary = ClaimSummary::from(&claim);
                    if let Ok(bytes) = bincode::serialize(&summary) {
                        summaries.push((seq, bytes));
                    }
                }

                start_seq = seq + 1;
                chunk_count += 1;
                if chunk_count >= chunk_size {
                    break;
                }
            }

            total_claims += chunk_count as u64;
            drop(rtxn);

            if chunk_count < chunk_size {
                break;
            }

            // Flush summaries periodically to avoid holding too much memory
            if summaries.len() >= chunk_size {
                let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;
                for (seq, bytes) in summaries.drain(..) {
                    self.dbs.claim_summaries.put(&mut wtxn, &seq, &bytes)
                        .map_err(|e| lmdb_err("put claim_summaries", e))?;
                }
                wtxn.commit().map_err(|e| lmdb_err("commit summaries", e))?;
            }
        }

        // Phase 2: Scan entities for entity_type_counts
        let mut entity_type_counts: HashMap<String, u64> = HashMap::new();
        {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let iter = self.dbs.entities.iter(&rtxn)
                .map_err(|e| lmdb_err("iter entities", e))?;
            for entry in iter {
                let (_id, data) = entry.map_err(|e| lmdb_err("read entity", e))?;
                if let Ok(entity) = bincode::deserialize::<EntityData>(data) {
                    *entity_type_counts.entry(entity.entity_type).or_insert(0) += 1;
                }
            }
        }

        // Phase 3: Clear counter DBs and write accumulated values
        let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;

        self.dbs.pred_type_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear pred_type", e))?;
        self.dbs.pred_id_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear pred_id", e))?;
        self.dbs.src_type_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear src_type", e))?;
        self.dbs.entity_type_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear entity_type", e))?;
        self.dbs.entity_pred_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear entity_pred", e))?;
        self.dbs.entity_src_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear entity_src", e))?;
        self.dbs.pred_pair_counts.clear(&mut wtxn).map_err(|e| lmdb_err("clear pred_pair", e))?;

        for (k, v) in &pred_type_counts {
            self.dbs.pred_type_counts.put(&mut wtxn, k, v)
                .map_err(|e| lmdb_err("put pred_type_counts", e))?;
        }
        for (k, v) in &pred_id_counts {
            self.dbs.pred_id_counts.put(&mut wtxn, k, v)
                .map_err(|e| lmdb_err("put pred_id_counts", e))?;
        }
        for (k, v) in &src_type_counts {
            self.dbs.src_type_counts.put(&mut wtxn, k, v)
                .map_err(|e| lmdb_err("put src_type_counts", e))?;
        }
        for (k, v) in &entity_type_counts {
            self.dbs.entity_type_counts.put(&mut wtxn, k, v)
                .map_err(|e| lmdb_err("put entity_type_counts", e))?;
        }
        for (k, v) in &entity_pred_counts {
            self.dbs.entity_pred_counts.put(&mut wtxn, k, v)
                .map_err(|e| lmdb_err("put entity_pred_counts", e))?;
        }
        for (k, (count, sum)) in &entity_src_counts {
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&count.to_le_bytes());
            buf[8..].copy_from_slice(&sum.to_le_bytes());
            self.dbs.entity_src_counts.put(&mut wtxn, k, &buf)
                .map_err(|e| lmdb_err("put entity_src_counts", e))?;
        }
        for (k, (count, sum)) in &pred_pair_counts {
            let mut buf = [0u8; 16];
            buf[..8].copy_from_slice(&count.to_le_bytes());
            buf[8..].copy_from_slice(&sum.to_le_bytes());
            self.dbs.pred_pair_counts.put(&mut wtxn, k, &buf)
                .map_err(|e| lmdb_err("put pred_pair_counts", e))?;
        }

        // Write remaining summaries
        for (seq, bytes) in &summaries {
            self.dbs.claim_summaries.put(&mut wtxn, seq, bytes)
                .map_err(|e| lmdb_err("put claim_summaries", e))?;
        }

        wtxn.commit().map_err(|e| lmdb_err("commit counters", e))?;

        log::info!("rebuild_counters: processed {} claims, {} entity types, {} predicate IDs",
            total_claims, entity_type_counts.len(), pred_id_counts.len());

        Ok(())
    }

    /// Merge all claims and entities from another LMDB database into this one.
    /// Stays entirely in Rust — no Python overhead. Respects bulk_load_mode.
    /// Returns the number of newly merged claims.
    pub fn merge_from(&mut self, source_path: &str) -> Result<usize, AttestError> {
        let src_path = PathBuf::from(source_path);
        if !src_path.exists() || !src_path.join("data.mdb").exists() {
            return Err(AttestError::Provenance(format!(
                "merge_from: source database not found: {source_path}"
            )));
        }

        // Auto-size map for source (read-only)
        let src_map_size = {
            let data_path = src_path.join("data.mdb");
            let file_size = std::fs::metadata(&data_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            let target = (file_size * 3 / 2).max(DEFAULT_MAP_SIZE);
            let gb = 1 << 30;
            ((target + gb - 1) / gb) * gb
        };

        let src_env = unsafe {
            EnvOpenOptions::new()
                .map_size(src_map_size)
                .max_dbs(MAX_DBS)
                .open(&src_path)
                .map_err(|e| lmdb_err("open source LMDB", e))?
        };

        let src_dbs = Self::create_databases(&src_env)?;

        // Phase 1: Merge entities
        {
            let src_rtxn = src_env.read_txn().map_err(|e| lmdb_err("begin_read source", e))?;
            let iter = src_dbs.entities.iter(&src_rtxn)
                .map_err(|e| lmdb_err("iter source entities", e))?;

            let mut batch: Vec<(String, String, String, HashMap<String, String>)> = Vec::with_capacity(10_000);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;

            for entry in iter {
                let (_id, data) = entry.map_err(|e| lmdb_err("read source entity", e))?;
                if let Ok(entity) = bincode::deserialize::<EntityData>(data) {
                    batch.push((
                        entity.id,
                        entity.entity_type,
                        entity.display_name,
                        entity.external_ids,
                    ));
                    if batch.len() >= 10_000 {
                        self.upsert_entities_batch(&batch, ts)?;
                        batch.clear();
                    }
                }
            }
            if !batch.is_empty() {
                self.upsert_entities_batch(&batch, ts)?;
            }
        }

        // Phase 2: Merge claims
        let target_env = self.env.as_ref().expect("LMDB environment already closed").clone();
        let chunk_size = 100_000usize;
        let mut start_seq: u64 = 0;
        let mut total_merged: usize = 0;

        loop {
            // Read chunk from source
            let src_rtxn = src_env.read_txn().map_err(|e| lmdb_err("begin_read source", e))?;
            let mut claims: Vec<(u64, Claim)> = Vec::with_capacity(chunk_size);

            let iter = src_dbs.claims.range(&src_rtxn, &(start_seq..))
                .map_err(|e| lmdb_err("range source claims", e))?;

            for entry in iter {
                let (seq, data) = entry.map_err(|e| lmdb_err("read source claim", e))?;
                start_seq = seq + 1;
                if let Ok(claim) = Self::decompress_claim(data) {
                    claims.push((seq, claim));
                }
                if claims.len() >= chunk_size {
                    break;
                }
            }

            if claims.is_empty() {
                break;
            }

            let is_last = claims.len() < chunk_size;
            drop(src_rtxn);

            // Write chunk to target, skipping duplicates
            let mut wtxn = target_env.write_txn()
                .map_err(|e| lmdb_err("begin_write target", e))?;

            let mut chunk_merged = 0usize;
            let mut seq = self.next_seq;

            for (_src_seq, claim) in &claims {
                // Skip if claim already exists in target
                if self.dbs.claim_id_idx.get(&wtxn, &claim.claim_id)
                    .ok().flatten().is_some()
                {
                    continue;
                }

                match self.write_claim_to_txn(&mut wtxn, seq, claim) {
                    Ok(_) => {
                        seq += 1;
                        chunk_merged += 1;
                    }
                    Err(e) => {
                        eprintln!("[merge_from] write_claim failed: {e}");
                    }
                }
            }

            wtxn.commit().map_err(|e| lmdb_err("commit merge", e))?;
            self.next_seq = seq;

            // Apply alias predicates
            for (_src_seq, claim) in &claims {
                self.apply_alias_predicate(claim);
            }

            total_merged += chunk_merged;

            if is_last {
                break;
            }
        }

        // Phase 3: Transfer status overrides (retracted/archived claims)
        {
            let src_rtxn = src_env.read_txn().map_err(|e| lmdb_err("begin_read source overrides", e))?;
            let iter = src_dbs.status_overrides.iter(&src_rtxn)
                .map_err(|e| lmdb_err("iter source status_overrides", e))?;

            let mut overrides: Vec<(String, ClaimStatus)> = Vec::new();
            for entry in iter {
                let (claim_id, status_bytes) = entry.map_err(|e| lmdb_err("read source override", e))?;
                if status_bytes.is_empty() { continue; }
                let status = Self::u8_to_status(status_bytes[0]);
                // Only transfer overrides for claims that exist in the target
                if self.claim_exists(claim_id) {
                    overrides.push((claim_id.to_string(), status));
                }
            }
            drop(src_rtxn);

            if !overrides.is_empty() {
                let n = self.update_claim_status_batch(&overrides)?;
                log::info!("merge_from: transferred {} status overrides", n);
            }
        }

        // Flush aliases/metadata
        self.flush_in_memory_state()?;

        // Close source env
        src_env.prepare_for_closing().wait();

        log::info!("merge_from: merged {} claims from {}", total_merged, source_path);
        Ok(total_merged)
    }

    // ── Analytics (Rust-native aggregation) ───────────────────────────

    /// Count claims per source_id by iterating the SOURCE_IDX DUP_SORT multimap.
    /// O(distinct_sources) — sub-millisecond on large databases.
    pub fn source_id_counts(&self) -> HashMap<String, u64> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };
        let mut result: HashMap<String, u64> = HashMap::new();
        let iter = match self.dbs.source_idx.iter(&rtxn) {
            Ok(i) => i,
            Err(_) => return result,
        };
        let mut last_key: Option<String> = None;
        let mut current_count: u64 = 0;
        for entry in iter {
            if let Ok((key, _seq)) = entry {
                if last_key.as_deref() == Some(key) {
                    current_count += 1;
                } else {
                    if let Some(ref prev_key) = last_key {
                        result.insert(prev_key.clone(), current_count);
                    }
                    last_key = Some(key.to_string());
                    current_count = 1;
                }
            }
        }
        if let Some(ref prev_key) = last_key {
            result.insert(prev_key.clone(), current_count);
        }
        result
    }

    pub fn predicate_counts(&self) -> HashMap<String, u64> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };
        // Try the fast counter table first
        let counts = Self::read_counts(&rtxn, &self.dbs.pred_id_counts);
        if !counts.is_empty() {
            return counts.into_iter().map(|(k, v)| (k, v as u64)).collect();
        }
        // Fallback: scan PREDICATE_IDX (slow for large DBs)
        let mut result: HashMap<String, u64> = HashMap::new();
        let iter = match self.dbs.predicate_idx.iter(&rtxn) {
            Ok(i) => i,
            Err(_) => return result,
        };
        let mut last_key: Option<String> = None;
        let mut current_count: u64 = 0;
        for entry in iter {
            if let Ok((key, _seq)) = entry {
                if last_key.as_deref() == Some(key) {
                    current_count += 1;
                } else {
                    if let Some(ref prev_key) = last_key {
                        result.insert(prev_key.clone(), current_count);
                    }
                    last_key = Some(key.to_string());
                    current_count = 1;
                }
            }
        }
        if let Some(ref prev_key) = last_key {
            result.insert(prev_key.clone(), current_count);
        }
        result
    }

    /// Backfill the pred_id_counts counter table from PREDICATE_IDX.
    /// This is needed for databases built before pred_id_counts was added.
    /// Returns the number of unique predicate IDs found.
    pub fn backfill_pred_id_counts(&mut self) -> Result<usize, AttestError> {
        let env = self.env.as_ref().expect("LMDB environment already closed").clone();
        let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;

        // Check if already populated
        let existing = Self::read_counts(&rtxn, &self.dbs.pred_id_counts);
        if !existing.is_empty() {
            return Ok(existing.len());
        }
        drop(rtxn);

        // Scan PREDICATE_IDX
        let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
        let mut counts: HashMap<String, u64> = HashMap::new();
        let iter = self.dbs.predicate_idx.iter(&rtxn)
            .map_err(|e| lmdb_err("iter predicate_idx", e))?;
        let mut last_key: Option<String> = None;
        let mut current_count: u64 = 0;
        for entry in iter {
            if let Ok((key, _)) = entry {
                if last_key.as_deref() == Some(key) {
                    current_count += 1;
                } else {
                    if let Some(ref prev_key) = last_key {
                        counts.insert(prev_key.clone(), current_count);
                    }
                    last_key = Some(key.to_string());
                    current_count = 1;
                }
            }
        }
        if let Some(ref prev_key) = last_key {
            counts.insert(prev_key.clone(), current_count);
        }
        drop(rtxn);

        // Persist
        let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;
        for (pred_id, count) in &counts {
            self.dbs.pred_id_counts.put(&mut wtxn, pred_id, count)
                .map_err(|e| lmdb_err("put pred_id_counts", e))?;
        }
        wtxn.commit().map_err(|e| lmdb_err("commit", e))?;

        Ok(counts.len())
    }

    /// Backfill the claim_summaries table from existing claims.
    /// Processes in 100K chunks to limit memory. Returns total summaries written.
    pub fn backfill_claim_summaries(&mut self) -> Result<u64, AttestError> {
        let env = self.env.as_ref().expect("LMDB environment already closed").clone();

        // Check if already populated: first AND last claim must have summaries
        {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let first = self.dbs.claims.first(&rtxn)
                .map_err(|e| lmdb_err("first claim", e))?;
            let last = self.dbs.claims.last(&rtxn)
                .map_err(|e| lmdb_err("last claim", e))?;
            match (first, last) {
                (None, _) | (_, None) => return Ok(0), // Empty database
                (Some((first_seq, _)), Some((last_seq, _))) => {
                    let first_ok = self.dbs.claim_summaries.get(&rtxn, &first_seq)
                        .map_err(|e| lmdb_err("get first summary", e))?
                        .is_some();
                    let last_ok = self.dbs.claim_summaries.get(&rtxn, &last_seq)
                        .map_err(|e| lmdb_err("get last summary", e))?
                        .is_some();
                    if first_ok && last_ok {
                        return Ok(0); // Already backfilled
                    }
                }
            }
        }

        let chunk_size = 100_000u64;
        let mut total: u64 = 0;
        let mut start_seq: u64 = 0;

        loop {
            // Read chunk
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let mut chunk: Vec<(u64, ClaimSummary)> = Vec::with_capacity(chunk_size as usize);

            let iter = self.dbs.claims.range(&rtxn, &(start_seq..))
                .map_err(|e| lmdb_err("range claims", e))?;

            for entry in iter {
                let (seq, data) = entry.map_err(|e| lmdb_err("read claim", e))?;
                if let Ok(claim) = Self::decompress_claim(data) {
                    chunk.push((seq, ClaimSummary::from(&claim)));
                }
                if chunk.len() >= chunk_size as usize {
                    start_seq = chunk.last().map(|(s, _)| s + 1).unwrap_or(0);
                    break;
                }
            }

            if chunk.is_empty() {
                break;
            }

            let is_last = chunk.len() < chunk_size as usize;
            let chunk_len = chunk.len() as u64;
            drop(rtxn);

            // Write chunk
            let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;
            for (seq, summary) in &chunk {
                let bytes = bincode::serialize(summary)
                    .map_err(|e| lmdb_err("serialize summary", e))?;
                self.dbs.claim_summaries.put(&mut wtxn, seq, &bytes)
                    .map_err(|e| lmdb_err("put claim_summaries", e))?;
            }
            wtxn.commit().map_err(|e| lmdb_err("commit", e))?;

            total += chunk_len;

            if is_last {
                break;
            }
        }

        Ok(total)
    }

    /// Backfill analytics counter tables (entity_pred_counts, entity_src_counts, pred_pair_counts)
    /// from claim_summaries. Processes in 100K chunks to limit memory.
    /// Returns total claims processed.
    pub fn backfill_analytics_counters(&mut self) -> Result<u64, AttestError> {
        let env = self.env.as_ref().expect("LMDB environment already closed").clone();

        // Check if already populated
        {
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let has_counters = self.dbs.entity_pred_counts.first(&rtxn)
                .ok().flatten().is_some();
            if has_counters {
                return Ok(0); // Already backfilled
            }
        }

        let chunk_size = 100_000u64;
        let mut total: u64 = 0;
        let mut start_seq: u64 = 0;

        loop {
            // Read chunk of summaries
            let rtxn = env.read_txn().map_err(|e| lmdb_err("begin_read", e))?;
            let mut chunk: Vec<ClaimSummary> = Vec::with_capacity(chunk_size as usize);

            let iter = self.dbs.claim_summaries.range(&rtxn, &(start_seq..))
                .map_err(|e| lmdb_err("range summaries", e))?;

            let mut last_seq = start_seq;
            for entry in iter {
                let (seq, data) = entry.map_err(|e| lmdb_err("read summary", e))?;
                last_seq = seq;
                if let Ok(summary) = bincode::deserialize::<ClaimSummary>(data) {
                    chunk.push(summary);
                }
                if chunk.len() >= chunk_size as usize {
                    break;
                }
            }

            if chunk.is_empty() {
                // No summaries — try falling back to claim table
                break;
            }

            let is_last = chunk.len() < chunk_size as usize;
            let chunk_len = chunk.len() as u64;
            start_seq = last_seq + 1;
            drop(rtxn);

            // Write counter updates
            let mut wtxn = env.write_txn().map_err(|e| lmdb_err("begin_write", e))?;
            for s in &chunk {
                // entity_pred_counts (truncate object_id for compound key)
                let key_subj = format!("{}\x1F{}", &s.subject_id, &s.predicate_id);
                Self::increment_counter(&mut wtxn, &self.dbs.entity_pred_counts, &key_subj)?;
                if s.object_id != s.subject_id {
                    let obj_id = Self::truncate_to_bytes(&s.object_id, LMDB_MAX_KEY);
                    let key_obj = format!("{}\x1F{}", obj_id, &s.predicate_id);
                    Self::increment_counter(&mut wtxn, &self.dbs.entity_pred_counts, &key_obj)?;
                }

                // entity_src_counts (truncate object_id for compound key)
                let key_subj_src = format!("{}\x1F{}", &s.subject_id, &s.source_id);
                Self::increment_conf_counter(&mut wtxn, &self.dbs.entity_src_counts, &key_subj_src, s.confidence)?;
                if s.object_id != s.subject_id {
                    let obj_id = Self::truncate_to_bytes(&s.object_id, 470);
                    let key_obj_src = format!("{}\x1F{}", obj_id, &s.source_id);
                    Self::increment_conf_counter(&mut wtxn, &self.dbs.entity_src_counts, &key_obj_src, s.confidence)?;
                }

                // pred_pair_counts (truncate entire compound key)
                let raw_key = format!("{}\x1F{}\x1F{}", &s.predicate_id, &s.subject_id, &s.object_id);
                let key_pair = Self::truncate_to_bytes(&raw_key, 500);
                Self::increment_conf_counter(&mut wtxn, &self.dbs.pred_pair_counts, key_pair, s.confidence)?;
            }
            wtxn.commit().map_err(|e| lmdb_err("commit", e))?;

            total += chunk_len;

            if is_last {
                break;
            }
        }

        Ok(total)
    }

    /// Find contradictions: (subject, object) pairs that appear under both pred_a and pred_b.
    /// Returns Vec<(subject_id, object_id, count_a, count_b, avg_conf_a, avg_conf_b)>.
    ///
    /// Uses pred_pair_counts counter table: prefix-scan "{pred_a}\x1F" to get all (subj,obj) pairs
    /// for pred_a, then point-lookup "{pred_b}\x1F{subj}\x1F{obj}" for each. O(pairs_in_pred_a).
    /// Falls back to summary scan when namespace/retraction filtering is active.
    pub fn find_contradictions(
        &self,
        pred_a: &str,
        pred_b: &str,
    ) -> Vec<(String, String, u64, u64, f64, f64)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Fast path: counter table
        if self.namespace_filter.is_empty() && self.retracted_ids.is_empty() {
            let prefix_a = format!("{}\x1F", pred_a);
            let mut pairs_a: HashMap<(String, String), (u64, f64)> = HashMap::new();
            let mut have_counters = false;

            if let Ok(iter) = self.dbs.pred_pair_counts.prefix_iter(&rtxn, &prefix_a) {
                for entry in iter {
                    if let Ok((key, data)) = entry {
                        if !key.starts_with(&prefix_a) {
                            break;
                        }
                        have_counters = true;
                        if data.len() >= 16 {
                            let count = u64::from_le_bytes(data[..8].try_into().unwrap());
                            let sum_conf = f64::from_le_bytes(data[8..16].try_into().unwrap());
                            // Key format: "{pred_a}\x1F{subject}\x1F{object}"
                            let rest = &key[prefix_a.len()..];
                            if let Some(sep) = rest.find('\x1F') {
                                let subj = &rest[..sep];
                                let obj = &rest[sep + 1..];
                                pairs_a.insert((subj.to_string(), obj.to_string()), (count, sum_conf));
                            }
                        }
                    }
                }
            }

            if have_counters {
                // Point-lookup each pair in pred_b
                let mut results = Vec::new();
                for ((subj, obj), (count_a, sum_conf_a)) in &pairs_a {
                    let key_b = format!("{}\x1F{}\x1F{}", pred_b, subj, obj);
                    if let Ok(Some(data)) = self.dbs.pred_pair_counts.get(&rtxn, &key_b) {
                        if data.len() >= 16 {
                            let count_b = u64::from_le_bytes(data[..8].try_into().unwrap());
                            let sum_conf_b = f64::from_le_bytes(data[8..16].try_into().unwrap());
                            if count_b > 0 {
                                results.push((
                                    subj.clone(),
                                    obj.clone(),
                                    *count_a,
                                    count_b,
                                    sum_conf_a / *count_a as f64,
                                    sum_conf_b / count_b as f64,
                                ));
                            }
                        }
                    }
                }
                results.sort_by(|a, b| (b.2 + b.3).cmp(&(a.2 + a.3)));
                return results;
            }
            // Fall through to slow path if counter table is empty
        }

        // Slow path: summary scan (needed for namespace/retraction filtering or pre-backfill)
        let seqs_a = Self::collect_seqs_for_key(&rtxn, &self.dbs.predicate_idx, pred_a);
        let mut pairs_a: HashMap<(String, String), (u64, f64)> = HashMap::with_capacity(seqs_a.len());
        for &seq in &seqs_a {
            if let Some(s) = self.load_summary_by_seq(&rtxn, seq) {
                if !self.should_include_claim(&s.claim_id) || !self.should_include_namespace(&s.namespace) {
                    continue;
                }
                let key = (s.subject_id, s.object_id);
                let entry = pairs_a.entry(key).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += s.confidence;
            }
        }

        let seqs_b = Self::collect_seqs_for_key(&rtxn, &self.dbs.predicate_idx, pred_b);
        let mut pairs_b: HashMap<(String, String), (u64, f64)> = HashMap::new();
        for &seq in &seqs_b {
            if let Some(s) = self.load_summary_by_seq(&rtxn, seq) {
                if !self.should_include_claim(&s.claim_id) || !self.should_include_namespace(&s.namespace) {
                    continue;
                }
                let key = (s.subject_id, s.object_id);
                if pairs_a.contains_key(&key) {
                    let entry = pairs_b.entry(key).or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += s.confidence;
                }
            }
        }

        let mut results = Vec::new();
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

    /// Gap analysis: for entities of a given type, find which expected predicates are missing.
    /// Returns Vec<(entity_id, Vec<missing_predicate_ids>)>.
    /// `limit` caps the number of entities returned (0 = all).
    ///
    /// Fast path uses entity_type_idx + entity_pred_counts counter table:
    /// O(entities_of_type × num_expected_predicates) point lookups instead of
    /// O(claims_per_predicate) sequential scans.
    pub fn gap_analysis(
        &self,
        entity_type: &str,
        expected_predicates: &[&str],
        limit: usize,
    ) -> Vec<(String, Vec<String>)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Fast path: entity_type_idx → entity_pred_counts point lookups
        if self.namespace_filter.is_empty() && self.retracted_ids.is_empty() {
            // Check if entity_pred_counts has data
            let have_counters = self.dbs.entity_pred_counts.first(&rtxn)
                .ok().flatten().is_some();

            if have_counters {
                // Get all entities of the requested type from entity_type_idx
                let entity_ids: Vec<String> = Self::collect_values_for_key(&rtxn, &self.dbs.entity_type_idx, entity_type);

                let mut gaps: Vec<(String, Vec<String>)> = Vec::new();
                for entity_id in &entity_ids {
                    let mut missing = Vec::new();
                    for pred_id in expected_predicates {
                        let key = format!("{}\x1F{}", entity_id, pred_id);
                        let count = self.dbs.entity_pred_counts.get(&rtxn, &key)
                            .ok().flatten().unwrap_or(0);
                        if count == 0 {
                            missing.push(pred_id.to_string());
                        }
                    }
                    // Only include entities that have at least one expected predicate
                    // (otherwise every entity without ANY of the predicates would show up)
                    if !missing.is_empty() && missing.len() < expected_predicates.len() {
                        gaps.push((entity_id.clone(), missing));
                    }
                }

                gaps.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                if limit > 0 && gaps.len() > limit {
                    gaps.truncate(limit);
                }
                return gaps;
            }
        }

        // Slow path: summary scan
        let mut entity_preds: HashMap<String, HashSet<String>> = HashMap::new();

        for pred_id in expected_predicates {
            let seqs = Self::collect_seqs_for_key(&rtxn, &self.dbs.predicate_idx, pred_id);
            for &seq in &seqs {
                if let Some(s) = self.load_summary_by_seq(&rtxn, seq) {
                    if !self.should_include_claim(&s.claim_id) || !self.should_include_namespace(&s.namespace) {
                        continue;
                    }
                    if s.subject_type == entity_type {
                        entity_preds.entry(s.subject_id.clone())
                            .or_default()
                            .insert(pred_id.to_string());
                    }
                    if s.object_type == entity_type {
                        entity_preds.entry(s.object_id)
                            .or_default()
                            .insert(pred_id.to_string());
                    }
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

        // Sort by most missing predicates first
        gaps.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        if limit > 0 && gaps.len() > limit {
            gaps.truncate(limit);
        }

        gaps
    }

    /// Get unique predicate IDs for a single entity.
    /// Uses entity_pred_counts counter table for O(num_predicates) performance.
    /// Falls back to summary scan when namespace/retraction filtering is active.
    pub fn entity_predicate_counts(&self, entity_id: &str) -> Vec<(String, u64)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Fast path: counter table (no filtering needed)
        if self.namespace_filter.is_empty() && self.retracted_ids.is_empty() {
            let prefix = format!("{}\x1F", entity_id);
            let mut result = Vec::new();
            if let Ok(iter) = self.dbs.entity_pred_counts.prefix_iter(&rtxn, &prefix) {
                for entry in iter {
                    if let Ok((key, count)) = entry {
                        if !key.starts_with(&prefix) {
                            break;
                        }
                        let pred_id = &key[prefix.len()..];
                        if count > 0 {
                            result.push((pred_id.to_string(), count));
                        }
                    }
                }
            }
            if !result.is_empty() {
                result.sort_by(|a, b| b.1.cmp(&a.1));
                return result;
            }
            // Fall through to slow path if counter table is empty (not yet backfilled)
        }

        // Slow path: summary scan (needed for namespace/retraction filtering or pre-backfill)
        let seqs = Self::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, entity_id);
        let mut counts: HashMap<String, u64> = HashMap::new();
        for &seq in &seqs {
            if let Some(s) = self.load_summary_by_seq(&rtxn, seq) {
                if !self.should_include_claim(&s.claim_id) || !self.should_include_namespace(&s.namespace) {
                    continue;
                }
                *counts.entry(s.predicate_id).or_insert(0) += 1;
            }
        }
        let mut result: Vec<_> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Hub consensus: for a given entity, get (source_id → count, avg_confidence).
    /// Uses entity_src_counts counter table for O(num_sources) performance.
    /// Falls back to summary scan when namespace/retraction filtering is active.
    pub fn entity_source_counts(&self, entity_id: &str) -> Vec<(String, u64, f64)> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Fast path: counter table
        if self.namespace_filter.is_empty() && self.retracted_ids.is_empty() {
            let prefix = format!("{}\x1F", entity_id);
            let mut result = Vec::new();
            if let Ok(iter) = self.dbs.entity_src_counts.prefix_iter(&rtxn, &prefix) {
                for entry in iter {
                    if let Ok((key, data)) = entry {
                        if !key.starts_with(&prefix) {
                            break;
                        }
                        if data.len() >= 16 {
                            let count = u64::from_le_bytes(data[..8].try_into().unwrap());
                            let sum_conf = f64::from_le_bytes(data[8..16].try_into().unwrap());
                            let source_id = &key[prefix.len()..];
                            if count > 0 {
                                result.push((source_id.to_string(), count, sum_conf / count as f64));
                            }
                        }
                    }
                }
            }
            if !result.is_empty() {
                result.sort_by(|a, b| b.1.cmp(&a.1));
                return result;
            }
        }

        // Slow path: summary scan
        let seqs = Self::collect_seqs_for_key(&rtxn, &self.dbs.entity_claims_idx, entity_id);
        let mut sources: HashMap<String, (u64, f64)> = HashMap::new();
        for &seq in &seqs {
            if let Some(s) = self.load_summary_by_seq(&rtxn, seq) {
                if !self.should_include_claim(&s.claim_id) || !self.should_include_namespace(&s.namespace) {
                    continue;
                }
                let entry = sources.entry(s.source_id).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += s.confidence;
            }
        }
        let mut result: Vec<_> = sources.into_iter()
            .map(|(src, (cnt, sum_conf))| (src, cnt, sum_conf / cnt as f64))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }

    /// Find entities with exactly one distinct source, where total claims >= min_claims.
    /// Single scan over the entity_src_counts table — O(table size), no per-entity roundtrip.
    pub fn find_single_source_entities(&self, min_claims: u64) -> Vec<String> {
        let rtxn = match self.env().read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // entity_src_counts keys: "{entity_id}\x1F{source_id}" → (count:u64, sum_conf:f64)
        // Accumulate per-entity: (total_claims, distinct_sources)
        let mut entity_stats: HashMap<String, (u64, u32)> = HashMap::new();

        if let Ok(iter) = self.dbs.entity_src_counts.iter(&rtxn) {
            for entry in iter {
                if let Ok((key, data)) = entry {
                    if let Some(sep) = key.find('\x1F') {
                        let entity_id = &key[..sep];
                        if data.len() >= 8 {
                            let count = u64::from_le_bytes(data[..8].try_into().unwrap());
                            if count > 0 {
                                let stats = entity_stats.entry(entity_id.to_string()).or_insert((0, 0));
                                stats.0 += count;
                                stats.1 += 1;
                            }
                        }
                    }
                }
            }
        }

        entity_stats.into_iter()
            .filter(|(_, (total, sources))| *sources == 1 && *total >= min_claims)
            .map(|(id, _)| id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::{EntityRef, PredicateRef, Provenance};

    fn make_claim(claim_id: &str, subj: &str, pred: &str, obj: &str) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: attest_core::compute_content_id(subj, pred, obj),
            subject: EntityRef {
                id: subj.to_string(),
                entity_type: "entity".to_string(),
                display_name: subj.to_string(),
                external_ids: HashMap::new(),
            },
            predicate: PredicateRef {
                id: pred.to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: obj.to_string(),
                entity_type: "entity".to_string(),
                display_name: obj.to_string(),
                external_ids: HashMap::new(),
            },
            confidence: 0.8,
            provenance: Provenance {
                source_type: "test".to_string(),
                source_id: "src".to_string(),
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

    #[test]
    fn test_lmdb_persistence_roundtrip() {
        let dir = std::env::temp_dir().join("attest_lmdb_roundtrip_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        // Write
        {
            let mut backend = LmdbBackend::open(db_path).unwrap();
            backend.upsert_entity("a", "entity", "A", None, 0);
            backend.insert_claim(make_claim("c1", "a", "rel", "a"), 0);
            assert!(backend.claim_exists("c1"), "claim should exist before close");
            assert_eq!(backend.count_claims(), 1);
            backend.close().unwrap();
        }

        // Read
        {
            let backend = LmdbBackend::open(db_path).unwrap();
            assert!(backend.claim_exists("c1"), "claim should exist after reopen");
            assert_eq!(backend.count_claims(), 1);
            let entity = backend.get_entity("a");
            assert!(entity.is_some(), "entity should exist after reopen");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_lmdb_basic_operations() {
        let dir = std::env::temp_dir().join("attest_lmdb_basic_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();

        // Entity CRUD
        backend.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        backend.upsert_entity("tp53", "gene", "TP53", None, 0);
        let entity = backend.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");

        // Claim insert
        assert!(backend.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53"), 0));
        assert!(backend.claim_exists("c1"));
        assert!(!backend.claim_exists("c99"));

        // Duplicate rejection
        assert!(!backend.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53"), 0));

        // Get claim
        let claim = backend.get_claim("c1").unwrap();
        assert_eq!(claim.claim_id, "c1");
        assert_eq!(claim.subject.id, "brca1");

        // Stats
        let stats = backend.stats();
        assert_eq!(stats.total_claims, 1);
        assert_eq!(stats.entity_count, 2);

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_lmdb_multimap_indexes() {
        // Verify DUP_SORT correctness: multiple claims for the same entity/content/source
        let dir = std::env::temp_dir().join("attest_lmdb_multimap_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();

        // Insert 3 claims involving "brca1"
        backend.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        backend.upsert_entity("tp53", "gene", "TP53", None, 0);
        backend.upsert_entity("egfr", "gene", "EGFR", None, 0);

        assert!(backend.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53"), 0));
        assert!(backend.insert_claim(make_claim("c2", "brca1", "inhibits", "egfr"), 0));
        assert!(backend.insert_claim(make_claim("c3", "tp53", "regulates", "egfr"), 0));

        // entity_claims_idx: brca1 should have 2 claims (c1 subject, c2 subject)
        let rtxn = backend.env().read_txn().unwrap();
        let brca1_seqs = LmdbBackend::collect_seqs_for_key(&rtxn, &backend.dbs.entity_claims_idx, "brca1");
        assert_eq!(brca1_seqs.len(), 2, "brca1 should appear in 2 claims");

        // tp53 should have 2 claims (c1 object, c3 subject)
        let tp53_seqs = LmdbBackend::collect_seqs_for_key(&rtxn, &backend.dbs.entity_claims_idx, "tp53");
        assert_eq!(tp53_seqs.len(), 2, "tp53 should appear in 2 claims");

        // egfr should have 2 claims (c2 object, c3 object)
        let egfr_seqs = LmdbBackend::collect_seqs_for_key(&rtxn, &backend.dbs.entity_claims_idx, "egfr");
        assert_eq!(egfr_seqs.len(), 2, "egfr should appear in 2 claims");
        drop(rtxn);

        // adjacency: brca1 should have 2 neighbors (tp53, egfr)
        let rtxn = backend.env().read_txn().unwrap();
        let brca1_neighbors = backend.neighbors(&rtxn, "brca1");
        assert_eq!(brca1_neighbors.len(), 2, "brca1 should have 2 neighbors");
        assert!(brca1_neighbors.contains("tp53"));
        assert!(brca1_neighbors.contains("egfr"));
        drop(rtxn);

        // BFS from brca1 at depth 1 should find claims c1 and c2
        let bfs = backend.bfs_claims("brca1", 1);
        assert_eq!(bfs.len(), 2, "BFS depth 1 from brca1 should find 2 claims");

        // claims_for should return both claims for brca1
        let claims = backend.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 2, "claims_for brca1 should return 2 claims");

        // entity_type_idx: "gene" type should have 3 entities
        assert_eq!(backend.count_entities(Some("gene"), 0), 3, "should count 3 gene entities");

        // list_entities for "gene" type
        let entities = backend.list_entities(Some("gene"), 0, 0, 0);
        assert_eq!(entities.len(), 3, "should list 3 gene entities");

        // search_entities: searching for "brca1" should find it
        let results = backend.search_entities("brca1", 10);
        assert!(!results.is_empty(), "search for 'brca1' should find results");
        assert_eq!(results[0].name, "BRCA1");

        // path_exists: brca1 → egfr (direct) and tp53 → egfr (direct)
        assert!(backend.path_exists("brca1", "egfr", 1), "direct path brca1→egfr");
        assert!(backend.path_exists("brca1", "egfr", 2), "path brca1→egfr within 2 hops");

        // get_adjacency_list should have entries for all 3 entities
        let adj = backend.get_adjacency_list();
        assert_eq!(adj.len(), 3, "adjacency list should have 3 entities");

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_claim_summary_analytics() {
        let dir = std::env::temp_dir().join("attest_lmdb_summary_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();

        // Insert claims with typed entities
        let mut c1 = make_claim("c1", "brca1", "binds_to", "tp53");
        c1.subject.entity_type = "gene".to_string();
        c1.object.entity_type = "gene".to_string();
        c1.provenance.source_id = "pubmed".to_string();
        c1.confidence = 0.9;

        let mut c2 = make_claim("c2", "brca1", "inhibits", "tp53");
        c2.subject.entity_type = "gene".to_string();
        c2.object.entity_type = "gene".to_string();
        c2.provenance.source_id = "string_db".to_string();
        c2.confidence = 0.7;

        let mut c3 = make_claim("c3", "brca1", "binds_to", "egfr");
        c3.subject.entity_type = "gene".to_string();
        c3.object.entity_type = "gene".to_string();
        c3.provenance.source_id = "pubmed".to_string();
        c3.confidence = 0.85;

        backend.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        backend.upsert_entity("tp53", "gene", "TP53", None, 0);
        backend.upsert_entity("egfr", "gene", "EGFR", None, 0);

        backend.insert_claim(c1, 0);
        backend.insert_claim(c2, 0);
        backend.insert_claim(c3, 0);

        // Verify summary table is populated (load_summary_by_seq should work)
        {
            let rtxn = backend.env().read_txn().unwrap();
            let summary = backend.load_summary_by_seq(&rtxn, 0).unwrap();
            assert_eq!(summary.claim_id, "c1");
            assert_eq!(summary.subject_id, "brca1");
            assert_eq!(summary.object_id, "tp53");
            assert_eq!(summary.predicate_id, "binds_to");
            assert_eq!(summary.source_id, "pubmed");
            assert!((summary.confidence - 0.9).abs() < 1e-9);
            assert_eq!(summary.subject_type, "gene");
        }

        // entity_predicate_counts uses summaries
        let pred_counts = backend.entity_predicate_counts("brca1");
        assert_eq!(pred_counts.len(), 2);
        // binds_to appears twice (c1, c3), inhibits once (c2)
        let binds_count = pred_counts.iter().find(|(p, _)| p == "binds_to").unwrap().1;
        let inhibits_count = pred_counts.iter().find(|(p, _)| p == "inhibits").unwrap().1;
        assert_eq!(binds_count, 2);
        assert_eq!(inhibits_count, 1);

        // entity_source_counts uses summaries
        let src_counts = backend.entity_source_counts("brca1");
        assert_eq!(src_counts.len(), 2);
        let pubmed = src_counts.iter().find(|(s, _, _)| s == "pubmed").unwrap();
        assert_eq!(pubmed.1, 2); // 2 claims from pubmed
        assert!((pubmed.2 - 0.875).abs() < 1e-9); // avg(0.9, 0.85)

        // gap_analysis uses summaries
        let gaps = backend.gap_analysis("gene", &["binds_to", "inhibits", "regulates"], 0);
        // brca1 has binds_to + inhibits → missing regulates
        // tp53 has binds_to + inhibits (as object) → missing regulates
        // egfr has binds_to (as object) → missing inhibits + regulates
        assert!(!gaps.is_empty());
        let egfr_gap = gaps.iter().find(|(e, _)| e == "egfr").unwrap();
        assert!(egfr_gap.1.contains(&"regulates".to_string()));
        assert!(egfr_gap.1.contains(&"inhibits".to_string()));

        // find_contradictions: binds_to vs inhibits for (brca1, tp53)
        let contradictions = backend.find_contradictions("binds_to", "inhibits");
        assert_eq!(contradictions.len(), 1);
        assert_eq!(contradictions[0].0, "brca1");
        assert_eq!(contradictions[0].1, "tp53");

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_claim_summary_backfill() {
        let dir = std::env::temp_dir().join("attest_lmdb_summary_backfill_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();
        backend.insert_claim(make_claim("c1", "a", "rel", "b"), 0);
        backend.insert_claim(make_claim("c2", "b", "rel", "c"), 0);

        // Clear summaries to simulate pre-v3 DB
        {
            let env = backend.env.as_ref().unwrap().clone();
            let mut wtxn = env.write_txn().unwrap();
            backend.dbs.claim_summaries.clear(&mut wtxn).unwrap();
            wtxn.commit().unwrap();
        }

        // Analytics still works via fallback (load_claim_by_seq)
        let pred_counts = backend.entity_predicate_counts("a");
        assert_eq!(pred_counts.len(), 1);
        assert_eq!(pred_counts[0].0, "rel");
        assert_eq!(pred_counts[0].1, 1);

        // Explicit backfill populates the summary table
        let n = backend.backfill_claim_summaries().unwrap();
        assert_eq!(n, 2);

        // Verify summaries are now populated
        {
            let rtxn = backend.env().read_txn().unwrap();
            let s0 = backend.load_summary_by_seq(&rtxn, 0).unwrap();
            assert_eq!(s0.claim_id, "c1");
            let s1 = backend.load_summary_by_seq(&rtxn, 1).unwrap();
            assert_eq!(s1.claim_id, "c2");
        }

        // Second backfill is a no-op (already populated)
        let n2 = backend.backfill_claim_summaries().unwrap();
        assert_eq!(n2, 0);

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_analytics_counter_tables() {
        let dir = std::env::temp_dir().join("attest_lmdb_counter_tables_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();

        // Insert claims with different predicates, sources, and entity pairs
        let mut c1 = make_claim("c1", "tp53", "upregulates", "mdm2");
        c1.provenance.source_id = "pubmed".to_string();
        c1.confidence = 0.9;
        c1.subject.entity_type = "gene".to_string();
        c1.object.entity_type = "gene".to_string();

        let mut c2 = make_claim("c2", "tp53", "downregulates", "mdm2");
        c2.provenance.source_id = "drugbank".to_string();
        c2.confidence = 0.7;
        c2.subject.entity_type = "gene".to_string();
        c2.object.entity_type = "gene".to_string();

        let mut c3 = make_claim("c3", "tp53", "upregulates", "bax");
        c3.provenance.source_id = "pubmed".to_string();
        c3.confidence = 0.85;
        c3.subject.entity_type = "gene".to_string();
        c3.object.entity_type = "gene".to_string();

        let mut c4 = make_claim("c4", "brca1", "upregulates", "mdm2");
        c4.provenance.source_id = "hetionet".to_string();
        c4.confidence = 0.6;
        c4.subject.entity_type = "gene".to_string();
        c4.object.entity_type = "gene".to_string();

        backend.upsert_entity("tp53", "gene", "TP53", None, 1000);
        backend.upsert_entity("mdm2", "gene", "MDM2", None, 1000);
        backend.upsert_entity("bax", "gene", "BAX", None, 1000);
        backend.upsert_entity("brca1", "gene", "BRCA1", None, 1000);

        backend.insert_claim(c1, 100000);
        backend.insert_claim(c2, 100000);
        backend.insert_claim(c3, 100000);
        backend.insert_claim(c4, 100000);

        // entity_predicate_counts — counter table populated at write time
        let pred_counts = backend.entity_predicate_counts("tp53");
        assert_eq!(pred_counts.len(), 2); // upregulates, downregulates
        // upregulates should have count 2 (c1 + c3 as subject)
        let upreg = pred_counts.iter().find(|(p, _)| p == "upregulates").unwrap();
        assert_eq!(upreg.1, 2);
        let downreg = pred_counts.iter().find(|(p, _)| p == "downregulates").unwrap();
        assert_eq!(downreg.1, 1);

        // mdm2 appears as object in c1, c2, c4 — should have 3 predicates
        let mdm2_preds = backend.entity_predicate_counts("mdm2");
        assert_eq!(mdm2_preds.len(), 2); // upregulates (c1+c4), downregulates (c2)
        let upreg_mdm2 = mdm2_preds.iter().find(|(p, _)| p == "upregulates").unwrap();
        assert_eq!(upreg_mdm2.1, 2);

        // entity_source_counts — counter table populated at write time
        let src_counts = backend.entity_source_counts("tp53");
        assert_eq!(src_counts.len(), 2); // pubmed, drugbank
        let pubmed = src_counts.iter().find(|(s, _, _)| s == "pubmed").unwrap();
        assert_eq!(pubmed.1, 2); // c1 + c3

        // find_contradictions — counter table
        let contradictions = backend.find_contradictions("upregulates", "downregulates");
        assert_eq!(contradictions.len(), 1); // tp53→mdm2 appears under both
        assert_eq!(contradictions[0].0, "tp53");
        assert_eq!(contradictions[0].1, "mdm2");
        assert_eq!(contradictions[0].2, 1); // upregulates count (only c1, not c4 which is brca1→mdm2)
        assert_eq!(contradictions[0].3, 1); // downregulates count

        // gap_analysis — uses entity_pred_counts
        let gaps = backend.gap_analysis("gene", &["upregulates", "downregulates"], 10);
        // tp53 has both, mdm2 has both, but bax only has upregulates, brca1 only has upregulates
        let bax_gap = gaps.iter().find(|(e, _)| e == "bax");
        assert!(bax_gap.is_some());
        assert!(bax_gap.unwrap().1.contains(&"downregulates".to_string()));

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_backfill_analytics_counters() {
        // Test that backfill produces same results as write-time counters
        let dir = std::env::temp_dir().join("attest_lmdb_backfill_counters_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();

        let mut c1 = make_claim("c1", "a", "treats", "b");
        c1.provenance.source_id = "src1".to_string();
        c1.confidence = 0.9;
        let mut c2 = make_claim("c2", "a", "causes", "b");
        c2.provenance.source_id = "src2".to_string();
        c2.confidence = 0.7;

        backend.upsert_entity("a", "drug", "Drug A", None, 1000);
        backend.upsert_entity("b", "disease", "Disease B", None, 1000);
        backend.insert_claim(c1, 100000);
        backend.insert_claim(c2, 100000);

        // Write-time counters populated
        let pred_counts = backend.entity_predicate_counts("a");
        assert_eq!(pred_counts.len(), 2);

        // Clear counter tables to simulate a pre-counter database
        {
            let env = backend.env.as_ref().unwrap().clone();
            let mut wtxn = env.write_txn().unwrap();
            backend.dbs.entity_pred_counts.clear(&mut wtxn).unwrap();
            backend.dbs.entity_src_counts.clear(&mut wtxn).unwrap();
            backend.dbs.pred_pair_counts.clear(&mut wtxn).unwrap();
            wtxn.commit().unwrap();
        }

        // After clearing, should fall back to slow path
        let pred_counts_slow = backend.entity_predicate_counts("a");
        assert_eq!(pred_counts_slow.len(), 2);

        // Now backfill
        let n = backend.backfill_analytics_counters().unwrap();
        assert_eq!(n, 2);

        // Counter tables repopulated — fast path
        let pred_counts_fast = backend.entity_predicate_counts("a");
        assert_eq!(pred_counts_fast.len(), 2);
        assert_eq!(pred_counts, pred_counts_fast);

        // Second backfill is a no-op
        let n2 = backend.backfill_analytics_counters().unwrap();
        assert_eq!(n2, 0);

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_bulk_load_mode_skips_counters() {
        let dir = std::env::temp_dir().join("attest_lmdb_bulk_mode_test");
        let _ = std::fs::remove_dir_all(&dir);
        let db_path = dir.to_str().unwrap();

        let mut backend = LmdbBackend::open(db_path).unwrap();
        backend.set_bulk_load_mode(true);

        // Insert claims in bulk mode
        backend.upsert_entity("a", "gene", "Alpha", None, 0);
        backend.upsert_entity("b", "disease", "Beta", None, 0);
        let claims = vec![
            make_claim("c1", "a", "associates_with", "b"),
            make_claim("c2", "b", "associates_with", "a"),
        ];
        let inserted = backend.insert_claims_batch(claims, 0);
        assert_eq!(inserted, 2);

        // Verify claims exist (primary indexes kept)
        assert!(backend.claim_exists("c1"));
        assert!(backend.claim_exists("c2"));
        assert_eq!(backend.count_claims(), 2);

        // Counter tables should be empty in bulk mode
        let stats = backend.stats();
        assert!(stats.predicate_types.is_empty(), "pred_type_counts should be empty in bulk mode");
        assert!(stats.source_types.is_empty(), "src_type_counts should be empty in bulk mode");
        // entity_type_counts also skipped
        assert!(stats.entity_types.is_empty(), "entity_type_counts should be empty in bulk mode");

        // Rebuild counters
        backend.set_bulk_load_mode(false);
        backend.rebuild_counters().unwrap();

        // Now stats should be correct
        let stats = backend.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 2);
        assert_eq!(*stats.predicate_types.get("relates_to").unwrap_or(&0), 2);
        assert_eq!(*stats.source_types.get("test").unwrap_or(&0), 2);
        assert_eq!(*stats.entity_types.get("gene").unwrap_or(&0), 1);
        assert_eq!(*stats.entity_types.get("disease").unwrap_or(&0), 1);

        // Analytics counters should also be rebuilt
        let pred_counts = backend.entity_predicate_counts("a");
        assert!(!pred_counts.is_empty(), "entity_pred_counts should be rebuilt");

        backend.close().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merge_from_basic() {
        let dir_src = std::env::temp_dir().join("attest_lmdb_merge_src");
        let dir_tgt = std::env::temp_dir().join("attest_lmdb_merge_tgt");
        let _ = std::fs::remove_dir_all(&dir_src);
        let _ = std::fs::remove_dir_all(&dir_tgt);

        // Build source DB
        {
            let mut src = LmdbBackend::open(dir_src.to_str().unwrap()).unwrap();
            src.upsert_entity("x", "gene", "GeneX", None, 0);
            src.upsert_entity("y", "disease", "DiseaseY", None, 0);
            src.insert_claims_batch(vec![
                make_claim("s1", "x", "associates_with", "y"),
                make_claim("s2", "y", "treats", "x"),
            ], 0);
            assert_eq!(src.count_claims(), 2);
            src.close().unwrap();
        }

        // Build target DB with one overlapping claim
        {
            let mut tgt = LmdbBackend::open(dir_tgt.to_str().unwrap()).unwrap();
            tgt.upsert_entity("z", "compound", "CompZ", None, 0);
            tgt.insert_claim(make_claim("t1", "z", "treats", "z"), 0);
            // Also add s1 (should be deduped during merge)
            tgt.upsert_entity("x", "gene", "GeneX", None, 0);
            tgt.upsert_entity("y", "disease", "DiseaseY", None, 0);
            tgt.insert_claim(make_claim("s1", "x", "associates_with", "y"), 0);
            assert_eq!(tgt.count_claims(), 2);

            // Merge
            let merged = tgt.merge_from(dir_src.to_str().unwrap()).unwrap();
            assert_eq!(merged, 1, "should merge 1 new claim (s2), skip duplicate s1");
            assert_eq!(tgt.count_claims(), 3, "target should have 3 total claims");

            // Verify all claims exist
            assert!(tgt.claim_exists("t1"));
            assert!(tgt.claim_exists("s1"));
            assert!(tgt.claim_exists("s2"));

            // Verify entities merged
            assert!(tgt.get_entity("x").is_some());
            assert!(tgt.get_entity("y").is_some());
            assert!(tgt.get_entity("z").is_some());

            tgt.close().unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir_src);
        let _ = std::fs::remove_dir_all(&dir_tgt);
    }

    #[test]
    fn test_bulk_mode_merge_rebuild_roundtrip() {
        let dir_a = std::env::temp_dir().join("attest_lmdb_roundtrip_a");
        let dir_b = std::env::temp_dir().join("attest_lmdb_roundtrip_b");
        let dir_target = std::env::temp_dir().join("attest_lmdb_roundtrip_target");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
        let _ = std::fs::remove_dir_all(&dir_target);

        // Build two source DBs with bulk mode
        {
            let mut a = LmdbBackend::open(dir_a.to_str().unwrap()).unwrap();
            a.set_bulk_load_mode(true);
            a.upsert_entity("g1", "gene", "Gene1", None, 0);
            a.upsert_entity("d1", "disease", "Disease1", None, 0);
            a.insert_claims_batch(vec![
                make_claim("a1", "g1", "associates_with", "d1"),
                make_claim("a2", "g1", "treats", "d1"),
            ], 0);
            a.close().unwrap();
        }
        {
            let mut b = LmdbBackend::open(dir_b.to_str().unwrap()).unwrap();
            b.set_bulk_load_mode(true);
            b.upsert_entity("g2", "gene", "Gene2", None, 0);
            b.upsert_entity("d1", "disease", "Disease1", None, 0);
            b.insert_claims_batch(vec![
                make_claim("b1", "g2", "associates_with", "d1"),
                make_claim("b2", "d1", "treats", "g2"),
            ], 0);
            b.close().unwrap();
        }

        // Create target, merge both, rebuild
        {
            let mut target = LmdbBackend::open(dir_target.to_str().unwrap()).unwrap();
            target.set_bulk_load_mode(true);

            let n1 = target.merge_from(dir_a.to_str().unwrap()).unwrap();
            assert_eq!(n1, 2);
            let n2 = target.merge_from(dir_b.to_str().unwrap()).unwrap();
            assert_eq!(n2, 2);

            assert_eq!(target.count_claims(), 4);

            // Rebuild counters
            target.set_bulk_load_mode(false);
            target.rebuild_counters().unwrap();

            // Verify stats are correct
            let stats = target.stats();
            assert_eq!(stats.total_claims, 4);
            assert_eq!(stats.entity_count, 3, "g1, g2, d1");
            assert_eq!(*stats.entity_types.get("gene").unwrap_or(&0), 2);
            assert_eq!(*stats.entity_types.get("disease").unwrap_or(&0), 1);

            // Verify all entities and claims present
            assert!(target.get_entity("g1").is_some());
            assert!(target.get_entity("g2").is_some());
            assert!(target.get_entity("d1").is_some());
            for id in &["a1", "a2", "b1", "b2"] {
                assert!(target.claim_exists(id), "claim {} should exist", id);
            }

            target.close().unwrap();
        }

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
        let _ = std::fs::remove_dir_all(&dir_target);
    }
}
