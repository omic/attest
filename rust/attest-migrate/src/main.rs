//! Migrate a redb-format .attest database to LMDB format.
//!
//! Usage: attest-migrate <path.attest> [--output <dir>]
//!
//! Reads all claims, entities, metadata, aliases, and status overrides from
//! a redb file and writes them into a new LMDB directory.  The original file
//! is preserved as `<path>.redb.bak`.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process;

use redb::{
    Database, ReadableTable, ReadableTableMetadata, ReadTransaction, TableDefinition,
};
use serde::Deserialize;

use attest_core::types::{
    Claim, ClaimStatus, EntityRef, Payload, PredicateRef, Provenance,
};
use attest_store::backend::lmdb::LmdbBackend;
use attest_store::entity_store::EntityData;
use attest_store::metadata::MetadataStore;
use attest_store::union_find::UnionFind;

// ── redb table definitions (must match the old backend) ──────────────

const CLAIMS: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");
const ENTITIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entities");
const META_BLOBS: TableDefinition<&str, &[u8]> = TableDefinition::new("meta_blobs");
const STATUS_OVERRIDES: TableDefinition<&str, u8> = TableDefinition::new("status_overrides");

// ── Compression handling ─────────────────────────────────────────────

const COMPRESSION_ZSTD: u8 = 0x01;

fn decompress_claim(data: &[u8]) -> Result<Claim, String> {
    if data.is_empty() {
        return Err("empty claim data".to_string());
    }
    let raw = match data[0] {
        COMPRESSION_ZSTD => zstd::decode_all(Cursor::new(&data[1..]))
            .map_err(|e| format!("zstd decompress: {e}"))?,
        0x00 => data[1..].to_vec(),
        _ => data.to_vec(), // Legacy: no header byte
    };
    if let Ok(claim) = bincode::deserialize::<Claim>(&raw) {
        return Ok(claim);
    }
    bincode::deserialize::<LegacyClaim>(&raw)
        .map(|lc| lc.into_claim())
        .map_err(|e| format!("bincode deserialize: {e}"))
}

/// Legacy Claim layout (v0.1.11 and earlier) — no `namespace` or `expires_at`.
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

// ── META_BLOBS keys ──────────────────────────────────────────────────

const META_KEY_ALIASES: &str = "aliases";
const META_KEY_METADATA: &str = "metadata";

// ── Helper: open a read txn ──────────────────────────────────────────

fn read_txn(db: &Database) -> Result<ReadTransaction, String> {
    db.begin_read().map_err(|e| format!("begin_read: {e}"))
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        eprintln!("Usage: attest-migrate <path.attest> [--output <dir>]");
        eprintln!();
        eprintln!("Migrates a redb-format .attest database to LMDB format.");
        eprintln!("The original file is preserved as <path>.redb.bak.");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --output <dir>   Write LMDB to a custom directory");
        eprintln!("                   (default: same path, converted to directory)");
        process::exit(1);
    }

    let redb_path = PathBuf::from(&args[1]);
    let output_path = if args.len() >= 4 && args[2] == "--output" {
        PathBuf::from(&args[3])
    } else {
        redb_path.clone()
    };

    if !redb_path.exists() {
        eprintln!("Error: File not found: {}", redb_path.display());
        process::exit(1);
    }
    if !redb_path.is_file() {
        eprintln!("Error: {} is not a file (already LMDB directory?)", redb_path.display());
        process::exit(1);
    }

    match migrate(&redb_path, &output_path) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}

fn migrate(redb_path: &Path, output_path: &Path) -> Result<(), String> {
    // ── 1. Open redb read-only ───────────────────────────────────────
    eprintln!("Opening redb database: {}", redb_path.display());
    let db = Database::open(redb_path)
        .map_err(|e| format!("failed to open redb: {e}"))?;

    // ── 2. Count claims + entities for progress ──────────────────────
    let (claim_count, entity_count) = {
        let txn = read_txn(&db)?;
        let claims_tbl = txn.open_table(CLAIMS).map_err(|e| format!("open CLAIMS: {e}"))?;
        let entities_tbl = txn.open_table(ENTITIES).map_err(|e| format!("open ENTITIES: {e}"))?;
        (claims_tbl.len().unwrap_or(0), entities_tbl.len().unwrap_or(0))
    };
    eprintln!("  Found {claim_count} claims, {entity_count} entities");

    // ── 3. Pre-flight: check backup path isn't occupied ──────────────
    if output_path == redb_path {
        let backup_path = redb_path.with_extension("redb.bak");
        if backup_path.exists() {
            return Err(format!(
                "Backup path already exists: {}. \
                 Remove it first or use --output to write to a different location.",
                backup_path.display()
            ));
        }
    }

    // ── 4. Create LMDB output in temp directory ──────────────────────
    let temp_lmdb_dir = output_path.with_extension("attest.lmdb.tmp");
    if temp_lmdb_dir.exists() {
        std::fs::remove_dir_all(&temp_lmdb_dir)
            .map_err(|e| format!("remove temp dir: {e}"))?;
    }

    let temp_str = temp_lmdb_dir.to_str()
        .ok_or_else(|| "invalid temp path".to_string())?;
    let mut lmdb = LmdbBackend::open(temp_str)
        .map_err(|e| format!("failed to create LMDB: {e}"))?;

    // ── 5. Read + write entities (preserving per-entity created_at) ──
    eprintln!("Migrating entities...");
    let mut entities_migrated = 0u64;
    {
        let txn = read_txn(&db)?;
        let entities_tbl = txn.open_table(ENTITIES).map_err(|e| format!("open ENTITIES: {e}"))?;

        let iter = entities_tbl.iter().map_err(|e| format!("iter: {e}"))?;
        for entry in iter {
            let guard = entry.map_err(|e| format!("entity iter: {e}"))?;
            let entity_id = guard.0.value().to_string();
            let bytes = guard.1.value();
            let data: EntityData = bincode::deserialize(bytes)
                .map_err(|e| format!("deserialize entity {entity_id}: {e}"))?;

            let ext_ids = if data.external_ids.is_empty() {
                None
            } else {
                Some(&data.external_ids)
            };
            // Use each entity's own created_at timestamp, not a batch-wide max.
            lmdb.upsert_entity(&data.id, &data.entity_type, &data.display_name, ext_ids, data.created_at);
            entities_migrated += 1;

            if entities_migrated % 100_000 == 0 {
                eprintln!("  {entities_migrated}/{entity_count} entities...");
            }
        }
    }
    eprintln!("  {entities_migrated} entities migrated");

    // ── 6. Read + write claims in batches ────────────────────────────
    eprintln!("Migrating claims...");
    let mut claims_migrated = 0u64;
    let mut claims_failed = 0u64;
    {
        let txn = read_txn(&db)?;
        let claims_tbl = txn.open_table(CLAIMS).map_err(|e| format!("open CLAIMS: {e}"))?;

        let batch_size = 100_000;
        let mut batch: Vec<Claim> = Vec::with_capacity(batch_size);

        let iter = claims_tbl.iter().map_err(|e| format!("iter: {e}"))?;
        for entry in iter {
            let guard = entry.map_err(|e| format!("claim iter: {e}"))?;
            let bytes = guard.1.value();
            match decompress_claim(bytes) {
                Ok(claim) => batch.push(claim),
                Err(e) => {
                    claims_failed += 1;
                    if claims_failed <= 10 {
                        eprintln!("  Warning: skipping unreadable claim: {e}");
                    }
                }
            }

            if batch.len() >= batch_size {
                let n = batch.len() as u64;
                lmdb.insert_claims_batch(std::mem::take(&mut batch), 0);
                claims_migrated += n;
                batch = Vec::with_capacity(batch_size);
                eprintln!("  {claims_migrated}/{claim_count} claims...");
            }
        }
        if !batch.is_empty() {
            claims_migrated += batch.len() as u64;
            lmdb.insert_claims_batch(batch, 0);
        }
    }
    if claims_failed > 10 {
        eprintln!("  ({} additional warnings suppressed)", claims_failed - 10);
    }
    eprintln!("  {claims_migrated} claims migrated ({claims_failed} failed)");

    // ── 7. Migrate metadata (vocabularies, predicates, schemas) ──────
    eprintln!("Migrating metadata...");
    {
        let txn = read_txn(&db)?;
        let meta_tbl = txn.open_table(META_BLOBS).map_err(|e| format!("open META_BLOBS: {e}"))?;

        if let Ok(Some(guard)) = meta_tbl.get(META_KEY_METADATA) {
            let bytes = guard.value();
            match bincode::deserialize::<MetadataStore>(bytes) {
                Ok(metadata) => {
                    let vocabs = metadata.vocabularies();
                    for (ns, vocab) in vocabs {
                        lmdb.register_vocabulary(ns, vocab.clone());
                    }
                    let pred_constraints = metadata.predicate_constraints();
                    for (pred_id, val) in &pred_constraints {
                        lmdb.register_predicate(pred_id, val.clone());
                    }
                    let schemas = metadata.payload_schemas();
                    for (schema_id, val) in &schemas {
                        lmdb.register_payload_schema(schema_id, val.clone());
                    }
                    eprintln!("  Vocabularies: {}, predicates: {}, schemas: {}",
                        vocabs.len(),
                        pred_constraints.len(),
                        schemas.len(),
                    );
                }
                Err(e) => {
                    eprintln!("  Warning: failed to deserialize metadata: {e}");
                    eprintln!("  (vocabularies, predicates, and schemas were NOT migrated)");
                }
            }
        }

        // Aliases — reconstructed from same_as/not_same_as claims during insert,
        // but log that we found them.
        if let Ok(Some(guard)) = meta_tbl.get(META_KEY_ALIASES) {
            let bytes = guard.value();
            if bincode::deserialize::<UnionFind>(bytes).is_ok() {
                eprintln!("  Alias data found (reconstructed from claims)");
            }
        }
    }

    // ── 8. Migrate status overrides ──────────────────────────────────
    eprintln!("Migrating status overrides...");
    let mut overrides_migrated = 0u64;
    {
        let txn = read_txn(&db)?;
        let status_tbl = match txn.open_table(STATUS_OVERRIDES) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("  No status overrides table found (skipping)");
                lmdb.checkpoint().map_err(|e| format!("checkpoint: {e}"))?;
                lmdb.close().map_err(|e| format!("close LMDB: {e}"))?;
                return finalize(redb_path, output_path, &temp_lmdb_dir, claim_count, entity_count);
            }
        };

        let mut updates: Vec<(String, ClaimStatus)> = Vec::new();
        let iter = status_tbl.iter().map_err(|e| format!("iter: {e}"))?;
        for entry in iter {
            let guard = entry.map_err(|e| format!("status iter: {e}"))?;
            let claim_id = guard.0.value().to_string();
            let status_byte = guard.1.value();
            let status = match status_byte {
                0 => ClaimStatus::Active,
                1 => ClaimStatus::Archived,
                2 => ClaimStatus::Tombstoned,
                3 => ClaimStatus::ProvenanceDegraded,
                _ => {
                    eprintln!("  Warning: unknown status byte {status_byte} for claim {claim_id}, skipping");
                    continue;
                }
            };
            if !matches!(status, ClaimStatus::Active) {
                updates.push((claim_id, status));
            }
        }

        if !updates.is_empty() {
            let n = updates.len();
            lmdb.update_claim_status_batch(&updates)
                .map_err(|e| format!("update_claim_status_batch: {e}"))?;
            overrides_migrated = n as u64;
        }
    }
    eprintln!("  {overrides_migrated} status overrides migrated");

    // ── 9. Flush and close ───────────────────────────────────────────
    lmdb.checkpoint().map_err(|e| format!("checkpoint: {e}"))?;
    lmdb.close().map_err(|e| format!("close LMDB: {e}"))?;

    finalize(redb_path, output_path, &temp_lmdb_dir, claim_count, entity_count)
}

fn finalize(
    redb_path: &Path,
    output_path: &Path,
    temp_lmdb_dir: &Path,
    claim_count: u64,
    entity_count: u64,
) -> Result<(), String> {
    if output_path == redb_path {
        let backup_path = redb_path.with_extension("redb.bak");
        // Pre-flight already verified backup_path doesn't exist.
        eprintln!("Backing up original to {}", backup_path.display());
        std::fs::rename(redb_path, &backup_path)
            .map_err(|e| format!("backup rename: {e}"))?;
        std::fs::rename(temp_lmdb_dir, output_path)
            .map_err(|e| format!("move LMDB dir: {e}"))?;
    } else {
        if output_path.exists() {
            return Err(format!("output path already exists: {}", output_path.display()));
        }
        std::fs::rename(temp_lmdb_dir, output_path)
            .map_err(|e| format!("move LMDB dir: {e}"))?;
    }

    eprintln!();
    eprintln!("Migration complete!");
    eprintln!("  Claims:   {claim_count}");
    eprintln!("  Entities: {entity_count}");
    eprintln!("  Output:   {}", output_path.display());

    Ok(())
}
