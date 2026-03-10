//! Migration from old SUBSTRT\0 bincode format to redb.
//!
//! Detects old-format files by magic bytes and migrates data into a new
//! redb database. The old file is renamed to `.v2.bak`.

use std::path::Path;

use attest_core::errors::AttestError;

use crate::file_format;

use super::memory::MemoryBackend;
use super::RedbBackend;

/// Check if a file uses the old SUBSTRT\0 bincode format.
pub fn needs_migration(path: &Path) -> bool {
    path.exists() && file_format::is_attest_file(path)
}

/// Migrate an old-format file to redb.
///
/// 1. Opens the old file with MemoryBackend (loads everything into memory)
/// 2. Creates a new redb database at a temp path
/// 3. Copies all entities, claims, metadata, and aliases
/// 4. Renames old file to `.v2.bak`, moves new redb file to original path
/// 5. Returns a RedbBackend opened at the original path
pub fn migrate_to_redb(db_path: &str) -> Result<RedbBackend, AttestError> {
    let path = Path::new(db_path);
    let backup_path = path.with_extension("attest.v2.bak");
    let temp_path = path.with_extension("attest.redb.tmp");

    // 1. Open old format
    let mut old = MemoryBackend::open(db_path)?;
    let stats = old.stats();
    log::info!(
        "Migrating database: {} claims, {} entities",
        stats.total_claims,
        stats.entity_count,
    );

    // 2. Create new redb at temp path
    let temp_str = temp_path.to_str().ok_or_else(|| {
        AttestError::Provenance("invalid temp path".to_string())
    })?;
    let mut redb = RedbBackend::open(temp_str)?;

    // 3. Copy entities (must be done before claims for entity claim counts)
    let entities = old.list_entities(None, 0);
    for entity in &entities {
        let external_ids: std::collections::HashMap<String, String> = entity.external_ids.clone();
        let ext_ref = if external_ids.is_empty() {
            None
        } else {
            Some(&external_ids)
        };
        // Preserve created_at from old EntityData
        let created_at = old.get_entity_data(&entity.id)
            .map(|d| d.created_at)
            .unwrap_or(0);
        redb.upsert_entity(&entity.id, &entity.entity_type, &entity.name, ext_ref, created_at);
    }

    // 4. Copy all claims in batch
    let claims = old.all_claims();
    let inserted = redb.insert_claims_batch(claims, 0);
    log::info!("Migrated {inserted} claims");

    // 5. Copy metadata
    let vocabs = old.get_registered_vocabularies().clone();
    for (ns, vocab) in vocabs {
        redb.register_vocabulary(&ns, vocab);
    }
    for (pred_id, constraints) in old.get_predicate_constraints() {
        redb.register_predicate(&pred_id, constraints);
    }
    for (schema_id, schema) in old.get_payload_schemas() {
        redb.register_payload_schema(&schema_id, schema);
    }

    // 6. Copy aliases — resolve all entity IDs through old backend
    // Aliases are already replayed when claims with same_as/not_same_as predicates
    // are inserted via insert_claims_batch, so no additional work needed.

    // Flush and close both
    redb.close()?;
    old.close()?;

    // 7. Rename: old → .v2.bak, temp → original
    std::fs::rename(path, &backup_path).map_err(|e| {
        AttestError::Provenance(format!("failed to backup old file: {e}"))
    })?;
    std::fs::rename(&temp_path, path).map_err(|e| {
        AttestError::Provenance(format!("failed to move new database: {e}"))
    })?;

    // Clean up lock and WAL files from old format
    let _ = std::fs::remove_file(path.with_extension("attest.lock"));
    crate::wal::Wal::remove(path);

    log::info!("Migration complete. Old file backed up to {}", backup_path.display());

    // 8. Reopen at original path
    RedbBackend::open(db_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

    fn make_claim(claim_id: &str, subj: &str, pred: &str, obj: &str) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: attest_core::compute_content_id(subj, pred, obj),
            subject: EntityRef {
                id: subj.to_string(),
                entity_type: "gene".to_string(),
                display_name: subj.to_uppercase(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: pred.to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: obj.to_string(),
                entity_type: "gene".to_string(),
                display_name: obj.to_uppercase(),
                external_ids: Default::default(),
            },
            confidence: 0.85,
            provenance: Provenance {
                source_type: "experimental".to_string(),
                source_id: "test_source".to_string(),
                method: None,
                chain: vec!["upstream".to_string()],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
        }
    }

    #[test]
    fn test_migrate_old_format_to_redb() {
        let dir = std::env::temp_dir().join("attest_migration_test.attest");
        let backup = dir.with_extension("attest.v2.bak");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(&backup);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

        let db_path = dir.to_str().unwrap();

        // Create old-format store using MemoryBackend
        {
            let mut old = MemoryBackend::open(db_path).unwrap();
            old.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            old.upsert_entity("tp53", "gene", "TP53", None, 0);
            old.upsert_entity("aspirin", "compound", "Aspirin", None, 0);
            old.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53"), 1000);
            old.insert_claim(make_claim("c2", "tp53", "inhibits", "aspirin"), 1000);
            old.register_vocabulary("bio", crate::metadata::Vocabulary {
                entity_types: vec!["gene".to_string()],
                predicate_types: vec!["binds_to".to_string()],
                source_types: vec!["experimental".to_string()],
            });
            old.close().unwrap();
        }

        // Verify it's old format
        assert!(needs_migration(&dir));

        // Migrate
        let mut redb = migrate_to_redb(db_path).unwrap();

        // Verify data
        assert_eq!(redb.stats().total_claims, 2);
        assert_eq!(redb.stats().entity_count, 3);
        assert!(redb.claim_exists("c1"));
        assert!(redb.claim_exists("c2"));

        // Verify entity
        let entity = redb.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");
        assert_eq!(entity.entity_type, "gene");

        // Verify provenance chain survived
        let chain = redb.get_claim_provenance_chain("c1");
        assert_eq!(chain, vec!["upstream"]);

        // Verify metadata
        assert!(redb.get_registered_vocabularies().contains_key("bio"));

        // Verify backup exists
        assert!(backup.exists());

        // No longer old format
        assert!(!needs_migration(&dir));

        redb.close().unwrap();

        // Clean up
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(&backup);
    }
}
