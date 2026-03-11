//! Write-Ahead Log for crash durability.
//!
//! Each claim insertion is appended to a `.wal` file and fsynced BEFORE
//! being applied to the in-memory store. On `close()` or `checkpoint()`,
//! the full store state is written to the `.attest` file and the WAL is
//! truncated. On `open()`, any WAL entries after the last checkpoint are
//! replayed.
//!
//! # File format
//!
//! ```text
//! [Header: 16 bytes]
//!   magic:           8 bytes  "ATTESTWL"
//!   version:         4 bytes  u32 LE
//!   reserved:        4 bytes
//! [Entries: repeated]
//!   entry_len:       4 bytes  u32 LE (length of bincode data)
//!   data:            entry_len bytes (bincode-serialized WalEntry)
//!   crc32:           4 bytes  CRC32 of data bytes
//! ```

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use attest_core::types::Claim;

const WAL_MAGIC: &[u8; 8] = b"ATTESTWL";
const WAL_VERSION: u32 = 1;
const WAL_HEADER_SIZE: usize = 16;

/// A single WAL entry. Currently only claims, but extensible.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum WalEntry {
    InsertClaim(Claim),
}

/// Write-ahead log handle.
pub struct Wal {
    #[allow(dead_code)]
    path: PathBuf,
    file: fs::File,
    entry_count: u64,
}

/// Error reading a WAL entry.
#[derive(Debug)]
pub enum WalReadError {
    /// Clean end of file (no more entries).
    Eof,
    /// Corrupt entry — stop replay here.
    Corrupt(String),
    /// I/O error.
    Io(io::Error),
}

impl From<io::Error> for WalReadError {
    fn from(e: io::Error) -> Self {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            WalReadError::Eof
        } else {
            WalReadError::Io(e)
        }
    }
}

impl Wal {
    /// Open or create a WAL file. Writes header if new.
    pub fn open(db_path: &Path) -> Result<Self, io::Error> {
        let wal_path = wal_path_for(db_path);
        let exists = wal_path.exists() && fs::metadata(&wal_path).map(|m| m.len() > 0).unwrap_or(false);

        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&wal_path)?;

        if !exists {
            // Write fresh header
            write_header(&mut file)?;
            file.sync_all()?;
        } else {
            // Validate existing header
            let mut header = [0u8; WAL_HEADER_SIZE];
            if file.read_exact(&mut header).is_err() || &header[0..8] != WAL_MAGIC {
                // Corrupt header — recreate
                file.set_len(0)?;
                file.seek(SeekFrom::Start(0))?;
                write_header(&mut file)?;
                file.sync_all()?;
            }
        }

        // Seek to end for appending
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            path: wal_path,
            file,
            entry_count: 0,
        })
    }

    /// Append a claim to the WAL and fsync.
    pub fn append_claim(&mut self, claim: &Claim) -> Result<(), io::Error> {
        self.append_claim_no_sync(claim)?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Append a claim to the WAL without fsyncing (for batch inserts).
    /// Caller must call `sync()` after the batch is complete.
    pub fn append_claim_no_sync(&mut self, claim: &Claim) -> Result<(), io::Error> {
        let entry = WalEntry::InsertClaim(claim.clone());
        let data = bincode::serialize(&entry)
            .map_err(|e| io::Error::other(e.to_string()))?;
        let crc = crc32fast::hash(&data);

        // Write: [u32 len][data][u32 crc]
        self.file.write_all(&(data.len() as u32).to_le_bytes())?;
        self.file.write_all(&data)?;
        self.file.write_all(&crc.to_le_bytes())?;

        self.entry_count += 1;
        Ok(())
    }

    /// Flush and fsync the WAL file.
    pub fn sync(&mut self) -> Result<(), io::Error> {
        self.file.sync_all()
    }

    /// Read all valid entries from the WAL file. Stops at first corrupt entry.
    pub fn read_entries(db_path: &Path) -> Result<Vec<WalEntry>, io::Error> {
        let wal_path = wal_path_for(db_path);
        if !wal_path.exists() {
            return Ok(vec![]);
        }

        let mut file = fs::File::open(&wal_path)?;
        let file_len = file.metadata()?.len();
        if file_len < WAL_HEADER_SIZE as u64 {
            return Ok(vec![]);
        }

        // Skip header
        let mut header = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut header)?;
        if &header[0..8] != WAL_MAGIC {
            log::warn!("WAL file has invalid magic — skipping replay");
            return Ok(vec![]);
        }

        let mut entries = Vec::new();
        loop {
            match read_one_entry(&mut file) {
                Ok(entry) => entries.push(entry),
                Err(WalReadError::Eof) => break,
                Err(WalReadError::Corrupt(msg)) => {
                    log::warn!(
                        "WAL entry {} corrupt ({}), replaying {} valid entries",
                        entries.len(),
                        msg,
                        entries.len()
                    );
                    break;
                }
                Err(WalReadError::Io(e)) => {
                    log::warn!("WAL I/O error after {} entries: {}", entries.len(), e);
                    break;
                }
            }
        }

        Ok(entries)
    }

    /// Truncate the WAL (called after checkpoint). Resets to header-only.
    pub fn truncate(&mut self) -> Result<(), io::Error> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.set_len(0)?;
        write_header(&mut self.file)?;
        self.file.sync_all()?;
        self.entry_count = 0;
        Ok(())
    }

    /// Number of entries written since last truncate.
    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    /// Delete the WAL file.
    pub fn remove(db_path: &Path) {
        let p = wal_path_for(db_path);
        let _ = fs::remove_file(p);
    }
}

/// Compute the WAL path from the DB path.
fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut p = db_path.as_os_str().to_owned();
    p.push(".wal");
    PathBuf::from(p)
}

fn write_header(file: &mut fs::File) -> Result<(), io::Error> {
    file.write_all(WAL_MAGIC)?;
    file.write_all(&WAL_VERSION.to_le_bytes())?;
    file.write_all(&0u32.to_le_bytes())?; // reserved
    Ok(())
}

fn read_one_entry(file: &mut fs::File) -> Result<WalEntry, WalReadError> {
    // Read entry length
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let entry_len = u32::from_le_bytes(len_buf) as usize;

    // Sanity check: max 256 MB per entry
    if entry_len > 256 * 1024 * 1024 {
        return Err(WalReadError::Corrupt(format!(
            "entry_len too large: {entry_len}"
        )));
    }

    // Read data
    let mut data = vec![0u8; entry_len];
    file.read_exact(&mut data).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            WalReadError::Corrupt("truncated entry data".to_string())
        } else {
            WalReadError::Io(e)
        }
    })?;

    // Read and verify CRC
    let mut crc_buf = [0u8; 4];
    file.read_exact(&mut crc_buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            WalReadError::Corrupt("truncated CRC".to_string())
        } else {
            WalReadError::Io(e)
        }
    })?;
    let expected_crc = u32::from_le_bytes(crc_buf);
    let actual_crc = crc32fast::hash(&data);
    if actual_crc != expected_crc {
        return Err(WalReadError::Corrupt(format!(
            "CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )));
    }

    // Deserialize
    bincode::deserialize(&data).map_err(|e| WalReadError::Corrupt(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

    fn make_test_claim(id: &str) -> Claim {
        Claim {
            claim_id: id.to_string(),
            content_id: format!("content_{id}"),
            subject: EntityRef {
                id: "a".to_string(),
                entity_type: "entity".to_string(),
                display_name: "A".to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: "rel".to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: "b".to_string(),
                entity_type: "entity".to_string(),
                display_name: "B".to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.9,
            provenance: Provenance {
                source_type: "test".to_string(),
                source_id: "wal_test".to_string(),
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

    #[test]
    fn test_wal_roundtrip() {
        let dir = std::env::temp_dir().join("wal_roundtrip_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        // Write entries
        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append_claim(&make_test_claim("c1")).unwrap();
            wal.append_claim(&make_test_claim("c2")).unwrap();
            assert_eq!(wal.entry_count(), 2);
        }

        // Read back
        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 2);
        match &entries[0] {
            WalEntry::InsertClaim(c) => assert_eq!(c.claim_id, "c1"),
        }
        match &entries[1] {
            WalEntry::InsertClaim(c) => assert_eq!(c.claim_id, "c2"),
        }

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_wal_truncate() {
        let dir = std::env::temp_dir().join("wal_truncate_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        let mut wal = Wal::open(&dir).unwrap();
        wal.append_claim(&make_test_claim("c1")).unwrap();
        wal.truncate().unwrap();
        assert_eq!(wal.entry_count(), 0);
        drop(wal);

        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 0);

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_wal_corrupt_entry_partial_replay() {
        let dir = std::env::temp_dir().join("wal_corrupt_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        // Write 2 good entries
        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append_claim(&make_test_claim("c1")).unwrap();
            wal.append_claim(&make_test_claim("c2")).unwrap();
        }

        // Corrupt the last few bytes (damages second entry's CRC)
        {
            let wal_path = wal_path_for(&dir);
            let mut data = std::fs::read(&wal_path).unwrap();
            let len = data.len();
            if len > 2 {
                data[len - 2] ^= 0xFF;
            }
            std::fs::write(&wal_path, &data).unwrap();
        }

        // Should recover first entry only
        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::InsertClaim(c) => assert_eq!(c.claim_id, "c1"),
        }

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_wal_empty_no_entries() {
        let dir = std::env::temp_dir().join("wal_empty_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        // Create empty WAL
        {
            let _wal = Wal::open(&dir).unwrap();
        }

        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 0);

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_wal_no_file() {
        let dir = std::env::temp_dir().join("wal_nofile_test.attest");
        Wal::remove(&dir);

        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn test_wal_multiple_entries_roundtrip() {
        let dir = std::env::temp_dir().join("wal_multi_roundtrip_3entry_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        // Write 3 entries
        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append_claim(&make_test_claim("m1")).unwrap();
            wal.append_claim(&make_test_claim("m2")).unwrap();
            wal.append_claim(&make_test_claim("m3")).unwrap();
            assert_eq!(wal.entry_count(), 3);
        }

        // Read all back, verify order and content
        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 3);
        let ids: Vec<&str> = entries
            .iter()
            .map(|e| match e {
                WalEntry::InsertClaim(c) => c.claim_id.as_str(),
            })
            .collect();
        assert_eq!(ids, vec!["m1", "m2", "m3"]);

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }

    #[test]
    fn test_wal_crc_mismatch() {
        let dir = std::env::temp_dir().join("wal_crc_mismatch_precise_test.attest");
        let _ = std::fs::remove_file(&dir);
        Wal::remove(&dir);

        // Write 2 entries
        {
            let mut wal = Wal::open(&dir).unwrap();
            wal.append_claim(&make_test_claim("good")).unwrap();
            wal.append_claim(&make_test_claim("bad")).unwrap();
        }

        // Corrupt the CRC of the second entry by reading the file and
        // overwriting the last 4 bytes (which are the CRC of the second entry).
        // Entry format after 16-byte header: [4-byte len][data][4-byte CRC]
        {
            let wal_path = wal_path_for(&dir);
            let mut data = std::fs::read(&wal_path).unwrap();
            let len = data.len();
            // Last 4 bytes are the CRC of the second entry — flip all bits
            data[len - 4] ^= 0xFF;
            data[len - 3] ^= 0xFF;
            data[len - 2] ^= 0xFF;
            data[len - 1] ^= 0xFF;
            std::fs::write(&wal_path, &data).unwrap();
        }

        // Should recover the first entry only; second is corrupt
        let entries = Wal::read_entries(&dir).unwrap();
        assert_eq!(entries.len(), 1);
        match &entries[0] {
            WalEntry::InsertClaim(c) => assert_eq!(c.claim_id, "good"),
        }

        Wal::remove(&dir);
        let _ = std::fs::remove_file(&dir);
    }
}
