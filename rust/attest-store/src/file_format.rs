//! File format for Attest database persistence.
//!
//! Layout:
//! ```text
//! [Header: 32 bytes]
//!   magic:      8 bytes  "SUBSTRT\0"
//!   version:    4 bytes  u32 LE
//!   flags:      4 bytes  reserved
//!   data_len:   8 bytes  u64 LE (length of data section)
//!   checksum:   4 bytes  CRC32 of data section
//!   reserved:   4 bytes
//! [Data section: variable]
//!   bincode-serialized store state (v2+) or JSON (v1)
//! ```

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use fs2::FileExt;

const MAGIC: &[u8; 8] = b"SUBSTRT\0";
/// v2: bincode serialization (v1 was JSON).
const VERSION: u32 = 2;
const HEADER_SIZE: usize = 32;

/// Errors specific to file I/O.
#[derive(Debug)]
pub enum FileError {
    Io(io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    ChecksumMismatch { expected: u32, actual: u32 },
    Deserialize(String),
    Locked,
}

impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::InvalidMagic => write!(f, "not an Attest database file (invalid magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported file version: {v}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected:#010x}, got {actual:#010x}")
            }
            Self::Deserialize(msg) => write!(f, "deserialization error: {msg}"),
            Self::Locked => write!(f, "database is locked by another process"),
        }
    }
}

impl std::error::Error for FileError {}

impl From<io::Error> for FileError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Write store state to a file with header and checksum.
///
/// Uses atomic write: data is written to a `.tmp` file first, then
/// renamed over the target. This prevents partial writes from corrupting
/// the database on crash.
pub fn write_store<T: serde::Serialize>(path: &Path, state: &T) -> Result<(), FileError> {
    let data = bincode::serialize(state).map_err(|e| FileError::Deserialize(e.to_string()))?;
    let checksum = crc32fast::hash(&data);

    // Write to temporary file first
    let tmp_path = path.with_extension("attest.tmp");
    {
        let mut file = fs::File::create(&tmp_path)?;

        // Write header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?; // flags
        file.write_all(&(data.len() as u64).to_le_bytes())?;
        file.write_all(&checksum.to_le_bytes())?;
        file.write_all(&0u32.to_le_bytes())?; // reserved

        // Write data
        file.write_all(&data)?;
        file.sync_all()?;
    }

    // Atomic rename
    fs::rename(&tmp_path, path)?;

    // Fsync the parent directory to ensure the new directory entry is durable.
    // Without this, a crash after rename can lose the entry on Linux/ext4.
    if let Some(parent) = path.parent() {
        if let Ok(dir) = fs::File::open(parent) {
            let _ = dir.sync_all();
        }
    }

    Ok(())
}

/// Acquire an exclusive file lock for single-writer access.
///
/// Returns the lock file handle. The lock is held as long as the handle
/// is alive. Drop it (or call `unlock()`) to release.
pub fn acquire_lock(path: &Path) -> Result<fs::File, FileError> {
    let lock_path = path.with_extension("attest.lock");
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;
    file.try_lock_exclusive().map_err(|_| FileError::Locked)?;
    Ok(file)
}

/// Release a file lock.
pub fn release_lock(file: &fs::File) {
    let _ = FileExt::unlock(file);
}

/// Read store state from a file, verifying header and checksum.
///
/// Supports both v1 (JSON) and v2 (bincode) formats for backward compatibility.
pub fn read_store<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, FileError> {
    let mut file = fs::File::open(path)?;

    // Read header
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    // Validate magic
    if &header[0..8] != MAGIC {
        return Err(FileError::InvalidMagic);
    }

    // Validate version
    let version = u32::from_le_bytes(
        header[8..12]
            .try_into()
            .map_err(|_| FileError::Io(io::Error::new(io::ErrorKind::InvalidData, "truncated header")))?,
    );
    if version > VERSION {
        return Err(FileError::UnsupportedVersion(version));
    }

    // Read data length and checksum
    let data_len_u64 = u64::from_le_bytes(
        header[16..24]
            .try_into()
            .map_err(|_| FileError::Io(io::Error::new(io::ErrorKind::InvalidData, "truncated header")))?,
    );
    let expected_checksum = u32::from_le_bytes(
        header[24..28]
            .try_into()
            .map_err(|_| FileError::Io(io::Error::new(io::ErrorKind::InvalidData, "truncated header")))?,
    );

    // Validate data_len against actual file size to prevent OOM on corrupt headers
    let file_size = file.metadata()?.len();
    let max_data_len = file_size.saturating_sub(HEADER_SIZE as u64);
    if data_len_u64 > max_data_len {
        return Err(FileError::Deserialize(format!(
            "data_len ({data_len_u64}) exceeds file size minus header ({max_data_len})"
        )));
    }
    let data_len = data_len_u64 as usize;

    // Read data
    let mut data = vec![0u8; data_len];
    file.read_exact(&mut data)?;

    // Verify checksum
    let actual_checksum = crc32fast::hash(&data);
    if actual_checksum != expected_checksum {
        return Err(FileError::ChecksumMismatch {
            expected: expected_checksum,
            actual: actual_checksum,
        });
    }

    // Deserialize based on version
    if version == 1 {
        // Legacy JSON format
        serde_json::from_slice(&data).map_err(|e| FileError::Deserialize(e.to_string()))
    } else {
        // v2+: bincode
        bincode::deserialize(&data).map_err(|e| FileError::Deserialize(e.to_string()))
    }
}

/// Check if a file is a valid Attest database.
pub fn is_attest_file(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).is_ok() && &magic == MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_roundtrip() {
        let dir = std::env::temp_dir().join("attest_test_roundtrip");
        let _ = fs::remove_file(&dir);

        let data: HashMap<String, Vec<i32>> = HashMap::from([
            ("a".to_string(), vec![1, 2, 3]),
            ("b".to_string(), vec![4, 5]),
        ]);

        write_store(&dir, &data).unwrap();
        let loaded: HashMap<String, Vec<i32>> = read_store(&dir).unwrap();
        assert_eq!(data, loaded);

        fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn test_invalid_magic() {
        let dir = std::env::temp_dir().join("attest_test_bad_magic");
        fs::write(&dir, b"NOT_ATTEST_xxxxxxxxxxxxxxxxxxxxxx").unwrap();
        assert!(matches!(read_store::<()>(&dir), Err(FileError::InvalidMagic)));
        fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn test_is_attest_file() {
        let dir = std::env::temp_dir().join("attest_test_is_file");
        write_store(&dir, &42u32).unwrap();
        assert!(is_attest_file(&dir));
        fs::remove_file(&dir).unwrap();

        assert!(!is_attest_file(Path::new("/nonexistent/path")));
    }
}
