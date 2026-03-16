//! Disk-backed writer queue for LMDB.
//!
//! Writers never block on LMDB. They append to a journal file, then the
//! writer thread drains all pending entries into a single LMDB write
//! transaction.
//!
//! # Architecture
//!
//! ```text
//! Caller → journal_append(entry) → fsync → condvar.notify()
//!                                         ↓
//!                              Writer Thread (background):
//!                              loop {
//!                                  condvar.wait()
//!                                  drain all entries from journal
//!                                  single LMDB write txn → commit
//!                                  truncate journal
//!                                  committed_ticket.store(max_ticket)
//!                                  waiters_condvar.notify_all()
//!                              }
//! ```
//!
//! # Crash recovery
//!
//! On open, any surviving journal entries are replayed into LMDB before
//! starting the writer thread.
//!
//! # File format
//!
//! Reuses the same framing as WAL: `[len:4][data:len][crc32:4]` entries
//! with magic `ATTESTJL`.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use attest_core::types::Claim;

const JOURNAL_MAGIC: &[u8; 8] = b"ATTESTJL";
const JOURNAL_VERSION: u32 = 1;
const JOURNAL_HEADER_SIZE: usize = 16;

/// Operations that can be queued in the writer journal.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum JournalEntry {
    /// Insert a single claim.
    InsertClaim(Claim),
    /// Insert a batch of claims.
    InsertClaimsBatch(Vec<Claim>),
    /// Update claim status: (claim_id, status_u8).
    UpdateClaimStatus(String, u8),
    /// Batch status updates.
    UpdateClaimStatusBatch(Vec<(String, u8)>),
    /// Upsert a single entity: (id, type, display_name, external_ids, timestamp).
    UpsertEntity(String, String, String, std::collections::HashMap<String, String>, i64),
    /// Upsert entities batch.
    UpsertEntitiesBatch(Vec<(String, String, String, std::collections::HashMap<String, String>)>, i64),
}

/// Handle to the on-disk journal file.
struct JournalFile {
    path: PathBuf,
    file: fs::File,
}

impl JournalFile {
    /// Open or create a journal file at `dir/journal`.
    fn open(dir: &Path) -> Result<Self, io::Error> {
        let path = dir.join("journal");
        let exists = path.exists() && fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false);

        let mut file = fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;

        if !exists {
            Self::write_header(&mut file)?;
            file.sync_all()?;
        } else {
            let mut header = [0u8; JOURNAL_HEADER_SIZE];
            if file.read_exact(&mut header).is_err() || &header[0..8] != JOURNAL_MAGIC {
                file.set_len(0)?;
                file.seek(SeekFrom::Start(0))?;
                Self::write_header(&mut file)?;
                file.sync_all()?;
            }
        }

        file.seek(SeekFrom::End(0))?;
        Ok(Self { path, file })
    }

    fn write_header(file: &mut fs::File) -> Result<(), io::Error> {
        file.write_all(JOURNAL_MAGIC)?;
        file.write_all(&JOURNAL_VERSION.to_le_bytes())?;
        file.write_all(&[0u8; 4])?; // reserved
        Ok(())
    }

    /// Append an entry and fsync.
    fn append(&mut self, entry: &JournalEntry) -> Result<(), io::Error> {
        let data = bincode::serialize(entry)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        let len = data.len() as u32;
        let crc = crc32fast::hash(&data);

        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&data)?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Read all entries from the journal (for replay).
    fn read_all(&mut self) -> Vec<JournalEntry> {
        let mut entries = Vec::new();
        if self.file.seek(SeekFrom::Start(JOURNAL_HEADER_SIZE as u64)).is_err() {
            return entries;
        }

        loop {
            let mut len_buf = [0u8; 4];
            if self.file.read_exact(&mut len_buf).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut data = vec![0u8; len];
            if self.file.read_exact(&mut data).is_err() {
                break;
            }

            let mut crc_buf = [0u8; 4];
            if self.file.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let stored_crc = u32::from_le_bytes(crc_buf);
            let computed_crc = crc32fast::hash(&data);
            if stored_crc != computed_crc {
                log::warn!("Journal CRC mismatch — stopping replay");
                break;
            }

            match bincode::deserialize(&data) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    log::warn!("Journal entry deserialize failed: {e}");
                    break;
                }
            }
        }

        entries
    }

    /// Truncate the journal (keep header only).
    fn truncate(&mut self) -> Result<(), io::Error> {
        self.file.set_len(JOURNAL_HEADER_SIZE as u64)?;
        self.file.seek(SeekFrom::Start(JOURNAL_HEADER_SIZE as u64))?;
        self.file.sync_all()?;
        Ok(())
    }

    /// Check if journal has entries beyond the header.
    fn has_entries(&self) -> bool {
        fs::metadata(&self.path)
            .map(|m| m.len() > JOURNAL_HEADER_SIZE as u64)
            .unwrap_or(false)
    }
}

/// Shared state between submitters and the writer thread.
struct WriterState {
    /// Pending entries to be drained by the writer thread.
    pending: Vec<(u64, JournalEntry)>,
    /// Whether the writer should shut down.
    shutdown: bool,
}

/// Disk-backed writer queue for LMDB.
///
/// Submitters call `submit()` to append an entry to the journal and
/// notify the writer thread. The writer thread batches entries into
/// a single LMDB write transaction.
pub struct WriterQueue {
    /// Monotonic ticket counter.
    next_ticket: AtomicU64,
    /// Last committed ticket number.
    committed_ticket: Arc<AtomicU64>,
    /// Shared state for the writer thread.
    state: Arc<(Mutex<WriterState>, Condvar)>,
    /// Condvar for callers waiting on read-after-write.
    waiters: Arc<(Mutex<()>, Condvar)>,
    /// Writer thread join handle.
    writer_handle: Option<thread::JoinHandle<()>>,
    /// Journal file (for appending from submitter side).
    journal: Mutex<JournalFile>,
    /// Whether the queue is running.
    running: Arc<AtomicBool>,
}

impl WriterQueue {
    /// Create a new writer queue.
    ///
    /// `db_dir` is the LMDB directory path (journal file lives inside).
    /// `apply_fn` is called by the writer thread to apply entries to the backend.
    pub fn new<F>(db_dir: &Path, apply_fn: F) -> Result<Self, io::Error>
    where
        F: Fn(Vec<JournalEntry>) + Send + 'static,
    {
        let journal = JournalFile::open(db_dir)?;

        // Replay any surviving journal entries
        let mut replay_journal = JournalFile::open(db_dir)?;
        if replay_journal.has_entries() {
            let entries = replay_journal.read_all();
            if !entries.is_empty() {
                log::info!("Replaying {} journal entries from crash recovery", entries.len());
                apply_fn(entries);
            }
            replay_journal.truncate()?;
        }

        let state = Arc::new((
            Mutex::new(WriterState {
                pending: Vec::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let committed_ticket = Arc::new(AtomicU64::new(0));
        let waiters = Arc::new((Mutex::new(()), Condvar::new()));
        let running = Arc::new(AtomicBool::new(true));

        // Clone Arcs for the writer thread
        let state_clone = state.clone();
        let committed_clone = committed_ticket.clone();
        let waiters_clone = waiters.clone();
        let running_clone = running.clone();
        let journal_path = db_dir.to_path_buf();

        let writer_handle = thread::Builder::new()
            .name("attest-writer".to_string())
            .spawn(move || {
                Self::writer_loop(
                    state_clone,
                    committed_clone,
                    waiters_clone,
                    running_clone,
                    journal_path,
                    apply_fn,
                );
            })
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        Ok(Self {
            next_ticket: AtomicU64::new(1),
            committed_ticket,
            state,
            waiters,
            writer_handle: Some(writer_handle),
            journal: Mutex::new(journal),
            running,
        })
    }

    /// Submit a write operation. Returns a ticket number.
    ///
    /// The entry is appended to the journal file (durable) and queued
    /// for the writer thread. Returns immediately — does NOT wait for
    /// the LMDB commit.
    pub fn submit(&self, entry: JournalEntry) -> Result<u64, io::Error> {
        let ticket = self.next_ticket.fetch_add(1, Ordering::SeqCst);

        // Append to journal (durable on disk before notifying writer)
        {
            let mut journal = self.journal.lock().unwrap();
            journal.append(&entry)?;
        }

        // Add to pending queue
        {
            let (lock, condvar) = &*self.state;
            let mut state = lock.lock().unwrap();
            state.pending.push((ticket, entry));
            condvar.notify_one();
        }

        Ok(ticket)
    }

    /// Wait until the given ticket has been committed to LMDB.
    ///
    /// Use this for read-after-write consistency: submit a write, then
    /// `wait_for(ticket)` before reading.
    pub fn wait_for(&self, ticket: u64) -> bool {
        let timeout = std::time::Duration::from_secs(30);
        let (lock, condvar) = &*self.waiters;
        let guard = lock.lock().unwrap();
        let result = condvar.wait_timeout_while(guard, timeout, |_| {
            self.committed_ticket.load(Ordering::SeqCst) < ticket
        });
        match result {
            Ok((_, timeout_result)) => !timeout_result.timed_out(),
            Err(_) => false,
        }
    }

    /// Check if a ticket has been committed.
    pub fn is_committed(&self, ticket: u64) -> bool {
        self.committed_ticket.load(Ordering::SeqCst) >= ticket
    }

    /// Shut down the writer thread gracefully.
    ///
    /// Drains all remaining entries, commits, then joins the thread.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        // Signal shutdown
        {
            let (lock, condvar) = &*self.state;
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            condvar.notify_one();
        }

        // Join the writer thread
        if let Some(handle) = self.writer_handle.take() {
            let _ = handle.join();
        }
    }

    /// The writer thread's main loop.
    fn writer_loop<F>(
        state: Arc<(Mutex<WriterState>, Condvar)>,
        committed_ticket: Arc<AtomicU64>,
        waiters: Arc<(Mutex<()>, Condvar)>,
        running: Arc<AtomicBool>,
        journal_path: PathBuf,
        apply_fn: F,
    ) where
        F: Fn(Vec<JournalEntry>),
    {
        loop {
            // Wait for entries or shutdown
            let batch: Vec<(u64, JournalEntry)>;
            let should_shutdown: bool;
            {
                let (lock, condvar) = &*state;
                let mut guard = lock.lock().unwrap();

                // Wait until there are pending entries or shutdown
                while guard.pending.is_empty() && !guard.shutdown {
                    guard = condvar.wait(guard).unwrap();
                }

                should_shutdown = guard.shutdown;
                batch = std::mem::take(&mut guard.pending);
            }

            if batch.is_empty() && should_shutdown {
                break;
            }

            if !batch.is_empty() {
                let max_ticket = batch.iter().map(|(t, _)| *t).max().unwrap_or(0);
                let entries: Vec<JournalEntry> = batch.into_iter().map(|(_, e)| e).collect();

                // Apply batch to LMDB
                apply_fn(entries);

                // Truncate journal (entries are now committed to LMDB)
                if let Ok(mut journal) = JournalFile::open(&journal_path) {
                    let _ = journal.truncate();
                }

                // Update committed ticket and notify waiters
                committed_ticket.store(max_ticket, Ordering::SeqCst);
                let (_, condvar) = &*waiters;
                condvar.notify_all();
            }

            if should_shutdown && !running.load(Ordering::SeqCst) {
                break;
            }
        }
    }
}

impl Drop for WriterQueue {
    fn drop(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_journal_roundtrip() {
        let dir = std::env::temp_dir().join("attest_journal_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let claim = Claim {
            claim_id: "test_claim".to_string(),
            content_id: "test_content".to_string(),
            subject: attest_core::types::EntityRef {
                id: "a".to_string(),
                entity_type: "entity".to_string(),
                display_name: "A".to_string(),
                external_ids: Default::default(),
            },
            predicate: attest_core::types::PredicateRef {
                id: "rel".to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: attest_core::types::EntityRef {
                id: "b".to_string(),
                entity_type: "entity".to_string(),
                display_name: "B".to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.9,
            provenance: attest_core::types::Provenance {
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
            status: attest_core::types::ClaimStatus::Active,
            namespace: String::new(),
            expires_at: 0,
        };

        // Write entries
        {
            let mut journal = JournalFile::open(&dir).unwrap();
            journal.append(&JournalEntry::InsertClaim(claim.clone())).unwrap();
            journal.append(&JournalEntry::UpdateClaimStatus("c1".to_string(), 2)).unwrap();
        }

        // Read back
        {
            let mut journal = JournalFile::open(&dir).unwrap();
            let entries = journal.read_all();
            assert_eq!(entries.len(), 2);
            match &entries[0] {
                JournalEntry::InsertClaim(c) => assert_eq!(c.claim_id, "test_claim"),
                _ => panic!("expected InsertClaim"),
            }
            match &entries[1] {
                JournalEntry::UpdateClaimStatus(id, status) => {
                    assert_eq!(id, "c1");
                    assert_eq!(*status, 2);
                }
                _ => panic!("expected UpdateClaimStatus"),
            }
        }

        // Truncate and verify empty
        {
            let mut journal = JournalFile::open(&dir).unwrap();
            journal.truncate().unwrap();
            let entries = journal.read_all();
            assert!(entries.is_empty());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_writer_queue_basic() {
        let dir = std::env::temp_dir().join("attest_writer_queue_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let apply_count = Arc::new(AtomicUsize::new(0));
        let apply_count_clone = apply_count.clone();

        let mut queue = WriterQueue::new(&dir, move |entries| {
            apply_count_clone.fetch_add(entries.len(), Ordering::SeqCst);
        }).unwrap();

        // Submit 3 entries
        let t1 = queue.submit(JournalEntry::UpdateClaimStatus("c1".to_string(), 0)).unwrap();
        let t2 = queue.submit(JournalEntry::UpdateClaimStatus("c2".to_string(), 0)).unwrap();
        let t3 = queue.submit(JournalEntry::UpdateClaimStatus("c3".to_string(), 0)).unwrap();

        // Wait for all to be committed
        assert!(queue.wait_for(t3), "all entries should be committed");
        assert!(queue.is_committed(t1));
        assert!(queue.is_committed(t2));
        assert!(queue.is_committed(t3));

        // Verify apply_fn was called with all entries
        assert_eq!(apply_count.load(Ordering::SeqCst), 3);

        queue.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_writer_queue_crash_recovery() {
        let dir = std::env::temp_dir().join("attest_writer_queue_crash_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // Simulate crash: write entries to journal but don't process them
        {
            let mut journal = JournalFile::open(&dir).unwrap();
            journal.append(&JournalEntry::UpdateClaimStatus("c1".to_string(), 2)).unwrap();
            journal.append(&JournalEntry::UpdateClaimStatus("c2".to_string(), 2)).unwrap();
            // Don't truncate — simulate crash
        }

        // Reopen — should replay the 2 entries
        let replayed = Arc::new(AtomicUsize::new(0));
        let replayed_clone = replayed.clone();

        let mut queue = WriterQueue::new(&dir, move |entries| {
            replayed_clone.fetch_add(entries.len(), Ordering::SeqCst);
        }).unwrap();

        // Give writer thread time to process
        thread::sleep(std::time::Duration::from_millis(50));

        // The 2 crash-recovery entries should have been replayed
        assert_eq!(replayed.load(Ordering::SeqCst), 2);

        queue.shutdown();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
