# Working with the Rust Storage Engine

## Architecture

AttestDB uses LMDB (via heed crate) as its sole storage backend. The Rust engine (`rust/attest-store/`) provides 1.3M claims/sec insert, sub-millisecond query (~12µs indexed lookups in-memory). Python accesses it through PyO3 bindings (`rust/attest-py/`).

## RustStore API

```python
from attest_py import RustStore

store = RustStore("path/to/db.attest")  # file-backed
store = RustStore(":memory:")            # in-memory (still persists on close())
```

**Critical:** `store.claims_for(entity_id, predicate_type, source_type, min_confidence)` requires 4 positional args. Pass `None` or `0.0` for optional filters:

```python
# Correct
claims = store.claims_for("TP53", None, None, 0.0)

# Wrong — will error
claims = store.claims_for("TP53")
```

Use `db.claims_for()` (AttestDB wrapper) instead, which handles defaults.

## File Locking (fs2)

Rust uses `fs2` advisory locks — **per-process**, not per-object. Two `RustStore` instances in the same process cannot open the same file.

Workaround for hooks/scripts running in the MCP server process: copy `.attest` + `.wal` to a temp file, open the copy.

```python
import shutil, tempfile
tmp = tempfile.mktemp(suffix=".attest")
shutil.copy2("db.attest", tmp)
shutil.copy2("db.attest.wal", tmp + ".wal")  # WAL has un-checkpointed claims
store = RustStore(tmp)
```

## LMDB Map Sizing

Map size auto-scales: 4× file size (minimum 10GB). On 64-bit systems this is virtual address space only — no actual RAM consumed. Old wheels used 1GB default which caused silent `MDB_MAP_FULL` errors on large DBs.

## Building attest-py

```bash
cd rust/attest-py
maturin build --release --interpreter ../../.venv/bin/python
pip install rust/target/wheels/attest_py-*.whl --force-reinstall
```

## Serialization

Bincode cannot serialize `serde_json::Value` (calls `deserialize_any`). Store JSON payloads as strings using `#[serde(with = "json_value_as_string")]`.

Adding fields to the `Claim` struct breaks bincode deserialization of old data (`#[serde(default)]` only works with self-describing formats). Fix: `LegacyClaim` fallback in `decompress_claim()`.

## Performance

- `claims_for()` on high-degree entities (e.g., EGFR = 48K claims) is expensive. Always pass `limit=N` (0 = no limit)
- Use `entity_predicate_counts()` for counts without materializing claims
- Cap analytics queries at 200-500 claims
- `insert_bulk()` chunks at 100K claims per transaction
