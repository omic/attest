"""SQLite-backed cache for LLM responses.

Keyed by a caller-provided content hash (sha256 of the prompt or any
stable identity string). Small, generic, reusable by any primitive that
wants to avoid re-running the same LLM call.

Schema is intentionally minimal: ``(key TEXT PRIMARY KEY, value TEXT,
created_at REAL)``. No TTL by default — callers purge manually if
needed.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


_DEFAULT_PATH = Path.home() / ".attestdb" / "llm_cache.sqlite"


class LLMCache:
    """Thin SQLite cache for LLM call results."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` is fine for small single-writer use;
        # the DB is protected by SQLite's own file locking.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "key TEXT PRIMARY KEY, "
            "value TEXT NOT NULL, "
            "created_at REAL NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, value, created_at) "
            "VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:  # pragma: no cover - defensive
        self.close()
