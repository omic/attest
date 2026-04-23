"""Unanswered-query log — demand-driven gap signal for the ask engine.

When ask_engine.ask() can't produce a confident answer (no entities found,
empty evidence, low aggregate confidence) the question is recorded here.
Distinct from analytics.blindspots() — that is structural (graph-level);
this is demand-driven (user-level).

Block's principle: failed compositions are the roadmap.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


_FAILURE_REASONS = frozenset({
    "no_entities",
    "no_evidence",
    "low_confidence",
    "no_answer",
    "fallback",
})


@dataclass
class UnansweredQuery:
    query_id: str
    question: str
    reason: str
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    n_citations: int = 0
    n_gaps: int = 0
    pipeline: str = ""
    created_at: float = 0.0


def _path_for(db) -> str:
    db_path = getattr(db, "_db_path", None) or getattr(db, "db_path", None)
    if not db_path or db_path == ":memory:":
        return ":memory:"
    base = db_path
    for suf in (".attest", ".substrate"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base + ".unanswered.sqlite"


class UnansweredLog:
    """SQLite-backed log of questions the ask engine couldn't answer."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS unanswered (
                query_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                reason TEXT NOT NULL,
                entities TEXT NOT NULL,
                confidence REAL NOT NULL,
                n_citations INTEGER NOT NULL,
                n_gaps INTEGER NOT NULL,
                pipeline TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unanswered_reason
            ON unanswered (reason, created_at)
        """)
        self._conn.commit()

    def record(self, q: UnansweredQuery) -> str:
        if q.reason not in _FAILURE_REASONS:
            raise ValueError(f"invalid unanswered reason: {q.reason}")
        if not q.query_id:
            q.query_id = str(uuid.uuid4())
        if not q.created_at:
            q.created_at = time.time()
        self._conn.execute(
            """
            INSERT INTO unanswered (
                query_id, question, reason, entities, confidence,
                n_citations, n_gaps, pipeline, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                q.query_id, q.question[:2000], q.reason,
                json.dumps(q.entities)[:2000],
                float(q.confidence), int(q.n_citations), int(q.n_gaps),
                q.pipeline, q.created_at,
            ),
        )
        self._conn.commit()
        return q.query_id

    def recent(
        self,
        limit: int = 100,
        reason: str | None = None,
        since: float | None = None,
    ) -> list[UnansweredQuery]:
        clauses: list[str] = []
        params: list = []
        if reason:
            clauses.append("reason = ?")
            params.append(reason)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM unanswered{where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._to_record(r) for r in rows]

    def summary(self, since: float | None = None) -> dict:
        clauses: list[str] = []
        params: list = []
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT reason, COUNT(*) AS n FROM unanswered{where} GROUP BY reason",
            params,
        ).fetchall()
        return {r["reason"]: r["n"] for r in rows}

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> UnansweredQuery:
        return UnansweredQuery(
            query_id=row["query_id"],
            question=row["question"],
            reason=row["reason"],
            entities=json.loads(row["entities"]) if row["entities"] else [],
            confidence=row["confidence"],
            n_citations=row["n_citations"],
            n_gaps=row["n_gaps"],
            pipeline=row["pipeline"],
            created_at=row["created_at"],
        )


def get_unanswered_log(db) -> UnansweredLog | None:
    """Lazy-init an UnansweredLog cached on the db instance."""
    existing = getattr(db, "_unanswered_log", None)
    if existing is not None:
        return existing
    try:
        log_obj = UnansweredLog(_path_for(db))
        try:
            db._unanswered_log = log_obj
        except Exception:
            pass
        return log_obj
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("unanswered log init failed: %s", exc)
        return None


def log_unanswered(
    db,
    *,
    question: str,
    reason: str,
    entities: list[str] | None = None,
    confidence: float = 0.0,
    n_citations: int = 0,
    n_gaps: int = 0,
    pipeline: str = "v2",
) -> str | None:
    """Best-effort: record one unanswered query. Returns query_id or None."""
    log_obj = get_unanswered_log(db)
    if log_obj is None:
        return None
    try:
        return log_obj.record(UnansweredQuery(
            query_id="",
            question=question,
            reason=reason,
            entities=entities or [],
            confidence=confidence,
            n_citations=n_citations,
            n_gaps=n_gaps,
            pipeline=pipeline,
        ))
    except Exception as exc:  # pragma: no cover — defensive
        log.debug("unanswered log write failed: %s", exc)
        return None
