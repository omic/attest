"""Materialized views — ranked leaderboards over entity metrics."""

from __future__ import annotations

import logging
import sqlite3
import uuid

log = logging.getLogger(__name__)


class MaterializedViewManager:
    """SQLite-backed materialized views for entity metric leaderboards."""

    def __init__(self, db_path: str, tenant_id: str = "default") -> None:
        self.db_path = db_path
        self.tenant_id = tenant_id
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS views (
                view_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                metric_claim_type TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                UNIQUE (tenant_id, entity_type, metric_claim_type)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS view_entries (
                view_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                value REAL NOT NULL,
                rank INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (view_id, entity_id),
                FOREIGN KEY (view_id) REFERENCES views (view_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_view_entries_rank
            ON view_entries (view_id, rank)
        """)
        self._conn.commit()

    def register_view(self, entity_type: str, metric_claim_type: str) -> str:
        """Create a new view definition. Returns the view_id."""
        # Check for existing view
        row = self._conn.execute(
            """
            SELECT view_id FROM views
            WHERE tenant_id = ? AND entity_type = ? AND metric_claim_type = ?
            """,
            (self.tenant_id, entity_type, metric_claim_type),
        ).fetchone()

        if row is not None:
            return row["view_id"]

        view_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO views (view_id, entity_type, metric_claim_type, tenant_id)
            VALUES (?, ?, ?, ?)
            """,
            (view_id, entity_type, metric_claim_type, self.tenant_id),
        )
        self._conn.commit()
        log.debug("Registered view %s: %s/%s", view_id, entity_type, metric_claim_type)
        return view_id

    def update_entry(
        self,
        entity_type: str,
        metric_claim_type: str,
        entity_id: str,
        value: float,
    ) -> None:
        """Upsert an entry and re-rank the view."""
        view_id = self._get_view_id(entity_type, metric_claim_type)
        if view_id is None:
            view_id = self.register_view(entity_type, metric_claim_type)

        # Upsert entry
        self._conn.execute(
            """
            INSERT INTO view_entries (view_id, entity_id, value, rank)
            VALUES (?, ?, ?, 0)
            ON CONFLICT (view_id, entity_id)
            DO UPDATE SET value = excluded.value
            """,
            (view_id, entity_id, value),
        )

        # Re-rank all entries for this view (descending by value)
        self._conn.execute(
            """
            UPDATE view_entries SET rank = (
                SELECT COUNT(*) + 1
                FROM view_entries AS ve2
                WHERE ve2.view_id = view_entries.view_id
                  AND ve2.value > view_entries.value
            )
            WHERE view_id = ?
            """,
            (view_id,),
        )
        self._conn.commit()

    def query_view(
        self,
        entity_type: str,
        metric_claim_type: str,
        top_n: int = 100,
    ) -> list[dict]:
        """Return top N entries sorted by rank."""
        view_id = self._get_view_id(entity_type, metric_claim_type)
        if view_id is None:
            return []

        rows = self._conn.execute(
            """
            SELECT entity_id, value, rank
            FROM view_entries
            WHERE view_id = ?
            ORDER BY rank ASC, entity_id ASC
            LIMIT ?
            """,
            (view_id, top_n),
        ).fetchall()

        return [
            {
                "entity_id": r["entity_id"],
                "value": r["value"],
                "rank": r["rank"],
            }
            for r in rows
        ]

    def view_exists(self, entity_type: str, metric_claim_type: str) -> bool:
        """Check if a view definition exists."""
        return self._get_view_id(entity_type, metric_claim_type) is not None

    def list_views(self) -> list[dict]:
        """List all view definitions for this tenant."""
        rows = self._conn.execute(
            """
            SELECT view_id, entity_type, metric_claim_type
            FROM views
            WHERE tenant_id = ?
            ORDER BY entity_type, metric_claim_type
            """,
            (self.tenant_id,),
        ).fetchall()

        return [
            {
                "view_id": r["view_id"],
                "entity_type": r["entity_type"],
                "metric_claim_type": r["metric_claim_type"],
            }
            for r in rows
        ]

    def auto_register_views(self, query_history: list[dict], min_frequency: int = 3) -> list[str]:
        """Analyze recent queries and auto-create views for common patterns.

        Each entry in *query_history* should have ``entity_type`` and
        ``metric_claim_type`` keys.  Patterns that appear at least
        *min_frequency* times and don't already have a view are registered.

        Returns the list of newly created view_ids.
        """
        # Count (entity_type, metric_claim_type) frequency
        counts: dict[tuple[str, str], int] = {}
        for entry in query_history:
            et = entry.get("entity_type")
            mct = entry.get("metric_claim_type")
            if et and mct:
                counts[(et, mct)] = counts.get((et, mct), 0) + 1

        new_view_ids: list[str] = []
        for (entity_type, metric_claim_type), count in counts.items():
            if count >= min_frequency and not self.view_exists(entity_type, metric_claim_type):
                vid = self.register_view(entity_type, metric_claim_type)
                new_view_ids.append(vid)
                log.info(
                    "Auto-registered view %s/%s (queried %d times)",
                    entity_type, metric_claim_type, count,
                )

        return new_view_ids

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def _get_view_id(self, entity_type: str, metric_claim_type: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT view_id FROM views
            WHERE tenant_id = ? AND entity_type = ? AND metric_claim_type = ?
            """,
            (self.tenant_id, entity_type, metric_claim_type),
        ).fetchone()
        return row["view_id"] if row else None
