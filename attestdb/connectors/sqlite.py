"""SQLite connector — maps query results to claims.

Uses the stdlib ``sqlite3`` module — no extra dependencies required.

Usage::

    from attestdb.connectors.sqlite import SQLiteConnector

    conn = SQLiteConnector(
        path="data/interactions.db",
        query="SELECT gene, relation, target FROM interactions",
        mapping={"subject": "gene", "predicate": "relation",
                 "object": "target"},
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Iterator

from attestdb.connectors.base import QueryConnector

logger = logging.getLogger(__name__)


class SQLiteConnector(QueryConnector):
    """Import claims from a SQLite database query with column mapping."""

    name = "sqlite"

    def __init__(
        self,
        path: str,
        query: str,
        mapping: dict,
        source_type: str = "database_import",
        **kwargs: object,
    ) -> None:
        super().__init__(
            query=query, mapping=mapping,
            source_type=source_type, **kwargs,
        )
        self.path = path

    def _open_cursor(self) -> Iterator[dict]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(self.query)
            for row in cursor:
                yield dict(row)
        finally:
            conn.close()
