"""PostgreSQL connector — maps query results to claims."""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import QueryConnector

logger = logging.getLogger(__name__)


class PostgresConnector(QueryConnector):
    """Import claims from a PostgreSQL query with column mapping."""

    name = "postgres"

    def __init__(
        self,
        dsn: str,
        query: str,
        mapping: dict,
        source_type: str = "database_import",
        **kwargs: object,
    ) -> None:
        super().__init__(
            query=query, mapping=mapping,
            source_type=source_type, **kwargs,
        )
        self.dsn = dsn

    def _open_cursor(self) -> Iterator[dict]:
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError(
                "pip install psycopg2-binary for PostgreSQL support"
            )

        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor,
            ) as cur:
                cur.execute(self.query)
                yield from cur
