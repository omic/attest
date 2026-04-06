"""MySQL connector — maps query results to claims."""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import QueryConnector

logger = logging.getLogger(__name__)


class MySQLConnector(QueryConnector):
    """Import claims from a MySQL query with column mapping."""

    name = "mysql"

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        database: str,
        query: str,
        mapping: dict,
        port: int = 3306,
        source_type: str = "database_import",
        **kwargs: object,
    ) -> None:
        super().__init__(
            query=query, mapping=mapping,
            source_type=source_type, **kwargs,
        )
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.port = port

    def _open_cursor(self) -> Iterator[dict]:
        try:
            import pymysql
            import pymysql.cursors
        except ImportError:
            raise ImportError("pip install pymysql for MySQL support")

        with pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            cursorclass=pymysql.cursors.DictCursor,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(self.query)
                yield from cur
