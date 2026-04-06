"""Microsoft SQL Server connector — maps query results to claims.

Usage::

    from attestdb.connectors.mssql import MSSQLConnector

    conn = MSSQLConnector(
        server="localhost",
        database="mydb",
        user="sa",
        password="YourStrong!Passw0rd",
        query="SELECT src, rel, dst FROM edges",
        mapping={"subject": "src", "predicate": "rel", "object": "dst"},
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import QueryConnector

logger = logging.getLogger(__name__)


class MSSQLConnector(QueryConnector):
    """Import claims from a SQL Server query with column mapping."""

    name = "mssql"

    def __init__(
        self,
        server: str,
        database: str,
        user: str,
        password: str,
        query: str,
        mapping: dict,
        port: int = 1433,
        source_type: str = "database_import",
        **kwargs: object,
    ) -> None:
        super().__init__(
            query=query, mapping=mapping,
            source_type=source_type, **kwargs,
        )
        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.port = port

    def _open_cursor(self) -> Iterator[dict]:
        try:
            import pymssql
        except ImportError:
            raise ImportError(
                "pip install pymssql for SQL Server support"
            )

        with pymssql.connect(
            server=self.server,
            user=self.user,
            password=self.password,
            database=self.database,
            port=self.port,
            as_dict=True,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(self.query)
                yield from cur
