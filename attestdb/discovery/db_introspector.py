"""Database schema introspection — reads catalog metadata into a normalized model.

Supports SQLite (via PRAGMAs) with postgres/mysql/mssql adapters planned.
Produces a DatabaseSchema that feeds into the classifier and compiler.

Usage::

    from attestdb.discovery.db_introspector import introspect_sqlite

    schema = introspect_sqlite("data/public_datasets/chinook.sqlite")
    for table in schema.tables:
        print(f"{table.name}: {len(table.columns)} cols, {len(table.foreign_keys)} FKs")
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Regex patterns for detecting semantic column types from sample values
_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_URL_RE = re.compile(r"^https?://")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I,
)
_PHONE_RE = re.compile(r"^[+]?\d[\d\s\-().]{6,}$")


@dataclass
class ColumnProfile:
    """Lightweight statistical profile for a column."""

    null_ratio: float = 0.0
    approx_distinct: int = 0
    distinct_ratio: float = 0.0
    top_values: list[tuple[str, int]] = field(default_factory=list)
    sample_values: list[str] = field(default_factory=list)
    regex_hits: dict[str, int] = field(default_factory=dict)
    min_value: str | None = None
    max_value: str | None = None


@dataclass
class Column:
    """A single column in a table."""

    name: str
    db_type: str
    nullable: bool
    default: str | None = None
    is_pk: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    ordinal: int = 0
    profile: ColumnProfile | None = None


@dataclass
class ForeignKey:
    """A foreign key relationship between tables."""

    source_table: str
    source_columns: list[str]
    target_table: str
    target_columns: list[str]
    constraint_name: str | None = None

    @property
    def is_self_ref(self) -> bool:
        return self.source_table == self.target_table


@dataclass
class Table:
    """A table with its columns, keys, and basic stats."""

    name: str
    schema_name: str | None = None
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    unique_constraints: list[list[str]] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    indexes: list[list[str]] = field(default_factory=list)
    estimated_row_count: int | None = None
    sample_rows: list[dict] = field(default_factory=list)

    @property
    def inbound_fk_count(self) -> int:
        """Set externally after full schema is built."""
        return getattr(self, "_inbound_fk_count", 0)

    @inbound_fk_count.setter
    def inbound_fk_count(self, value: int) -> None:
        self._inbound_fk_count = value

    def column_by_name(self, name: str) -> Column | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    @property
    def fk_column_names(self) -> set[str]:
        names: set[str] = set()
        for fk in self.foreign_keys:
            names.update(fk.source_columns)
        return names

    @property
    def non_pk_non_fk_columns(self) -> list[Column]:
        pk_set = set(self.primary_key)
        fk_set = self.fk_column_names
        return [
            c for c in self.columns
            if c.name not in pk_set and c.name not in fk_set
        ]


@dataclass
class DatabaseSchema:
    """Complete introspected schema for a database."""

    dialect: Literal["sqlite", "postgres", "mysql", "mssql"]
    database: str
    tables: list[Table] = field(default_factory=list)

    def table_by_name(self, name: str) -> Table | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None

    def inbound_fks(self, table_name: str) -> list[ForeignKey]:
        """All FKs from other tables pointing at this table."""
        result = []
        for t in self.tables:
            for fk in t.foreign_keys:
                if fk.target_table == table_name:
                    result.append(fk)
        return result


def _profile_column_values(
    values: list, col_name: str,
) -> ColumnProfile:
    """Build a ColumnProfile from a list of raw column values."""
    total = len(values)
    if total == 0:
        return ColumnProfile()

    nulls = sum(1 for v in values if v is None)
    non_null = [v for v in values if v is not None]
    str_values = [str(v) for v in non_null]
    distinct = len(set(str_values))

    # Top values (up to 10)
    from collections import Counter
    counts = Counter(str_values)
    top = counts.most_common(10)

    # Regex hits on non-null string values
    regex_hits: dict[str, int] = {}
    for sv in str_values[:200]:
        if _EMAIL_RE.match(sv):
            regex_hits["email"] = regex_hits.get("email", 0) + 1
        if _URL_RE.match(sv):
            regex_hits["url"] = regex_hits.get("url", 0) + 1
        if _UUID_RE.match(sv):
            regex_hits["uuid"] = regex_hits.get("uuid", 0) + 1
        if _PHONE_RE.match(sv):
            regex_hits["phone"] = regex_hits.get("phone", 0) + 1

    # Min/max for sortable values
    min_val = min(str_values) if str_values else None
    max_val = max(str_values) if str_values else None

    # Sample values (up to 5 distinct)
    seen: set[str] = set()
    samples: list[str] = []
    for sv in str_values:
        if sv not in seen and len(samples) < 5:
            seen.add(sv)
            samples.append(sv)

    return ColumnProfile(
        null_ratio=nulls / total if total > 0 else 0.0,
        approx_distinct=distinct,
        distinct_ratio=distinct / len(non_null) if non_null else 0.0,
        top_values=top,
        sample_values=samples,
        regex_hits=regex_hits,
        min_value=min_val,
        max_value=max_val,
    )


def introspect_sqlite(path: str, sample_limit: int = 200) -> DatabaseSchema:
    """Introspect a SQLite database and return a DatabaseSchema.

    Reads table structure via PRAGMAs, samples rows, and profiles columns.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    schema = DatabaseSchema(dialect="sqlite", database=path)

    try:
        # Get all user tables (exclude sqlite internals)
        table_names = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]

        for tname in table_names:
            table = _introspect_sqlite_table(conn, tname, sample_limit)
            schema.tables.append(table)

        # Compute inbound FK counts
        for table in schema.tables:
            table.inbound_fk_count = len(schema.inbound_fks(table.name))

    finally:
        conn.close()

    logger.info(
        "Introspected %s: %d tables, %d total FKs",
        path,
        len(schema.tables),
        sum(len(t.foreign_keys) for t in schema.tables),
    )
    return schema


def _introspect_sqlite_table(
    conn: sqlite3.Connection, table_name: str, sample_limit: int,
) -> Table:
    """Introspect a single SQLite table."""
    table = Table(name=table_name)

    # Columns via PRAGMA table_info
    pk_cols: list[str] = []
    for row in conn.execute(f"PRAGMA table_info([{table_name}])"):
        col = Column(
            name=row["name"],
            db_type=row["type"] or "TEXT",
            nullable=not bool(row["notnull"]),
            default=row["dflt_value"],
            is_pk=bool(row["pk"]),
            ordinal=row["cid"],
        )
        if col.is_pk:
            pk_cols.append(col.name)
        table.columns.append(col)
    table.primary_key = pk_cols

    # Foreign keys via PRAGMA foreign_key_list
    fk_groups: dict[int, dict] = {}
    for row in conn.execute(f"PRAGMA foreign_key_list([{table_name}])"):
        fk_id = row["id"]
        if fk_id not in fk_groups:
            fk_groups[fk_id] = {
                "target_table": row["table"],
                "source_cols": [],
                "target_cols": [],
            }
        fk_groups[fk_id]["source_cols"].append(row["from"])
        fk_groups[fk_id]["target_cols"].append(row["to"])

    for fk_id, fk_data in fk_groups.items():
        fk = ForeignKey(
            source_table=table_name,
            source_columns=fk_data["source_cols"],
            target_table=fk_data["target_table"],
            target_columns=fk_data["target_cols"],
        )
        table.foreign_keys.append(fk)
        # Mark FK columns
        for col_name in fk_data["source_cols"]:
            col = table.column_by_name(col_name)
            if col:
                col.is_indexed = True  # FKs are typically indexed

    # Indexes and unique constraints via PRAGMA index_list
    for idx_row in conn.execute(f"PRAGMA index_list([{table_name}])"):
        idx_name = idx_row["name"]
        is_unique = bool(idx_row["unique"])
        idx_cols = []
        for col_row in conn.execute(f"PRAGMA index_info([{idx_name}])"):
            idx_cols.append(col_row["name"])
            col = table.column_by_name(col_row["name"])
            if col:
                col.is_indexed = True
                if is_unique:
                    col.is_unique = True
        table.indexes.append(idx_cols)
        if is_unique:
            table.unique_constraints.append(idx_cols)

    # Row count
    count_row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM [{table_name}]"
    ).fetchone()
    table.estimated_row_count = count_row["cnt"] if count_row else 0

    # Sample rows + column profiling
    sample_rows = [
        dict(row) for row in conn.execute(
            f"SELECT * FROM [{table_name}] LIMIT ?", (sample_limit,)
        )
    ]
    table.sample_rows = sample_rows[:20]  # Keep 20 for the schema object

    # Profile each column from the full sample
    if sample_rows:
        for col in table.columns:
            values = [row.get(col.name) for row in sample_rows]
            col.profile = _profile_column_values(values, col.name)

    return table


# ======================================================================
# PostgreSQL introspection
# ======================================================================

def introspect_postgres(
    dsn: str,
    schema_name: str = "public",
    sample_limit: int = 200,
) -> DatabaseSchema:
    """Introspect a PostgreSQL database via information_schema + pg_catalog."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError("pip install psycopg2-binary for PostgreSQL introspection")

    conn = psycopg2.connect(dsn)
    schema = DatabaseSchema(dialect="postgres", database=dsn.split("/")[-1] if "/" in dsn else dsn)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Tables
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                "ORDER BY table_name",
                (schema_name,),
            )
            table_names = [row["table_name"] for row in cur.fetchall()]

            for tname in table_names:
                table = _introspect_pg_table(cur, tname, schema_name, sample_limit)
                schema.tables.append(table)

            for table in schema.tables:
                table.inbound_fk_count = len(schema.inbound_fks(table.name))
    finally:
        conn.close()

    logger.info(
        "Introspected postgres %s.%s: %d tables, %d total FKs",
        schema.database, schema_name, len(schema.tables),
        sum(len(t.foreign_keys) for t in schema.tables),
    )
    return schema


def _introspect_pg_table(cur, table_name: str, schema_name: str, sample_limit: int) -> Table:
    """Introspect a single PostgreSQL table."""
    table = Table(name=table_name, schema_name=schema_name)

    # Columns
    cur.execute(
        "SELECT column_name, data_type, is_nullable, column_default, ordinal_position "
        "FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s "
        "ORDER BY ordinal_position",
        (schema_name, table_name),
    )
    for row in cur.fetchall():
        table.columns.append(Column(
            name=row["column_name"],
            db_type=row["data_type"],
            nullable=row["is_nullable"] == "YES",
            default=row["column_default"],
            ordinal=row["ordinal_position"],
        ))

    # Primary key
    cur.execute(
        "SELECT kcu.column_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "  AND tc.table_schema = kcu.table_schema "
        "WHERE tc.table_schema = %s AND tc.table_name = %s "
        "  AND tc.constraint_type = 'PRIMARY KEY' "
        "ORDER BY kcu.ordinal_position",
        (schema_name, table_name),
    )
    pk_cols = [row["column_name"] for row in cur.fetchall()]
    table.primary_key = pk_cols
    for col_name in pk_cols:
        col = table.column_by_name(col_name)
        if col:
            col.is_pk = True

    # Foreign keys
    cur.execute(
        "SELECT kcu.column_name AS source_column, "
        "  ccu.table_name AS target_table, "
        "  ccu.column_name AS target_column, "
        "  tc.constraint_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "  AND tc.table_schema = kcu.table_schema "
        "JOIN information_schema.constraint_column_usage ccu "
        "  ON tc.constraint_name = ccu.constraint_name "
        "  AND tc.table_schema = ccu.table_schema "
        "WHERE tc.table_schema = %s AND tc.table_name = %s "
        "  AND tc.constraint_type = 'FOREIGN KEY'",
        (schema_name, table_name),
    )
    fk_groups: dict[str, dict] = {}
    for row in cur.fetchall():
        cname = row["constraint_name"]
        if cname not in fk_groups:
            fk_groups[cname] = {
                "target_table": row["target_table"],
                "source_cols": [],
                "target_cols": [],
            }
        fk_groups[cname]["source_cols"].append(row["source_column"])
        fk_groups[cname]["target_cols"].append(row["target_column"])

    for cname, fk_data in fk_groups.items():
        table.foreign_keys.append(ForeignKey(
            source_table=table_name,
            source_columns=fk_data["source_cols"],
            target_table=fk_data["target_table"],
            target_columns=fk_data["target_cols"],
            constraint_name=cname,
        ))

    # Unique constraints
    cur.execute(
        "SELECT kcu.column_name, tc.constraint_name "
        "FROM information_schema.table_constraints tc "
        "JOIN information_schema.key_column_usage kcu "
        "  ON tc.constraint_name = kcu.constraint_name "
        "  AND tc.table_schema = kcu.table_schema "
        "WHERE tc.table_schema = %s AND tc.table_name = %s "
        "  AND tc.constraint_type = 'UNIQUE' "
        "ORDER BY tc.constraint_name, kcu.ordinal_position",
        (schema_name, table_name),
    )
    uq_groups: dict[str, list[str]] = {}
    for row in cur.fetchall():
        uq_groups.setdefault(row["constraint_name"], []).append(row["column_name"])
    table.unique_constraints = list(uq_groups.values())
    for cols in table.unique_constraints:
        for col_name in cols:
            col = table.column_by_name(col_name)
            if col:
                col.is_unique = True

    # Row count estimate (no full scan)
    cur.execute(
        "SELECT n_live_tup FROM pg_stat_user_tables "
        "WHERE schemaname = %s AND relname = %s",
        (schema_name, table_name),
    )
    row = cur.fetchone()
    table.estimated_row_count = int(row["n_live_tup"]) if row else 0

    # Sample rows + column profiling
    cur.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT %s', (sample_limit,))
    sample_rows = [dict(r) for r in cur.fetchall()]
    table.sample_rows = sample_rows[:20]

    if sample_rows:
        for col in table.columns:
            values = [row.get(col.name) for row in sample_rows]
            col.profile = _profile_column_values(values, col.name)

    return table


# ======================================================================
# MySQL introspection
# ======================================================================

def introspect_mysql(
    host: str,
    user: str,
    password: str,
    database: str,
    port: int = 3306,
    sample_limit: int = 200,
) -> DatabaseSchema:
    """Introspect a MySQL database via information_schema."""
    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        raise ImportError("pip install pymysql for MySQL introspection")

    conn = pymysql.connect(
        host=host, user=user, password=password,
        database=database, port=port,
        cursorclass=pymysql.cursors.DictCursor,
    )
    schema = DatabaseSchema(dialect="mysql", database=database)

    try:
        with conn.cursor() as cur:
            # Tables
            cur.execute(
                "SELECT table_name, table_rows "
                "FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
                "ORDER BY table_name",
                (database,),
            )
            table_info = {row["table_name"]: row for row in cur.fetchall()}

            for tname in sorted(table_info):
                table = _introspect_mysql_table(
                    cur, tname, database,
                    int(table_info[tname].get("table_rows") or 0),
                    sample_limit,
                )
                schema.tables.append(table)

            for table in schema.tables:
                table.inbound_fk_count = len(schema.inbound_fks(table.name))
    finally:
        conn.close()

    logger.info(
        "Introspected mysql %s: %d tables, %d total FKs",
        database, len(schema.tables),
        sum(len(t.foreign_keys) for t in schema.tables),
    )
    return schema


def _introspect_mysql_table(
    cur, table_name: str, database: str, est_rows: int, sample_limit: int,
) -> Table:
    """Introspect a single MySQL table."""
    table = Table(name=table_name, schema_name=database)
    table.estimated_row_count = est_rows

    # Columns
    cur.execute(
        "SELECT column_name, data_type, is_nullable, column_default, "
        "  ordinal_position, column_key "
        "FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s "
        "ORDER BY ordinal_position",
        (database, table_name),
    )
    pk_cols = []
    for row in cur.fetchall():
        is_pk = row["column_key"] == "PRI"
        col = Column(
            name=row["column_name"],
            db_type=row["data_type"],
            nullable=row["is_nullable"] == "YES",
            default=row["column_default"],
            is_pk=is_pk,
            is_unique=row["column_key"] == "UNI",
            ordinal=row["ordinal_position"],
        )
        if is_pk:
            pk_cols.append(col.name)
        table.columns.append(col)
    table.primary_key = pk_cols

    # Foreign keys
    cur.execute(
        "SELECT constraint_name, column_name, "
        "  referenced_table_name, referenced_column_name "
        "FROM information_schema.key_column_usage "
        "WHERE table_schema = %s AND table_name = %s "
        "  AND referenced_table_name IS NOT NULL "
        "ORDER BY constraint_name, ordinal_position",
        (database, table_name),
    )
    fk_groups: dict[str, dict] = {}
    for row in cur.fetchall():
        cname = row["constraint_name"]
        if cname not in fk_groups:
            fk_groups[cname] = {
                "target_table": row["referenced_table_name"],
                "source_cols": [],
                "target_cols": [],
            }
        fk_groups[cname]["source_cols"].append(row["column_name"])
        fk_groups[cname]["target_cols"].append(row["referenced_column_name"])

    for cname, fk_data in fk_groups.items():
        table.foreign_keys.append(ForeignKey(
            source_table=table_name,
            source_columns=fk_data["source_cols"],
            target_table=fk_data["target_table"],
            target_columns=fk_data["target_cols"],
            constraint_name=cname,
        ))

    # Indexes
    cur.execute(
        "SELECT index_name, column_name, non_unique "
        "FROM information_schema.statistics "
        "WHERE table_schema = %s AND table_name = %s "
        "ORDER BY index_name, seq_in_index",
        (database, table_name),
    )
    idx_groups: dict[str, tuple[list[str], bool]] = {}
    for row in cur.fetchall():
        iname = row["index_name"]
        if iname not in idx_groups:
            idx_groups[iname] = ([], not bool(row["non_unique"]))
        idx_groups[iname][0].append(row["column_name"])

    for iname, (cols, is_unique) in idx_groups.items():
        if iname != "PRIMARY":
            table.indexes.append(cols)
            if is_unique:
                table.unique_constraints.append(cols)
        for col_name in cols:
            col = table.column_by_name(col_name)
            if col:
                col.is_indexed = True

    # Sample rows + profiling
    cur.execute(f"SELECT * FROM `{table_name}` LIMIT %s", (sample_limit,))
    sample_rows = [dict(r) for r in cur.fetchall()]
    table.sample_rows = sample_rows[:20]

    if sample_rows:
        for col in table.columns:
            values = [row.get(col.name) for row in sample_rows]
            col.profile = _profile_column_values(values, col.name)

    return table


# ======================================================================
# MSSQL introspection
# ======================================================================

def introspect_mssql(
    server: str,
    database: str,
    user: str,
    password: str,
    schema_name: str = "dbo",
    sample_limit: int = 200,
) -> DatabaseSchema:
    """Introspect a Microsoft SQL Server database via sys catalog views."""
    try:
        import pymssql
    except ImportError:
        raise ImportError("pip install pymssql for MSSQL introspection")

    conn = pymssql.connect(server=server, user=user, password=password, database=database, as_dict=True)
    schema = DatabaseSchema(dialect="mssql", database=database)

    try:
        with conn.cursor() as cur:
            # Tables
            cur.execute(
                "SELECT t.name AS table_name, "
                "  SUM(p.rows) AS row_estimate "
                "FROM sys.tables t "
                "JOIN sys.schemas s ON t.schema_id = s.schema_id "
                "LEFT JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0, 1) "
                "WHERE s.name = %s "
                "GROUP BY t.name "
                "ORDER BY t.name",
                (schema_name,),
            )
            table_info = {row["table_name"]: row for row in cur.fetchall()}

            for tname in sorted(table_info):
                table = _introspect_mssql_table(
                    cur, tname, schema_name,
                    int(table_info[tname].get("row_estimate") or 0),
                    sample_limit,
                )
                schema.tables.append(table)

            for table in schema.tables:
                table.inbound_fk_count = len(schema.inbound_fks(table.name))
    finally:
        conn.close()

    logger.info(
        "Introspected mssql %s.%s: %d tables, %d total FKs",
        database, schema_name, len(schema.tables),
        sum(len(t.foreign_keys) for t in schema.tables),
    )
    return schema


def _introspect_mssql_table(
    cur, table_name: str, schema_name: str, est_rows: int, sample_limit: int,
) -> Table:
    """Introspect a single MSSQL table."""
    table = Table(name=table_name, schema_name=schema_name)
    table.estimated_row_count = est_rows

    # Columns
    cur.execute(
        "SELECT c.name AS column_name, ty.name AS data_type, "
        "  c.is_nullable, c.column_id, c.max_length "
        "FROM sys.columns c "
        "JOIN sys.types ty ON c.user_type_id = ty.user_type_id "
        "JOIN sys.tables t ON c.object_id = t.object_id "
        "JOIN sys.schemas s ON t.schema_id = s.schema_id "
        "WHERE s.name = %s AND t.name = %s "
        "ORDER BY c.column_id",
        (schema_name, table_name),
    )
    for row in cur.fetchall():
        table.columns.append(Column(
            name=row["column_name"],
            db_type=row["data_type"],
            nullable=bool(row["is_nullable"]),
            ordinal=row["column_id"],
        ))

    # Primary key
    cur.execute(
        "SELECT col.name AS column_name "
        "FROM sys.key_constraints kc "
        "JOIN sys.index_columns ic ON kc.parent_object_id = ic.object_id "
        "  AND kc.unique_index_id = ic.index_id "
        "JOIN sys.columns col ON ic.object_id = col.object_id "
        "  AND ic.column_id = col.column_id "
        "JOIN sys.tables t ON kc.parent_object_id = t.object_id "
        "JOIN sys.schemas s ON t.schema_id = s.schema_id "
        "WHERE s.name = %s AND t.name = %s AND kc.type = 'PK' "
        "ORDER BY ic.key_ordinal",
        (schema_name, table_name),
    )
    pk_cols = [row["column_name"] for row in cur.fetchall()]
    table.primary_key = pk_cols
    for col_name in pk_cols:
        col = table.column_by_name(col_name)
        if col:
            col.is_pk = True

    # Foreign keys
    cur.execute(
        "SELECT fk.name AS constraint_name, "
        "  pc.name AS source_column, "
        "  rt.name AS target_table, "
        "  rc.name AS target_column "
        "FROM sys.foreign_keys fk "
        "JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id "
        "JOIN sys.columns pc ON fkc.parent_object_id = pc.object_id "
        "  AND fkc.parent_column_id = pc.column_id "
        "JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id "
        "JOIN sys.columns rc ON fkc.referenced_object_id = rc.object_id "
        "  AND fkc.referenced_column_id = rc.column_id "
        "JOIN sys.tables pt ON fk.parent_object_id = pt.object_id "
        "JOIN sys.schemas s ON pt.schema_id = s.schema_id "
        "WHERE s.name = %s AND pt.name = %s "
        "ORDER BY fk.name, fkc.constraint_column_id",
        (schema_name, table_name),
    )
    fk_groups: dict[str, dict] = {}
    for row in cur.fetchall():
        cname = row["constraint_name"]
        if cname not in fk_groups:
            fk_groups[cname] = {
                "target_table": row["target_table"],
                "source_cols": [],
                "target_cols": [],
            }
        fk_groups[cname]["source_cols"].append(row["source_column"])
        fk_groups[cname]["target_cols"].append(row["target_column"])

    for cname, fk_data in fk_groups.items():
        table.foreign_keys.append(ForeignKey(
            source_table=table_name,
            source_columns=fk_data["source_cols"],
            target_table=fk_data["target_table"],
            target_columns=fk_data["target_cols"],
            constraint_name=cname,
        ))

    # Sample rows + profiling
    cur.execute(f"SELECT TOP {int(sample_limit)} * FROM [{schema_name}].[{table_name}]")
    sample_rows = [dict(r) for r in cur.fetchall()]
    table.sample_rows = sample_rows[:20]

    if sample_rows:
        for col in table.columns:
            values = [row.get(col.name) for row in sample_rows]
            col.profile = _profile_column_values(values, col.name)

    return table
