"""Database extraction orchestrator — executes compiled plans into claims.

Supports SQLite, PostgreSQL, MySQL, and MSSQL. Each engine has a convenience
wrapper; the core pipeline (classify → compile → execute) is engine-agnostic.

Usage::

    from attestdb.discovery.db_extract import extract_sqlite, extract_postgres

    result = extract_sqlite("chinook.sqlite", db)
    result = extract_postgres("postgresql://user:pass@host/db", db)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Iterator

from attestdb.discovery.db_introspector import DatabaseSchema
from attestdb.discovery.db_classifier import classify_schema
from attestdb.discovery.db_compiler import compile_extraction_plans, ExtractionPlan

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)


def _run_pipeline(
    schema: DatabaseSchema,
    execute_query: Callable[[str], list[dict]],
    db: AttestDB,
    t0: float,
    use_llm: bool = False,
) -> dict:
    """Core pipeline: classify → compile → execute. Engine-agnostic."""
    classified = classify_schema(schema, use_llm=use_llm)
    compilation = compile_extraction_plans(classified)

    plan_results = []
    total_claims = 0

    for plan in compilation.plans:
        try:
            rows = execute_query(plan.query)
        except Exception as exc:
            logger.warning("Plan %s failed: %s", plan.description, exc)
            plan_results.append({
                "kind": plan.kind,
                "description": plan.description,
                "claims_ingested": 0,
                "confidence": plan.confidence,
                "error": str(exc),
            })
            continue

        claims_count = _ingest_rows(rows, plan, db)
        plan_results.append({
            "kind": plan.kind,
            "description": plan.description,
            "claims_ingested": claims_count,
            "confidence": plan.confidence,
        })
        total_claims += claims_count

    elapsed = time.monotonic() - t0

    return {
        "database": schema.database,
        "dialect": schema.dialect,
        "tables_total": len(schema.tables),
        "tables_scanned": len(classified.entity_tables) + len(classified.junction_tables),
        "entity_tables": len(classified.entity_tables),
        "junction_tables": len(classified.junction_tables),
        "lookup_tables": len(classified.lookup_tables),
        "extraction_plans": len(compilation.plans),
        "total_claims": total_claims,
        "plan_results": plan_results,
        "review_items": [
            {
                "table": ri.table,
                "field": ri.field,
                "kind": ri.kind,
                "chosen": ri.chosen,
                "candidates": ri.candidates,
                "confidence": ri.confidence,
            }
            for ri in compilation.review_items
        ],
        "classifications": {
            ct.table.name: {
                "kind": ct.kind,
                "entity_type": ct.entity_type,
                "display_columns": ct.display_columns,
                "confidence": ct.confidence,
            }
            for ct in classified.tables
        },
        "elapsed_s": round(elapsed, 2),
    }


# ======================================================================
# SQLite
# ======================================================================

def extract_sqlite(
    path: str,
    db: AttestDB,
    sample_limit: int = 200,
    use_llm: bool = False,
) -> dict:
    """Full pipeline for SQLite: introspect → classify → compile → extract.

    Args:
        use_llm: If True, use LLM oracle for ambiguous predicates and entity types.
    """
    import sqlite3
    from attestdb.discovery.db_introspector import introspect_sqlite

    t0 = time.monotonic()
    schema = introspect_sqlite(path, sample_limit=sample_limit)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    def execute_query(sql: str) -> list[dict]:
        return [dict(row) for row in conn.execute(sql)]

    try:
        return _run_pipeline(schema, execute_query, db, t0, use_llm=use_llm)
    finally:
        conn.close()


# ======================================================================
# PostgreSQL
# ======================================================================

def extract_postgres(
    dsn: str,
    db: AttestDB,
    schema_name: str = "public",
    sample_limit: int = 200,
) -> dict:
    """Full pipeline for PostgreSQL: introspect → classify → compile → extract."""
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        raise ImportError("pip install psycopg2-binary for PostgreSQL extraction")

    from attestdb.discovery.db_introspector import introspect_postgres

    t0 = time.monotonic()
    schema = introspect_postgres(dsn, schema_name=schema_name, sample_limit=sample_limit)

    conn = psycopg2.connect(dsn)

    def execute_query(sql: str) -> list[dict]:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    try:
        return _run_pipeline(schema, execute_query, db, t0)
    finally:
        conn.close()


# ======================================================================
# MySQL
# ======================================================================

def extract_mysql(
    host: str,
    user: str,
    password: str,
    database: str,
    db: AttestDB,
    port: int = 3306,
    sample_limit: int = 200,
) -> dict:
    """Full pipeline for MySQL: introspect → classify → compile → extract."""
    try:
        import pymysql
        import pymysql.cursors
    except ImportError:
        raise ImportError("pip install pymysql for MySQL extraction")

    from attestdb.discovery.db_introspector import introspect_mysql

    t0 = time.monotonic()
    schema = introspect_mysql(
        host=host, user=user, password=password,
        database=database, port=port, sample_limit=sample_limit,
    )

    conn = pymysql.connect(
        host=host, user=user, password=password,
        database=database, port=port,
        cursorclass=pymysql.cursors.DictCursor,
    )

    def execute_query(sql: str) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    try:
        return _run_pipeline(schema, execute_query, db, t0)
    finally:
        conn.close()


# ======================================================================
# MSSQL
# ======================================================================

def extract_mssql(
    server: str,
    database: str,
    user: str,
    password: str,
    db: AttestDB,
    schema_name: str = "dbo",
    sample_limit: int = 200,
) -> dict:
    """Full pipeline for MSSQL: introspect → classify → compile → extract."""
    try:
        import pymssql
    except ImportError:
        raise ImportError("pip install pymssql for MSSQL extraction")

    from attestdb.discovery.db_introspector import introspect_mssql

    t0 = time.monotonic()
    schema = introspect_mssql(
        server=server, database=database, user=user, password=password,
        schema_name=schema_name, sample_limit=sample_limit,
    )

    conn = pymssql.connect(
        server=server, user=user, password=password,
        database=database, as_dict=True,
    )

    def execute_query(sql: str) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]

    try:
        return _run_pipeline(schema, execute_query, db, t0)
    finally:
        conn.close()


# ======================================================================
# Claim extraction helpers (shared across all engines)
# ======================================================================

def _ingest_rows(
    rows: list[dict],
    plan: ExtractionPlan,
    db: AttestDB,
) -> int:
    """Convert rows to claims and ingest them."""
    if not rows:
        return 0

    claims = []
    if plan.kind == "entity_attributes":
        claims = _extract_entity_attributes(rows, plan)
    elif plan.kind in (
        "direct_relationship", "self_ref_relationship", "lookup_relationship",
    ):
        claims = _extract_relationship(rows, plan)
    elif plan.kind == "junction_relationship":
        claims = _extract_junction(rows, plan)

    if not claims:
        return 0

    result = db.ingest_batch(claims)
    return result.ingested


def _make_display_name(row: dict, key: str = "subject_display") -> str:
    """Extract display name from a row, handling None."""
    val = row.get(key)
    if val is None:
        pk_val = row.get(key.replace("_display", "_pk"), "")
        return str(pk_val)
    return str(val)


def _build_external_ids(
    row: dict, ext_map: dict[str, str],
) -> dict[str, str]:
    """Build external_ids dict from a row using the plan's ext_id mapping."""
    ext: dict[str, str] = {}
    for ext_type, col_alias in ext_map.items():
        val = row.get(col_alias)
        if val is not None and str(val).strip():
            ext[ext_type] = str(val)
    return ext


def _extract_entity_attributes(rows: list[dict], plan: ExtractionPlan) -> list:
    """Extract entity attribute claims from query results."""
    from attestdb.core.types import ClaimInput

    claims = []
    for row in rows:
        subject = _make_display_name(row, "subject_display")
        if not subject.strip():
            continue

        subj_ext = _build_external_ids(row, plan.external_id_columns)

        for col_name, predicate in plan.attribute_predicates:
            value = row.get(col_name)
            if value is None or str(value).strip() == "":
                continue

            ext_ids = {"subject": subj_ext} if subj_ext else None
            claims.append(ClaimInput(
                subject=(subject, plan.entity_type),
                predicate=(predicate, _predicate_type(predicate)),
                object=(str(value), "entity"),
                provenance={
                    "source_type": "database_import",
                    "source_id": f"{plan.provenance_id}:{col_name}",
                },
                confidence=plan.confidence,
                external_ids=ext_ids,
            ))

    return claims


def _extract_relationship(rows: list[dict], plan: ExtractionPlan) -> list:
    """Extract relationship claims from FK join results."""
    from attestdb.core.types import ClaimInput

    claims = []
    for row in rows:
        subject = _make_display_name(row, "subject_display")
        obj = _make_display_name(row, "object_display")
        if not subject.strip() or not obj.strip():
            continue

        subj_ext = _build_external_ids(row, plan.external_id_columns)
        ext_ids = {"subject": subj_ext} if subj_ext else None

        claims.append(ClaimInput(
            subject=(subject, plan.entity_type),
            predicate=(plan.predicate, _predicate_type(plan.predicate)),
            object=(obj, plan.target_entity_type or "entity"),
            provenance={
                "source_type": "database_import",
                "source_id": plan.provenance_id,
            },
            confidence=plan.confidence,
            external_ids=ext_ids,
        ))

    return claims


def _extract_junction(rows: list[dict], plan: ExtractionPlan) -> list:
    """Extract junction relationship claims (plus extra column claims)."""
    from attestdb.core.types import ClaimInput

    claims = []
    for row in rows:
        subject = _make_display_name(row, "subject_display")
        obj = _make_display_name(row, "object_display")
        if not subject.strip() or not obj.strip():
            continue

        claims.append(ClaimInput(
            subject=(subject, plan.entity_type),
            predicate=(plan.predicate, _predicate_type(plan.predicate)),
            object=(obj, plan.target_entity_type or "entity"),
            provenance={
                "source_type": "database_import",
                "source_id": plan.provenance_id,
            },
            confidence=plan.confidence,
        ))

        for col_name, predicate in plan.attribute_predicates:
            value = row.get(col_name)
            if value is None or str(value).strip() == "":
                continue

            assoc_id = f"{subject}::{obj}"
            claims.append(ClaimInput(
                subject=(assoc_id, "association"),
                predicate=(predicate, _predicate_type(predicate)),
                object=(str(value), "entity"),
                provenance={
                    "source_type": "database_import",
                    "source_id": f"{plan.provenance_id}:{col_name}",
                },
                confidence=plan.confidence,
            ))

    return claims


def _predicate_type(predicate: str) -> str:
    """Map a predicate to its type category."""
    try:
        from attestdb.connectors.predicates import predicate_type
        return predicate_type(predicate)
    except (ImportError, KeyError):
        if predicate.startswith("has_") or predicate.startswith("is_"):
            return "has_attribute"
        if predicate in (
            "reports_to", "works_in", "belongs_to", "assigned_to",
            "created_by", "managed_by", "led_by", "supported_by",
            "billed_to", "appears_on",
        ):
            return "directional"
        return "relates_to"
