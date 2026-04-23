"""Extraction plan compiler — turns classified schema into SQL + mapping plans.

Takes a ClassifiedSchema and produces ExtractionPlan objects that compile
down to (query, mapping) pairs compatible with QueryConnector.

Usage::

    from attestdb.discovery.db_compiler import compile_extraction_plans

    plans = compile_extraction_plans(classified_schema)
    for plan in plans:
        print(f"{plan.kind}: {plan.description}")
        print(f"  SQL: {plan.query[:80]}...")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from attestdb.discovery.db_classifier import (
    ClassifiedSchema,
    ClassifiedTable,
    RelationshipIntent,
)

logger = logging.getLogger(__name__)


def _quoter(dialect: str):
    """Return a quoting function for the given SQL dialect."""
    if dialect in ("postgres",):
        return lambda name: f'"{name}"'
    if dialect in ("mysql",):
        return lambda name: f"`{name}`"
    # sqlite, mssql — bracket quoting
    return lambda name: f"[{name}]"

PlanKind = Literal[
    "entity_attributes",
    "direct_relationship",
    "self_ref_relationship",
    "junction_relationship",
    "lookup_relationship",
]


@dataclass
class ExtractionPlan:
    """A single extraction job: query + mapping + metadata."""

    kind: PlanKind
    description: str
    query: str
    source_table: str
    entity_type: str
    display_columns: list[str]
    predicate: str | None = None
    target_table: str | None = None
    target_entity_type: str | None = None
    target_display_columns: list[str] = field(default_factory=list)
    attribute_predicates: list[tuple[str, str]] = field(default_factory=list)
    external_id_columns: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.9
    timestamp_column: str | None = None

    @property
    def provenance_id(self) -> str:
        """Source identifier for claim provenance."""
        if self.target_table:
            return f"db:{self.source_table}.{self.predicate}->{self.target_table}"
        return f"db:{self.source_table}"


@dataclass
class ReviewItem:
    """An uncertain inference that needs human review."""

    table: str
    field: str
    kind: str
    chosen: str
    candidates: list[str]
    confidence: float
    reason: str


@dataclass
class CompilationResult:
    """Output of the extraction plan compiler."""

    plans: list[ExtractionPlan] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)


def compile_extraction_plans(
    classified: ClassifiedSchema,
) -> CompilationResult:
    """Compile a classified schema into extraction plans."""
    result = CompilationResult()
    q = _quoter(classified.source.dialect)

    for ct in classified.entity_tables:
        # Entity attribute plan
        attr_plan = _compile_entity_attributes(ct, classified, q)
        if attr_plan:
            result.plans.append(attr_plan)

        # Relationship plans (one per FK)
        for rel in ct.relationships:
            plan = _compile_relationship(ct, rel, classified, q)
            if plan:
                result.plans.append(plan)

            # Surface low-confidence predicates for review
            if rel.confidence < 0.8:
                result.review_items.append(ReviewItem(
                    table=ct.table.name,
                    field=rel.fk.source_columns[0],
                    kind="predicate",
                    chosen=rel.predicate,
                    candidates=rel.predicate_candidates,
                    confidence=rel.confidence,
                    reason=(
                        f"{ct.entity_type}.{rel.fk.source_columns[0]} "
                        f"-> {rel.target_entity_type}"
                    ),
                ))

    # Junction table plans
    for ct in classified.junction_tables:
        plans = _compile_junction(ct, classified, q)
        result.plans.extend(plans)

    logger.info(
        "Compiled %d extraction plans (%d entity, %d relationship, %d junction), "
        "%d review items",
        len(result.plans),
        sum(1 for p in result.plans if p.kind == "entity_attributes"),
        sum(1 for p in result.plans if p.kind in (
            "direct_relationship", "self_ref_relationship", "lookup_relationship",
        )),
        sum(1 for p in result.plans if p.kind == "junction_relationship"),
        len(result.review_items),
    )
    return result


def _display_expr(table_alias: str, display_cols: list[str], q) -> str:
    """Build a SQL expression for the display name."""
    if len(display_cols) == 0:
        return f"{table_alias}.rowid"
    if len(display_cols) == 1:
        return f"{table_alias}.{q(display_cols[0])}"
    # Concatenate multiple columns (e.g., FirstName || ' ' || LastName)
    parts = [f"{table_alias}.{q(c)}" for c in display_cols]
    return " || ' ' || ".join(parts)


def _compile_entity_attributes(
    ct: ClassifiedTable,
    classified: ClassifiedSchema,
    q,
) -> ExtractionPlan | None:
    """Compile an entity attribute extraction plan."""
    # Collect attribute columns
    attrs = []
    for cr in ct.column_roles:
        if cr.role == "attribute" and cr.predicate:
            attrs.append((cr.column.name, cr.predicate))

    if not attrs and not ct.display_columns:
        return None

    # Build SELECT
    alias = "t"
    pk_col = ct.table.primary_key[0] if ct.table.primary_key else "rowid"
    display_expr = _display_expr(alias, ct.display_columns, q)

    select_parts = [
        f"{alias}.{q(pk_col)} AS subject_pk",
        f"{display_expr} AS subject_display",
    ]

    # External IDs
    ext_id_map: dict[str, str] = {}
    for cr in ct.column_roles:
        if cr.role == "external_id" and cr.external_id_type:
            col_alias = f"ext_{cr.external_id_type}"
            select_parts.append(f"{alias}.{q(cr.column.name)} AS {col_alias}")
            ext_id_map[cr.external_id_type] = col_alias

    # Attribute columns
    for col_name, predicate in attrs:
        select_parts.append(f"{alias}.{q(col_name)} AS {q(col_name)}")

    # Timestamp
    ts_col = ct.timestamp_column
    if ts_col:
        select_parts.append(f"{alias}.{q(ts_col)} AS record_ts")

    query = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {q(ct.table.name)} {alias}"
    )

    return ExtractionPlan(
        kind="entity_attributes",
        description=f"{ct.table.name} entity attributes ({len(attrs)} attrs)",
        query=query,
        source_table=ct.table.name,
        entity_type=ct.entity_type,
        display_columns=ct.display_columns,
        attribute_predicates=attrs,
        external_id_columns=ext_id_map,
        confidence=ct.confidence,
        timestamp_column="record_ts" if ts_col else None,
    )


def _compile_relationship(
    ct: ClassifiedTable,
    rel: RelationshipIntent,
    classified: ClassifiedSchema,
    q,
) -> ExtractionPlan | None:
    """Compile a FK relationship extraction plan."""
    target_ct = classified.by_name(rel.fk.target_table)
    if not target_ct:
        return None

    src_alias = "s"
    tgt_alias = "t"
    src_pk = ct.table.primary_key[0] if ct.table.primary_key else "rowid"
    tgt_pk = target_ct.table.primary_key[0] if target_ct.table.primary_key else "rowid"
    src_display = _display_expr(src_alias, ct.display_columns, q)
    tgt_display = _display_expr(tgt_alias, target_ct.display_columns, q)

    # FK column(s)
    fk_col = rel.fk.source_columns[0]
    tgt_col = rel.fk.target_columns[0]

    # Join type
    fk_column_obj = ct.table.column_by_name(fk_col)
    join_type = "LEFT JOIN" if (fk_column_obj and fk_column_obj.nullable) else "INNER JOIN"

    # Determine plan kind
    if rel.fk.is_self_ref:
        kind: PlanKind = "self_ref_relationship"
    elif rel.is_lookup:
        kind = "lookup_relationship"
    else:
        kind = "direct_relationship"

    # Build SELECT
    select_parts = [
        f"{src_alias}.{q(src_pk)} AS subject_pk",
        f"{src_display} AS subject_display",
        f"{tgt_alias}.{q(tgt_pk)} AS object_pk",
        f"{tgt_display} AS object_display",
    ]

    # Source external IDs
    ext_id_map: dict[str, str] = {}
    for cr in ct.column_roles:
        if cr.role == "external_id" and cr.external_id_type:
            col_alias = f"subj_ext_{cr.external_id_type}"
            select_parts.append(
                f"{src_alias}.{q(cr.column.name)} AS {col_alias}"
            )
            ext_id_map[cr.external_id_type] = col_alias

    query = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {q(ct.table.name)} {src_alias} "
        f"{join_type} {q(rel.fk.target_table)} {tgt_alias} "
        f"ON {src_alias}.{q(fk_col)} = {tgt_alias}.{q(tgt_col)} "
        f"WHERE {src_alias}.{q(fk_col)} IS NOT NULL"
    )

    return ExtractionPlan(
        kind=kind,
        description=(
            f"{ct.table.name}.{fk_col} -> {rel.fk.target_table} "
            f"({rel.predicate})"
        ),
        query=query,
        source_table=ct.table.name,
        entity_type=ct.entity_type,
        display_columns=ct.display_columns,
        predicate=rel.predicate,
        target_table=rel.fk.target_table,
        target_entity_type=rel.target_entity_type,
        target_display_columns=target_ct.display_columns,
        external_id_columns=ext_id_map,
        confidence=rel.confidence,
    )


def _compile_junction(
    ct: ClassifiedTable,
    classified: ClassifiedSchema,
    q,
) -> list[ExtractionPlan]:
    """Compile junction table into relationship plans."""
    if len(ct.table.foreign_keys) < 2:
        return []

    plans = []
    fks = ct.table.foreign_keys

    # Get the two parent tables
    left_fk = fks[0]
    right_fk = fks[1]

    left_ct = classified.by_name(left_fk.target_table)
    right_ct = classified.by_name(right_fk.target_table)
    if not left_ct or not right_ct:
        return []

    left_pk = left_ct.table.primary_key[0] if left_ct.table.primary_key else "rowid"
    right_pk = right_ct.table.primary_key[0] if right_ct.table.primary_key else "rowid"
    left_display = _display_expr("l", left_ct.display_columns, q)
    right_display = _display_expr("r", right_ct.display_columns, q)

    # Infer predicate from junction context
    predicate = _junction_predicate(left_ct, right_ct, ct.table.name)

    # Build the join query
    select_parts = [
        f"l.{q(left_pk)} AS subject_pk",
        f"{left_display} AS subject_display",
        f"r.{q(right_pk)} AS object_pk",
        f"{right_display} AS object_display",
    ]

    # Extra columns from junction table (non-FK, non-PK)
    extra_cols = ct.table.non_pk_non_fk_columns
    for col in extra_cols:
        select_parts.append(f"j.{q(col.name)} AS {q(col.name)}")

    query = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM {q(ct.table.name)} j "
        f"INNER JOIN {q(left_fk.target_table)} l "
        f"ON j.{q(left_fk.source_columns[0])} = l.{q(left_fk.target_columns[0])} "
        f"INNER JOIN {q(right_fk.target_table)} r "
        f"ON j.{q(right_fk.source_columns[0])} = r.{q(right_fk.target_columns[0])}"
    )

    extra_predicates = [
        (col.name, f"has_{col.name.lower()}")
        for col in extra_cols
    ]

    plans.append(ExtractionPlan(
        kind="junction_relationship",
        description=(
            f"{ct.table.name}: {left_fk.target_table} "
            f"{predicate} {right_fk.target_table}"
        ),
        query=query,
        source_table=ct.table.name,
        entity_type=left_ct.entity_type,
        display_columns=left_ct.display_columns,
        predicate=predicate,
        target_table=right_fk.target_table,
        target_entity_type=right_ct.entity_type,
        target_display_columns=right_ct.display_columns,
        attribute_predicates=extra_predicates,
        confidence=0.85,
    ))

    return plans


def _junction_predicate(
    left_ct: ClassifiedTable,
    right_ct: ClassifiedTable,
    junction_name: str,
) -> str:
    """Infer predicate for a junction table relationship."""
    left_type = left_ct.entity_type
    right_type = right_ct.entity_type
    jname = junction_name.lower()

    # Specific patterns
    if "skill" in jname:
        return "has_skill"
    if "assign" in jname:
        return "assigned_to"
    if "member" in jname:
        return "member_of"
    if "playlist" in jname and "track" in jname:
        return "contains_track"
    if "invoice" in jname and "line" in jname:
        return "includes_item"

    # Generic: left "has" right or left "related_to" right
    if right_type in ("media", "product", "capability"):
        return f"includes_{right_ct.table.name.lower()}"
    if right_type == "person":
        return "involves_person"

    return "related_to"
