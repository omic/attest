"""Table and column classification for introspected database schemas.

Classifies tables as entity, junction, lookup, audit, or internal using
structural heuristics. Infers display columns, entity types, external ID
candidates, and attribute predicates. LLM assists only for ambiguous cases.

Usage::

    from attestdb.discovery.db_introspector import introspect_sqlite
    from attestdb.discovery.db_classifier import classify_schema

    schema = introspect_sqlite("chinook.sqlite")
    classified = classify_schema(schema)
    for tc in classified.tables:
        print(f"{tc.table.name}: {tc.kind} ({tc.confidence:.2f})")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from attestdb.discovery.db_introspector import (
    Column,
    DatabaseSchema,
    ForeignKey,
    Table,
)

logger = logging.getLogger(__name__)

TableKind = Literal["entity", "junction", "lookup", "audit", "internal"]

# Framework/internal table name patterns
_INTERNAL_PATTERNS = [
    r"^sqlite_", r"^django_", r"^alembic_", r"^schema_migrations$",
    r"^flyway_", r"^auth_", r"^cache_", r"^sessions?$", r"^celery_",
    r"^_prisma_", r"__", r"^knex_", r"^typeorm_",
]
_INTERNAL_RE = [re.compile(p, re.I) for p in _INTERNAL_PATTERNS]

# Audit/log table name patterns
_AUDIT_PATTERNS = [
    r"audit", r"_log$", r"_history$", r"_event", r"_changelog",
    r"^activity_", r"_version$",
]
_AUDIT_RE = [re.compile(p, re.I) for p in _AUDIT_PATTERNS]

# Display column name hints (ordered by priority)
_DISPLAY_HINTS = [
    "name", "full_name", "display_name", "title", "label",
    "company_name", "subject", "code", "number",
]

# External ID column name hints
_EXTERNAL_ID_HINTS = [
    "email", "email_address", "mail", "username", "user_name",
    "login", "handle", "domain", "website", "url",
    "phone", "fax", "employee_number", "invoice_number",
]

# Timestamp column hints for incremental sync
_TIMESTAMP_HINTS = [
    "updated_at", "modified_at", "last_modified", "last_updated",
    "changed_at", "modified_date", "update_date",
]
_CREATED_HINTS = ["created_at", "created_date", "creation_date"]

# Entity type inference from table names
_TABLE_TYPE_MAP: dict[str, str] = {
    "employee": "person", "employees": "person",
    "staff": "person", "user": "person", "users": "person",
    "person": "person", "people": "person", "contact": "person",
    "contacts": "person", "member": "person", "members": "person",
    "customer": "person", "customers": "person",
    "artist": "person", "artists": "person",
    "author": "person", "authors": "person",
    "department": "organization", "departments": "organization",
    "team": "organization", "teams": "organization",
    "division": "organization", "org": "organization",
    "company": "company", "companies": "company",
    "organization": "company", "organizations": "company",
    "account": "company", "accounts": "company",
    "client": "company", "clients": "company",
    "vendor": "company", "vendors": "company",
    "supplier": "company", "suppliers": "company",
    "project": "project", "projects": "project",
    "initiative": "project", "campaign": "project",
    "invoice": "invoice", "invoices": "invoice",
    "order": "invoice", "orders": "invoice",
    "payment": "invoice", "payments": "invoice",
    "product": "product", "products": "product",
    "item": "product", "items": "product",
    "sku": "product",
    "ticket": "ticket", "tickets": "ticket",
    "case": "ticket", "cases": "ticket",
    "issue": "ticket", "issues": "ticket",
    "incident": "ticket", "incidents": "ticket",
    "album": "collection", "albums": "collection",
    "playlist": "collection", "playlists": "collection",
    "track": "media", "tracks": "media", "song": "media",
    "genre": "category", "genres": "category",
    "category": "category", "categories": "category",
    "type": "category", "types": "category",
    "skill": "capability", "skills": "capability",
}

# Predicate hints from column name stems
_PREDICATE_HINTS: dict[str, str] = {
    "title": "has_title",
    "name": "has_name",
    "email": "has_email",
    "phone": "has_phone",
    "fax": "has_fax",
    "address": "located_at",
    "city": "located_in",
    "state": "located_in",
    "country": "located_in",
    "salary": "has_salary",
    "budget": "has_budget",
    "amount": "has_amount",
    "total": "has_total",
    "price": "has_price",
    "cost": "has_cost",
    "status": "has_status",
    "priority": "has_priority",
    "category": "categorized_as",
    "industry": "operates_in",
    "description": "described_as",
    "notes": "has_notes",
    "company": "works_at",
    "birthdate": "born_on",
    "birth_date": "born_on",
    "hiredate": "hired_on",
    "hire_date": "hired_on",
    "start_date": "started_on",
    "end_date": "ended_on",
    "due_date": "due_on",
    "close_date": "closes_on",
    "paid_date": "paid_on",
    "created_at": "created_on",
    "composer": "composed_by",
    "milliseconds": "has_duration_ms",
    "bytes": "has_size_bytes",
    "postalcode": "has_postal_code",
    "postal_code": "has_postal_code",
}


@dataclass
class ColumnClassification:
    """Classification of a single column's role."""

    column: Column
    role: Literal["pk", "fk", "display", "attribute", "external_id", "timestamp", "skip"]
    predicate: str | None = None
    external_id_type: str | None = None
    confidence: float = 1.0


@dataclass
class RelationshipIntent:
    """A discovered FK relationship with predicate candidates."""

    fk: ForeignKey
    source_entity_type: str
    target_entity_type: str
    predicate: str
    predicate_candidates: list[str] = field(default_factory=list)
    is_lookup: bool = False
    confidence: float = 0.8


@dataclass
class ClassifiedTable:
    """A table with its classification and inferred semantics."""

    table: Table
    kind: TableKind
    entity_type: str = "entity"
    display_columns: list[str] = field(default_factory=list)
    column_roles: list[ColumnClassification] = field(default_factory=list)
    relationships: list[RelationshipIntent] = field(default_factory=list)
    external_id_columns: list[str] = field(default_factory=list)
    timestamp_column: str | None = None
    confidence: float = 0.9
    evidence: list[str] = field(default_factory=list)


@dataclass
class ClassifiedSchema:
    """Fully classified database schema."""

    source: DatabaseSchema
    tables: list[ClassifiedTable] = field(default_factory=list)

    def by_name(self, name: str) -> ClassifiedTable | None:
        for ct in self.tables:
            if ct.table.name == name:
                return ct
        return None

    @property
    def entity_tables(self) -> list[ClassifiedTable]:
        return [ct for ct in self.tables if ct.kind == "entity"]

    @property
    def junction_tables(self) -> list[ClassifiedTable]:
        return [ct for ct in self.tables if ct.kind == "junction"]

    @property
    def lookup_tables(self) -> list[ClassifiedTable]:
        return [ct for ct in self.tables if ct.kind == "lookup"]

    @property
    def review_items(self) -> list[ClassifiedTable]:
        return [ct for ct in self.tables if ct.confidence < 0.6]


def classify_schema(
    schema: DatabaseSchema, use_llm: bool = False,
) -> ClassifiedSchema:
    """Classify all tables in a DatabaseSchema.

    Args:
        schema: Introspected database schema.
        use_llm: If True, use LLM oracle for low-confidence inferences
                 (entity types, FK predicates, display columns).
    """
    result = ClassifiedSchema(source=schema)

    # Pass 1: classify each table
    for table in schema.tables:
        ct = _classify_table(table, schema)
        result.tables.append(ct)

    # Pass 1b: LLM-assisted entity type resolution for low-confidence tables
    if use_llm:
        _llm_resolve_entity_types(result, schema)

    # Pass 2: resolve relationship predicates using classified entity types
    for ct in result.tables:
        if ct.kind in ("entity", "junction"):
            ct.relationships = _build_relationships(ct, result, schema)

    # Pass 2b: LLM-assisted predicate resolution for low-confidence FKs
    if use_llm:
        _llm_resolve_predicates(result, schema)

    logger.info(
        "Classified %d tables: %d entity, %d junction, %d lookup, "
        "%d audit, %d internal",
        len(result.tables),
        len(result.entity_tables),
        len(result.junction_tables),
        len(result.lookup_tables),
        sum(1 for ct in result.tables if ct.kind == "audit"),
        sum(1 for ct in result.tables if ct.kind == "internal"),
    )
    return result


def _classify_table(table: Table, schema: DatabaseSchema) -> ClassifiedTable:
    """Classify a single table using structural heuristics."""
    # Check internal/framework first
    for pattern in _INTERNAL_RE:
        if pattern.search(table.name):
            return ClassifiedTable(
                table=table, kind="internal",
                confidence=0.99,
                evidence=[f"name matches internal pattern"],
            )

    # Check audit/log
    for pattern in _AUDIT_RE:
        if pattern.search(table.name):
            return ClassifiedTable(
                table=table, kind="audit",
                confidence=0.85,
                evidence=[f"name matches audit pattern"],
            )

    # Check junction table
    if _is_junction(table, schema):
        ct = ClassifiedTable(
            table=table, kind="junction",
            confidence=0.9,
            evidence=["2 FKs, few non-FK columns"],
        )
        ct.entity_type = "association"
        return ct

    # Check lookup table
    if _is_lookup(table, schema):
        ct = ClassifiedTable(
            table=table, kind="lookup",
            confidence=0.85,
            evidence=["small row count, few columns, referenced by others"],
        )
        ct.entity_type = "category"
        ct.display_columns = _find_display_columns(table)
        return ct

    # Default: entity table
    ct = ClassifiedTable(table=table, kind="entity")
    ct.entity_type = _infer_entity_type(table)
    ct.display_columns = _find_display_columns(table)
    ct.external_id_columns = _find_external_ids(table)
    ct.timestamp_column = _find_timestamp_column(table)
    ct.column_roles = _classify_columns(table, ct)
    ct.confidence = 0.9
    ct.evidence = [
        f"{len(table.columns)} columns",
        f"{table.estimated_row_count} rows",
        f"display={ct.display_columns}",
        f"type={ct.entity_type}",
    ]
    return ct


def _is_junction(table: Table, schema: DatabaseSchema) -> bool:
    """Check if table is a junction/bridge table."""
    if len(table.foreign_keys) < 2:
        return False

    # Count distinct target tables
    targets = {fk.target_table for fk in table.foreign_keys if not fk.is_self_ref}
    if len(targets) < 2:
        return False

    # Composite PK made of FK columns?
    if len(table.primary_key) >= 2:
        pk_set = set(table.primary_key)
        fk_cols = table.fk_column_names
        if pk_set <= fk_cols:
            return True

    # Few non-FK, non-PK columns?
    extra = table.non_pk_non_fk_columns
    if len(extra) <= 3:
        return True

    return False


def _is_lookup(table: Table, schema: DatabaseSchema) -> bool:
    """Check if table is a lookup/reference table."""
    if table.estimated_row_count is None:
        return False

    # Small row count
    if table.estimated_row_count > 100:
        return False

    # Few columns (id + label + maybe description)
    if len(table.columns) > 4:
        return False

    # No outbound FKs (or only self-ref)
    outbound = [fk for fk in table.foreign_keys if not fk.is_self_ref]
    if outbound:
        return False

    # Referenced by at least one other table
    inbound = schema.inbound_fks(table.name)
    if not inbound:
        return False

    # Has a likely label/name column
    for col in table.columns:
        if col.name.lower() in ("name", "label", "title", "code", "description"):
            return True

    return False


def _infer_entity_type(table: Table) -> str:
    """Infer entity type from table name."""
    name_lower = table.name.lower()
    if name_lower in _TABLE_TYPE_MAP:
        return _TABLE_TYPE_MAP[name_lower]

    # Try singular form
    if name_lower.endswith("s") and name_lower[:-1] in _TABLE_TYPE_MAP:
        return _TABLE_TYPE_MAP[name_lower[:-1]]

    # Try removing common prefixes/suffixes
    for prefix in ("tbl_", "t_", "dim_", "fact_"):
        stripped = name_lower.removeprefix(prefix)
        if stripped in _TABLE_TYPE_MAP:
            return _TABLE_TYPE_MAP[stripped]

    return "entity"


def _find_display_columns(table: Table) -> list[str]:
    """Find the best display column(s) for a table."""
    candidates: list[tuple[str, float]] = []

    for col in table.columns:
        if col.is_pk:
            continue
        score = _score_display_column(col, table)
        if score > 0:
            candidates.append((col.name, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Check for FirstName + LastName pattern
    col_names_lower = {c.name.lower(): c.name for c in table.columns}
    if "firstname" in col_names_lower and "lastname" in col_names_lower:
        return [col_names_lower["firstname"], col_names_lower["lastname"]]

    if not candidates or candidates[0][1] < 6.0:
        # No strong display candidate — fall back to PK
        return table.primary_key[:1]

    return [candidates[0][0]]


def _score_display_column(col: Column, table: Table) -> float:
    """Score a column as a potential display name."""
    score = 0.0
    name_lower = col.name.lower()

    # Name hint bonus
    for i, hint in enumerate(_DISPLAY_HINTS):
        if hint in name_lower:
            score += 10.0 - i * 0.5
            break

    # String type bonus
    if "char" in col.db_type.lower() or "text" in col.db_type.lower():
        score += 3.0

    # Non-null bonus
    if col.profile and col.profile.null_ratio < 0.05:
        score += 2.0

    # High uniqueness bonus
    if col.profile and col.profile.distinct_ratio > 0.8:
        score += 2.0

    # Readable length (not too short, not too long)
    if col.profile and col.profile.sample_values:
        avg_len = sum(len(s) for s in col.profile.sample_values) / len(col.profile.sample_values)
        if 3 < avg_len < 100:
            score += 1.0

    # Penalty for ID-like names that aren't display
    if name_lower.endswith("_id") or name_lower == "id":
        score -= 5.0

    # Penalty for address sub-fields (billing*, shipping*, mailing*)
    if any(name_lower.startswith(p) for p in (
        "billing", "shipping", "mailing", "postal", "zip",
    )):
        score -= 5.0

    # Penalty for date/time columns as display names
    if "date" in col.db_type.lower() or "time" in col.db_type.lower():
        score -= 3.0

    return score


def _find_external_ids(table: Table) -> list[str]:
    """Find columns that could serve as cross-system identifiers."""
    results = []
    for col in table.columns:
        name_lower = col.name.lower()
        # Name-based detection
        for hint in _EXTERNAL_ID_HINTS:
            if hint in name_lower:
                results.append(col.name)
                break
        else:
            # Regex-based detection from profiled values
            if col.profile and col.profile.regex_hits:
                if col.profile.regex_hits.get("email", 0) > 0:
                    results.append(col.name)
    return results


def _find_timestamp_column(table: Table) -> str | None:
    """Find the best timestamp column for incremental sync."""
    col_names_lower = {c.name.lower(): c.name for c in table.columns}

    # Check update timestamps first
    for hint in _TIMESTAMP_HINTS:
        if hint in col_names_lower:
            return col_names_lower[hint]

    # Fall back to created timestamps
    for hint in _CREATED_HINTS:
        if hint in col_names_lower:
            return col_names_lower[hint]

    return None


def _classify_columns(
    table: Table, ct: ClassifiedTable,
) -> list[ColumnClassification]:
    """Classify each column's role in the entity."""
    pk_set = set(table.primary_key)
    fk_set = table.fk_column_names
    display_set = set(ct.display_columns)
    ext_id_set = set(ct.external_id_columns)
    ts_col = ct.timestamp_column
    results = []

    for col in table.columns:
        if col.name in pk_set:
            results.append(ColumnClassification(column=col, role="pk"))
        elif col.name in fk_set:
            results.append(ColumnClassification(column=col, role="fk"))
        elif col.name in display_set:
            results.append(ColumnClassification(column=col, role="display"))
        elif col.name in ext_id_set:
            ext_type = _infer_external_id_type(col)
            results.append(ColumnClassification(
                column=col, role="external_id",
                external_id_type=ext_type,
            ))
        elif col.name == ts_col:
            results.append(ColumnClassification(column=col, role="timestamp"))
        else:
            predicate = _infer_attribute_predicate(col)
            if predicate:
                results.append(ColumnClassification(
                    column=col, role="attribute",
                    predicate=predicate,
                ))
            else:
                results.append(ColumnClassification(column=col, role="skip"))

    return results


def _infer_external_id_type(col: Column) -> str:
    """Infer the type of external ID from column name/values."""
    name_lower = col.name.lower()
    if "email" in name_lower or "mail" in name_lower:
        return "email"
    if "phone" in name_lower or "fax" in name_lower:
        return "phone"
    if "url" in name_lower or "website" in name_lower:
        return "url"
    if "username" in name_lower or "login" in name_lower:
        return "username"
    return "identifier"


def _infer_attribute_predicate(col: Column) -> str | None:
    """Infer a predicate for an attribute column."""
    name_lower = col.name.lower()

    # Direct match
    if name_lower in _PREDICATE_HINTS:
        return _PREDICATE_HINTS[name_lower]

    # Partial match
    for hint, pred in _PREDICATE_HINTS.items():
        if hint in name_lower:
            return pred

    # Type-based inference
    db_type_lower = col.db_type.lower()
    if "date" in db_type_lower or "time" in db_type_lower:
        return f"has_{name_lower}"
    if "numeric" in db_type_lower or "decimal" in db_type_lower:
        return f"has_{name_lower}"
    if "int" in db_type_lower and not name_lower.endswith("id"):
        return f"has_{name_lower}"

    # String columns that aren't display or external ID
    if "char" in db_type_lower or "text" in db_type_lower:
        return f"has_{name_lower}"

    return None


def _build_relationships(
    ct: ClassifiedTable,
    classified: ClassifiedSchema,
    schema: DatabaseSchema,
) -> list[RelationshipIntent]:
    """Build relationship intents for a table's foreign keys."""
    relationships = []

    for fk in ct.table.foreign_keys:
        target_ct = classified.by_name(fk.target_table)
        if not target_ct:
            continue

        is_lookup = target_ct.kind == "lookup"
        target_type = target_ct.entity_type

        predicate = _infer_fk_predicate(
            fk, ct.entity_type, target_type, is_lookup,
        )
        candidates = _fk_predicate_candidates(
            fk, ct.entity_type, target_type,
        )

        confidence = 0.9 if fk.is_self_ref else 0.85
        if is_lookup:
            confidence = 0.95  # Lookup relationships are structural

        relationships.append(RelationshipIntent(
            fk=fk,
            source_entity_type=ct.entity_type,
            target_entity_type=target_type,
            predicate=predicate,
            predicate_candidates=candidates,
            is_lookup=is_lookup,
            confidence=confidence,
        ))

    return relationships


def _infer_fk_predicate(
    fk: ForeignKey,
    source_type: str,
    target_type: str,
    is_lookup: bool,
) -> str:
    """Infer the best predicate for a FK relationship."""
    col_stem = fk.source_columns[0].lower()

    # Remove _id suffix
    stem = re.sub(r"_?id$", "", col_stem)

    # Self-referential patterns
    if fk.is_self_ref:
        if "manager" in stem or "supervisor" in stem:
            return "reports_to"
        if "parent" in stem:
            return "parent_of"
        if "reportsto" in stem:
            return "reports_to"
        return "reports_to"

    # Lookup table -> use has_<column_stem>
    if is_lookup:
        return f"has_{stem}" if stem else "has_type"

    # Specific FK name patterns
    if "department" in stem:
        if source_type == "person":
            return "works_in"
        return "belongs_to"
    if "lead" in stem or "head" in stem:
        return "led_by"
    if "manager" in stem:
        return "managed_by"
    if "owner" in stem:
        return "owned_by"
    if "assignee" in stem or "assigned" in stem:
        return "assigned_to"
    if "customer" in stem or "client" in stem:
        if source_type == "invoice":
            return "billed_to"
        return "for_customer"
    if "project" in stem:
        return "for_project"
    if "account" in stem:
        if "manager" in col_stem:
            return "managed_by"
        return "belongs_to"
    if "artist" in stem:
        return "created_by"
    if "album" in stem:
        return "appears_on"
    if "support" in stem and "rep" in col_stem:
        return "supported_by"

    # Generic: use target entity type
    if target_type == "person":
        return f"involves_{stem}" if stem else "involves_person"
    if target_type in ("company", "organization"):
        return "belongs_to"

    return f"related_to_{stem}" if stem else "related_to"


def _fk_predicate_candidates(
    fk: ForeignKey,
    source_type: str,
    target_type: str,
) -> list[str]:
    """Generate candidate predicates for review."""
    primary = _infer_fk_predicate(fk, source_type, target_type, False)
    candidates = [primary]

    # Add generic alternatives
    if target_type == "person":
        candidates.extend(["assigned_to", "reported_by", "created_by", "managed_by"])
    elif target_type in ("company", "organization"):
        candidates.extend(["belongs_to", "works_in", "member_of", "part_of"])
    else:
        candidates.extend(["belongs_to", "related_to", "part_of"])

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:5]


# ======================================================================
# LLM-assisted resolution (called only for low-confidence inferences)
# ======================================================================

def _llm_resolve_entity_types(
    classified: ClassifiedSchema,
    schema: DatabaseSchema,
) -> None:
    """Use LLM to resolve ambiguous entity types on low-confidence tables."""
    from attestdb.discovery.db_llm_oracle import resolve_table_type, LLM_THRESHOLD

    for ct in classified.entity_tables:
        if ct.confidence >= LLM_THRESHOLD:
            continue

        # Build FK neighbor list
        fk_neighbors = []
        for fk in ct.table.foreign_keys:
            fk_neighbors.append(fk.target_table)
        for t in schema.tables:
            for fk in t.foreign_keys:
                if fk.target_table == ct.table.name:
                    fk_neighbors.append(t.name)

        sample = ct.table.sample_rows[0] if ct.table.sample_rows else None

        decision = resolve_table_type(
            table_name=ct.table.name,
            columns=[c.name for c in ct.table.columns],
            column_types=[c.db_type for c in ct.table.columns],
            fk_neighbors=fk_neighbors,
            row_count=ct.table.estimated_row_count or 0,
            sample_row=sample,
            rule_guess=ct.entity_type,
            rule_confidence=ct.confidence,
        )

        if decision.source == "llm":
            logger.info(
                "LLM resolved %s entity type: %s -> %s (conf=%.2f, %s)",
                ct.table.name, ct.entity_type, decision.chosen,
                decision.confidence, decision.reasoning,
            )
            ct.entity_type = decision.chosen
            ct.confidence = decision.confidence
            ct.evidence.append(f"llm: {decision.reasoning}")


def _llm_resolve_predicates(
    classified: ClassifiedSchema,
    schema: DatabaseSchema,
) -> None:
    """Use LLM to resolve ambiguous FK predicates on low-confidence relationships."""
    from attestdb.discovery.db_llm_oracle import resolve_fk_predicate, LLM_THRESHOLD

    for ct in classified.entity_tables:
        for rel in ct.relationships:
            if rel.confidence >= LLM_THRESHOLD:
                continue

            target_ct = classified.by_name(rel.fk.target_table)
            if not target_ct:
                continue

            # Get sample rows
            src_sample = ct.table.sample_rows[0] if ct.table.sample_rows else None
            tgt_sample = target_ct.table.sample_rows[0] if target_ct.table.sample_rows else None

            decision = resolve_fk_predicate(
                source_table=ct.table.name,
                target_table=rel.fk.target_table,
                fk_column=rel.fk.source_columns[0],
                target_column=rel.fk.target_columns[0],
                source_type=ct.entity_type,
                target_type=target_ct.entity_type,
                source_columns=[c.name for c in ct.table.columns],
                target_columns=[c.name for c in target_ct.table.columns],
                source_sample=src_sample,
                target_sample=tgt_sample,
                rule_guess=rel.predicate,
                rule_confidence=rel.confidence,
                candidates=rel.predicate_candidates,
            )

            if decision.source == "llm":
                logger.info(
                    "LLM resolved %s.%s -> %s: %s -> %s (conf=%.2f, %s)",
                    ct.table.name, rel.fk.source_columns[0],
                    rel.fk.target_table,
                    rel.predicate, decision.chosen,
                    decision.confidence, decision.reasoning,
                )
                rel.predicate = decision.chosen
                rel.confidence = decision.confidence
