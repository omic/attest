"""Schema Reasoner — bridges schema discovery to connector mapping configs.

Closes five gaps between the discovery module (FieldProfile, SemanticMapping,
SchemaMap) and the connector framework (QueryConnector/CSV/HTTP mapping dicts):

1. schema_map_to_mapping() — convert a SchemaMap into a connector mapping dict
2. propose_mapping() — LLM generates a mapping from schema + vocabulary
3. export_vocabulary_for_llm() — vocabulary + constraints in LLM-friendly format
4. validate_mapping() — check a mapping against vocabulary constraints
5. execute_mapping() — unified entry point: data source + mapping → ClaimInput list

Uses the existing cheap LLM fallback chain (groq/gemini) — not Anthropic.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

from attestdb.core.types import ClaimInput
from attestdb.discovery.analyzer import SemanticMapping
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)


# ── Path resolution for nested data ──────────────────────────────────


def _resolve_path(obj, path: str):
    """Resolve a dot-separated path against a dict/list structure.

    Falls back to flat key lookup when path has no dots, so this is
    backward-compatible with row.get(key) for flat CSV/SQL data.

    >>> _resolve_path({"a": {"b": 1}}, "a.b")
    1
    >>> _resolve_path({"name": "Acme"}, "name")
    'Acme'
    """
    if not path:
        return obj
    # Fast path: no dots means flat key lookup
    if "." not in path and isinstance(obj, dict):
        return obj.get(path)
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, (list, tuple)) and key.isdigit():
            obj = obj[int(key)]
        else:
            return None
        if obj is None:
            return None
    return obj


# ── Semantic type → predicate mapping ────────────────────────────────
# Maps discovery semantic types to AttestDB predicates and entity types.

_SEMANTIC_TO_PREDICATE: dict[str, str] = {
    "revenue": "has_metric",
    "revenue.arr": "has_metric",
    "revenue.mrr": "has_metric",
    "churn": "has_metric",
    "churn.rate": "has_metric",
    "satisfaction": "has_metric",
    "satisfaction.nps": "has_metric",
    "satisfaction.health": "has_metric",
    "pipeline.stage": "has_status",
    "status": "has_status",
    "ownership": "assigned_to",
    "classification.industry": "classified_as",
    "classification.segment": "classified_as",
    "classification.type": "classified_as",
    "geography": "located_in",
    "contact.email": "has_attribute",
    "contact.phone": "has_attribute",
    "entity.name": "has_attribute",
    "reference.url": "has_attribute",
    "text.description": "has_attribute",
    "financial.amount": "has_metric",
    "metric.count": "has_metric",
    "metric.score": "has_metric",
    "contract.value": "has_metric",
    "renewal": "has_status",
}

_SEMANTIC_TO_ENTITY_TYPE: dict[str, str] = {
    "ownership": "person",
    "geography": "location",
    "contact.email": "contact",
    "contact.phone": "contact",
}


# ── Gap 1: Schema-to-mapping bridge ──────────────────────────────────


def schema_map_to_mapping(
    schema_map: SchemaMap,
    *,
    subject_field: str | None = None,
    min_confidence: float = 0.6,
) -> list[dict]:
    """Convert a SchemaMap into connector mapping dicts.

    For each semantic mapping with sufficient confidence, generates a
    mapping dict compatible with QueryConnector/CSVConnector/HTTPConnector.

    Args:
        schema_map: Discovered schema with field profiles and semantic mappings.
        subject_field: Override for the subject column. If None, auto-detects
            the best entity key field (ID-like column with high cardinality).
        min_confidence: Minimum semantic mapping confidence to include.

    Returns:
        List of mapping dicts, one per extractable claim pattern.
    """
    # Auto-detect subject field if not provided
    if subject_field is None:
        subject_field = _detect_subject_field(schema_map)

    if not subject_field:
        logger.warning("No subject field detected in schema %s", schema_map.source_id)
        return []

    mappings = []
    for sm in schema_map.semantic_mappings:
        if sm.confidence < min_confidence:
            continue
        if sm.semantic_type in ("unknown", "identifier"):
            continue
        if sm.field_name == subject_field:
            continue

        predicate = _SEMANTIC_TO_PREDICATE.get(sm.semantic_type, "has_attribute")
        object_type = _SEMANTIC_TO_ENTITY_TYPE.get(sm.semantic_type, "entity")

        mapping = {
            "subject": subject_field,
            "subject_type": "entity",
            "predicate": sm.field_name,
            "predicate_type": predicate,
            "object": sm.field_name,
            "object_type": object_type,
        }
        mappings.append(mapping)

    return mappings


def _detect_subject_field(schema_map: SchemaMap) -> str | None:
    """Find the best entity key field in a schema map."""
    import re
    _ID_PATTERNS = [
        re.compile(r"(?:^|[_\s])id$", re.IGNORECASE),
        re.compile(r"_id$", re.IGNORECASE),
        re.compile(r"(?:account|customer|contact|user|org|company|name)(?:_?id)?$", re.IGNORECASE),
    ]

    # First: check semantic mappings for "identifier" or "entity.name"
    for sm in schema_map.semantic_mappings:
        if sm.semantic_type == "entity.name" and sm.confidence > 0.7:
            return sm.field_name

    # Second: check field profiles for ID-like patterns with high cardinality
    best_candidate = None
    best_score = 0.0
    for fp in schema_map.field_profiles:
        score = 0.0
        for pattern in _ID_PATTERNS:
            if pattern.search(fp.field_name):
                score = fp.fill_rate * min(fp.cardinality / 10.0, 1.0)
                break
        if score > best_score:
            best_score = score
            best_candidate = fp.field_name

    # Third: fall back to first column with "name" in it
    if best_candidate is None:
        for fp in schema_map.field_profiles:
            if "name" in fp.field_name.lower():
                return fp.field_name

    return best_candidate


# ── Enterprise object templates ────────────────────────────────────────
# Reference patterns for the LLM — what good enterprise mappings look like.
# Each template: field patterns that signal this object type, expected
# predicates, and external_ids hints for cross-system entity resolution.

_ENTERPRISE_TEMPLATES: dict[str, dict] = {
    "deal_opportunity": {
        "description": "Sales deal, opportunity, or quote",
        "field_signals": ["stage", "stagename", "amount", "close", "pipeline",
                          "opportunity", "deal", "quote", "forecast"],
        "subject_type": "opportunity",
        "expected_mappings": [
            {"predicate": "has_status", "fields": "stage, stagename, status, pipeline_stage"},
            {"predicate": "has_metric", "fields": "amount, value, revenue, arr, deal_size"},
            {"predicate": "owned_by", "fields": "owner, rep, account_executive, salesperson",
             "object_type": "person"},
            {"predicate": "belongs_to", "fields": "account, company, customer, organization",
             "object_type": "company"},
            {"predicate": "closes_on", "fields": "close_date, expected_close, end_date"},
        ],
        "skip_fields": "id, created_date, modified_date, system_modstamp, created_by_id",
    },
    "ticket_case": {
        "description": "Support ticket, case, incident, or service request",
        "field_signals": ["priority", "assignee", "reporter", "incident",
                          "ticket", "case", "severity", "resolution"],
        "subject_type": "ticket",
        "expected_mappings": [
            {"predicate": "has_status", "fields": "status, state, resolution, phase"},
            {"predicate": "has_priority", "fields": "priority, severity, urgency, impact"},
            {"predicate": "assigned_to", "fields": "assignee, assigned_to, owner, handler",
             "object_type": "person"},
            {"predicate": "reported_by", "fields": "reporter, opened_by, requester, creator",
             "object_type": "person"},
            {"predicate": "classified_as", "fields": "category, type, issue_type, service"},
        ],
        "skip_fields": "id, created, updated, sys_id, number",
    },
    "contact_person": {
        "description": "Contact, person, user, or team member",
        "field_signals": ["email", "first_name", "last_name", "title",
                          "job_title", "phone", "role"],
        "subject_type": "person",
        "expected_mappings": [
            {"predicate": "works_at", "fields": "company, organization, account, employer",
             "object_type": "company"},
            {"predicate": "has_role", "fields": "title, job_title, role, position"},
            {"predicate": "has_status", "fields": "status, lifecycle_stage, lead_status"},
            {"predicate": "part_of", "fields": "department, team, group, division"},
            {"predicate": "located_in", "fields": "city, country, region, office",
             "object_type": "location"},
        ],
        "skip_fields": "id, created_date, modified_date, photo_url",
        "external_ids_hint": "Include person_name and email as external_ids for "
                             "cross-system entity resolution.",
    },
    "account_company": {
        "description": "Account, company, organization, or customer",
        "field_signals": ["industry", "employees", "annual_revenue", "website",
                          "account", "company", "organization"],
        "subject_type": "company",
        "expected_mappings": [
            {"predicate": "classified_as", "fields": "industry, sector, segment, type"},
            {"predicate": "has_metric", "fields": "employees, headcount, revenue, arr, size"},
            {"predicate": "located_in", "fields": "country, region, city, headquarters",
             "object_type": "location"},
            {"predicate": "owned_by", "fields": "owner, account_manager, csm",
             "object_type": "person"},
            {"predicate": "has_status", "fields": "status, tier, health_score, rating"},
        ],
        "skip_fields": "id, created_date, modified_date, website, logo_url",
    },
    "employee": {
        "description": "Employee, team member, or workforce record",
        "field_signals": ["manager", "hire_date", "salary", "department",
                          "employee", "worker", "staff", "compensation"],
        "subject_type": "person",
        "expected_mappings": [
            {"predicate": "part_of", "fields": "department, team, division, business_unit"},
            {"predicate": "reports_to", "fields": "manager, supervisor, direct_manager",
             "object_type": "person"},
            {"predicate": "has_role", "fields": "title, position, job_family, level"},
            {"predicate": "located_in", "fields": "location, office, city, site",
             "object_type": "location"},
            {"predicate": "has_status", "fields": "status, employment_status, active"},
        ],
        "skip_fields": "id, ssn, tax_id, bank_account, created_date",
        "external_ids_hint": "Include person_name and email as external_ids.",
    },
    "invoice_order": {
        "description": "Invoice, purchase order, transaction, or billing record",
        "field_signals": ["invoice", "order", "total", "line_item",
                          "payment", "billing", "amount_due"],
        "subject_type": "transaction",
        "expected_mappings": [
            {"predicate": "has_metric", "fields": "total, amount, subtotal, tax, amount_due"},
            {"predicate": "has_status", "fields": "status, payment_status, state"},
            {"predicate": "belongs_to", "fields": "customer, account, client, buyer",
             "object_type": "company"},
            {"predicate": "classified_as", "fields": "type, category, payment_method"},
            {"predicate": "issued_on", "fields": "date, invoice_date, created, due_date"},
        ],
        "skip_fields": "id, internal_id, created_date, modified_date",
    },
    "product_item": {
        "description": "Product, SKU, service, or catalog item",
        "field_signals": ["sku", "product", "price", "catalog",
                          "item", "upc", "inventory"],
        "subject_type": "product",
        "expected_mappings": [
            {"predicate": "classified_as", "fields": "category, type, family, line"},
            {"predicate": "has_metric", "fields": "price, cost, msrp, quantity, weight"},
            {"predicate": "has_status", "fields": "status, availability, active, lifecycle"},
            {"predicate": "part_of", "fields": "collection, bundle, suite, parent"},
            {"predicate": "described_by", "fields": "description, summary, features"},
        ],
        "skip_fields": "id, internal_id, created_date, modified_date, image_url",
    },
    "project_sprint": {
        "description": "Project, sprint, initiative, or work stream",
        "field_signals": ["project", "sprint", "milestone", "initiative",
                          "epic", "roadmap", "program"],
        "subject_type": "project",
        "expected_mappings": [
            {"predicate": "has_status", "fields": "status, state, phase, progress"},
            {"predicate": "owned_by", "fields": "owner, lead, pm, project_manager",
             "object_type": "person"},
            {"predicate": "part_of", "fields": "program, portfolio, parent, team"},
            {"predicate": "has_metric", "fields": "budget, hours, velocity, completion_pct"},
            {"predicate": "classified_as", "fields": "type, category, priority"},
        ],
        "skip_fields": "id, created_date, modified_date, url",
    },
}


def _match_enterprise_templates(field_names: list[str]) -> list[str]:
    """Match schema field names against enterprise templates.

    Returns template keys sorted by match score (best first), up to 2.
    """
    lower_fields = {f.lower().replace(" ", "_") for f in field_names}
    scores: list[tuple[str, int]] = []
    for key, tmpl in _ENTERPRISE_TEMPLATES.items():
        hits = sum(1 for sig in tmpl["field_signals"]
                   if any(sig in f for f in lower_fields))
        if hits >= 2:
            scores.append((key, hits))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in scores[:2]]


# ── Gap 2: LLM mapping proposal ─────────────────────────────────────


_MAPPING_PROPOSAL_PROMPT = """\
You are a data mapping expert for enterprise systems. Given a source data \
schema and a target knowledge graph vocabulary, propose a JSON claim mapping \
configuration that extracts maximum relationship value from each record.

The target system stores claims as (subject, predicate, object) triples. \
Each predicate describes a specific kind of relationship. Choosing the \
right predicate is critical — do NOT default to "associated_with" or \
"relates_to" when a more specific predicate fits.

PREDICATE SELECTION GUIDE (use the most specific match):

- has_metric — numeric measurements, scores, amounts, percentages, counts \
  (revenue, AUM, CVSS score, NPS, return_pct, headcount, deal_size)
- has_status — state/status/phase/stage values \
  (status, stage, remediation_status, certification_status, risk_rating)
- has_priority — urgency/importance level \
  (priority, severity, urgency, impact, criticality)
- assigned_to — person responsible for something \
  (assignee, owner, assigned_to, handler, account_executive)
- owned_by — person who owns or manages a record \
  (owner, relationship_manager, csm, account_manager, rep)
- reported_by — person who created/reported something \
  (reporter, opened_by, requester, creator, submitted_by)
- classified_as — category, type, or classification \
  (industry, department, portfolio_type, contract_type, severity, segment)
- located_in — geographic location \
  (country, region, city, governing_law, jurisdiction)
- works_at — employment or organizational membership \
  (company, organization, employer, account)
- has_role — job title or functional role \
  (title, job_title, role, position)
- managed_by — management/supervisory relationship \
  (manager, supervisor, team_lead)
- described_by — free text descriptions \
  (description, notes, comments, finding_text, summary)
- part_of — hierarchical membership \
  (department, division, section, parent_org, team, group)
- belongs_to — ownership or containment relationship \
  (account, customer, parent, organization)
- reports_to — reporting chain \
  (reports_to, direct_report, supervisor)
- relates_to — ONLY as a last resort when no specific predicate fits

FIELD FILTERING — skip these, they are not meaningful facts:
- Internal IDs: fields ending in _id, sys_id, guid (except the subject identifier)
- Audit timestamps: created_date, modified_date, system_modstamp, last_login
- Internal metadata: created_by_id, last_modified_by_id, is_deleted, record_type_id
- URLs and API references: url, self, attributes.url, photo_url, logo_url

EXTERNAL_IDS — for cross-system entity resolution:
When the object is a person (assigned_to, owned_by, reported_by, etc.), include \
an "external_ids" field mapping person_name and email fields from the source data. \
When the object is a company (belongs_to, works_at), include company_name. \
Use dot notation for nested fields (e.g., "Owner.Name", "Account.Email").

MULTI-ENTITY EXTRACTION:
Each record may reference multiple entities (e.g., a deal references an account, \
an owner, and a contact). Map ALL meaningful entity references — produce 4-6 \
mappings per record for rich relationship extraction. Use dot notation for nested \
fields in API data (e.g., "Account.Name", "Owner.Email").

Respond with EXACTLY one JSON object:
{
  "subject_field": "the column that identifies the primary entity",
  "subject_type": "entity type for the subject (e.g., opportunity, ticket, person)",
  "mappings": [
    {
      "field": "source column or dot.path",
      "predicate_type": "predicate from the guide above",
      "object_type": "entity type for the value",
      "confidence": 0.85,
      "reasoning": "brief explanation",
      "external_ids": {"object": {"person_name": "field.path"}}
    }
  ]
}

Rules:
- Use predicates from the PREDICATE SELECTION GUIDE above.
- Prefer specific predicates over generic ones.
- Skip internal/audit fields per the FIELD FILTERING guide.
- Extract 4-6 mappings per record — enough for rich relationships.
- Include external_ids for person and company references.
- Set confidence based on how certain the mapping is (0.5-1.0).
"""


def propose_mapping(
    schema_summary: dict,
    vocabulary: dict,
    domain_context: str = "",
) -> dict:
    """Use an LLM to propose a claim mapping from schema + vocabulary.

    Args:
        schema_summary: Dict with "fields" (list of field descriptions)
            and optionally "sample_rows" (list of dicts).
        vocabulary: Output of export_vocabulary_for_llm().
        domain_context: Optional domain description (e.g., "healthcare compliance").

    Returns:
        Dict with "subject_field", "subject_type", and "mappings" list.
        Returns empty dict on LLM failure.
    """
    from attestdb.discovery.analyzer import _get_llm_client

    client, model = _get_llm_client()
    if client is None:
        logger.warning("No LLM available for mapping proposal")
        return {}

    # Build the user prompt
    parts = [
        "SOURCE SCHEMA:",
        json.dumps(schema_summary, indent=2, default=str),
        "",
        "TARGET VOCABULARY:",
        json.dumps(vocabulary, indent=2),
    ]
    if domain_context:
        parts.extend(["", f"DOMAIN CONTEXT: {domain_context}"])

    # Inject matching enterprise templates as reference patterns
    field_names = [f.get("name", "") if isinstance(f, dict) else str(f)
                   for f in schema_summary.get("fields", [])]
    matched = _match_enterprise_templates(field_names)
    if matched:
        parts.append("")
        parts.append("REFERENCE PATTERNS (use as guidance, not rigid rules):")
        for key in matched:
            tmpl = _ENTERPRISE_TEMPLATES[key]
            parts.append(f"\n  {tmpl['description'].upper()}:")
            parts.append(f"  Subject type: {tmpl['subject_type']}")
            for em in tmpl["expected_mappings"]:
                obj_type = em.get("object_type", "entity")
                parts.append(f"  - {em['predicate']} → {em['fields']} (object_type: {obj_type})")
            if "external_ids_hint" in tmpl:
                parts.append(f"  Note: {tmpl['external_ids_hint']}")

    user_prompt = "\n".join(parts)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _MAPPING_PROPOSAL_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.1,
            timeout=30,
        )
        content = response.choices[0].message.content or ""

        # Extract JSON
        # Extract JSON from LLM response — handle markdown fences,
        # leading text, and multiple JSON blocks
        if "```" in content:
            parts = content.split("```")
            # Take the first fenced block
            if len(parts) >= 2:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            return json.loads(content[start:end + 1])
        logger.warning("LLM response contained no JSON object")
    except json.JSONDecodeError as exc:
        logger.warning("LLM mapping proposal JSON parse failed: %s", exc)
    except Exception as exc:
        logger.warning("LLM mapping proposal failed: %s", exc)

    return {}


def propose_and_validate(
    rows: list[dict],
    vocabulary: dict | None = None,
    domain_context: str = "",
    db=None,
    max_preview: int = 50,
) -> tuple[list[dict], PreviewResult]:
    """End-to-end: propose mappings via LLM, validate against real data.

    1. Builds schema summary from sample rows
    2. Calls propose_mapping() to get LLM-proposed mappings
    3. Converts to connector format
    4. Runs preview_batch() to validate against the actual data
    5. Returns (mappings, preview_result) for human review

    Args:
        rows: Data rows (all of them or a representative sample).
        vocabulary: AttestDB vocabulary dict. Auto-exported from db if omitted.
        domain_context: Optional domain hint for the LLM.
        db: Optional AttestDB instance for vocabulary export.
        max_preview: Max rows to use in preview validation.

    Returns:
        (mappings, preview_result) — mappings ready for execute_rows_batch()
        if preview_result.warnings is empty. Inspect warnings before proceeding.
    """
    if not rows:
        return [], PreviewResult(
            total_rows=0, total_mappings=0, expected_claims=0,
            produced_claims=0, dropped=0, drop_rate=0.0,
            sample_claims=[], field_coverage={}, warnings=["No rows provided"],
        )

    # Build schema summary from data
    sample = rows[:5]
    fields = []
    for key in rows[0].keys():
        vals = [str(r.get(key, ""))[:100] for r in sample if r.get(key)]
        fields.append({"name": key, "samples": vals[:3]})

    schema_summary = {"fields": fields, "sample_rows": sample}

    # Get vocabulary
    if vocabulary is None:
        vocabulary = export_vocabulary_for_llm(db)

    # Propose mappings
    proposal = propose_mapping(schema_summary, vocabulary, domain_context)
    mappings = proposal_to_connector_mappings(proposal)

    if not mappings:
        return [], PreviewResult(
            total_rows=len(rows), total_mappings=0, expected_claims=0,
            produced_claims=0, dropped=0, drop_rate=0.0,
            sample_claims=[], field_coverage={},
            warnings=["LLM proposed no mappings. Check schema_summary and vocabulary."],
        )

    # Validate against real data
    preview_rows = rows[:max_preview]
    result = preview_batch(preview_rows, mappings)

    return mappings, result


def proposal_to_connector_mappings(proposal: dict) -> list[dict]:
    """Convert an LLM proposal into connector-compatible mapping dicts.

    Args:
        proposal: Output of propose_mapping().

    Returns:
        List of mapping dicts for QueryConnector/CSVConnector.
    """
    if not proposal or "mappings" not in proposal:
        return []

    subject_field = proposal.get("subject_field", "")
    subject_type = proposal.get("subject_type", "entity")
    mappings = []

    for m in proposal["mappings"]:
        mapping = {
            "subject": subject_field,
            "subject_type": subject_type,
            "predicate": m["field"],
            "predicate_type": m.get("predicate_type", "has_attribute"),
            "object": m["field"],
            "object_type": m.get("object_type", "entity"),
        }
        # Pass through external_ids from LLM proposal
        if "external_ids" in m:
            mapping["external_ids"] = m["external_ids"]
        mappings.append(mapping)

    return mappings


# ── Gap 3: Vocabulary export for LLM consumption ────────────────────


def export_vocabulary_for_llm(db=None) -> dict:
    """Export AttestDB's vocabulary in a format suitable for LLM prompts.

    When a db with a PredicateStore is available, exports the live predicate
    catalog with descriptions, data patterns, and usage counts. Falls back
    to static vocabulary constants when no store is available.

    Args:
        db: Optional AttestDB instance for live vocabulary data.

    Returns:
        Dict with entity_types, predicate_types, source_types, and
        predicate_catalog (descriptions and data patterns for LLM guidance).
    """
    from attestdb.core.vocabulary import (
        BUILT_IN_ENTITY_TYPES,
        BUILT_IN_SOURCE_TYPES,
    )

    result: dict = {
        "entity_types": sorted(BUILT_IN_ENTITY_TYPES),
        "source_types": sorted(BUILT_IN_SOURCE_TYPES),
    }

    # Use PredicateStore catalog if available — live, self-describing
    pred_store = getattr(db, "_predicate_store", None) if db else None
    if pred_store:
        catalog = pred_store.export_catalog()
        result["predicate_catalog"] = catalog
        result["predicate_types"] = sorted(catalog.keys())
    else:
        # Fallback to static constants
        from attestdb.core.vocabulary import (
            BUILT_IN_PREDICATE_TYPES,
            OPPOSITE_PREDICATES,
            STANDARD_PREDICATES,
        )
        result["predicate_types"] = sorted(BUILT_IN_PREDICATE_TYPES)
        result["opposite_predicates"] = dict(OPPOSITE_PREDICATES)
        result["standard_predicates"] = (
            sorted(STANDARD_PREDICATES)
            if hasattr(STANDARD_PREDICATES, "__iter__") and not isinstance(STANDARD_PREDICATES, dict)
            else list(STANDARD_PREDICATES)
        )

    if db is not None:
        try:
            schema = db.schema()
            all_entity_types = set(schema.entity_types.keys())
            result["entity_types"] = sorted(all_entity_types)

            if schema.relationship_patterns:
                result["relationship_patterns"] = [
                    {
                        "subject_type": rp.subject_type,
                        "predicate": rp.predicate,
                        "object_type": rp.object_type,
                        "count": rp.count,
                    }
                    for rp in schema.relationship_patterns[:50]
                ]
        except Exception as exc:
            logger.debug("Failed to get live schema: %s", exc)

    return result


# ── Gap 4: Mapping validation ────────────────────────────────────────


@dataclass
class MappingValidationResult:
    """Result of validating a mapping against vocabulary."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_mapping(mapping: dict, vocabulary: dict) -> MappingValidationResult:
    """Check if a mapping is valid against the AttestDB vocabulary.

    Args:
        mapping: A connector mapping dict.
        vocabulary: Output of export_vocabulary_for_llm().

    Returns:
        MappingValidationResult with errors and warnings.
    """
    errors = []
    warnings = []
    entity_types = set(vocabulary.get("entity_types", []))
    predicate_types = set(vocabulary.get("predicate_types", []))

    # Check required fields
    for required in ("subject", "object"):
        if required not in mapping:
            errors.append(f"Missing required field: {required}")

    # Check entity types
    subj_type = mapping.get("subject_type", "entity")
    obj_type = mapping.get("object_type", "entity")
    if entity_types and subj_type not in entity_types:
        warnings.append(
            f"subject_type '{subj_type}' not in vocabulary "
            f"(known: {sorted(entity_types)[:10]})"
        )
    if entity_types and obj_type not in entity_types:
        warnings.append(
            f"object_type '{obj_type}' not in vocabulary "
            f"(known: {sorted(entity_types)[:10]})"
        )

    # Check predicate type
    pred_type = mapping.get("predicate_type", "relates_to")
    if predicate_types and pred_type not in predicate_types:
        warnings.append(
            f"predicate_type '{pred_type}' not in vocabulary "
            f"(known: {sorted(predicate_types)[:10]})"
        )

    return MappingValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


# ── Dry-run preview: validate mappings against sample data ────────────


@dataclass
class PreviewResult:
    """Result of a dry-run mapping preview."""
    total_rows: int
    total_mappings: int
    expected_claims: int
    produced_claims: int
    dropped: int
    drop_rate: float
    sample_claims: list[dict]
    field_coverage: dict[str, float]  # mapping field → % of rows with data
    warnings: list[str]


def preview_batch(
    rows: list[dict],
    mappings: list[dict],
    *,
    max_sample: int = 10,
    source_type: str = "preview",
    source_id: str = "preview",
) -> PreviewResult:
    """Dry-run mappings against data to validate before ingestion.

    Runs all mappings against all rows without touching the database.
    Reports drop rates, field coverage, and a sample of what claims
    would look like. Use this before execute_rows_batch() to catch
    bad mappings early.

    Args:
        rows: Data rows to preview.
        mappings: Proposed mapping dicts.
        max_sample: Max sample claims to include in result.
        source_type: Source type label (not persisted).
        source_id: Source ID label (not persisted).

    Returns:
        PreviewResult with stats, sample claims, and warnings.
    """
    claims = execute_rows_batch(
        rows, mappings,
        source_type=source_type, source_id=source_id,
        include_payload=True,
    )

    expected = len(rows) * len(mappings)
    produced = len(claims)
    dropped = expected - produced
    drop_rate = dropped / expected if expected > 0 else 0.0

    # Field coverage: for each mapping field, what % of rows have non-empty data
    field_coverage: dict[str, float] = {}
    for m in mappings:
        for role in ("subject", "object"):
            col = m.get(role, "")
            if col:
                hits = sum(1 for r in rows if _resolve_path(r, col))
                field_coverage[f"{m.get('predicate', '?')}.{role}={col}"] = (
                    hits / len(rows) if rows else 0.0
                )

    # Sample claims for human review
    sample_claims = []
    for c in claims[:max_sample]:
        sample_claims.append({
            "subject": c.subject[0],
            "predicate": c.predicate[0],
            "object": c.object[0],
            "confidence": c.confidence,
            "payload_schema": c.payload.get("schema_ref") if c.payload else None,
        })

    # Generate warnings
    warnings = []
    if drop_rate > 0.5:
        warnings.append(
            f"High drop rate: {drop_rate:.0%} of expected claims were dropped "
            f"({dropped}/{expected}). Check mapping field names against data columns."
        )
    if drop_rate == 1.0:
        warnings.append(
            "ALL claims dropped. Mappings likely have wrong field names. "
            f"Available columns: {list(rows[0].keys()) if rows else '(no rows)'}"
        )
    for key, coverage in field_coverage.items():
        if coverage < 0.5:
            warnings.append(
                f"Low coverage: {key} has data in only {coverage:.0%} of rows"
            )

    # Check for duplicate subjects across all claims (possible mapping error)
    subj_pred_counts: dict[str, int] = {}
    for c in claims:
        k = f"{c.subject[0]}|{c.predicate[0]}"
        subj_pred_counts[k] = subj_pred_counts.get(k, 0) + 1
    dup_count = sum(1 for v in subj_pred_counts.values() if v > 1)
    if dup_count > len(subj_pred_counts) * 0.5 and dup_count > 5:
        warnings.append(
            f"{dup_count} subject+predicate pairs appear multiple times. "
            "This may indicate duplicate mappings or a non-unique subject field."
        )

    return PreviewResult(
        total_rows=len(rows),
        total_mappings=len(mappings),
        expected_claims=expected,
        produced_claims=produced,
        dropped=dropped,
        drop_rate=drop_rate,
        sample_claims=sample_claims,
        field_coverage=field_coverage,
        warnings=warnings,
    )


# ── Gap 5: Generic mapping executor ─────────────────────────────────


def execute_csv_mapping(
    csv_path: str,
    mapping: dict,
    *,
    source_type: str = "csv_import",
    source_id: str = "",
    default_confidence: float = 0.7,
) -> list[ClaimInput]:
    """Execute a mapping against a CSV file, producing ClaimInput objects.

    Args:
        csv_path: Path to CSV file.
        mapping: Connector mapping dict with subject, predicate, object fields.
        source_type: Source type for provenance.
        source_id: Source identifier. Defaults to filename.
        default_confidence: Confidence when not specified in mapping.

    Returns:
        List of ClaimInput objects ready for db.ingest_batch().
    """
    if not source_id:
        source_id = f"csv:{os.path.basename(csv_path)}"

    claims: list[ClaimInput] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            try:
                claim = _row_to_claim_input(
                    row, mapping, source_type, source_id, default_confidence,
                )
                if claim is not None:
                    claims.append(claim)
            except Exception as exc:
                logger.warning("Row %d: skipping: %s", row_num, exc)

    logger.info(
        "execute_csv_mapping: %s → %d claims from %s",
        csv_path, len(claims), mapping.get("subject", "?"),
    )
    return claims


def execute_rows_mapping(
    rows: list[dict],
    mapping: dict,
    *,
    source_type: str = "data_import",
    source_id: str = "import",
    default_confidence: float = 0.7,
) -> list[ClaimInput]:
    """Execute a mapping against a list of row dicts.

    Args:
        rows: List of dict-like rows (from database, API, etc.).
        mapping: Connector mapping dict.
        source_type: Source type for provenance.
        source_id: Source identifier.
        default_confidence: Default confidence.

    Returns:
        List of ClaimInput objects.
    """
    claims: list[ClaimInput] = []
    for i, row in enumerate(rows):
        try:
            claim = _row_to_claim_input(
                row, mapping, source_type, source_id, default_confidence,
            )
            if claim is not None:
                claims.append(claim)
        except Exception as exc:
            logger.warning("Row %d: skipping: %s", i, exc)

    return claims


def _row_to_claim_input(
    row: dict,
    mapping: dict,
    source_type: str,
    source_id: str,
    default_confidence: float,
    include_payload: bool = False,
) -> ClaimInput | None:
    """Convert a single row to a ClaimInput using a mapping dict.

    Supports dot-notation field paths for nested data (e.g., "Account.Name").

    When include_payload=True, attaches the source row as a claim payload
    with schema_ref="{source_type}/{source_id}". If the mapping contains
    a "record_id" field path, that value is included for direct lookback.
    """
    subj_col = mapping.get("subject")
    obj_col = mapping.get("object")

    if not subj_col or not obj_col:
        logger.warning(
            "Mapping missing 'subject' or 'object' key (got keys: %s). "
            "Note: use 'subject'/'object', not 'subject_path'/'object_path'.",
            list(mapping.keys()),
        )
        return None

    subj_raw = _resolve_path(row, subj_col)
    obj_raw = _resolve_path(row, obj_col)
    subj_val = str(subj_raw).strip() if subj_raw is not None else ""
    obj_val = str(obj_raw).strip() if obj_raw is not None else ""

    if not subj_val or not obj_val:
        logger.debug(
            "Skipping claim: empty %s (column=%r) in row",
            "subject" if not subj_val else "object",
            subj_col if not subj_val else obj_col,
        )
        return None

    # Predicate: either a column value or a static string
    pred_col = mapping.get("predicate", "")
    pred_raw = _resolve_path(row, pred_col) if pred_col else None
    if pred_raw is not None:
        pred_val = str(pred_raw).strip()
    else:
        pred_val = pred_col  # Static predicate

    if not pred_val:
        pred_val = "has_attribute"

    # Confidence: from column or default
    conf = default_confidence
    conf_col = mapping.get("confidence")
    if conf_col:
        conf_raw = _resolve_path(row, conf_col)
        if conf_raw is not None:
            try:
                conf = float(conf_raw)
            except (ValueError, TypeError):
                pass

    # External IDs: resolve field paths from mapping
    ext_ids = None
    ext_ids_spec = mapping.get("external_ids")
    if ext_ids_spec:
        ext_ids = {}
        for role in ("subject", "object"):
            role_spec = ext_ids_spec.get(role)
            if role_spec and isinstance(role_spec, dict):
                resolved = {}
                for key, field_path in role_spec.items():
                    val = _resolve_path(row, field_path)
                    if val is not None:
                        resolved[key] = str(val).strip()
                if resolved:
                    ext_ids[role] = resolved
        if not ext_ids:
            ext_ids = None

    # Payload: attach source record for lookback to originating system
    payload = None
    if include_payload:
        schema_ref = mapping.get("schema_ref") or f"{source_type}/{source_id}"
        payload_data = {}
        # Include a record ID if the mapping specifies one
        record_id_field = mapping.get("record_id")
        if record_id_field:
            rid = _resolve_path(row, record_id_field)
            if rid is not None:
                payload_data["record_id"] = str(rid).strip()
        # Attach the source row (only JSON-serializable values)
        for k, v in row.items():
            try:
                if isinstance(v, (str, int, float, bool)) or v is None:
                    payload_data[k] = v
                else:
                    payload_data[k] = str(v)
            except Exception:
                pass
        payload = {"schema_ref": schema_ref, "data": payload_data}

    return ClaimInput(
        subject=(subj_val, mapping.get("subject_type", "entity")),
        predicate=(pred_val, mapping.get("predicate_type", "has_attribute")),
        object=(obj_val, mapping.get("object_type", "entity")),
        provenance={"source_type": source_type, "source_id": source_id},
        confidence=conf,
        external_ids=ext_ids,
        payload=payload,
    )


# ── Batch execution: multiple mappings in a single pass ───────────────


def execute_rows_batch(
    rows: list[dict],
    mappings: list[dict],
    *,
    source_type: str = "data_import",
    source_id: str = "import",
    default_confidence: float = 0.7,
    include_payload: bool = False,
) -> list[ClaimInput]:
    """Execute multiple mappings against rows in a single pass.

    For each row, applies every mapping and collects all resulting claims.
    Produces N * M claims max (N rows, M mappings) in one iteration.

    Args:
        rows: List of row dicts (from CSV, database, API, etc.).
        mappings: List of connector mapping dicts.
        source_type: Source type for provenance.
        source_id: Source identifier.
        default_confidence: Default confidence when not specified.
        include_payload: Attach source row data as claim payload for
            lookback to the originating system.

    Returns:
        List of ClaimInput objects.
    """
    claims: list[ClaimInput] = []
    expected = len(rows) * len(mappings)
    dropped = 0
    errors = 0
    for i, row in enumerate(rows):
        for mapping in mappings:
            try:
                claim = _row_to_claim_input(
                    row, mapping, source_type, source_id, default_confidence,
                    include_payload=include_payload,
                )
                if claim is not None:
                    claims.append(claim)
                else:
                    dropped += 1
            except Exception as exc:
                errors += 1
                logger.warning("Row %d mapping %s: skipping: %s",
                               i, mapping.get("predicate_type", "?"), exc)
    if dropped or errors:
        logger.info(
            "execute_rows_batch: %d/%d claims produced (%d dropped, %d errors)",
            len(claims), expected, dropped, errors,
        )
    return claims


def execute_csv_batch(
    csv_path: str,
    mappings: list[dict],
    *,
    source_type: str = "csv_import",
    source_id: str = "",
    default_confidence: float = 0.7,
    include_payload: bool = False,
) -> list[ClaimInput]:
    """Execute multiple mappings against a CSV file in a single pass.

    Opens the CSV once and applies all mappings to each row.

    Args:
        csv_path: Path to CSV file.
        mappings: List of connector mapping dicts.
        source_type: Source type for provenance.
        source_id: Source identifier. Defaults to filename.
        default_confidence: Default confidence.
        include_payload: Attach source row data as claim payload.

    Returns:
        List of ClaimInput objects.
    """
    if not source_id:
        source_id = f"csv:{os.path.basename(csv_path)}"

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    claims = execute_rows_batch(
        rows, mappings,
        source_type=source_type,
        source_id=source_id,
        default_confidence=default_confidence,
        include_payload=include_payload,
    )
    logger.info(
        "execute_csv_batch: %s → %d claims from %d mappings × %d rows",
        csv_path, len(claims), len(mappings), len(rows),
    )
    return claims


# ── Mapping persistence ──────────────────────────────────────────────


def _schema_fingerprint(field_names: list[str]) -> str:
    """Compute a stable fingerprint from sorted field names.

    Used to detect when a source schema has changed, invalidating
    cached mapping proposals.
    """
    canonical = "|".join(sorted(f.strip().lower() for f in field_names))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def save_proposal(
    proposal: dict,
    source_id: str,
    field_names: list[str],
    cache_dir: str,
    *,
    validated: bool = False,
) -> str:
    """Save a mapping proposal to disk for reuse.

    Args:
        proposal: LLM proposal dict (from propose_mapping).
        source_id: Identifier for the data source.
        field_names: Schema field names (for fingerprinting).
        cache_dir: Directory to save the proposal.
        validated: Whether a human has reviewed and approved this proposal.

    Returns:
        Path to the saved proposal file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = source_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path = os.path.join(cache_dir, f"proposal_{safe_name}.json")

    envelope = {
        "source_id": source_id,
        "schema_fingerprint": _schema_fingerprint(field_names),
        "field_names": field_names,
        "proposal": proposal,
        "validated": validated,
        "created_at": time.time(),
    }
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2, default=str)
    logger.info("Saved mapping proposal to %s", path)
    return path


def load_proposal(
    source_id: str,
    field_names: list[str],
    cache_dir: str,
) -> dict | None:
    """Load a cached mapping proposal if the schema fingerprint matches.

    Returns:
        The proposal dict if found and valid, None otherwise.
    """
    safe_name = source_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path = os.path.join(cache_dir, f"proposal_{safe_name}.json")

    if not os.path.exists(path):
        return None

    try:
        with open(path) as f:
            envelope = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load cached proposal %s: %s", path, exc)
        return None

    # Invalidate if schema changed
    if envelope.get("schema_fingerprint") != _schema_fingerprint(field_names):
        logger.info("Schema fingerprint mismatch for %s — re-proposing", source_id)
        return None

    logger.info("Loaded cached mapping proposal for %s", source_id)
    return envelope.get("proposal")


# ── Convenience: end-to-end schema reasoning ─────────────────────────


def _build_schema_summary(
    source_name: str,
    headers: list[str],
    sample_rows: list[dict],
) -> dict:
    """Build a schema summary dict from headers and sample rows."""
    return {
        "source": source_name,
        "fields": [
            {"name": h, "sample_values": [r.get(h, "") for r in sample_rows[:3]]}
            for h in headers
        ],
        "sample_rows": sample_rows[:3],
        "total_columns": len(headers),
    }


def _propose_or_load(
    schema_summary: dict,
    headers: list[str],
    source_id: str,
    db,
    domain_context: str,
    cache_dir: str | None,
) -> dict:
    """Propose a mapping via LLM, or load from cache if available."""
    # Try cache first
    if cache_dir:
        cached = load_proposal(source_id, headers, cache_dir)
        if cached:
            return cached

    vocabulary = export_vocabulary_for_llm(db)
    proposal = propose_mapping(schema_summary, vocabulary, domain_context)

    # Save to cache
    if proposal and cache_dir:
        save_proposal(proposal, source_id, headers, cache_dir)

    return proposal


def reason_and_map_csv(
    csv_path: str,
    db=None,
    *,
    domain_context: str = "",
    source_type: str = "csv_import",
    cache_dir: str | None = None,
) -> tuple[list[ClaimInput], dict]:
    """End-to-end: discover schema from CSV, propose mapping, execute.

    Args:
        csv_path: Path to CSV file.
        db: Optional AttestDB instance for vocabulary context.
        domain_context: Optional domain hint for the LLM.
        source_type: Source type for provenance.
        cache_dir: Optional directory for caching mapping proposals.

    Returns:
        Tuple of (list of ClaimInput, proposal dict from LLM).
    """
    # Step 1: Read CSV headers and sample rows
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        sample_rows = []
        for i, row in enumerate(reader):
            sample_rows.append(dict(row))
            if i >= 4:
                break

    source_id = f"csv:{os.path.basename(csv_path)}"
    schema_summary = _build_schema_summary(
        os.path.basename(csv_path), headers, sample_rows,
    )

    # Step 2+3: Propose or load from cache
    proposal = _propose_or_load(
        schema_summary, headers, source_id, db, domain_context, cache_dir,
    )
    if not proposal:
        logger.warning("LLM mapping proposal returned empty; cannot proceed")
        return [], {}

    # Step 4: Convert proposal to connector mappings
    connector_mappings = proposal_to_connector_mappings(proposal)
    if not connector_mappings:
        logger.warning("No valid mappings from LLM proposal")
        return [], proposal

    # Step 5: Execute all mappings in a single pass over the CSV
    all_claims = execute_csv_batch(
        csv_path, connector_mappings,
        source_type=source_type, source_id=source_id,
    )

    logger.info(
        "reason_and_map_csv: %s → %d claims from %d mappings",
        csv_path, len(all_claims), len(connector_mappings),
    )
    return all_claims, proposal


def reason_and_map_rows(
    rows: list[dict],
    source_id: str = "api_import",
    db=None,
    *,
    domain_context: str = "",
    source_type: str = "api_import",
    cache_dir: str | None = None,
) -> tuple[list[ClaimInput], dict]:
    """End-to-end: schema-reason over in-memory rows, propose mapping, execute.

    For use with API responses, database query results, or any row-based data
    already in memory.

    Args:
        rows: List of row dicts.
        source_id: Identifier for the data source.
        db: Optional AttestDB instance for vocabulary context.
        domain_context: Optional domain hint for the LLM.
        source_type: Source type for provenance.
        cache_dir: Optional directory for caching mapping proposals.

    Returns:
        Tuple of (list of ClaimInput, proposal dict from LLM).
    """
    if not rows:
        return [], {}

    # Derive headers from first row
    headers = list(rows[0].keys())
    sample_rows = rows[:5]
    schema_summary = _build_schema_summary(source_id, headers, sample_rows)

    proposal = _propose_or_load(
        schema_summary, headers, source_id, db, domain_context, cache_dir,
    )
    if not proposal:
        logger.warning("LLM mapping proposal returned empty; cannot proceed")
        return [], {}

    connector_mappings = proposal_to_connector_mappings(proposal)
    if not connector_mappings:
        logger.warning("No valid mappings from LLM proposal")
        return [], proposal

    all_claims = execute_rows_batch(
        rows, connector_mappings,
        source_type=source_type, source_id=source_id,
    )

    logger.info(
        "reason_and_map_rows: %s → %d claims from %d mappings",
        source_id, len(all_claims), len(connector_mappings),
    )
    return all_claims, proposal
