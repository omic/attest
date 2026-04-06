"""Auto-discovery pipeline for connected data sources.

Samples data, analyzes fields, and maps schemas to semantic types
for automatic claim extraction.  Includes schema drift detection,
versioning, and template migration for continuous re-discovery.
"""

from __future__ import annotations

from attestdb.discovery.aligner import AlignedFieldGroup, UnifiedSchema, align_schemas
from attestdb.discovery.analyzer import SemanticMapping, detect_deprecated_fields, infer_semantics
from attestdb.discovery.claim_templates import ClaimTemplate, generate_claim_templates, templates_to_report
from attestdb.discovery.drift_detector import (
    DriftMonitor,
    DriftReport,
    SchemaChange,
    detect_drift,
    detect_drift_from_connector,
)
from attestdb.discovery.sampler import FieldProfile, analyze_fields, detect_pii, sample_source
from attestdb.discovery.schema_map import SchemaMap
from attestdb.discovery.schema_versioning import SchemaVersion, SchemaVersionStore
from attestdb.discovery.template_migrator import (
    TemplateMigrationAction,
    TemplateMigrationResult,
    migrate_templates,
    validate_templates,
)
from attestdb.discovery.unstructured import ExtractionConfig, UnstructuredClaim, UnstructuredSource, extract_claims

__all__ = [
    "AlignedFieldGroup",
    "ClaimTemplate",
    "DriftMonitor",
    "DriftReport",
    "ExtractionConfig",
    "FieldProfile",
    "SchemaChange",
    "SchemaMap",
    "SchemaVersion",
    "SchemaVersionStore",
    "SemanticMapping",
    "TemplateMigrationAction",
    "TemplateMigrationResult",
    "UnifiedSchema",
    "UnstructuredClaim",
    "UnstructuredSource",
    "align_schemas",
    "analyze_fields",
    "detect_deprecated_fields",
    "detect_drift",
    "detect_drift_from_connector",
    "detect_pii",
    "extract_claims",
    "generate_claim_templates",
    "infer_semantics",
    "migrate_templates",
    "sample_source",
    "templates_to_report",
    "validate_templates",
]
