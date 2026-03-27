"""Shared dataclasses defining the contract between infrastructure and intelligence."""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field


class Role(enum.Enum):
    ADMIN = "admin"      # Full access: ingest, retract, query, grant/revoke
    WRITER = "writer"    # Ingest + query (no retract, no grant/revoke)
    READER = "reader"    # Query only


class SensitivityLevel(enum.IntEnum):
    """Claim sensitivity classification for access control."""
    SHARED = -1         # Visible across all tenants (public knowledge)
    PUBLIC = 0          # Visible to all principals with database access
    INTERNAL = 1        # Visible within the owning org/tenant only
    CONFIDENTIAL = 2    # Visible to claim owner + explicit ACL + org admins
    RESTRICTED = 3      # Visible to claim owner + explicit ACL only
    REDACTED = 4        # Structure visible, content replaced with "[REDACTED]"


class ClaimStatus(enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"
    PROVENANCE_DEGRADED = "provenance_degraded"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    DISPUTED = "disputed"


@dataclass
class EntityRef:
    id: str
    entity_type: str
    display_name: str = ""
    external_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class PredicateRef:
    id: str
    predicate_type: str


@dataclass
class Provenance:
    source_type: str
    source_id: str
    method: str | None = None
    chain: list[str] = field(default_factory=list)
    model_version: str | None = None
    organization: str | None = None


@dataclass
class Payload:
    schema_ref: str
    data: dict


@dataclass
class Principal:
    """Identity for access control decisions."""
    principal_id: str
    principal_type: str = "user"  # "user" | "agent" | "api_key" | "org" | "team"
    org_id: str | None = None
    roles: list[str] = field(default_factory=list)
    clearance: SensitivityLevel = SensitivityLevel.PUBLIC


@dataclass
class Claim:
    claim_id: str
    content_id: str
    subject: EntityRef
    predicate: PredicateRef
    object: EntityRef
    confidence: float
    provenance: Provenance
    embedding: list[float] | None = None
    payload: Payload | None = None
    timestamp: int = 0
    status: ClaimStatus = ClaimStatus.ACTIVE
    namespace: str = ""
    expires_at: int = 0  # Nanosecond timestamp. 0 = never expires.
    # Security fields (Python-only, not persisted in Rust engine)
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    owner: str | None = None
    acl: list[str] = field(default_factory=list)


@dataclass
class ClaimInput:
    """Input structure for ingesting a new claim."""
    subject: tuple[str, str]  # (id, entity_type)
    predicate: tuple[str, str]  # (id, predicate_type)
    object: tuple[str, str]  # (id, entity_type)
    provenance: dict[str, object]
    confidence: float | None = None
    embedding: list[float] | None = None
    payload: dict | None = None
    timestamp: int | None = None
    external_ids: dict[str, dict[str, str]] | None = None
    namespace: str = ""
    ttl_seconds: int = 0  # Time-to-live in seconds. 0 = never expires.
    # Security fields
    sensitivity: SensitivityLevel | None = None  # None = auto-classify
    owner: str | None = None
    acl: list[str] | None = None


@dataclass
class EntitySummary:
    id: str
    name: str
    entity_type: str
    external_ids: dict[str, str] = field(default_factory=dict)
    claim_count: int = 0


@dataclass
class Relationship:
    predicate: str
    target: EntitySummary
    confidence: float
    n_independent_sources: int = 1
    source_types: list[str] = field(default_factory=list)
    latest_claim_timestamp: int = 0
    payload: dict | None = None
    is_symmetric: bool = False
    is_inverse: bool = False
    is_composed: bool = False
    composition_chain: list[str] = field(default_factory=list)


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    description: str = ""
    evidence_a: int = 0
    evidence_b: int = 0
    status: str = "unresolved"  # "unresolved", "a_preferred", "b_preferred"


@dataclass
class QuantitativeClaim:
    predicate: str
    target: str
    value: float
    unit: str
    metric: str = ""
    source_type: str = ""
    confidence: float = 0.0


@dataclass
class ContextFrame:
    focal_entity: EntitySummary
    direct_relationships: list[Relationship] = field(default_factory=list)
    quantitative_data: list[QuantitativeClaim] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    narrative: str = ""
    provenance_summary: dict[str, float] = field(default_factory=dict)
    claim_count: int = 0
    confidence_range: tuple[float, float] = (0.0, 0.0)
    topic_membership: list[str] = field(default_factory=list)
    open_inquiries: list[str] = field(default_factory=list)


@dataclass
class BatchResult:
    ingested: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


# --- Topic Thread types ---


@dataclass
class ThreadState:
    """Persisted as a Payload on a thread claim."""
    thread_id: str
    seed_entities: list[str]
    seed_query: str | None = None
    traversal_depth: int = 2
    min_confidence: float = 0.5
    source_filter: list[str] | None = None
    exclude_source_types: list[str] | None = None
    claim_ids: list[str] = field(default_factory=list)
    frontier_entities: list[str] = field(default_factory=list)
    created_at: int = 0
    last_accessed: int = 0
    parent_thread_id: str | None = None
    summary: str | None = None
    open_questions: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    contradiction_details: list[dict] = field(default_factory=list)
    key_findings_threshold: float = 0.8
    stale: bool = False


@dataclass
class ThreadContext:
    """Return type for thread operations."""
    thread_id: str
    claims: list["Claim"] = field(default_factory=list)
    summary: str = ""
    open_questions: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    frontier: list[EntitySummary] = field(default_factory=list)
    claim_count: int = 0
    entity_count: int = 0
    seed_entities: list[str] = field(default_factory=list)
    # Diff fields (populated by resume_thread)
    new_since_last: list["Claim"] = field(default_factory=list)
    removed_since_last: list[str] = field(default_factory=list)


# --- Paper Audit types ---


@dataclass
class PaperContent:
    """Resolved content from a paper source."""
    paper_id: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    references: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ClaimAuditDetail:
    """Per-claim detail in a paper audit."""
    claim_id: str
    claim_summary: str = ""
    verdict: str = ""
    tier: int = 1
    cost_usd: float = 0.0
    section: str = ""


@dataclass
class PaperAuditReport:
    """Result of auditing a scientific paper."""
    paper_id: str
    title: str = ""
    source: str = ""
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_failed: int = 0
    claims_disputed: int = 0
    claims_pending: int = 0
    overall_score: float = 0.0
    total_cost_usd: float = 0.0
    flagged_issues: list[str] = field(default_factory=list)
    claim_details: list[ClaimAuditDetail] = field(default_factory=list)
    thread_id: str | None = None


@dataclass
class PipelineResult:
    """Result of executing a reproducibility pipeline."""
    success: bool = False
    output: dict = field(default_factory=dict)
    execution_time_seconds: float = 0.0
    environment_hash: str = ""
    errors: list[str] = field(default_factory=list)
    cost_usd: float = 0.0


# --- Knowledge Topology types ---


@dataclass
class Community:
    id: str
    members: list[str]
    resolution: float
    density: float
    dominant_entity_types: dict[str, int] = field(default_factory=dict)
    label: str = ""


@dataclass
class TopicNode:
    id: str
    label: str
    level: int
    entities: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    density: float = 0.0


@dataclass
class DensityMapEntry:
    topic_id: str
    topic_label: str
    level: int
    internal_density: float
    entity_count: int
    claim_count: int = 0


@dataclass
class CrossDomainBridge:
    entity_id: str
    entity_type: str
    communities: list[str]
    bridge_score: float
    explanation: str = ""


@dataclass
class QueryProfile:
    focal_entity: str
    total_candidates: int
    after_scoring: int
    after_budget: int
    elapsed_ms: float
    depth: int
    filters_applied: dict = field(default_factory=dict)


@dataclass
class RetractResult:
    source_id: str
    reason: str
    retracted_count: int
    claim_ids: list[str]


@dataclass
class QualityReport:
    total_claims: int = 0
    total_entities: int = 0
    entity_type_counts: dict[str, int] = field(default_factory=dict)
    single_source_entity_count: int = 0
    stale_entity_count: int = 0
    gap_count: int = 0
    confidence_alert_count: int = 0
    avg_claims_per_entity: float = 0.0
    source_type_distribution: dict[str, int] = field(default_factory=dict)
    predicate_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class RelationshipPattern:
    """A (subject_type, predicate, object_type) pattern with occurrence count."""

    subject_type: str
    predicate: str
    object_type: str
    count: int = 0


@dataclass
class SchemaDescriptor:
    """Metadata descriptor of the knowledge graph — like Neo4j's db.schema().

    Shows what entity types exist, what predicates connect them,
    relationship patterns, source types, and counts.
    """

    entity_types: dict[str, int] = field(default_factory=dict)        # type → count
    predicate_types: dict[str, int] = field(default_factory=dict)     # predicate → count
    source_types: dict[str, int] = field(default_factory=dict)        # source_type → count
    relationship_patterns: list[RelationshipPattern] = field(default_factory=list)
    total_claims: int = 0
    total_entities: int = 0
    registered_vocabularies: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Schema: {self.total_entities} entities, {self.total_claims} claims"]
        if self.entity_types:
            lines.append("Entity types:")
            for t, c in sorted(self.entity_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {t}: {c}")
        if self.relationship_patterns:
            lines.append("Relationship patterns:")
            for rp in sorted(self.relationship_patterns, key=lambda x: -x.count)[:20]:
                lines.append(
                    f"  ({rp.subject_type})-[{rp.predicate}]"
                    f"->({rp.object_type}): {rp.count}"
                )
        if self.source_types:
            lines.append("Source types:")
            for s, c in sorted(self.source_types.items(), key=lambda x: -x[1]):
                lines.append(f"  {s}: {c}")
        return "\n".join(lines)


@dataclass
class CascadeResult:
    source_retract: RetractResult
    degraded_claim_ids: list[str] = field(default_factory=list)
    degraded_count: int = 0


@dataclass
class DownstreamNode:
    claim_id: str
    dependents: list["DownstreamNode"] = field(default_factory=list)


@dataclass
class KnowledgeHealth:
    """Quantified health metrics for the knowledge graph."""
    total_claims: int = 0
    total_entities: int = 0
    avg_confidence: float = 0.0
    multi_source_ratio: float = 0.0       # % of entities with >1 source
    corroboration_ratio: float = 0.0      # % of content_ids with >1 claim
    avg_provenance_depth: float = 0.0     # avg chain length
    source_diversity: int = 0             # unique source_type count
    freshness_score: float = 0.0          # 0-1, based on recency of claims
    confidence_trend: float = 0.0         # positive = improving over time
    knowledge_density: float = 0.0        # claims_per_entity
    health_score: float = 0.0             # 0-100 composite


@dataclass
class PathStep:
    """A single hop in a path between two entities."""
    entity_id: str
    entity_type: str
    predicate: str
    confidence: float
    source_types: list[str] = field(default_factory=list)


@dataclass
class PathResult:
    """A complete path between two entities with aggregated metadata."""
    steps: list[PathStep] = field(default_factory=list)
    total_confidence: float = 0.0
    length: int = 0


# --- New API types (Stage 7) ---


@dataclass
class ImpactReport:
    """Impact analysis for a source: what claims depend on it."""
    source_id: str
    direct_claims: int = 0
    downstream_claims: int = 0
    affected_entities: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)


@dataclass
class BlindspotMap:
    """Knowledge gaps and single-source vulnerabilities."""
    single_source_entities: list[str] = field(default_factory=list)
    knowledge_gaps: list[dict] = field(default_factory=list)
    low_confidence_areas: list[dict] = field(default_factory=list)
    unresolved_warnings: list[dict] = field(default_factory=list)


@dataclass
class ConsensusReport:
    """Consensus analysis for a topic: agreement across sources."""
    topic: str = ""
    total_claims: int = 0
    unique_sources: int = 0
    avg_confidence: float = 0.0
    agreement_ratio: float = 0.0
    claims_by_source: dict[str, int] = field(default_factory=dict)
    corroborated_content_ids: list[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    """A single entry in the mutation audit log."""
    event: str  # claim_ingested, source_retracted, status_changed, namespace_set
    timestamp: int = 0
    actor: str = ""
    claim_id: str = ""
    source_id: str = ""
    namespace: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class AuditTrail:
    """Full provenance audit for a claim."""
    claim_id: str = ""
    content_id: str = ""
    corroborating_claims: list[str] = field(default_factory=list)
    provenance_chain: list[str] = field(default_factory=list)
    downstream_dependents: int = 0
    source_type: str = ""
    source_id: str = ""
    confidence: float = 0.0
    timestamp: int = 0


@dataclass
class DriftReport:
    """Knowledge drift over a time period."""
    period_days: int = 30
    new_claims: int = 0
    new_entities: int = 0
    retracted_claims: int = 0
    confidence_delta: float = 0.0
    new_source_types: list[str] = field(default_factory=list)
    entity_count_before: int = 0
    entity_count_after: int = 0
    claim_count_before: int = 0
    claim_count_after: int = 0


@dataclass
class HypotheticalReport:
    """What-if analysis for a hypothetical claim."""
    would_corroborate: bool = False
    existing_corroborations: int = 0
    fills_gap: bool = False
    content_id: str = ""
    related_entities: list[str] = field(default_factory=list)


@dataclass
class IndirectEvidence:
    """Multi-hop evidence for/against a hypothesis found in the sandbox."""
    path: list[str]           # entity IDs along the path
    predicates: list[str]     # predicates at each hop
    predicted_predicate: str  # composed result from PREDICATE_COMPOSITION
    confidence: float         # product of hop confidences
    direction: str = "neutral"  # "supporting" | "contradicting" | "neutral"


@dataclass
class SandboxVerdict:
    """Rich analysis result from HypotheticalContext.analyze()."""
    hypothesis: str
    verdict: str = "insufficient_data"  # "supported" | "contradicted" | "plausible" | "insufficient_data"
    # Direct evidence
    direct_contradictions: list[tuple[str, str]] = field(default_factory=list)
    direct_corroborations: int = 0
    # Multi-hop evidence
    indirect_evidence: list[IndirectEvidence] = field(default_factory=list)
    predicted_predicate: str = ""
    # Gap analysis
    gaps_closed: int = 0
    gap_descriptions: list[str] = field(default_factory=list)
    # Suggestions
    follow_up_hypotheses: list[str] = field(default_factory=list)
    # Summary
    confidence_score: float = 0.0
    explanation: str = ""


@dataclass
class Prediction:
    """A novel predicted relationship from causal composition."""
    target: str                  # target entity ID
    predicted_predicate: str     # composed causal predicate
    supporting_paths: int        # number of supporting causal paths
    opposing_paths: int          # number of opposing causal paths
    consensus: float             # supporting / total ratio
    intermediaries: int          # unique intermediary entities
    is_gap: bool                 # True if no direct connection exists at all
    existing_predicates: list[str] = field(default_factory=list)  # existing non-causal preds
    evidence: list[IndirectEvidence] = field(default_factory=list)  # top paths


# --- Verification types ---


@dataclass
class VerificationResult:
    """Result of a single verification check."""
    check_name: str           # e.g., "provenance_check", "consistency_check"
    passed: bool
    confidence: float         # 0.0–1.0 — how confident the check is
    evidence: list[str] = field(default_factory=list)   # human-readable evidence
    claim_ids: list[str] = field(default_factory=list)   # related claims found
    metadata: dict = field(default_factory=dict)          # check-specific data


@dataclass
class VerificationVerdict:
    """Aggregated result of the verification pipeline."""
    verdict: str              # "VERIFIED" | "FAILED" | "DISPUTED"
    overall_confidence: float
    check_results: list[VerificationResult] = field(default_factory=list)
    recommended_status: str = "verified"  # the claim status to transition to
    tier_used: int = 1
    total_cost_usd: float = 0.0
    checks_skipped: list[str] = field(default_factory=list)
    verification_completeness: float = 1.0  # 1.0 if all checks ran


# --- Autonomous research types ---


@dataclass
class ResearchQuestion:
    """A question generated from a knowledge gap or blindspot."""
    entity_id: str
    entity_type: str
    gap_type: str           # "missing_predicate", "single_source", "low_confidence", "manual"
    question: str
    predicate_hint: str = ""
    inquiry_claim_id: str = ""


@dataclass
class ResearchResult:
    """Outcome of researching a single question."""
    question: ResearchQuestion
    claims_ingested: int = 0
    claims_rejected: int = 0
    inquiry_resolved: bool = False
    source: str = ""        # "llm", "search_fn", "slack"
    raw_response: str = ""


@dataclass
class InvestigationReport:
    """Summary of a full investigate() cycle."""
    questions_generated: int = 0
    questions_researched: int = 0
    claims_ingested: int = 0
    inquiries_resolved: int = 0
    results: list = field(default_factory=list)
    blindspot_before: int = 0
    blindspot_after: int = 0


@dataclass
class CloseGapsReport:
    """Summary of a close_gaps() cycle — hypothesis-driven or blindspot-driven."""
    hypothesis: str = ""
    verdict_before: str = ""
    verdict_after: str = ""
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    gaps_identified: int = 0
    gaps_researched: int = 0
    claims_ingested: int = 0
    investigations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    blindspot_before: int = 0
    blindspot_after: int = 0


# --- Knowledge-Intelligence API types ---


@dataclass
class EvidenceChain:
    """A chain of evidence hops supporting or contradicting a hypothesis."""
    steps: list[PathStep] = field(default_factory=list)
    propagated_confidence: float = 0.0
    direction: str = "supporting"  # "supporting" | "contradicting"
    summary: str = ""


@dataclass
class ConfidenceGap:
    """A gap or weakness in the evidence chain."""
    from_entity: str
    to_entity: str
    issue: str  # "missing" | "weak"
    current_confidence: float = 0.0
    explanation: str = ""


@dataclass
class HypothesisVerdict:
    """Result of evaluating a natural-language hypothesis against the graph."""
    hypothesis: str
    # "supported" | "contradicted" | "partial" | "unsupported" | "insufficient_data"
    verdict: str = "unsupported"
    verdict_confidence: float = 0.0
    supporting_chains: list[EvidenceChain] = field(default_factory=list)
    contradicting_chains: list[EvidenceChain] = field(default_factory=list)
    confidence_gaps: list[ConfidenceGap] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    entities_missing: list[str] = field(default_factory=list)
    suggested_next_steps: list[str] = field(default_factory=list)


@dataclass
class ConfidenceChange:
    """A confidence shift for a specific relationship over time."""
    content_id: str
    predicate: str
    target_entity: str
    confidence_before: float
    confidence_after: float
    delta: float
    reason: str = ""  # "new_corroboration" | "source_retracted"


@dataclass
class EvolutionReport:
    """Entity-specific knowledge evolution over time."""
    entity_id: str
    since_timestamp: int = 0
    new_connections: list[str] = field(default_factory=list)
    new_claims: int = 0
    retracted_claims: int = 0
    confidence_changes: list[ConfidenceChange] = field(default_factory=list)
    new_corroborations: int = 0
    source_types_before: list[str] = field(default_factory=list)
    source_types_after: list[str] = field(default_factory=list)
    new_source_types: list[str] = field(default_factory=list)
    trajectory: str = "stable"  # "growing" | "stable" | "declining"
    total_claims_before: int = 0
    total_claims_after: int = 0


@dataclass
class Investigation:
    """A prioritized investigation recommendation."""
    entity_id: str
    entity_type: str
    reason: str
    # "single_source" | "confidence_gap" | "missing_predicate"
    # | "predicted_link" | "stale"
    signal_type: str
    priority_score: float = 0.0
    affected_downstream: int = 0
    suggested_action: str = ""


@dataclass
class ReasoningHop:
    """A single hop in a reasoning chain with full evidence."""
    from_entity: str
    to_entity: str
    predicate: str
    confidence: float
    source_types: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    evidence_text: str = ""
    has_contradiction: bool = False
    contradiction_predicate: str = ""


@dataclass
class SourceOverlap:
    """Source overlap between two hops in a reasoning chain."""
    hop_a_index: int
    hop_b_index: int
    shared_sources: list[str] = field(default_factory=list)
    discount_factor: float = 1.0


@dataclass
class ReasoningChain:
    """A reasoning chain with source-overlap-discounted confidence."""
    hops: list[ReasoningHop] = field(default_factory=list)
    raw_confidence: float = 0.0
    source_overlaps: list[SourceOverlap] = field(default_factory=list)
    chain_confidence: float = 0.0
    length: int = 0
    reliability_score: float = 0.0


# --- Discovery engine types ---


@dataclass
class Discovery:
    """A proactive hypothesis generated from graph structure."""
    hypothesis: str                # "BRCA1 may inhibit apoptosis via p53"
    evidence_summary: str          # "3 bridging paths through p53..."
    discovery_type: str            # "bridge_prediction" | "cross_domain" | "chain_completion"
    confidence: float              # 0-1
    novelty_score: float           # 0-1
    entities: list[str]
    supporting_paths: list[dict] = field(default_factory=list)
    suggested_action: str = ""     # "Search for evidence of X inhibits Y"
    predicted_predicate: str = ""  # "inhibits"
    entity_types: dict[str, str] = field(default_factory=dict)


@dataclass
class Analogy:
    """A structural analogy: A:B :: C:D predicted from embeddings."""
    entity_a: str                  # Source A
    entity_b: str                  # Source B
    entity_c: str                  # Analog of A
    entity_d: str                  # Analog of B (predicted)
    predicted_predicate: str
    score: float
    explanation: str
    source_predicates: list[str] = field(default_factory=list)
    entity_types: dict[str, str] = field(default_factory=dict)


# --- Crown jewels types ---


@dataclass
class BeliefChange:
    """A single belief that changed between two time periods."""
    content_id: str
    subject: str
    predicate: str
    object: str
    change_type: str       # "new" | "strengthened" | "weakened" | "contradicted"
    claims_before: int
    claims_after: int
    confidence_before: float
    confidence_after: float
    new_sources: list[str] = field(default_factory=list)


@dataclass
class KnowledgeDiff:
    """Knowledge diff between two time periods — like git diff for knowledge."""
    since: int
    until: int
    new_beliefs: list[BeliefChange] = field(default_factory=list)
    strengthened: list[BeliefChange] = field(default_factory=list)
    weakened: list[BeliefChange] = field(default_factory=list)
    new_contradictions: list[BeliefChange] = field(default_factory=list)
    new_entities: list[str] = field(default_factory=list)
    new_sources: list[str] = field(default_factory=list)
    total_new_claims: int = 0
    total_retracted: int = 0
    net_confidence: float = 0.0
    summary: str = ""


@dataclass
class ContradictionSide:
    """One side of a contradiction with evidence metrics."""
    predicate: str
    claim_count: int
    source_count: int
    source_types: list[str] = field(default_factory=list)
    avg_confidence: float = 0.0
    corroboration_count: int = 0
    newest_timestamp: int = 0
    evidence_weight: float = 0.0


@dataclass
class ContradictionAnalysis:
    """Analysis of a single contradiction between opposing predicates."""
    subject: str
    object: str
    side_a: ContradictionSide = field(default_factory=lambda: ContradictionSide("", 0, 0))
    side_b: ContradictionSide = field(default_factory=lambda: ContradictionSide("", 0, 0))
    winner: str = "ambiguous"     # "side_a" | "side_b" | "ambiguous"
    margin: float = 0.0
    resolution: str = "unresolved"  # "a_preferred" | "b_preferred" | "unresolved"
    explanation: str = ""


@dataclass
class ContradictionReport:
    """Report on all contradictions found and resolved."""
    total_found: int = 0
    resolved: int = 0
    ambiguous: int = 0
    analyses: list[ContradictionAnalysis] = field(default_factory=list)
    claims_added: int = 0


@dataclass
class ConnectionLoss:
    """A pair of entities that lose connectivity in a simulation."""
    entity_a: str
    entity_b: str
    lost_predicates: list[str] = field(default_factory=list)


@dataclass
class ConfidenceShift:
    """A confidence change for a content_id in a simulation."""
    content_id: str
    subject: str
    predicate: str
    object: str
    confidence_before: float
    confidence_after: float


@dataclass
class SimulationReport:
    """Result of a counterfactual simulation — what-if analysis."""
    scenario: str = ""
    claims_affected: int = 0
    claims_removed: int = 0
    claims_degraded: int = 0
    entities_affected: list[str] = field(default_factory=list)
    entities_now_orphaned: int = 0
    entities_now_single_source: int = 0
    connection_losses: list[ConnectionLoss] = field(default_factory=list)
    confidence_shifts: list[ConfidenceShift] = field(default_factory=list)
    new_contradictions: list[str] = field(default_factory=list)
    new_corroborations: int = 0
    gaps_closed: int = 0
    risk_score: float = 0.0
    risk_level: str = "low"
    summary: str = ""


@dataclass
class Citation:
    """A cited claim with provenance metadata."""
    claim_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.0
    source_id: str = ""
    source_type: str = ""
    corroboration_count: int = 0


@dataclass
class AskResult:
    """Structured answer from db.ask() with citations, contradictions, and gaps.

    Dict-compatible for backward compatibility — supports r["answer"], r.get("key"), etc.
    """
    answer: str | None = None
    citations: list[Citation] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    entities: list = field(default_factory=list)
    evidence: str = ""
    meta: dict = field(default_factory=dict)

    # Dict-like backward compatibility
    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def keys(self):
        return ("answer", "citations", "contradictions", "gaps", "entities", "evidence", "meta")


@dataclass
class BriefSection:
    """A section of a knowledge brief covering a cluster of entities."""
    title: str
    entities: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    avg_confidence: float = 0.0


@dataclass
class KnowledgeBrief:
    """Structured research brief compiled from the knowledge graph."""
    topic: str = ""
    sections: list[BriefSection] = field(default_factory=list)
    executive_summary: str = ""
    total_entities: int = 0
    total_claims_cited: int = 0
    total_sources: int = 0
    avg_confidence: float = 0.0
    total_contradictions: int = 0
    total_gaps: int = 0
    strongest_findings: list[str] = field(default_factory=list)
    weakest_areas: list[str] = field(default_factory=list)


@dataclass
class ExplanationStep:
    """A single hop in an explain_why chain."""
    from_entity: str
    to_entity: str
    predicate: str
    confidence: float
    source_summary: str = ""
    evidence_text: str = ""
    has_contradiction: bool = False


@dataclass
class Explanation:
    """Full provenance-traced explanation of how two entities are connected."""
    entity_a: str
    entity_b: str
    connected: bool = False
    steps: list[ExplanationStep] = field(default_factory=list)
    chain_confidence: float = 0.0
    narrative: str = ""
    alternative_paths: int = 0
    source_count: int = 0


@dataclass
class ForecastConnection:
    """A predicted future connection for an entity."""
    target_entity: str
    target_type: str
    predicted_predicate: str
    score: float = 0.0
    reason: str = ""
    evidence_entities: list[str] = field(default_factory=list)


@dataclass
class Forecast:
    """Predicted next connections for an entity based on graph structure."""
    entity_id: str
    predictions: list[ForecastConnection] = field(default_factory=list)
    growth_rate: float = 0.0       # connections per month
    trajectory: str = "stable"     # "growing" | "stable" | "declining"
    total_current_connections: int = 0


@dataclass
class AutodidactConfig:
    """Configuration for the autodidact self-learning daemon."""

    interval_seconds: float = 3600
    max_questions_per_cycle: int = 5
    max_llm_calls_per_day: int = 100
    max_cost_per_day: float = 1.00     # USD hard cap — stops when reached
    gap_types: list[str] | None = None
    entity_types: list[str] | None = None
    use_curator: bool = True
    jitter: float = 0.1
    negative_result_limit: int = 3
    enabled_triggers: list[str] = field(
        default_factory=lambda: ["timer", "retraction", "inquiry"]
    )
    # Per-operation cost estimates (USD). Override if your LLM provider differs.
    cost_search_paid: float = 0.002       # Perplexity/Serper per call
    cost_ingest_text: float = 0.005       # LLM extraction call
    cost_curator_per_claim: float = 0.001  # Curator triage per claim
    trigger_cooldown: float = 60.0        # Seconds between event-triggered cycles
    max_history: int = 100                # Max cycle reports kept in memory


@dataclass
class CycleReport:
    """Report from a single autodidact cycle."""

    cycle_number: int
    started_at: float
    finished_at: float
    tasks_generated: int = 0
    tasks_researched: int = 0
    claims_ingested: int = 0
    claims_rejected: int = 0
    negative_results: int = 0
    llm_calls: int = 0
    estimated_cost: float = 0.0        # USD estimated for this cycle
    blindspot_before: int = 0
    blindspot_after: int = 0
    trigger: str = "timer"
    errors: list[str] = field(default_factory=list)


@dataclass
class AutodidactStatus:
    """Status of the autodidact daemon."""

    enabled: bool = False
    running: bool = False
    cycle_count: int = 0
    total_claims_ingested: int = 0
    total_llm_calls_today: int = 0
    estimated_cost_today: float = 0.0  # USD estimated spend today
    max_cost_per_day: float = 1.00     # USD cap
    budget_exhausted: bool = False
    last_cycle: CycleReport | None = None
    next_cycle_at: float = 0.0


@dataclass
class RegimeShift:
    """A detected shift in temporal pattern."""

    index: int
    timestamp: int  # nanoseconds
    confidence: float
    direction: str = ""  # "up", "down", or ""


@dataclass
class VelocityStats:
    """Rate-of-change statistics."""

    mean_velocity: float = 0.0
    max_velocity: float = 0.0
    min_velocity: float = 0.0
    std_velocity: float = 0.0
    mean_acceleration: float = 0.0


@dataclass
class DetectedCycle:
    """A detected periodic pattern."""

    period: float  # in time-bucket units
    power: float
    is_harmonic: bool = False
    harmonic_of: float | None = None


@dataclass
class TemporalResult:
    """Result of temporal analysis on an entity's claims."""

    entity_id: str
    analysis_type: str  # "regime_shifts", "velocity", "cycles", "summary"
    num_claims: int = 0
    time_span_days: float = 0.0
    bucket_size: str = ""  # "day", "week", "month"
    num_buckets: int = 0

    # Regime shifts
    regime_shifts: list[RegimeShift] = field(default_factory=list)
    num_shifts: int = 0

    # Velocity
    velocity: VelocityStats | None = None

    # Cycles
    cycles: list[DetectedCycle] = field(default_factory=list)
    dominant_period: float | None = None

    # Overall
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class JudgeVote:
    """A single provider's judgment on whether the group has converged."""
    provider: str
    converged: bool                  # does this provider think we've converged?
    best_provider: str = ""          # who gave the best answer?
    rating: int = 0                  # 1-10 agreement rating
    critique: str = ""               # what's still wrong / missing


@dataclass
class ProviderResponse:
    """A single LLM provider's response in a consensus round."""
    provider: str
    model: str
    response: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    error: str = ""
    round_number: int = 1
    cost_usd: float = 0.0


@dataclass
class AgentConsensusResult:
    """Result of multi-LLM consensus: synthesized answer with provenance."""
    question: str
    consensus: str                          # final synthesized answer
    confidence: float                       # 0-1 agreement level
    rounds: int                             # how many rounds it took
    providers_used: list[str] = field(default_factory=list)
    responses: list[ProviderResponse] = field(default_factory=list)
    dissents: list[str] = field(default_factory=list)
    votes: list[JudgeVote] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    converged: bool = False


@dataclass
class TrustRule:
    """One rule in a trust policy. First matching rule wins."""
    action: str = "include"  # "include" or "exclude"
    source_types: list[str] | None = None
    source_ids: list[str] | None = None
    predicates: list[str] | None = None
    min_confidence: float | None = None
    min_corroboration: int | None = None


@dataclass
class MergeConflict:
    """A belief where two databases disagree."""
    content_id: str
    subject: str
    predicate: str
    object: str
    self_confidence: float
    other_confidence: float
    self_sources: int
    other_sources: int


@dataclass
class MergeReport:
    """Comparison of two knowledge bases — what each knows that the other doesn't."""
    self_unique_beliefs: int = 0
    other_unique_beliefs: int = 0
    shared_beliefs: int = 0
    conflicts: list[MergeConflict] = field(default_factory=list)
    self_unique_entities: list[str] = field(default_factory=list)
    other_unique_entities: list[str] = field(default_factory=list)
    shared_entities: list[str] = field(default_factory=list)
    self_total_claims: int = 0
    other_total_claims: int = 0
    summary: str = ""


# --- Dict-to-dataclass converters for Rust PyO3 layer ---


def claim_from_dict(d: dict) -> Claim:
    """Convert a dict from the Rust store into a Claim dataclass."""
    try:
        return _claim_from_dict_inner(d)
    except KeyError as exc:
        raise ValueError(
            f"claim_from_dict: missing required key {exc} in claim dict "
            f"(keys present: {sorted(d.keys())})"
        ) from exc


def _claim_from_dict_inner(d: dict) -> Claim:
    """Inner implementation — raises KeyError on missing fields."""
    subj = d["subject"]
    subject = EntityRef(
        id=subj["id"],
        entity_type=subj["entity_type"],
        display_name=subj.get("display_name") or subj.get("name", ""),
        external_ids=subj.get("external_ids", {}),
    )

    pred = d["predicate"]
    predicate = PredicateRef(id=pred["id"], predicate_type=pred["predicate_type"])

    obj = d["object"]
    object_ = EntityRef(
        id=obj["id"],
        entity_type=obj["entity_type"],
        display_name=obj.get("display_name") or obj.get("name", ""),
        external_ids=obj.get("external_ids", {}),
    )

    prov = d["provenance"]
    provenance = Provenance(
        source_type=prov["source_type"],
        source_id=prov["source_id"],
        method=prov.get("method"),
        chain=prov.get("chain", []),
        model_version=prov.get("model_version"),
        organization=prov.get("organization"),
    )

    payload = None
    if d.get("payload") is not None:
        pl = d["payload"]
        if "schema_ref" in pl:
            if "data_json" in pl:
                try:
                    data = json.loads(pl["data_json"])
                except (json.JSONDecodeError, TypeError):
                    data = {}
            else:
                data = pl.get("data", {})
            payload = Payload(schema_ref=pl["schema_ref"], data=data)

    status_str = d.get("status", "active")
    try:
        status = ClaimStatus(status_str)
    except ValueError:
        status = ClaimStatus.ACTIVE

    # Security fields (Python-only, not in Rust engine)
    sens_raw = d.get("sensitivity", 0)
    try:
        sensitivity = SensitivityLevel(sens_raw) if isinstance(sens_raw, int) else SensitivityLevel[sens_raw.upper()]
    except (ValueError, KeyError):
        sensitivity = SensitivityLevel.PUBLIC

    return Claim(
        claim_id=d["claim_id"],
        content_id=d["content_id"],
        subject=subject,
        predicate=predicate,
        object=object_,
        confidence=d["confidence"],
        provenance=provenance,
        payload=payload,
        timestamp=d["timestamp"],
        status=status,
        namespace=d.get("namespace", ""),
        expires_at=d.get("expires_at", 0),
        sensitivity=sensitivity,
        owner=d.get("owner"),
        acl=d.get("acl", []),
    )


@dataclass
class EvalItem:
    """A single evaluation question with expected answer."""
    question: str
    expected_answer: str
    supporting_claims: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    domain: str = ""
    eval_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalSet:
    """A versioned set of evaluation items for a domain."""
    eval_id: str
    items: list[EvalItem] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    created_at: int = 0
    version: int = 1
    n_easy: int = 0
    n_medium: int = 0
    n_hard: int = 0


@dataclass
class EvalResult:
    """Result of an agent taking an eval."""
    eval_id: str
    agent_id: str
    score: float = 0.0
    total: int = 0
    correct: int = 0
    per_item: list[dict] = field(default_factory=list)
    domain_scores: dict[str, float] = field(default_factory=dict)
    type_scores: dict[str, float] = field(default_factory=dict)
    elapsed: float = 0.0
    timestamp: int = 0


def entity_summary_from_dict(d: dict) -> EntitySummary:
    """Convert a dict from the Rust store into an EntitySummary dataclass."""
    return EntitySummary(
        id=d["id"],
        name=d["name"],
        entity_type=d["entity_type"],
        external_ids=d.get("external_ids", {}),
        claim_count=d.get("claim_count", 0),
    )
