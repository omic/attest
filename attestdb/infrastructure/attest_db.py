"""Top-level AttestDB class — the user-facing API."""

from __future__ import annotations

import builtins
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.connectors.base import Connector
    from attestdb.core.types import (
        AuditTrail,
        AutodidactConfig,
        AutodidactStatus,
        BlindspotMap,
        ConsensusReport,
        CycleReport,
        DriftReport,
        HypotheticalReport,
        ImpactReport,
        InvestigationReport,
        ResearchResult,
    )
    from attestdb.infrastructure.autodidact import AutodidactDaemon

from attestdb.core.hashing import compute_chain_hash, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.infrastructure.event_bus import EventType
from attestdb.core.types import (
    Analogy,
    AskResult,
    AuditEvent,
    BatchResult,
    Role,
    BeliefChange,
    BriefSection,
    CascadeResult,
    Citation,
    Claim,
    ClaimInput,
    ClaimStatus,
    ConfidenceChange,
    ConfidenceGap,
    ConfidenceShift,
    ConnectionLoss,
    ContextFrame,
    ContradictionAnalysis,
    ContradictionReport,
    ContradictionSide,
    CrossDomainBridge,
    DensityMapEntry,
    Discovery,
    DownstreamNode,
    EntitySummary,
    EvidenceChain,
    EvolutionReport,
    Explanation,
    ExplanationStep,
    Forecast,
    ForecastConnection,
    HypothesisVerdict,
    Investigation,
    KnowledgeBrief,
    KnowledgeDiff,
    KnowledgeHealth,
    MergeConflict,
    MergeReport,
    PathResult,
    QualityReport,
    QueryProfile,
    ReasoningChain,
    ReasoningHop,
    RelationshipPattern,
    RetractResult,
    SchemaDescriptor,
    SimulationReport,
    SourceOverlap,
    TopicNode,
    claim_from_dict,
    entity_summary_from_dict,
)
from attestdb.infrastructure.embedding_index import EmbeddingIndex, MultiSpaceEmbeddingIndex
from attestdb.infrastructure.entity_manager import EntityManager
from attestdb.infrastructure.ingestion import IngestionPipeline
from attestdb.infrastructure.query_engine import QueryEngine
from attestdb.infrastructure.tracing import span as _span, set_attribute as _set_attr, record_exception as _record_exc

logger = logging.getLogger(__name__)

# ── Defaults (extracted from method signatures) ─────────────────────────
DEFAULT_ITER_BATCH_SIZE = 10_000


def _claim_is_expired(claim, now_ns: int) -> bool:
    """Return True if a claim has a TTL and has expired."""
    expires_at = getattr(claim, "expires_at", 0) or 0
    return expires_at > 0 and now_ns >= expires_at
DEFAULT_MAX_CLAIMS = 500
DEFAULT_MAX_TOKENS = 4000
FRESHNESS_HALF_LIFE_DAYS = 30


def _make_store(db_path: str, *, read_only: bool = False):
    """Create the raw Rust store backend."""
    from attest_rust import RustStore
    if db_path == ":memory:":
        return RustStore.in_memory()
    rust_path = db_path if db_path.endswith((".substrate", ".attest")) else db_path + ".attest"
    if read_only:
        return RustStore.open_read_only(rust_path)
    return RustStore(rust_path)


class HypotheticalContext:
    """Sandbox for speculative what-if reasoning.

    Created by AttestDB.hypothetical(). Supports context manager protocol.
    """

    def __init__(self, sandbox: "AttestDB", parent: "AttestDB", description: str):
        self.sandbox = sandbox
        self._parent = parent
        self.description = description
        self._parent_claim_ids = {
            d["claim_id"]
            for d in parent._store.all_claims(0, 0)
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.sandbox.close()

    @property
    def new_claims(self) -> list[Claim]:
        """Claims in sandbox that aren't in parent (by claim_id)."""
        result = []
        for d in self.sandbox._store.all_claims(0, 0):
            if d["claim_id"] not in self._parent_claim_ids:
                result.append(claim_from_dict(d))
        return result

    def find_contradictions(self) -> list[tuple[Claim, Claim]]:
        """Find pairs where same (subject, object) have opposing predicates."""
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        contradictions = []
        new = self.new_claims
        for nc in new:
            pred = nc.predicate.id
            opposite = OPPOSITE_PREDICATES.get(pred)
            if not opposite:
                continue
            # Check sandbox for claims with opposite predicate on same entities
            for d in self.sandbox._store.claims_for(nc.subject.id, None, None, 0.0):
                existing = claim_from_dict(d)
                if existing.claim_id == nc.claim_id:
                    continue
                if (existing.predicate.id == opposite
                        and existing.object.id == nc.object.id):
                    contradictions.append((nc, existing))
        return contradictions

    def promote(self, min_confidence: float = 0.0) -> int:
        """Merge new claims back to parent. Returns count promoted."""
        count = 0
        for claim in self.new_claims:
            if claim.confidence < min_confidence:
                continue
            self._parent.ingest(
                subject=(claim.subject.id, claim.subject.entity_type),
                predicate=(claim.predicate.id, claim.predicate.predicate_type),
                object=(claim.object.id, claim.object.entity_type),
                provenance={
                    "source_type": claim.provenance.source_type,
                    "source_id": claim.provenance.source_id,
                    "method": "hypothetical_reasoning",
                },
                confidence=claim.confidence,
            )
            count += 1
        return count

    def report(self) -> str:
        """Summary of new claims and contradictions found."""
        new = self.new_claims
        contradictions = self.find_contradictions()
        lines = [f'Hypothesis: "{self.description}"']
        lines.append(f"New claims in sandbox: {len(new)}")
        for c in new:
            lines.append(f"  + {c.subject.id} —{c.predicate.id}→ {c.object.id} (conf: {c.confidence:.2f})")
        if contradictions:
            lines.append(f"Contradictions found: {len(contradictions)}")
            for nc, existing in contradictions:
                lines.append(
                    f"  ✗ {nc.predicate.id} vs {existing.predicate.id}"
                    f" on {nc.subject.id}→{nc.object.id}"
                    f" (existing conf: {existing.confidence:.2f})"
                )
        else:
            lines.append("Contradictions: none")
        return "\n".join(lines)

    def analyze(self):
        """Rich analysis of all new claims in the sandbox.

        Returns a SandboxVerdict with direct contradictions, multi-hop evidence,
        gap analysis, follow-up suggestions, and an overall verdict — all purely
        graph-structural, no LLM calls.
        """
        from attestdb.core.types import IndirectEvidence, SandboxVerdict
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES, compose_predicates

        new = self.new_claims
        if not new:
            return SandboxVerdict(hypothesis=self.description, explanation="No claims to analyze.")

        all_direct_contradictions: list[tuple[str, str]] = []
        total_corroborations = 0
        all_indirect: list[IndirectEvidence] = []
        gaps_closed = 0
        gap_descs: list[str] = []
        follow_ups: list[str] = []

        for nc in new:
            subj_id = nc.subject.id
            obj_id = nc.object.id
            hyp_pred = nc.predicate.id

            # 1. Direct contradictions
            opposite = OPPOSITE_PREDICATES.get(hyp_pred)
            if opposite:
                for d in self.sandbox._store.claims_for(subj_id, None, None, 0.0):
                    existing = claim_from_dict(d)
                    if existing.claim_id == nc.claim_id:
                        continue
                    if existing.predicate.id == opposite and existing.object.id == obj_id:
                        all_direct_contradictions.append((hyp_pred, existing.predicate.id))

            # 2. Direct corroborations (same content_id in parent)
            content_id = nc.content_id
            parent_matches = self._parent.claims_by_content_id(content_id)
            total_corroborations += len(parent_matches)

            # 3. Multi-hop evidence via find_paths on parent
            try:
                paths = self._parent.find_paths(subj_id, obj_id, max_depth=3, top_k=5)
            except Exception:
                paths = []

            for path in paths:
                if len(path.steps) < 2:
                    continue
                entity_ids = [path.steps[0].entity_id] + [s.entity_id for s in path.steps[1:]]
                preds = [s.predicate for s in path.steps]
                # Compose predicates along the chain
                composed = preds[0]
                for p in preds[1:]:
                    composed = compose_predicates(composed, p)
                # Determine direction
                if composed == hyp_pred:
                    direction = "supporting"
                elif OPPOSITE_PREDICATES.get(composed) == hyp_pred or OPPOSITE_PREDICATES.get(hyp_pred) == composed:
                    direction = "contradicting"
                else:
                    direction = "neutral"
                all_indirect.append(IndirectEvidence(
                    path=entity_ids,
                    predicates=preds,
                    predicted_predicate=composed,
                    confidence=path.total_confidence,
                    direction=direction,
                ))

            # 4. Gap detection — does this hypothesis bridge disconnected entities?
            subj_exists = self._parent._store.get_entity(
                normalize_entity_id(subj_id)) is not None
            obj_exists = self._parent._store.get_entity(
                normalize_entity_id(obj_id)) is not None
            if subj_exists and obj_exists:
                if not self._parent.path_exists(subj_id, obj_id, max_depth=2):
                    gaps_closed += 1
                    gap_descs.append(
                        f"{subj_id} and {obj_id} exist but are not directly connected"
                    )

            # 5. Follow-up suggestions
            follow_ups.extend(self._suggest_nearby(subj_id, obj_id))

        # Deduplicate follow-ups
        seen = set()
        unique_follow_ups = []
        for f in follow_ups:
            if f not in seen:
                seen.add(f)
                unique_follow_ups.append(f)
        follow_ups = unique_follow_ups[:5]

        # Best predicted predicate from indirect evidence
        supporting = [ie for ie in all_indirect if ie.direction == "supporting"]
        contradicting = [ie for ie in all_indirect if ie.direction == "contradicting"]
        if supporting:
            predicted = supporting[0].predicted_predicate
        elif all_indirect:
            predicted = all_indirect[0].predicted_predicate
        else:
            predicted = ""

        # Score and verdict
        score = 0.5  # base
        score += min(total_corroborations * 0.15, 0.3)
        score += min(len(supporting) * 0.1, 0.2)
        score -= min(len(all_direct_contradictions) * 0.25, 0.5)
        score -= min(len(contradicting) * 0.1, 0.2)
        score += min(gaps_closed * 0.05, 0.1)
        score = max(0.0, min(1.0, score))

        if all_direct_contradictions:
            verdict = "contradicted"
        elif total_corroborations > 0 and not contradicting:
            verdict = "supported"
        elif supporting or gaps_closed > 0:
            verdict = "plausible"
        else:
            verdict = "insufficient_data"

        # Build explanation
        parts = []
        if all_direct_contradictions:
            parts.append(f"{len(all_direct_contradictions)} direct contradiction(s)")
        if total_corroborations:
            parts.append(f"{total_corroborations} existing corroboration(s)")
        if supporting:
            parts.append(f"{len(supporting)} supporting indirect path(s)")
        if contradicting:
            parts.append(f"{len(contradicting)} contradicting indirect path(s)")
        neutral = [ie for ie in all_indirect if ie.direction == "neutral"]
        if neutral and not supporting and not contradicting:
            parts.append(f"{len(neutral)} indirect path(s) (neutral)")
        if gaps_closed:
            parts.append(f"closes {gaps_closed} gap(s)")
        explanation = "; ".join(parts) if parts else "No evidence found in graph."

        return SandboxVerdict(
            hypothesis=self.description,
            verdict=verdict,
            direct_contradictions=all_direct_contradictions,
            direct_corroborations=total_corroborations,
            indirect_evidence=all_indirect,
            predicted_predicate=predicted,
            gaps_closed=gaps_closed,
            gap_descriptions=gap_descs,
            follow_up_hypotheses=follow_ups,
            confidence_score=score,
            explanation=explanation,
        )

    def _suggest_nearby(self, subj_id: str, obj_id: str) -> list[str]:
        """Find entities 1-hop from subject/object in parent that are unconnected."""
        subj_neighbors: set[str] = set()
        obj_neighbors: set[str] = set()
        subj_raw = self._parent._store.claims_for(subj_id, None, None, 0.0)
        if len(subj_raw) > 200:
            subj_raw = subj_raw[:200]
        for d in subj_raw:
            c = claim_from_dict(d)
            subj_neighbors.add(c.object.id if c.subject.id == normalize_entity_id(subj_id) else c.subject.id)
        obj_raw = self._parent._store.claims_for(obj_id, None, None, 0.0)
        if len(obj_raw) > 200:
            obj_raw = obj_raw[:200]
        for d in obj_raw:
            c = claim_from_dict(d)
            obj_neighbors.add(c.object.id if c.subject.id == normalize_entity_id(obj_id) else c.subject.id)

        suggestions = []
        for sn in subj_neighbors:
            if sn == obj_id or sn == subj_id:
                continue
            for on in obj_neighbors:
                if on == subj_id or on == obj_id or on == sn:
                    continue
                if not self._parent.path_exists(sn, on, max_depth=1):
                    suggestions.append(f"{sn} may relate to {on}")
                    if len(suggestions) >= 5:
                        return suggestions
        return suggestions

    def ingest_and_analyze(self, subject, predicate, object_, confidence=0.6,
                           source_type="hypothesis", source_id="agent"):
        """Add a hypothesis claim to the sandbox and immediately analyze.

        Args:
            subject: (id, entity_type) tuple
            predicate: (id, predicate_type) tuple
            object_: (id, entity_type) tuple
            confidence: claim confidence (default 0.6)
            source_type: provenance source type
            source_id: provenance source id

        Returns:
            SandboxVerdict with full analysis.
        """
        self.sandbox.ingest(
            subject=subject,
            predicate=predicate,
            object=object_,
            provenance={"source_type": source_type, "source_id": source_id},
            confidence=confidence,
        )
        return self.analyze()

    def report_dict(self) -> dict:
        """Machine-readable report for MCP serialization."""
        from dataclasses import asdict
        verdict = self.analyze()
        return asdict(verdict)


class AttestDB:
    """Main entry point for Attest — wraps all infrastructure components."""

    @classmethod
    def open_read_only(cls, db_path: str) -> "AttestDB":
        """Open a database in native read-only mode (no lock contention).

        Uses LMDB's read-only mode which acquires only a shared lock.
        Multiple readers can coexist. All write operations will raise
        an error.
        """
        instance = cls.__new__(cls)
        instance._db_path = db_path
        instance._embedding_dim = None
        instance._embedding_spaces = None
        instance._strict = False
        instance._store = _make_store(db_path, read_only=True)
        # Read-only: no sidecar migration needed (status is in Rust engine)
        instance._multi_embedding_index = None
        instance._embedding_index = None
        instance._pipeline = None
        instance._py_status_overrides = {}
        instance._load_py_status_overrides()

        def _converting(d):
            claim = claim_from_dict(d)
            override = instance._py_status_overrides.get(claim.claim_id)
            if override is not None:
                claim.status = override
            return claim
        instance._query_engine = QueryEngine(
            instance._store,
            claim_converter=_converting,
        )
        # Composed subsystem objects (read-only: minimal setup)
        from attestdb.infrastructure.audit_log import AuditLog, OpsLog
        from attestdb.infrastructure.webhooks import WebhookManager
        from attestdb.infrastructure.rbac import RBACManager
        from attestdb.infrastructure.event_bus import EventBus

        instance._audit = AuditLog(db_path, True)  # read-only = in-memory audit
        instance._ops_log = OpsLog(db_path, True)
        instance._webhooks_mgr = WebhookManager(None)
        instance._rbac_mgr = RBACManager(None, instance._audit, lambda: instance._store.get_namespace_filter())
        instance._events = EventBus(instance._webhooks_mgr, lambda cid: instance.get_claim(cid))
        instance._entity_mgr = EntityManager(instance)

        instance._scheduler = None
        instance._purge_timer = None
        instance._purge_interval = 0
        instance._chain_log = []
        instance._last_chain_hash = "genesis"
        instance._autodidact = None
        instance._federation = None
        instance._decay_config = None
        instance._trust_policy = None
        from attestdb.infrastructure.intelligence_gateway import IntelligenceGateway
        instance._intel = IntelligenceGateway(instance._store, db=instance)
        instance._cache = {}
        instance._cache_ts = {}
        instance._verification_budget = None
        instance._verification_spent = 0.0
        from attestdb.core.security import SecurityConfig, SecurityAuditLogger
        instance._security = SecurityConfig()
        instance._security_meta = {}
        instance._security_path = None
        instance._security_audit = SecurityAuditLogger("./attestdb_audit.log", enabled=False)
        instance._entity_thread_index = {}
        from attestdb.infrastructure.topic_threads import TopicThreads
        instance._threads = TopicThreads(instance)
        instance._compute_backend = None
        from attestdb.infrastructure.analytics import AnalyticsEngine
        instance._analytics = AnalyticsEngine(instance)
        from attestdb.infrastructure.verification_engine import VerificationEngine
        instance._verification = VerificationEngine(instance)
        from attestdb.infrastructure.security import SecurityLayer
        instance._security_layer = SecurityLayer(instance)
        from attestdb.infrastructure.ask_engine import AskEngine
        instance._ask_engine = AskEngine(instance)
        from attestdb.infrastructure.eval_engine import EvalEngine
        instance._eval_engine = EvalEngine(instance)
        from attestdb.infrastructure.agent_registry import AgentRegistry
        instance._agent_registry = AgentRegistry(instance)
        return instance

    def __init__(
        self,
        db_path: str,
        embedding_dim: int | None = 768,
        strict: bool = False,
        embedding_spaces: dict[str, int] | None = None,
    ):
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._strict = strict
        self._embedding_spaces = embedding_spaces

        self._store = _make_store(db_path)

        # Migrate legacy retraction sidecar to Rust engine (one-time)
        self._migrate_retraction_sidecar()

        # Auto-backfill analytics indexes on first open (idempotent, no-op if already done)
        if hasattr(self._store, "backfill_pred_id_counts"):
            try:
                self._store.backfill_pred_id_counts()
            except Exception:
                pass  # Non-critical — analytics degrade gracefully
        if hasattr(self._store, "backfill_claim_summaries"):
            try:
                self._store.backfill_claim_summaries()
            except Exception:
                pass

        # Embedding index — multi-space or single-space
        self._multi_embedding_index: MultiSpaceEmbeddingIndex | None = None
        self._embedding_index: EmbeddingIndex | None = None
        if embedding_spaces:
            # Multi-space mode
            manifest_path = db_path + ".usearch.manifest.json"
            if os.path.exists(manifest_path) or os.path.exists(db_path + ".usearch"):
                self._multi_embedding_index = MultiSpaceEmbeddingIndex.load_from(db_path)
                # Ensure all requested spaces exist
                for name, ndim in embedding_spaces.items():
                    if name not in self._multi_embedding_index._spaces:
                        self._multi_embedding_index._spaces[name] = EmbeddingIndex(ndim=ndim)
                        self._multi_embedding_index._dims[name] = ndim
            else:
                self._multi_embedding_index = MultiSpaceEmbeddingIndex(spaces=embedding_spaces)
            # Default space acts as the single embedding_index for backward compat
            self._embedding_index = self._multi_embedding_index._spaces.get("default")
            if self._embedding_index:
                self._embedding_dim = self._multi_embedding_index._dims.get("default", embedding_dim)
        elif embedding_dim:
            self._embedding_index = EmbeddingIndex(ndim=embedding_dim)
            index_path = db_path + ".usearch"
            if os.path.exists(index_path):
                self._embedding_index = EmbeddingIndex.load_from(index_path)

        # Ingestion pipeline
        self._pipeline = IngestionPipeline(
            self._store, self._embedding_index, embedding_dim, strict
        )

        # Python-level status overlays for statuses the Rust fast path doesn't track.
        # The Rust engine's apply_status_overlay_fast only handles tombstoned via
        # retracted_ids HashSet. Other status overlays (archived, disputed, etc.) are
        # only read by get_claim() (which checks the LMDB status_overrides table).
        # For bulk reads (claims_for, all_claims), we apply the overlay in Python.
        self._py_status_overrides: dict[str, ClaimStatus] = {}
        self._load_py_status_overrides()

        # Query engine — converter applies Python-level status overlays
        def _converting(d):
            claim = claim_from_dict(d)
            override = self._py_status_overrides.get(claim.claim_id)
            if override is not None:
                claim.status = override
            return claim
        self._query_engine = QueryEngine(self._store, claim_converter=_converting)

        # Composed subsystem objects
        from attestdb.infrastructure.audit_log import AuditLog, OpsLog
        from attestdb.infrastructure.webhooks import WebhookManager
        from attestdb.infrastructure.rbac import RBACManager
        from attestdb.infrastructure.event_bus import EventBus

        is_memory = self._is_memory_db
        self._audit = AuditLog(self._db_path, is_memory)
        self._ops_log = OpsLog(self._db_path, is_memory)
        webhooks_path = None if is_memory else self._db_path + ".webhooks.json"
        failure_log_path = None if is_memory else self._db_path + ".webhooks_failed.jsonl"
        self._webhooks_mgr = WebhookManager(webhooks_path, failure_log_path)
        rbac_path = None if is_memory else self._db_path + ".rbac.json"
        self._rbac_mgr = RBACManager(rbac_path, self._audit, lambda: self._store.get_namespace_filter())
        self._events = EventBus(self._webhooks_mgr, lambda cid: self.get_claim(cid))
        self._entity_mgr = EntityManager(self)

        # Continuous ingestion scheduler (lazy-init)
        self._scheduler = None

        # Auto-purge timer for TTL expiry
        self._purge_timer: threading.Timer | None = None
        self._purge_interval: float = 0

        # Tamper-evident audit chain (Merkle hash chain over claim IDs)
        self._chain_log: list[tuple[str, str]] = []  # [(claim_id, chain_hash), ...]
        self._last_chain_hash: str = "genesis"
        self._load_chain_log()

        # Entity aliases (runtime synonym table)
        self._load_entity_aliases()

        # Autodidact daemon (lazy-init)
        self._autodidact: "AutodidactDaemon | None" = None

        # Proactive intelligence heartbeat (lazy-init)
        self._heartbeat = None
        self._proactive_hooks = None

        # Federation daemon (lazy-init)
        self._federation = None

        # Confidence decay config (query-time only)
        self._decay_config: "DecayConfig | None" = None

        # Verification budget (per-session cost tracking)
        self._verification_budget: float | None = None  # None = no limit
        self._verification_spent: float = 0.0

        # Trust policy (in-memory session state)
        self._trust_policy: list | None = None

        # Intelligence gateway (curator, text extractor, insight engine, etc.)
        from attestdb.infrastructure.intelligence_gateway import IntelligenceGateway
        self._intel = IntelligenceGateway(self._store, db=self)

        # Security layer
        from attestdb.core.security import SecurityConfig, SecurityAuditLogger
        self._security = SecurityConfig()
        self._security_meta: dict[str, dict] = {}  # claim_id → {sensitivity, owner, acl}
        self._security_path: str | None = None
        self._security_audit = SecurityAuditLogger("./attestdb_audit.log", enabled=False)
        from attestdb.infrastructure.security import SecurityLayer
        self._security_layer = SecurityLayer(self)
        if not self._is_memory_db:
            self._security_path = self._db_path + ".security.json"
            self._load_security_meta()

        # Enterprise RBAC — groups, policies, secure store proxy
        from attestdb.infrastructure.policy_store import PolicyStore
        from attestdb.infrastructure.access_resolver import AccessResolver
        from attestdb.infrastructure.secure_store_proxy import SecureStoreProxy
        policy_path = None if self._is_memory_db else self._db_path + ".policy.json"
        self._policy_store = PolicyStore(policy_path, audit_log=self._audit)
        self._access_resolver = AccessResolver(self._policy_store)
        self._secure_proxy = SecureStoreProxy(self._store, self._access_resolver)

        # Entity-thread index (entity_id → set of thread_ids)
        self._entity_thread_index: dict[str, set[str]] = {}

        # Topic threads subsystem
        from attestdb.infrastructure.topic_threads import TopicThreads
        self._threads = TopicThreads(self)

        # Compute backend for reproducibility checks (Phase 3)
        self._compute_backend = None

        # TTL cache for expensive analytical methods
        self._cache: dict[str, object] = {}
        self._cache_ts: dict[str, float] = {}

        # Analytics engine (read-only analytical methods)
        from attestdb.infrastructure.analytics import AnalyticsEngine
        self._analytics = AnalyticsEngine(self)

        # Verification engine
        from attestdb.infrastructure.verification_engine import VerificationEngine
        self._verification = VerificationEngine(self)

        # Ask engine (question-answering subsystem)
        from attestdb.infrastructure.ask_engine import AskEngine
        self._ask_engine = AskEngine(self, ops_callback=self._ops_log.write)

        # Eval engine (domain evaluation generation and scoring)
        from attestdb.infrastructure.eval_engine import EvalEngine
        self._eval_engine = EvalEngine(self)

        # Agent registry (agent registration and leaderboard)
        from attestdb.infrastructure.agent_registry import AgentRegistry
        self._agent_registry = AgentRegistry(self)

    # --- TTL cache helper ---

    def _cached(self, key: str, fn, ttl: float = 120) -> object:
        """Return cached result or call fn() on cache miss. Thread-safe."""
        now = time.time()
        if key in self._cache and (now - self._cache_ts.get(key, 0)) < ttl:
            return self._cache[key]
        result = fn()
        self._cache[key] = result
        self._cache_ts[key] = now
        return result

    def _invalidate_cache(self) -> None:
        """Clear all cached analytical results (called on writes)."""
        self._cache.clear()
        self._cache_ts.clear()

    # --- Security layer (delegated to SecurityLayer) ---

    def configure_security(self, **kwargs) -> None:
        """Configure security settings."""
        return self._security_layer.configure_security(**kwargs)

    def _load_security_meta(self) -> None:
        """Load security metadata sidecar."""
        return self._security_layer._load_security_meta()

    def _save_security_meta(self) -> None:
        """Persist security metadata sidecar."""
        return self._security_layer._save_security_meta()

    def _apply_security_overlay(self, claim):
        """Apply security metadata overlay from sidecar to a Claim object."""
        return self._security_layer._apply_security_overlay(claim)

    def _set_claim_security(self, claim_id, sensitivity, owner, acl):
        """Store security metadata for a claim."""
        return self._security_layer._set_claim_security(claim_id, sensitivity, owner, acl)

    def _security_filter(self, claims, principal=None):
        """Apply access control filtering + redaction to a list of claims."""
        return self._security_layer._security_filter(claims, principal)

    # --- Retraction sidecar migration ---

    def _migrate_retraction_sidecar(self) -> None:
        """One-time migration: import legacy .retracted.json into Rust engine."""
        if self._is_memory_db:
            return
        sidecar = self._db_path + ".retracted.json"
        if not os.path.exists(sidecar):
            return
        try:
            with builtins.open(sidecar) as f:
                overrides = json.load(f)
            if overrides:
                updates = [(cid, status) for cid, status in overrides.items()]
                self._store.update_claim_status_batch(updates)
                logger.info("Migrated %d retraction overrides from sidecar to Rust engine", len(updates))
            os.rename(sidecar, sidecar + ".migrated")
        except Exception as e:
            logger.warning("Failed to migrate retraction sidecar: %s", e)

    def _load_py_status_overrides(self) -> None:
        """Load non-tombstone status overrides from Rust store into Python dict.

        The Rust engine's fast overlay path only handles tombstoned claims via
        an in-memory HashSet. Other status overlays (archived, verification_failed,
        etc.) are stored in the LMDB status_overrides table but only read by
        get_claim(). This method bootstraps a Python-side dict so that bulk read
        paths (query engine, iter_claims) can apply the full overlay.
        """
        if not hasattr(self._store, 'get_status_overrides'):
            return
        try:
            overrides = self._store.get_status_overrides()
            for claim_id, status_str in overrides.items():
                if status_str != "tombstoned":  # tombstoned handled by Rust fast path
                    try:
                        self._py_status_overrides[claim_id] = ClaimStatus(status_str)
                    except ValueError:
                        pass
        except Exception:
            pass  # Graceful degradation — overlays will be applied per get_claim

    def _update_py_status_overlay(self, claim_id: str, status: str) -> None:
        """Track a status change in the Python overlay dict."""
        if status == "tombstoned":
            self._py_status_overrides.pop(claim_id, None)
        elif status == "active":
            self._py_status_overrides.pop(claim_id, None)
        else:
            try:
                self._py_status_overrides[claim_id] = ClaimStatus(status)
            except ValueError:
                pass

    # --- Embedding provider ---

    def configure_embeddings(self, provider: str = "openai", **kwargs) -> None:
        """Configure automatic embedding generation on ingest.

        Args:
            provider: "openai" (uses OPENAI_API_KEY env var) or a callable (text -> list[float])
            **kwargs: api_key, model, dimensions (for openai provider)

        Example:
            db.configure_embeddings("openai")  # Uses OPENAI_API_KEY, text-embedding-3-small
            db.configure_embeddings("openai", model="text-embedding-3-large", dimensions=1536)
            db.configure_embeddings(provider=my_embed_fn)  # Custom callable
        """
        if callable(provider):
            self._pipeline._embed_fn = provider
            return

        if provider == "openai":
            api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Pass api_key= or set the environment variable."
                )
            model = kwargs.get("model", "text-embedding-3-small")
            dimensions = kwargs.get("dimensions", self._embedding_dim or 768)

            # Update embedding_dim to match if not set or different
            if self._embedding_dim is None or self._embedding_dim != dimensions:
                self._embedding_dim = dimensions
                self._embedding_index = EmbeddingIndex(ndim=dimensions)
                self._pipeline._embedding_index = self._embedding_index
                self._pipeline._embedding_dim = dimensions

            import openai
            client = openai.OpenAI(api_key=api_key)

            def _openai_embed(text: str) -> list[float]:
                resp = client.embeddings.create(
                    input=text, model=model, dimensions=dimensions,
                )
                return resp.data[0].embedding

            self._pipeline._embed_fn = _openai_embed
        else:
            raise ValueError(f"Unknown embedding provider: {provider!r}. Use 'openai' or a callable.")

    def configure_decay(
        self,
        half_life_days: int = 365,
        predicate_half_lives: dict[str, int] | None = None,
        enabled: bool = True,
    ) -> None:
        """Configure time-weighted confidence decay.

        Decay is applied at query time only — stored claims are never
        modified.  Older evidence naturally loses weight, encouraging
        the autodidact daemon to refresh stale knowledge.

        Args:
            half_life_days: Default half-life for all predicates.
            predicate_half_lives: Per-predicate overrides
                (e.g. ``{"has_status": 30, "interacts_with": 730}``).
            enabled: Set ``False`` to disable decay without clearing config.

        Example::

            db.configure_decay(half_life_days=365)
            db.configure_decay(
                half_life_days=365,
                predicate_half_lives={"has_status": 30},
            )
        """
        from attestdb.core.confidence import DecayConfig

        self._decay_config = DecayConfig(
            default_half_life_days=half_life_days,
            predicate_half_lives=predicate_half_lives or {},
            enabled=enabled,
        )
        self._query_engine.set_decay_config(self._decay_config)

    @property
    def decay_config(self):
        """Current decay configuration, or ``None`` if not configured."""
        return self._decay_config

    # --- Tamper-evident chain ---

    @property
    def ingestion_pipeline(self) -> IngestionPipeline:
        """Access the ingestion pipeline for registering quality gates."""
        return self._pipeline

    @property
    def _is_memory_db(self) -> bool:
        return self._db_path == ":memory:"

    def _chain_log_path(self) -> str:
        return self._db_path + ".chain.json"

    def _load_chain_log(self) -> None:
        if self._is_memory_db:
            return
        path = self._chain_log_path()
        if os.path.exists(path):
            with builtins.open(path) as f:
                data = json.load(f)
            self._chain_log = [(e["claim_id"], e["chain_hash"]) for e in data]
            if self._chain_log:
                self._last_chain_hash = self._chain_log[-1][1]

    def _save_chain_log(self) -> None:
        if self._is_memory_db:
            return
        path = self._chain_log_path()
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    [{"claim_id": cid, "chain_hash": ch} for cid, ch in self._chain_log],
                    f,
                )
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _append_chain(self, claim_id: str) -> str:
        """Append a claim to the Merkle chain. Returns the new chain hash."""
        chain_hash = compute_chain_hash(self._last_chain_hash, claim_id)
        self._chain_log.append((claim_id, chain_hash))
        self._last_chain_hash = chain_hash
        return chain_hash

    def _flush_chain_log(self) -> None:
        """Persist the chain log to disk. Called by close() and snapshot()."""
        if self._is_memory_db:
            return
        self._save_chain_log()

    def verify_integrity(self) -> dict:
        """Verify the tamper-evident Merkle hash chain.

        Returns a dict with keys:
          - ``valid`` (bool): True if the chain is intact.
          - ``length`` (int): Number of entries in the chain.
          - ``error`` (str | None): Description of the first violation, if any.
        """
        prev = "genesis"
        for i, (claim_id, expected_hash) in enumerate(self._chain_log):
            actual = compute_chain_hash(prev, claim_id)
            if actual != expected_hash:
                return {
                    "valid": False,
                    "length": len(self._chain_log),
                    "error": f"Chain broken at position {i}: claim {claim_id}",
                }
            prev = expected_hash
        return {"valid": True, "length": len(self._chain_log), "error": None}

    def get_claim(self, claim_id: str) -> "Claim | None":
        """Get a single claim by ID. O(1) via Rust index lookup."""
        d = self._store.get_claim(claim_id)
        if d is None:
            return None
        claim = claim_from_dict(d)
        self._apply_security_overlay(claim)
        return claim

    def _all_claims(self) -> list[Claim]:
        """Return all claims via Rust store."""
        claims = [claim_from_dict(d) for d in self._store.all_claims()]
        return self._security_filter(claims)

    def iter_claims(self, batch_size: int = DEFAULT_ITER_BATCH_SIZE, exclude_expired: bool = False):
        """Yield all claims by paging through the Rust store.

        Generator that fetches ``batch_size`` claims at a time, avoiding
        materializing the entire claim set in memory.

        Args:
            exclude_expired: If True, skip claims whose ``expires_at`` has passed.
        """
        now_ns = int(time.time() * 1_000_000_000) if exclude_expired else 0
        offset = 0
        batch = self._store.all_claims(offset, batch_size)
        while True:
            if not batch:
                break
            for d in batch:
                claim = claim_from_dict(d)
                if exclude_expired and _claim_is_expired(claim, now_ns):
                    continue
                yield claim
            if len(batch) < batch_size:
                break
            offset += batch_size
            batch = self._store.all_claims(offset, batch_size)

    def count_claims(self) -> int:
        """Return the total number of claims without materializing them."""
        return self._store.count_claims()

    # --- Database lifecycle ---

    def close(self) -> None:
        """Close the database and persist indexes."""
        with _span("attestdb.close"):
            try:
                if getattr(self, "_heartbeat", None):
                    self.disable_heartbeat()
                if self._autodidact:
                    self._autodidact.stop()
                    self._autodidact = None
                if self._federation:
                    self._federation.stop()
                    self._federation = None
                if self._purge_timer:
                    self._purge_timer.cancel()
                    self._purge_timer = None
                if self._scheduler:
                    self._scheduler.stop_all()
                    self._scheduler = None
                self._webhooks_mgr.shutdown()
                self._flush_chain_log()
                if self._multi_embedding_index and len(self._multi_embedding_index) > 0:
                    self._multi_embedding_index.save(self._db_path)
                elif self._embedding_index and len(self._embedding_index) > 0:
                    self._embedding_index.save(self._db_path + ".usearch")
            finally:
                self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # --- Namespace isolation ---

    def set_namespace(self, namespace: str) -> None:
        """Filter all queries to a single namespace.

        Claims without a matching namespace are invisible to queries,
        entity listings, and claim iteration.  Pass an empty string
        to clear the filter (all namespaces visible).
        """
        if namespace:
            self._store.set_namespace_filter([namespace])
        else:
            self._store.set_namespace_filter([])

    def set_namespaces(self, namespaces: list[str]) -> None:
        """Filter queries to multiple namespaces.

        Pass an empty list to clear the filter.
        """
        self._store.set_namespace_filter(namespaces)

    def get_namespaces(self) -> list[str]:
        """Return the current namespace filter (empty = all visible)."""
        return self._store.get_namespace_filter()

    # --- Audit log ---

    def set_actor(self, actor: str) -> None:
        """Set the current actor identity for audit logging.

        All subsequent mutations (ingest, retract, etc.) will be
        attributed to this actor in the audit log.
        """
        self._audit.set_actor(actor)

    def _audit_write(self, event: str, **kwargs) -> None:
        """Append an audit event to the JSONL log."""
        self._audit.write(event, **kwargs)

    def audit_log(
        self,
        since: int = 0,
        event_type: str | None = None,
        actor: str | None = None,
        limit: int = 1000,
    ) -> list[AuditEvent]:
        """Query the audit log.

        Args:
            since: Nanosecond timestamp. Returns events after this time.
            event_type: Filter to a specific event type (e.g. "claim_ingested").
            actor: Filter to a specific actor.
            limit: Max events to return.

        Returns:
            List of AuditEvent, oldest first.
        """
        return self._audit.query(since, event_type, actor, limit)

    def ops_log(
        self,
        since: float = 0.0,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list:
        """Query the operational event log.

        Args:
            since: Unix timestamp. Returns events after this time.
            event_type: Filter (e.g. "curator_triage", "ask_query").
            limit: Max events to return.

        Returns:
            List of OpsEvent, oldest first.
        """
        return self._ops_log.query(since, event_type, limit)

    # --- RBAC (role-based access control) ---

    def enable_rbac(self) -> None:
        """Enable RBAC enforcement. Once enabled, all mutations require a
        matching grant for the current actor + active namespace."""
        self._rbac_mgr.enable()

    def disable_rbac(self) -> None:
        """Disable RBAC enforcement (all operations permitted)."""
        self._rbac_mgr.disable()

    def grant(self, principal: str, namespace: str, role: str | Role) -> None:
        """Grant a role to a principal on a namespace.

        Args:
            principal: User or service identity (e.g. "alice@corp.com").
            namespace: Namespace to grant access to. Use "*" for all namespaces.
            role: One of "admin", "writer", "reader" (or Role enum).

        Example::

            db.grant("alice@corp.com", "team_a", "writer")
            db.grant("bob@corp.com", "*", "admin")
        """
        self._rbac_mgr.grant(principal, namespace, role)

    def revoke(self, principal: str, namespace: str) -> None:
        """Revoke a principal's access to a namespace.

        Args:
            principal: User or service identity.
            namespace: Namespace to revoke. Use "*" to revoke global access.
        """
        self._rbac_mgr.revoke(principal, namespace)

    def list_grants(self, principal: str | None = None) -> dict:
        """List all RBAC grants, optionally filtered by principal.

        Returns dict: {principal: {namespace: role}}.
        """
        return self._rbac_mgr.list_grants(principal)

    def _get_actor_role(self, namespace: str = "") -> str | None:
        """Get the current actor's effective role for a namespace."""
        return self._rbac_mgr.get_actor_role(namespace)

    # ── Enterprise RBAC (groups, policies, secure proxy) ────────────

    def enable_enterprise_rbac(self) -> None:
        """Enable enterprise RBAC with group-based policies and secure proxy.

        When enabled, all claim-returning store methods are filtered
        through the SecureStoreProxy based on the active user's entitlement.
        Auto-imports legacy RBAC grants as PolicyRules on first enable.
        """
        # Auto-import legacy RBAC grants if they exist
        if self._rbac_mgr._enabled:
            grants = self._rbac_mgr.list_grants()
            if grants:
                self._policy_store.import_legacy_rbac(grants)
        self._secure_proxy.enable()

    def disable_enterprise_rbac(self) -> None:
        """Disable enterprise RBAC (passthrough mode)."""
        self._secure_proxy.disable()

    def set_user(self, principal_id: str, org_id: str = "") -> None:
        """Set the active user for enterprise RBAC filtering.

        Resolves the user's groups and policies into an Entitlement,
        then activates namespace + sensitivity filtering on all queries.

        Args:
            principal_id: User identity (e.g. "alice@corp.com").
            org_id: Organization/tenant ID.
        """
        principal = self._access_resolver.build_principal(principal_id, org_id)
        self._secure_proxy.set_principal(principal)
        # Also set actor for audit trail
        self.set_actor(principal_id)

    def create_group(self, group_id: str, org_id: str = "",
                     display_name: str = "", parent_group_id: str | None = None) -> None:
        """Create a team/organizational unit for access control."""
        from attestdb.core.types import Group
        self._policy_store.create_group(
            Group(group_id=group_id, org_id=org_id,
                  display_name=display_name or group_id,
                  parent_group_id=parent_group_id),
            actor=self._audit.actor or "",
        )

    def add_to_group(self, principal_id: str, group_id: str,
                     role_in_group: str = "member") -> None:
        """Add a user to a group."""
        from attestdb.core.types import GroupMembership
        self._policy_store.add_member(
            GroupMembership(principal_id=principal_id, group_id=group_id,
                           role_in_group=role_in_group),
            actor=self._audit.actor or "",
        )

    def remove_from_group(self, principal_id: str, group_id: str) -> None:
        """Remove a user from a group."""
        self._policy_store.remove_member(
            principal_id, group_id,
            actor=self._audit.actor or "",
        )

    def add_policy(self, rule_id: str, group_ids: list[str] | None = None,
                   namespaces: list[str] | None = None,
                   actions: list[str] | None = None,
                   sensitivity_max: str = "RESTRICTED",
                   effect: str = "allow", priority: int = 100,
                   org_id: str = "", description: str = "") -> None:
        """Add a declarative access policy rule.

        Args:
            rule_id: Unique rule identifier.
            group_ids: Groups this rule applies to.
            namespaces: Namespaces granted. [] = all.
            actions: "read", "write", "admin". [] = all.
            sensitivity_max: Max sensitivity level (default RESTRICTED).
            effect: "allow" or "deny".
            priority: Higher = evaluated first. Deny at same priority wins.
        """
        from attestdb.core.types import PolicyRule, SensitivityLevel
        level = SensitivityLevel[sensitivity_max.upper()] if isinstance(sensitivity_max, str) else sensitivity_max
        self._policy_store.add_rule(
            PolicyRule(
                rule_id=rule_id, org_id=org_id, priority=priority,
                effect=effect, group_ids=group_ids or [],
                namespaces=namespaces or [],
                sensitivity_max=level,
                actions=actions or [],
                description=description,
            ),
            actor=self._audit.actor or "",
        )

    def remove_policy(self, rule_id: str) -> None:
        """Remove a policy rule."""
        self._policy_store.remove_rule(rule_id, actor=self._audit.actor or "")

    def delete_group(self, group_id: str) -> None:
        """Delete a group and its memberships/policies."""
        self._policy_store.delete_group(group_id, actor=self._audit.actor or "")

    def get_group_members(self, group_id: str) -> list:
        """List members of a group."""
        return self._policy_store.get_group_members(group_id)

    def get_user_groups(self, principal_id: str) -> list:
        """List groups a user belongs to."""
        return self._policy_store.get_memberships(principal_id)

    def list_groups(self, org_id: str = "") -> list:
        """List all groups."""
        return self._policy_store.list_groups(org_id)

    def list_policies(self, org_id: str = "") -> list:
        """List all policy rules."""
        return self._policy_store.list_rules(org_id)

    def get_entitlement(self, principal_id: str, org_id: str = ""):
        """Resolve a user's effective access rights."""
        return self._access_resolver.resolve_entitlement(principal_id, org_id)

    def _check_permission(self, required_role: str) -> None:
        """Raise PermissionError if RBAC is enabled and actor lacks permission.

        Role hierarchy: admin > writer > reader.
        """
        self._rbac_mgr.check_permission(required_role)

    def _set_include_retracted(self, include: bool) -> None:
        """Set retraction visibility on the Rust store."""
        self._store.set_include_retracted(include)

    # --- Change feed ---

    def changes(self, since: int = 0, limit: int = 1000) -> tuple[list[Claim], int]:
        """Return claims ingested after timestamp `since`.

        Returns (claims, cursor) where cursor is the max timestamp seen.
        Use the returned cursor as `since` in the next call for
        cursor-based pagination::

            cursor = 0
            while True:
                claims, cursor = db.changes(since=cursor)
                for c in claims:
                    process(c)
                time.sleep(5)

        Args:
            since: Nanosecond timestamp. Returns claims with timestamp > since.
            limit: Max claims to return per call.

        Returns:
            Tuple of (claims, new_cursor). If no new claims, cursor == since.
        """
        max_ts = int(time.time() * 1_000_000_000)
        raw = self._store.claims_in_range(since + 1, max_ts)
        claims = [claim_from_dict(d) for d in raw]
        # Sort by timestamp ascending for deterministic ordering
        claims.sort(key=lambda c: c.timestamp)
        if limit and len(claims) > limit:
            claims = claims[:limit]
        new_cursor = claims[-1].timestamp if claims else since
        return claims, new_cursor

    # --- Topic subscriptions ---

    def subscribe(
        self,
        subscriber_id: str,
        topics: list[str],
        callback,
        predicates: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        """Register interest in topics. Callback fires on matching ingested claims.

        Args:
            subscriber_id: Identifier for the subscribing agent.
            topics: Entity IDs to watch (normalized for matching).
            callback: Called with (claim, sub_id, subscriber_id) on match.
            predicates: If set, only match claims with these predicate IDs.
            min_confidence: Minimum confidence to trigger notification.

        Returns:
            Subscription ID (use to unsubscribe).
        """
        return self._events.subscribe(subscriber_id, topics, callback, predicates, min_confidence)

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscription. Returns True if it existed."""
        return self._events.unsubscribe(sub_id)

    def _check_subscriptions(self, claim_id: str, claim_input: ClaimInput) -> None:
        """Check all subscriptions against a newly ingested claim."""
        self._events.check_subscriptions(claim_id, claim_input)

    # --- Hypothetical reasoning ---

    def hypothetical_sandbox(self, description: str, seed_entities: list[str] | None = None):
        """Create a lightweight sandbox for speculative reasoning.

        Returns a HypotheticalContext (use as context manager) with:
        - .sandbox: an in-memory AttestDB preloaded with relevant claims
        - .promote(): merge surviving claims back to parent
        - .report(): summary of new claims, contradictions, gaps

        Usage::

            with db.hypothetical_sandbox("BRCA1 inhibits PD-L1", seed_entities=["BRCA1", "PD-L1"]) as h:
                h.sandbox.ingest(subject=("BRCA1", "gene"), ...)
                contradictions = h.find_contradictions()
                if not contradictions:
                    h.promote()
        """
        sandbox = AttestDB(":memory:", embedding_dim=None)

        # Determine seed entities
        entities = seed_entities or []
        if not entities:
            # Try to extract entity IDs from description
            for word in description.replace(",", " ").split():
                word = word.strip()
                if word and len(word) > 1:
                    results = self._store.search_entities(word, 3)
                    for r in results:
                        entities.append(r["id"])

        # Copy claims within 2 hops of seed entities
        seen_claims = set()
        for entity_id in entities:
            for depth_hop in range(2):
                hop_entities = [entity_id] if depth_hop == 0 else list(neighbors)
                neighbors = set()
                for eid in hop_entities:
                    eid_claims = self._store.claims_for(eid, None, None, 0.0)
                    if len(eid_claims) > 500:
                        eid_claims = eid_claims[:500]
                    for cd in eid_claims:
                        cid = cd["claim_id"]
                        if cid in seen_claims:
                            continue
                        seen_claims.add(cid)
                        claim = claim_from_dict(cd)
                        sandbox.ingest(
                            subject=(claim.subject.id, claim.subject.entity_type),
                            predicate=(claim.predicate.id, claim.predicate.predicate_type),
                            object=(claim.object.id, claim.object.entity_type),
                            provenance={
                                "source_type": claim.provenance.source_type,
                                "source_id": claim.provenance.source_id,
                                "method": claim.provenance.method,
                            },
                            confidence=claim.confidence,
                            timestamp=claim.timestamp,
                        )
                        neighbors.add(claim.subject.id)
                        neighbors.add(claim.object.id)

        return HypotheticalContext(sandbox=sandbox, parent=self, description=description)

    # --- Trust policies ---

    def set_trust_policy(self, rules: list) -> None:
        """Set trust policy for current session. Filters claims_for() results.

        Each rule is a TrustRule (from core.types). First matching rule wins.
        Claims that match no rule are included by default.
        """
        self._trust_policy = rules

    def clear_trust_policy(self) -> None:
        """Remove trust policy — all claims visible."""
        self._trust_policy = None

    def _apply_trust_filter(self, claims: list[Claim]) -> list[Claim]:
        """Apply trust policy to a list of claims. Returns filtered list."""
        if not self._trust_policy:
            return claims
        result = []
        for claim in claims:
            action = self._match_trust_rule(claim)
            if action == "include":
                result.append(claim)
        return result

    def _match_trust_rule(self, claim: Claim) -> str:
        """Find the first matching trust rule for a claim. Returns action."""
        from attestdb.core.types import TrustRule
        for rule in self._trust_policy:
            matched = True
            if rule.source_types is not None:
                if claim.provenance.source_type not in rule.source_types:
                    matched = False
            if rule.source_ids is not None:
                if claim.provenance.source_id not in rule.source_ids:
                    matched = False
            if rule.predicates is not None:
                if claim.predicate.id not in rule.predicates:
                    matched = False
            if rule.min_confidence is not None:
                if claim.confidence < rule.min_confidence:
                    matched = False
            if rule.min_corroboration is not None:
                corr = len(self.claims_by_content_id(claim.content_id))
                if corr < rule.min_corroboration:
                    matched = False
            if matched:
                return rule.action
        return "include"  # default: include if no rule matches

    # --- Event hooks ---

    def on(self, event: str, callback) -> None:
        """Register a callback for a lifecycle event.

        Use :class:`~attestdb.infrastructure.event_bus.EventType` constants
        for event names (e.g. ``EventType.CLAIM_INGESTED``).
        """
        self._events.on(event, callback)

    def off(self, event: str, callback) -> None:
        """Remove a registered callback."""
        self._events.off(event, callback)

    def _fire(self, event: str, **kwargs) -> None:
        """Fire all callbacks for an event and POST to registered webhooks."""
        self._events.fire(event, **kwargs)

    # --- Webhooks ---

    def register_webhook(
        self, url: str, events: list[str] | None = None, secret: str | None = None,
    ) -> None:
        """Register an HTTP endpoint to receive event notifications.

        Args:
            url: The URL to POST event payloads to.
            events: List of event names to subscribe to. None = all events.
            secret: Optional HMAC-SHA256 secret for payload signing.
        """
        self._webhooks_mgr.register(url, events, secret)

    def remove_webhook(self, url: str) -> bool:
        """Remove a registered webhook by URL. Returns True if found."""
        return self._webhooks_mgr.remove(url)

    def list_webhooks(self) -> list[dict]:
        """Return registered webhooks (secrets are masked)."""
        return self._webhooks_mgr.list()

    # --- Auto-purge ---

    def enable_auto_purge(self, interval_seconds: float = 3600) -> None:
        """Start a background timer that periodically purges expired claims.

        Args:
            interval_seconds: Seconds between purge runs (default 1 hour).
        """
        if self._purge_timer:
            self._purge_timer.cancel()
        self._purge_interval = interval_seconds

        def _run_purge():
            try:
                purged = self.purge_expired()
                if purged:
                    logger.info("Auto-purge: tombstoned %d expired claims", purged)
            except Exception as exc:
                logger.warning("Auto-purge failed: %s", exc)
            finally:
                if self._purge_interval > 0:
                    self._purge_timer = threading.Timer(self._purge_interval, _run_purge)
                    self._purge_timer.daemon = True
                    self._purge_timer.start()

        self._purge_timer = threading.Timer(interval_seconds, _run_purge)
        self._purge_timer.daemon = True
        self._purge_timer.start()

    # --- Autodidact self-learning ---

    def enable_autodidact(
        self,
        interval: float = 3600,
        max_llm_calls_per_day: int = 100,
        max_questions_per_cycle: int = 5,
        max_cost_per_day: float = 1.00,
        search_fn=None,
        sources: str = "auto",
        connectors: list[str] | None = None,
        gap_types: list[str] | None = None,
        entity_types: list[str] | None = None,
        use_curator: bool = True,
        jitter: float = 0.1,
        negative_result_limit: int = 3,
        enabled_triggers: list[str] | None = None,
        trigger_cooldown: float = 60.0,
    ) -> "AutodidactStatus":
        """Start the autodidact self-learning daemon.

        The daemon runs in a background thread, continuously detecting
        knowledge gaps and researching them via registered evidence sources
        or the enterprise Researcher.

        Args:
            interval: Seconds between cycles (default 1 hour).
            max_llm_calls_per_day: Daily LLM call budget (resets at midnight).
            max_questions_per_cycle: Max tasks per cycle.
            max_cost_per_day: Estimated dollar cap per day (default $1.00).
            search_fn: Optional callable(str) -> str for evidence retrieval.
                Overrides auto-registration of built-in sources.
            sources: Source registration mode. "auto" registers PubMed,
                Semantic Scholar, and paid sources if API keys available.
                "none" disables auto-registration (gap detection only).
            connectors: Optional list of connector names (e.g. ["slack", "github"])
                to register as evidence sources. These connectors must have saved
                credentials or be pre-configured. Only connectors with search
                support are registered.
            gap_types: Filter task types (single_source, low_confidence, etc).
            entity_types: Filter by entity type.
            use_curator: Triage extracted claims through curator.
            jitter: Random jitter fraction on sleep interval.
            negative_result_limit: Skip entities with >= N negative results.
            enabled_triggers: Event triggers (default: timer, retraction, inquiry).
            trigger_cooldown: Seconds between event-triggered cycles (default 60).

        Returns:
            AutodidactStatus after starting.
        """
        from attestdb.core.types import AutodidactConfig
        from attestdb.infrastructure.autodidact import AutodidactDaemon

        if self._autodidact:
            self._autodidact.stop()

        config = AutodidactConfig(
            interval_seconds=interval,
            max_llm_calls_per_day=max_llm_calls_per_day,
            max_questions_per_cycle=max_questions_per_cycle,
            max_cost_per_day=max_cost_per_day,
            gap_types=gap_types,
            entity_types=entity_types,
            use_curator=use_curator,
            jitter=jitter,
            negative_result_limit=negative_result_limit,
            enabled_triggers=enabled_triggers or ["timer", "retraction", "inquiry"],
            trigger_cooldown=trigger_cooldown,
        )
        self._autodidact = AutodidactDaemon(self, config)

        if search_fn is not None:
            self._autodidact.register_source("default", search_fn)
        elif sources == "auto":
            # Auto-register built-in sources based on available API keys
            from attestdb.infrastructure.evidence_sources import register_default_sources

            env_path = os.path.join(os.getcwd(), ".env") if not self._is_memory_db else None
            register_default_sources(self._autodidact, env_path=env_path)

        # Register connector-based evidence sources
        if connectors:
            self._register_connector_sources(connectors)

        self._autodidact.start()
        return self._autodidact.status

    def _register_connector_sources(self, connector_names: list[str]) -> None:
        """Instantiate named connectors and register searchable ones as evidence sources."""
        from attestdb.infrastructure.evidence_sources import register_connector_sources

        instances = []
        for name in connector_names:
            try:
                conn = self.connect(name)
                instances.append(conn)
            except Exception as exc:
                logger.warning("Could not instantiate connector '%s' for autodidact: %s", name, exc)
        if instances:
            registered = register_connector_sources(self._autodidact, instances)
            if registered:
                logger.info("Autodidact connector sources: %s", registered)

    def disable_autodidact(self) -> None:
        """Stop the autodidact daemon."""
        if self._autodidact:
            self._autodidact.stop()
            self._autodidact = None

    def autodidact_status(self) -> "AutodidactStatus":
        """Return current autodidact daemon status."""
        from attestdb.core.types import AutodidactStatus

        if not self._autodidact:
            return AutodidactStatus()
        return self._autodidact.status

    def autodidact_run_now(self) -> None:
        """Trigger an immediate autodidact cycle."""
        if self._autodidact:
            self._autodidact.run_now()

    def autodidact_cost_estimate(self, cycles: int = 24) -> dict:
        """Estimate cost for N autodidact cycles without running them.

        Args:
            cycles: Number of cycles to estimate (default 24 = 1 day at 1hr interval).

        Returns:
            Dict with per_cycle, daily, monthly estimates and breakdown.
            Returns empty dict if autodidact is not enabled.
        """
        if not self._autodidact:
            return {}
        return self._autodidact.cost_estimate(cycles)

    def autodidact_history(self, limit: int = 10) -> list["CycleReport"]:
        """Return recent autodidact cycle reports.

        Args:
            limit: Max number of reports to return (most recent first).
        """
        if not self._autodidact:
            return []
        history = self._autodidact.history
        return list(reversed(history[-limit:]))

    # --- Proactive Intelligence Heartbeat ---

    def enable_heartbeat(
        self,
        cycle_interval: float = 30.0,
        hot_threshold: float = 0.65,
        decay_half_life_hours: float = 336.0,
        composite_synthesis_budget: float = 10.0,
        max_consolidation_batch: int = 64,
        working_memory_max_claims: int = 512,
    ) -> dict:
        """Start the proactive intelligence heartbeat.

        The heartbeat runs a background Perceive->Plan->Act cycle that
        maintains claim lifecycle (decay, tiers), synthesizes composites,
        consolidates near-duplicates, detects gaps, and tracks access.

        Args:
            cycle_interval: Seconds between cycles (default 30).
            hot_threshold: Decay score threshold for hot tier (default 0.65).
            decay_half_life_hours: Default half-life for decay (default 336 = 2 weeks).
            composite_synthesis_budget: Max seconds per cycle on synthesis.
            max_consolidation_batch: Claims per consolidation batch.
            working_memory_max_claims: Max claims in hot tier.

        Returns:
            Status dict with running state.
        """
        from attestdb.intelligence.heartbeat import HeartbeatConfig, HeartbeatScheduler
        from attestdb.intelligence.proactive_hooks import ProactiveHooks

        if self._heartbeat:
            self._heartbeat.stop()
        if self._proactive_hooks:
            self._proactive_hooks.uninstall()

        config = HeartbeatConfig(
            cycle_interval_seconds=cycle_interval,
            hot_threshold=hot_threshold,
            decay_half_life_hours=decay_half_life_hours,
            composite_synthesis_budget_seconds=composite_synthesis_budget,
            max_consolidation_batch=max_consolidation_batch,
            working_memory_max_claims=working_memory_max_claims,
        )

        self._heartbeat = HeartbeatScheduler(self, config=config)
        self._heartbeat.start()

        self._proactive_hooks = ProactiveHooks(self, self._heartbeat)
        self._proactive_hooks.install()

        status = self._heartbeat.get_status()
        return {"running": status.running, "cycle_interval": cycle_interval}

    def disable_heartbeat(self) -> None:
        """Stop the proactive intelligence heartbeat."""
        if self._proactive_hooks:
            self._proactive_hooks.uninstall()
            self._proactive_hooks = None
        if self._heartbeat:
            self._heartbeat.stop()
            self._heartbeat = None

    def heartbeat_status(self) -> dict:
        """Return current heartbeat status."""
        if not self._heartbeat:
            return {"running": False, "cycle_count": 0}
        status = self._heartbeat.get_status()
        return {
            "running": status.running,
            "cycle_count": status.cycle_count,
            "last_cycle_at": status.last_cycle_at,
            "last_cycle_duration_ms": round(status.last_cycle_duration_ms, 1),
            "tier_distribution": status.tier_distribution,
            "tracked_claims": status.tracked_claims,
            "hot_claims": status.hot_claims,
            "stale_composites": status.stale_composites,
        }

    def heartbeat_run_now(self) -> None:
        """Trigger an immediate heartbeat cycle."""
        if self._heartbeat:
            self._heartbeat.run_now()

    # --- Federation ---

    def enable_federation(
        self,
        remote_url: str,
        api_key: str,
        interval: float = 300,
        conflict_policy: str = "keep_higher_confidence",
        namespace: str | None = None,
    ) -> "FederationStatus":
        """Start continuous federation with a remote AttestDB instance.

        Runs a background thread that periodically syncs claims bidirectionally.
        Cursor is persisted in a `.federation.json` sidecar for resume across restarts.

        Args:
            remote_url: Base URL of the remote API (e.g. "https://api.attestdb.com").
            api_key: Bearer token for authentication.
            interval: Seconds between sync cycles (default 5 min).
            conflict_policy: "keep_higher_confidence" | "keep_newer" | "keep_both".
            namespace: If set, only sync claims in this namespace.

        Returns:
            FederationStatus after starting.
        """
        from attestdb.infrastructure.agents import FederationDaemon

        if self._federation:
            self._federation.stop()

        self._federation = FederationDaemon(
            self,
            remote_url=remote_url,
            api_key=api_key,
            interval=interval,
            conflict_policy=conflict_policy,
            namespace=namespace,
        )
        self._federation.start()
        return self._federation.status

    def disable_federation(self) -> None:
        """Stop the federation daemon."""
        if self._federation:
            self._federation.stop()
            self._federation = None

    def federation_status(self) -> "FederationStatus":
        """Return current federation daemon status."""
        from attestdb.infrastructure.agents import FederationStatus

        if not self._federation:
            return FederationStatus()
        return self._federation.status

    # --- Backup / Restore ---

    def snapshot(self, dest_path: str) -> str:
        """Copy store files and embedding index to dest_path. DB stays open.

        Returns dest_path.
        """
        with _span("attestdb.snapshot", {"dest_path": dest_path}):
            return self._snapshot_inner(dest_path)

    def _snapshot_inner(self, dest_path: str) -> str:
        os.makedirs(dest_path, exist_ok=True)

        # Persist embedding index first so files are up-to-date
        if self._multi_embedding_index and len(self._multi_embedding_index) > 0:
            self._multi_embedding_index.save(self._db_path)
        elif self._embedding_index and len(self._embedding_index) > 0:
            self._embedding_index.save(self._db_path + ".usearch")

        # Flush Rust store to disk (close + reopen) so we can copy the file
        store_path = self._store._path if hasattr(self._store, '_path') else None
        self._store.close()
        try:
            if store_path:
                src = store_path
            elif self._db_path.endswith((".substrate", ".attest")):
                src = self._db_path
            else:
                src = self._db_path + ".attest"
            if os.path.exists(src):
                dst = os.path.join(dest_path, os.path.basename(src))
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        finally:
            # Always reopen the store so DB remains usable
            self._store = _make_store(self._db_path)
            self._pipeline._store = self._store
            self._query_engine._store = self._store

        # Retraction status is now stored in the Rust engine (no sidecar needed)

        # Flush + copy chain log
        self._flush_chain_log()
        chain_path = self._chain_log_path()
        if os.path.exists(chain_path):
            shutil.copy2(chain_path, os.path.join(dest_path, os.path.basename(chain_path)))

        # Copy embedding index files (single-space and multi-space)
        for ext in (".usearch", ".usearch.json", ".usearch.manifest.json"):
            src = self._db_path + ext
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest_path, os.path.basename(src)))
        # Copy multi-space files (.usearch.<space> and .usearch.<space>.json)
        if self._multi_embedding_index:
            for space_name in self._multi_embedding_index.space_names:
                for ext in (f".usearch.{space_name}", f".usearch.{space_name}.json"):
                    src = self._db_path + ext
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(dest_path, os.path.basename(src)))

        self._fire(EventType.SNAPSHOT_CREATED, dest_path=dest_path)
        return dest_path

    @staticmethod
    def restore(
        src_path: str,
        dest_path: str,
        embedding_dim: int | None = 768,
    ) -> "AttestDB":
        """Restore an AttestDB from a snapshot directory.

        Copies snapshot files to dest_path, opens and returns a new AttestDB.
        """
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

        store_ext = ".attest"

        # Copy store files from snapshot
        for entry in os.listdir(src_path):
            src_file = os.path.join(src_path, entry)
            if entry.endswith(store_ext):
                dst = dest_path + store_ext if not dest_path.endswith(store_ext) else dest_path
                if os.path.isdir(src_file):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src_file, dst)
                else:
                    shutil.copy2(src_file, dst)
            elif entry.endswith(".chain.json"):
                dst = dest_path + ".chain.json"
                shutil.copy2(src_file, dst)
            elif entry.endswith(".usearch") or entry.endswith(".usearch.json"):
                dot_idx = entry.find(".")
                if dot_idx == -1:
                    continue  # skip files without an extension
                dst = dest_path + entry[dot_idx:]
                shutil.copy2(src_file, dst)

        return AttestDB(dest_path, embedding_dim=embedding_dim)

    # --- Vocabulary registration ---

    def register_vocabulary(self, namespace: str, vocab: dict) -> None:
        self._store.register_vocabulary(namespace, vocab)
        # Also register predicate constraints if included in the vocab dict
        for pred_id, constraints in vocab.get("predicate_constraints", {}).items():
            self._store.register_predicate(pred_id, constraints)
        self._pipeline.invalidate_vocab_caches()
        self._intel.invalidate_text_extractor()

    def register_predicate(self, predicate_id: str, constraints: dict) -> None:
        self._store.register_predicate(predicate_id, constraints)
        self._intel.invalidate_text_extractor()

    def register_payload_schema(self, schema_id: str, schema: dict) -> None:
        self._store.register_payload_schema(schema_id, schema)

    # --- Claim ingestion ---

    def ingest(
        self,
        subject: "tuple[str, str] | ClaimInput" = None,
        predicate: tuple[str, str] | None = None,
        object: tuple[str, str] | None = None,
        provenance: dict | None = None,
        confidence: float | None = None,
        embedding: list[float] | None = None,
        payload: dict | None = None,
        timestamp: int | None = None,
        external_ids: dict | None = None,
        namespace: str = "",
        ttl_seconds: int = 0,
        verify: bool = False,
        sensitivity: "SensitivityLevel | None" = None,
        owner: str | None = None,
        acl: list[str] | None = None,
    ) -> str:
        """Ingest a single claim. Returns claim_id.

        Accepts either a ClaimInput object or individual fields::

            db.ingest(claim_input)
            db.ingest(subject=(...), predicate=(...), object=(...), provenance={...})
            db.ingest(subject=(...), ..., namespace="team_a", ttl_seconds=3600)
            db.ingest(subject=(...), ..., verify=True)  # auto-verify after ingest
            db.ingest(subject=(...), ..., sensitivity=SensitivityLevel.CONFIDENTIAL)

        If ``verify=True``, runs the verification pipeline after ingestion and
        updates the claim status to the recommended status (verified/disputed/
        verification_failed).

        If ``sensitivity=None``, auto-classifies based on content scanning.
        """
        self._check_permission("writer")
        if isinstance(subject, ClaimInput):
            ci = subject
            # Extract security fields from ClaimInput if set
            if sensitivity is None and hasattr(ci, "sensitivity"):
                sensitivity = ci.sensitivity
            if owner is None and hasattr(ci, "owner"):
                owner = ci.owner
            if acl is None and hasattr(ci, "acl"):
                acl = ci.acl
        else:
            ci = ClaimInput(
                subject=subject,
                predicate=predicate,
                object=object,
                provenance=provenance,
                confidence=confidence,
                embedding=embedding,
                payload=payload,
                timestamp=timestamp,
                external_ids=external_ids,
                namespace=namespace,
                ttl_seconds=ttl_seconds,
            )
        with _span("attestdb.ingest", {"subject": ci.subject[0], "predicate": ci.predicate[0]}):
            claim_id = self._pipeline.ingest(ci)
            self._append_chain(claim_id)
            self._flush_chain_log()
            _set_attr("claim_id", claim_id)
        self._invalidate_cache()

        # --- Security: classify + injection scan ---
        claim_obj = self.get_claim(claim_id)
        if claim_obj:
            from attestdb.core.security import auto_classify_sensitivity, scan_for_injection
            from attestdb.core.types import SensitivityLevel as SL

            # Injection detection (runs always — cheap, purely additive)
            if self._security.injection_detection:
                scan_result = scan_for_injection(claim_obj)
                if scan_result["risk"] != "none":
                    claim_obj._injection_risk = scan_result["risk"]
                    claim_obj._injection_patterns = scan_result["patterns"]
                    # High risk → auto-bump sensitivity
                    if scan_result["risk"] == "high" and (sensitivity is None or sensitivity < SL.RESTRICTED):
                        sensitivity = SL.RESTRICTED

            # Auto-classify sensitivity
            if sensitivity is None and self._security.auto_classify:
                sensitivity = auto_classify_sensitivity(claim_obj, self._security)
            if sensitivity is None:
                sensitivity = self._security.default_sensitivity

            # Set owner from actor if not specified
            if owner is None and self._audit.actor:
                owner = self._audit.actor

            # Persist security metadata
            self._set_claim_security(claim_id, sensitivity, owner, acl or [])
            if self._security_path:
                self._save_security_meta()

        self._audit_write(
            "claim_ingested",
            claim_id=claim_id,
            namespace=ci.namespace if hasattr(ci, "namespace") else "",
            subject=ci.subject[0],
            predicate=ci.predicate[0],
            object=ci.object[0],
        )
        self._fire(EventType.CLAIM_INGESTED, claim_id=claim_id, claim_input=ci)
        self._check_subscriptions(claim_id, ci)
        # Check corroboration (use normalized predicate to match stored claims)
        from attestdb.core.vocabulary import normalize_predicate
        content_id = compute_content_id(
            normalize_entity_id(ci.subject[0]),
            normalize_predicate(ci.predicate[0]),
            normalize_entity_id(ci.object[0]),
        )
        existing = self.claims_by_content_id(content_id)
        if len(existing) > 1:
            self._fire(EventType.CLAIM_CORROBORATED, content_id=content_id, count=len(existing))
        # Check inquiry matches
        matches = self.check_inquiry_matches(
            subject_id=ci.subject[0], object_id=ci.object[0], predicate_id=ci.predicate[0],
        )
        for m in matches:
            self._fire(EventType.INQUIRY_MATCHED, inquiry_id=m, claim_id=claim_id)
        # Optional verification
        if verify:
            try:
                if claim_obj is None:
                    claim_obj = self.get_claim(claim_id)
                if claim_obj:
                    verdict = self.verify_claim(claim_obj)
                    self._store.update_claim_status(claim_id, verdict.recommended_status)
            except Exception as e:
                logger.warning("Post-ingest verification failed: %s", e)
        return claim_id

    def ingest_batch(self, claims: list[ClaimInput]) -> BatchResult:
        self._check_permission("writer")
        with _span("attestdb.ingest_batch", {"batch_size": len(claims)}):
            result = self._pipeline.ingest_batch(claims, on_ingested=self._append_chain)
            self._flush_chain_log()
            _set_attr("ingested", result.ingested)
            _set_attr("duplicates", result.duplicates)
        if result.ingested > 0:
            self._invalidate_cache()
            self._audit_write(
                "batch_ingested",
                count=result.ingested,
                duplicates=result.duplicates,
                errors=len(result.errors),
            )
        return result

    # --- Querying ---

    def query(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        llm_narrative: bool = False,
        confidence_threshold: float = 0.0,
        predicate_types: list[str] | None = None,
        include_archived: bool = False,
    ) -> ContextFrame:
        with _span("attestdb.query", {"focal_entity": focal_entity, "depth": depth}):
            frame = self._query_engine.query(
                focal_entity, depth, min_confidence, exclude_source_types,
                max_claims, max_tokens,
                confidence_threshold=confidence_threshold,
                predicate_types=predicate_types,
                include_archived=include_archived,
            )
            _set_attr("claim_count", frame.claim_count)
            _set_attr("relationship_count", len(frame.direct_relationships))
        # Enrich with topology if computed
        if hasattr(self, "_topology") and self._topology is not None:
            frame.topic_membership = self._get_topic_membership(frame.focal_entity.id)
        # Generate narrative
        if llm_narrative:
            try:
                from attestdb_enterprise.narrative import generate_llm_narrative
            except ImportError:
                raise ImportError(
                    "LLM narrative requires attestdb-enterprise: "
                    "pip install attestdb-enterprise"
                )
            frame.narrative = generate_llm_narrative(
                frame,
                model=self._intel._curator_model,
                api_key=self._intel._curator_api_key,
                env_path=self._intel._curator_env_path,
            )
        elif not frame.narrative:
            try:
                from attestdb.intelligence.narrative import generate_narrative
                frame.narrative = generate_narrative(frame)
            except ImportError:
                pass  # narrative generation requires attestdb-intelligence
        # Proactive intelligence: record access and attach freshness warnings
        if getattr(self, "_proactive_hooks", None):
            try:
                self._proactive_hooks.post_query(focal_entity, frame)
            except Exception:
                pass  # hooks must never break queries
        return frame

    def _get_topic_membership(self, entity_id: str) -> list[str]:
        """Find community IDs for entity across all resolutions."""
        memberships = []
        for res, communities in self._topology.communities.items():
            for comm in communities:
                if entity_id in comm.members:
                    memberships.append(comm.id)
        return memberships

    # --- Explain ---

    def explain(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        confidence_threshold: float = 0.0,
        predicate_types: list[str] | None = None,
    ) -> tuple[ContextFrame, QueryProfile]:
        """Like query() but also returns a QueryProfile with timing and counts."""
        return self._query_engine.explain(
            focal_entity, depth, min_confidence, exclude_source_types,
            max_claims, max_tokens,
            confidence_threshold=confidence_threshold,
            predicate_types=predicate_types,
        )

    # --- Time-travel ---

    def at(self, timestamp: int):
        """Return a read-only view of the database as of the given timestamp."""
        from attestdb.infrastructure.snapshot import AttestDBSnapshot
        return AttestDBSnapshot(self, timestamp)

    # --- Semantic search ---

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        space: str = "default",
    ) -> list[tuple[str, float]]:
        if self._multi_embedding_index is not None:
            return self._multi_embedding_index.search(query_embedding, top_k, space=space)
        if self._embedding_index is None:
            return []
        return self._embedding_index.search(query_embedding, top_k)

    def get_embedding(self, claim_id: str) -> list[float] | None:
        """Retrieve the stored embedding for a claim_id."""
        if self._embedding_index is None:
            return None
        vec = self._embedding_index.get(claim_id)
        if vec is None:
            return None
        return vec.tolist()

    def embedding_similarity(self, entity_a: str, entity_b: str) -> float:
        """Compute cosine similarity between two entities' embeddings.

        Looks up each entity's embedding from the embedding index.  Checks for
        structural embeddings (``_struct_{entity_id}``) first, then falls back
        to averaging the embeddings of claims associated with the entity.

        Returns a float in [-1, 1], or 0.0 if either entity has no embedding.
        """
        if self._embedding_index is None:
            return 0.0

        vec_a = self._entity_embedding(entity_a)
        vec_b = self._entity_embedding(entity_b)
        if vec_a is None or vec_b is None:
            return 0.0

        return float(self._cosine_similarity(vec_a, vec_b))

    # -- helpers for embedding_similarity --

    def _entity_embedding(self, entity_id: str):
        """Return the embedding vector for an entity, or None."""
        import numpy as np

        norm_id = normalize_entity_id(entity_id)

        # 1. Structural embedding (from generate_structural_embeddings)
        struct_key = f"_struct_{norm_id}"
        struct_target = self._structural_embedding_target()
        if struct_target is not None:
            vec = struct_target.get(struct_key)
            if vec is not None:
                return vec
        # Also check default space (legacy: struct embeddings stored there)
        if self._embedding_index is not None and self._embedding_index is not struct_target:
            vec = self._embedding_index.get(struct_key)
            if vec is not None:
                return vec

        # 2. Average of claim embeddings for this entity
        if self._embedding_index is None:
            return None
        raw_claims = self._store.claims_for(norm_id, None, None, 0.0)
        if len(raw_claims) > 200:
            raw_claims = raw_claims[:200]
        vecs = []
        for c in raw_claims:
            cid = c.get("claim_id") or c.get("id")
            if cid:
                v = self._embedding_index.get(cid)
                if v is not None:
                    vecs.append(v)
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two vectors (numpy arrays)."""
        import numpy as np

        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # --- Text and hybrid search ---

    def text_search(
        self,
        query: str,
        entity_type: str | None = None,
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> list[Claim]:
        """Search claims by text query using BM25 entity name matching.

        Uses Rust search_entities() for BM25-ranked entity retrieval,
        then enriches with claims filtered by entity_type and min_confidence.
        """
        with _span("attestdb.text_search", {"query": query, "top_k": top_k}):
            now_ns = int(time.time() * 1_000_000_000)
            # Over-fetch entities to allow for filtering
            raw_entities = self._store.search_entities(query, top_k * 3)
            entities = [entity_summary_from_dict(d) for d in raw_entities]

            if entity_type:
                entities = [e for e in entities if e.entity_type == entity_type]

            seen_claim_ids: set[str] = set()
            results: list[Claim] = []
            for entity in entities:
                raw_claims = self._store.claims_for(entity.id, None, None, 0.0)
                if len(raw_claims) > 500:
                    raw_claims = raw_claims[:500]
                for d in raw_claims:
                    claim = claim_from_dict(d)
                    if claim.claim_id in seen_claim_ids:
                        continue
                    if claim.confidence < min_confidence:
                        continue
                    if claim.status != ClaimStatus.ACTIVE:
                        continue
                    if _claim_is_expired(claim, now_ns):
                        continue
                    seen_claim_ids.add(claim.claim_id)
                    results.append(claim)
                    if len(results) >= top_k:
                        _set_attr("result_count", len(results))
                        return results
            _set_attr("result_count", len(results))
            return results

    def hybrid_search(
        self,
        query: str,
        alpha: float = 0.7,
        top_k: int = 20,
        entity_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[tuple[Claim, float]]:
        """Hybrid text + embedding search with Reciprocal Rank Fusion.

        Args:
            query: Natural language search query.
            alpha: Weight for text vs embedding. 0=pure embedding, 1=pure text.
            top_k: Number of results to return.
            entity_type: Optional entity type filter.
            min_confidence: Minimum confidence threshold.

        Returns list of (Claim, fused_score) tuples sorted by score descending.
        Falls back to text-only if no embedding provider is configured.
        """
        with _span("attestdb.hybrid_search", {"query": query, "alpha": alpha, "top_k": top_k}):
            # Text search
            text_results = self.text_search(
                query, entity_type=entity_type, min_confidence=min_confidence,
                top_k=top_k * 2,
            )
            text_rank: dict[str, int] = {
                c.claim_id: i for i, c in enumerate(text_results)
            }
            claim_map: dict[str, Claim] = {c.claim_id: c for c in text_results}

            # Embedding search (if configured)
            embed_rank: dict[str, int] = {}
            if self._pipeline and self._pipeline._embed_fn and self._embedding_index:
                try:
                    query_vec = self._pipeline._embed_fn(query)
                    emb_results = self._embedding_index.search(query_vec, top_k * 2)
                    for i, (claim_id, _dist) in enumerate(emb_results):
                        embed_rank[claim_id] = i
                        if claim_id not in claim_map:
                            c = self.get_claim(claim_id)
                            if c and c.status == ClaimStatus.ACTIVE and c.confidence >= min_confidence:
                                if not entity_type or c.subject.entity_type == entity_type or c.object.entity_type == entity_type:
                                    claim_map[claim_id] = c
                except Exception as e:
                    _record_exc(e)
                    logger.warning("Embedding search failed in hybrid_search: %s", e)

            # RRF fusion
            k = 60  # RRF constant
            scores: dict[str, float] = {}
            for cid in claim_map:
                t_score = alpha * (1.0 / (k + text_rank[cid])) if cid in text_rank else 0.0
                e_score = (1.0 - alpha) * (1.0 / (k + embed_rank[cid])) if cid in embed_rank else 0.0
                scores[cid] = t_score + e_score

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            _set_attr("result_count", len(ranked))
            return [(claim_map[cid], score) for cid, score in ranked if cid in claim_map]

    # --- Entity operations (delegated to EntityManager) ---

    def resolve(self, entity_id: str) -> str:
        """Resolve an entity ID to its canonical form via union-find."""
        return self._entity_mgr.resolve(entity_id)

    def get_entity(self, entity_id: str) -> EntitySummary | None:
        """Return an EntitySummary for the given entity, or None."""
        return self._entity_mgr.get_entity(entity_id)

    def list_entities(
        self,
        entity_type: str | None = None,
        min_claims: int = 0,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[EntitySummary]:
        """List entities, optionally filtered by type and min claim count."""
        return self._entity_mgr.list_entities(entity_type, min_claims, offset, limit)

    def count_entities(self, entity_type: str | None = None, min_claims: int = 0) -> int:
        """Return the number of entities without materializing them."""
        return self._entity_mgr.count_entities(entity_type, min_claims)

    # --- Entity resolution ---

    def enable_entity_resolution(self, mode: str = "external_ids") -> None:
        """Enable entity resolution during ingestion.

        Modes: "external_ids" (exact + ext ID), "fuzzy" (+ text search), "full" (reserved).
        Must be called explicitly -- resolution is opt-in.
        """
        self._entity_mgr.enable_entity_resolution(mode)

    def resolve_entity(
        self,
        name: str,
        entity_type: str = "",
        external_ids: dict[str, str] | None = None,
    ) -> tuple[str | None, float]:
        """Manually resolve an entity name against existing entities.

        Returns (existing_entity_id, confidence) or (None, 0.0).
        """
        return self._entity_mgr.resolve_entity(name, entity_type, external_ids)

    def merge_entities(self, entity_a: str, entity_b: str, reason: str = "") -> str:
        """Merge two entities via same_as claim. Returns claim_id.

        The existing union-find handles downstream alias resolution.
        """
        return self._entity_mgr.merge_entities(entity_a, entity_b, reason)

    def find_duplicate_entities(
        self, min_confidence: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Scan for potential duplicate entities.

        Returns list of (entity_a, entity_b, confidence) sorted by confidence desc.
        """
        return self._entity_mgr.find_duplicate_entities(min_confidence)

    def auto_merge_duplicates(
        self, min_confidence: float = 0.95,
    ) -> dict:
        """Auto-merge detected duplicate entities above confidence threshold.

        Calls find_duplicate_entities(), ingests same_as claims for each pair.
        Returns dict with merged_count, skipped_count, claim_ids, merged_pairs.
        """
        return self._entity_mgr.auto_merge_duplicates(min_confidence)

    def search_entities(self, query: str, top_k: int = 10) -> list[EntitySummary]:
        """Search entities by text matching on id and display_name."""
        return self._entity_mgr.search_entities(query, top_k)

    # --- Entity aliases ---

    def add_entity_alias(self, alias: str, canonical: str) -> None:
        """Register a runtime entity alias.

        After this call, any ingested claim mentioning *alias* (after
        normalize_entity_id) will be stored under *canonical* instead.

        Runtime aliases are persisted to a ``.aliases.json`` sidecar
        file alongside the database.  They take precedence over the
        built-in ``ENTITY_ALIAS_MAP`` in ``vocabulary.py``.
        """
        normalized_alias = normalize_entity_id(alias)
        normalized_canonical = normalize_entity_id(canonical)
        self._pipeline._entity_aliases[normalized_alias] = normalized_canonical
        self._save_entity_aliases()

    def remove_entity_alias(self, alias: str) -> bool:
        """Remove a runtime entity alias.  Returns True if it existed."""
        normalized = normalize_entity_id(alias)
        removed = self._pipeline._entity_aliases.pop(normalized, None) is not None
        if removed:
            self._save_entity_aliases()
        return removed

    def get_entity_aliases(self) -> dict[str, str]:
        """Return a copy of the runtime entity alias map."""
        return dict(self._pipeline._entity_aliases)

    def _aliases_path(self) -> str | None:
        if self._is_memory_db:
            return None
        return self._db_path + ".aliases.json"

    def _load_entity_aliases(self) -> None:
        path = self._aliases_path()
        if path and os.path.exists(path):
            with builtins.open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._pipeline._entity_aliases = data

    def _save_entity_aliases(self) -> None:
        path = self._aliases_path()
        if not path:
            return
        aliases = self._pipeline._entity_aliases
        if aliases:
            with builtins.open(path, "w") as f:
                json.dump(aliases, f, indent=2, sort_keys=True)
        elif os.path.exists(path):
            os.remove(path)

    # --- Ask (question answering) — delegated to AskEngine ---

    def _get_llm_client(self):
        """Return (client, model) from the text extractor, or (None, None)."""
        return self._ask_engine._get_llm_client()

    def _llm_call(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str | None:
        """Single LLM call. Returns response text or None."""
        return self._ask_engine._llm_call(prompt, max_tokens, temperature)

    def _cluster_entities(self, entity_ids: list[str]) -> list[list[str]]:
        """Cluster candidate entities by 2-hop graph connectivity."""
        return self._ask_engine._cluster_entities(entity_ids)

    def _label_cluster(self, cluster: list[str], entity_map: dict[str, "EntitySummary"]) -> str:
        """Generate a human-readable label for a cluster of entities."""
        return self._ask_engine._label_cluster(cluster, entity_map)

    def ask(self, question: str, top_k: int = 10) -> AskResult:
        """Answer a natural-language question using the knowledge graph.

        Fast path (search hits >= 3): search -> evidence -> single LLM call.
        Slow path (few search hits): adds type catalog + LLM type selection.

        Returns AskResult with structured citations, contradictions, and gaps.
        Dict-compatible for backward compat (r["answer"] works).
        """
        return self._ask_engine.ask(question, top_k)

    def _gather_clustered_evidence(
        self,
        clusters: list[list[str]],
        entity_map: dict[str, "EntitySummary"],
        max_rels: int = 400,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build evidence organized by topic cluster."""
        return self._ask_engine._gather_clustered_evidence(clusters, entity_map, max_rels, collect_citations)

    def _ask_type_expand(
        self,
        question: str,
        search_hits: list,
        top_k: int,
    ) -> tuple:
        """Slow path: use type catalog + LLM to find relevant entity types."""
        return self._ask_engine._ask_type_expand(question, search_hits, top_k)

    def _gather_evidence(
        self,
        entity_ids: list[str],
        max_rels: int = 60,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build rich evidence from claims for a set of entities."""
        return self._ask_engine._gather_evidence(entity_ids, max_rels, collect_citations)

    # --- Claim operations ---

    def claims_for(
        self,
        entity_id: str,
        predicate_type: str | None = None,
        source_type: str | None = None,
        min_confidence: float = 0.0,
        principal: "Principal | None" = None,
    ) -> list[Claim]:
        claims = [
            claim_from_dict(d)
            for d in self._store.claims_for(entity_id, predicate_type, source_type, min_confidence)
        ]
        claims = self._apply_trust_filter(claims)
        return self._security_filter(claims, principal)

    def claims_by_content_id(self, content_id: str) -> list[Claim]:
        claims = [claim_from_dict(d) for d in self._store.claims_by_content_id(content_id)]
        return self._security_filter(claims)

    def claims_by_source_id(self, source_id: str) -> list[Claim]:
        """Return all claims with the given source_id via Rust index (O(k))."""
        claims = [claim_from_dict(d) for d in self._store.claims_by_source_id(source_id)]
        return self._security_filter(claims)

    def corroboration_report(self, min_sources: int = 2) -> dict:
        """Report on corroboration status across the knowledge graph."""
        return self._analytics.corroboration_report(min_sources)

    def diagnose_corroboration(self, content_id: str) -> dict:
        """Show how corroboration is counted for a specific content_id."""
        return self._analytics.diagnose_corroboration(content_id)

    def claims_for_predicate(self, predicate_id: str) -> list[Claim]:
        """Return all claims with the given predicate id via Rust index (O(k))."""
        claims = [claim_from_dict(d) for d in self._store.claims_by_predicate_id(predicate_id)]
        return self._security_filter(claims)

    # --- Fork / Merge ---

    def fork(self, name: str, dest_dir: str | None = None) -> "AttestDB":
        """Create a branched copy of this database for isolated experimentation.

        The fork is a full independent AttestDB backed by a copy of the
        current store file.  Ingest, query, retract — everything works.
        When done, call ``db.merge(fork)`` to promote the fork's new
        claims into the main database.

        Args:
            name: Branch name (used as filename suffix).
            dest_dir: Directory for the fork file. Defaults to same dir as main DB.

        Returns:
            A new AttestDB instance pointing at the forked store.

        Example::

            fork = db.fork("experiment_1")
            fork.ingest(...)
            report = db.merge(fork)
        """
        with _span("attestdb.fork", {"name": name}):
            return self._fork_inner(name, dest_dir)

    def _fork_inner(self, name: str, dest_dir: str | None) -> "AttestDB":
        if self._is_memory_db:
            raise ValueError("Cannot fork an in-memory database.")

        # Determine source file
        store_ext = ".attest"
        src = self._db_path if self._db_path.endswith(store_ext) else self._db_path + store_ext

        # Determine destination
        if dest_dir is None:
            dest_dir = os.path.dirname(src) or "."
        fork_base = os.path.join(dest_dir, f"{os.path.basename(self._db_path)}.fork.{name}")
        fork_file = fork_base if fork_base.endswith(store_ext) else fork_base + store_ext

        # Flush and copy
        self._store.close()
        try:
            if os.path.isdir(src):
                if os.path.exists(fork_file):
                    shutil.rmtree(fork_file)
                shutil.copytree(src, fork_file)
            else:
                shutil.copy2(src, fork_file)
        finally:
            self._store = _make_store(self._db_path)
            self._pipeline._store = self._store
            self._query_engine._store = self._store

        fork_db = AttestDB(fork_base, embedding_dim=self._embedding_dim, strict=self._strict)
        self._audit_write("fork_created", fork_name=name, fork_path=fork_file)
        return fork_db

    def merge(self, fork: "AttestDB", dry_run: bool = False) -> MergeReport:
        """Merge claims from a forked database into this one.

        Only claims that exist in the fork but NOT in the main DB
        are merged (by claim_id).  Duplicate claim_ids are skipped.

        Args:
            fork: A forked AttestDB (typically created via ``db.fork()``).
            dry_run: If True, compute the report without actually merging.

        Returns:
            MergeReport with counts of unique, shared, and conflicting beliefs.
        """
        with _span("attestdb.merge", {"dry_run": dry_run}):
            return self._merge_inner(fork, dry_run)

    def _merge_inner(self, fork: "AttestDB", dry_run: bool) -> MergeReport:
        self._check_permission("admin")

        # Collect content_ids and claim_ids from both sides
        main_claims = {c.claim_id: c for c in self.iter_claims()}
        fork_claims = {c.claim_id: c for c in fork.iter_claims()}

        main_content = {}  # content_id → (max_confidence, source_count)
        for c in main_claims.values():
            existing = main_content.get(c.content_id)
            if existing is None:
                main_content[c.content_id] = (c.confidence, 1)
            else:
                main_content[c.content_id] = (max(existing[0], c.confidence), existing[1] + 1)

        fork_content = {}
        for c in fork_claims.values():
            existing = fork_content.get(c.content_id)
            if existing is None:
                fork_content[c.content_id] = (c.confidence, 1)
            else:
                fork_content[c.content_id] = (max(existing[0], c.confidence), existing[1] + 1)

        # Identify new claims (in fork but not in main)
        new_claim_ids = set(fork_claims.keys()) - set(main_claims.keys())

        # Identify conflicts (same content_id, different confidence)
        shared_content = set(main_content.keys()) & set(fork_content.keys())
        conflicts = []
        for cid in shared_content:
            mc, ms = main_content[cid]
            fc, fs = fork_content[cid]
            if abs(mc - fc) > 0.05:  # Meaningful confidence difference
                # Find a representative claim for labels
                rep = next((c for c in fork_claims.values() if c.content_id == cid), None)
                if rep:
                    conflicts.append(MergeConflict(
                        content_id=cid,
                        subject=rep.subject.id,
                        predicate=rep.predicate.id,
                        object=rep.object.id,
                        self_confidence=mc,
                        other_confidence=fc,
                        self_sources=ms,
                        other_sources=fs,
                    ))

        # Entity diff
        main_entities = {e.id for e in self.list_entities()}
        fork_entities = {e.id for e in fork.list_entities()}

        report = MergeReport(
            self_unique_beliefs=len(set(main_claims.keys()) - set(fork_claims.keys())),
            other_unique_beliefs=len(new_claim_ids),
            shared_beliefs=len(set(main_claims.keys()) & set(fork_claims.keys())),
            conflicts=conflicts,
            self_unique_entities=sorted(main_entities - fork_entities),
            other_unique_entities=sorted(fork_entities - main_entities),
            shared_entities=sorted(main_entities & fork_entities),
            self_total_claims=len(main_claims),
            other_total_claims=len(fork_claims),
        )

        if not dry_run and new_claim_ids:
            # Actually merge: insert new claims from fork
            new_claims = [fork_claims[cid] for cid in new_claim_ids]
            for claim in new_claims:
                self._store.insert_claim(claim)
                self._append_chain(claim.claim_id)
            self._flush_chain_log()
            self._audit_write(
                "fork_merged",
                merged_count=len(new_claims),
                conflicts=len(conflicts),
                fork_path=fork._db_path,
            )

        return report

    # --- Retraction ---

    def retract(self, source_id: str, reason: str) -> RetractResult:
        """Retract all claims from a source. Tombstones active claims and creates audit trail."""
        self._check_permission("admin")
        with _span("attestdb.retract", {"source_id": source_id}):
            claim_ids = self._store.retract_source(source_id)
            _set_attr("retracted_count", len(claim_ids))
        if claim_ids:
            # Remove retracted claims from embedding index
            if self._embedding_index is not None:
                for cid in claim_ids:
                    self._embedding_index.remove(cid)
            # Ingest a retraction meta-claim for audit trail
            self._pipeline.ingest(ClaimInput(
                subject=(source_id, "document"),
                predicate=("retracted", "retracted"),
                object=(source_id, "document"),
                provenance={"source_type": "human_annotation", "source_id": source_id},
                payload={"schema": "", "data": {"reason": reason}},
            ))
        self._invalidate_cache()
        result = RetractResult(source_id, reason, len(claim_ids), claim_ids)
        self._audit_write(
            "source_retracted",
            source_id=source_id,
            reason=reason,
            retracted_count=len(claim_ids),
        )
        self._fire(EventType.SOURCE_RETRACTED, source_id=source_id, reason=reason, claim_ids=claim_ids)
        return result

    # --- Provenance cascade ---

    def _build_reverse_provenance_index(self) -> dict[str, list[str]]:
        """Scan all claims, build {claim_id: [dependent_claim_ids]}."""
        reverse_index: dict[str, list[str]] = {}
        for claim in self.iter_claims():
            for upstream_id in claim.provenance.chain:
                reverse_index.setdefault(upstream_id, []).append(claim.claim_id)
        return reverse_index

    def purge_expired(self) -> int:
        """Tombstone all claims whose expires_at has passed. Returns count of purged claims."""
        self._check_permission("admin")
        now_ns = int(time.time() * 1_000_000_000)
        purged = 0
        for claim in self.iter_claims():
            if _claim_is_expired(claim, now_ns) and claim.status == ClaimStatus.ACTIVE:
                self._store.update_claim_status(claim.claim_id, "tombstoned")
                purged += 1
                self._audit_write(
                    "claim_expired",
                    claim_id=claim.claim_id,
                    namespace=claim.namespace,
                )
        return purged

    def purge_source(self, source_id: str) -> int:
        """Physically delete all claims from a source from all indexes.

        Unlike retract() which tombstones claims (preserving append-only
        semantics and allowing undo), purge_source() permanently removes
        data. Use for maintenance: bad bulk loads, data cleanup, disk recovery.

        Returns the number of claims deleted. The DB file size won't shrink
        until LMDB reclaims pages on subsequent writes.
        """
        deleted = self._store.purge_source(source_id)
        self._invalidate_cache()
        return deleted

    # --- Archival lifecycle ---

    def archive(
        self,
        claim_ids: list[str] | None = None,
        criteria: dict | None = None,
        dry_run: bool = True,
    ) -> dict:
        """Move claims from active graph to archived status.

        Two modes:
        1. Explicit: pass claim_ids to archive specific claims
        2. Criteria-based: pass criteria dict to archive matching claims

        criteria options:
            status: list[str] — archive claims with these statuses
                (default: ["provenance_degraded", "verification_failed"])
            max_age_days: int — archive claims older than this
                (only applies to claims matching status filter)
            max_confidence: float — archive claims below this confidence
            source_types: list[str] — only archive from these source types
            exclude_verified: bool — never archive VERIFIED claims (default: True)

        dry_run: if True, return what would be archived without changing anything.
        """
        self._check_permission("admin")

        exclude_verified = True
        if criteria:
            exclude_verified = criteria.get("exclude_verified", True)

        # Collect candidate claim IDs
        candidates: list[Claim] = []
        if claim_ids:
            for cid in claim_ids:
                claim = self.get_claim(cid)
                if claim is not None:
                    candidates.append(claim)
        elif criteria:
            status_filter = set(criteria.get("status", ["provenance_degraded", "verification_failed"]))
            max_age_days = criteria.get("max_age_days")
            max_confidence = criteria.get("max_confidence")
            source_types = set(criteria.get("source_types", []))
            now_ns = int(time.time() * 1_000_000_000)
            age_cutoff_ns = now_ns - (max_age_days * 86400 * 1_000_000_000) if max_age_days is not None else 0

            for claim in self.iter_claims():
                # Re-fetch via get_claim to pick up status overlays
                full_claim = self.get_claim(claim.claim_id)
                if full_claim is None:
                    continue
                if full_claim.status.value not in status_filter:
                    continue
                if max_age_days is not None and full_claim.timestamp > age_cutoff_ns:
                    continue  # too recent
                if max_confidence is not None and full_claim.confidence > max_confidence:
                    continue
                if source_types and full_claim.provenance.source_type not in source_types:
                    continue
                candidates.append(full_claim)
        else:
            return {"archived_count": 0, "archived_claim_ids": [], "skipped_count": 0, "dry_run": dry_run}

        # Filter out protected claims
        protected_statuses = set()
        if exclude_verified:
            protected_statuses.add(ClaimStatus.VERIFIED)

        archived_ids: list[str] = []
        skipped = 0
        for claim in candidates:
            if claim.status in protected_statuses:
                skipped += 1
                continue
            if claim.status == ClaimStatus.ARCHIVED:
                skipped += 1
                continue  # already archived
            archived_ids.append(claim.claim_id)

        if not dry_run and archived_ids:
            updates = [(cid, "archived") for cid in archived_ids]
            self._store.update_claim_status_batch(updates)
            for cid in archived_ids:
                self._update_py_status_overlay(cid, "archived")
            self._invalidate_cache()
            for cid in archived_ids:
                self._audit_write("claim_archived", claim_id=cid)

        return {
            "archived_count": len(archived_ids),
            "archived_claim_ids": archived_ids,
            "skipped_count": skipped,
            "dry_run": dry_run,
        }

    def auto_archive(self, dry_run: bool = True) -> dict:
        """Run automatic archival based on default thresholds.

        Archives:
        - PROVENANCE_DEGRADED claims older than 90 days
        - VERIFICATION_FAILED claims older than 30 days

        Respects protected claims (verified are never archived).
        Designed to run daily via cron or scheduler.
        """
        degraded_result = self.archive(
            criteria={
                "status": ["provenance_degraded"],
                "max_age_days": 90,
                "exclude_verified": True,
            },
            dry_run=dry_run,
        )
        failed_result = self.archive(
            criteria={
                "status": ["verification_failed"],
                "max_age_days": 30,
                "exclude_verified": True,
            },
            dry_run=dry_run,
        )
        all_ids = degraded_result["archived_claim_ids"] + failed_result["archived_claim_ids"]
        return {
            "archived_count": len(all_ids),
            "archived_claim_ids": all_ids,
            "skipped_count": degraded_result["skipped_count"] + failed_result["skipped_count"],
            "dry_run": dry_run,
        }

    def retract_cascade(self, source_id: str, reason: str) -> CascadeResult:
        """Retract all claims from a source and mark downstream dependents as degraded."""
        retract_result = self.retract(source_id, reason)
        if not retract_result.claim_ids:
            return CascadeResult(source_retract=retract_result)

        reverse_index = self._build_reverse_provenance_index()

        # BFS downstream from retracted claim_ids
        degraded: list[str] = []
        queue = deque(retract_result.claim_ids)
        visited = set(retract_result.claim_ids)
        while queue:
            current = queue.popleft()
            for dependent_id in reverse_index.get(current, []):
                if dependent_id not in visited:
                    visited.add(dependent_id)
                    degraded.append(dependent_id)
                    queue.append(dependent_id)

        if degraded:
            updates = [(cid, "provenance_degraded") for cid in degraded]
            self._store.update_claim_status_batch(updates)

        return CascadeResult(
            source_retract=retract_result,
            degraded_claim_ids=degraded,
            degraded_count=len(degraded),
        )

    def trace_downstream(self, claim_id: str) -> DownstreamNode:
        """Build a tree of all claims that transitively depend on this claim."""
        reverse_index = self._build_reverse_provenance_index()

        def _build_tree(cid: str, visited: set[str]) -> DownstreamNode:
            node = DownstreamNode(claim_id=cid)
            visited.add(cid)
            for dep_id in reverse_index.get(cid, []):
                if dep_id not in visited:
                    node.dependents.append(_build_tree(dep_id, visited))
            return node

        return _build_tree(claim_id, set())

    def weak_foundations(
        self,
        max_source_confidence: float = 0.3,
        min_downstream: int = 5,
    ) -> list[dict]:
        """Find low-confidence claims that many other claims depend on.

        Scans all claims to find those with confidence below
        *max_source_confidence* that have at least *min_downstream*
        transitive dependents via provenance chain references.

        Returns a list of dicts with keys: claim_id, confidence,
        downstream_count, subject, predicate, object.
        """
        reverse_index = self._build_reverse_provenance_index()

        results = []
        for claim in self.iter_claims():
            if claim.confidence >= max_source_confidence:
                continue

            # BFS to count transitive dependents
            visited: set[str] = set()
            queue = list(reverse_index.get(claim.claim_id, []))
            while queue:
                dep_id = queue.pop()
                if dep_id in visited:
                    continue
                visited.add(dep_id)
                queue.extend(reverse_index.get(dep_id, []))

            if len(visited) >= min_downstream:
                results.append({
                    "claim_id": claim.claim_id,
                    "confidence": claim.confidence,
                    "downstream_count": len(visited),
                    "subject": claim.subject.id,
                    "predicate": claim.predicate.id,
                    "object": claim.object.id,
                })

        return results

    def recompute_confidence(
        self,
        exclude_sources: list[str] | None = None,
        exclude_source_types: list[str] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict:
        """Recompute confidence scores grouped by content_id.

        Iterates all claims, excludes any matching *exclude_sources* (by
        ``source_id``) or *exclude_source_types* (by ``source_type``), groups
        the remaining claims by ``content_id``, and computes a weighted max
        confidence per group.

        This is a **read-only** operation — the underlying store is not
        mutated.  The returned dict contains the recomputed mapping so the
        caller can apply the results as needed.

        Args:
            exclude_sources: Source IDs to skip.
            exclude_source_types: Source types to skip.
            weights: Maps ``source_type`` to a float multiplier applied to
                each claim's confidence before taking the group max.
                Unlisted source types default to ``1.0``.

        Returns:
            ``{"claims_processed": int, "claims_excluded": int,
              "groups_recomputed": int,
              "confidence_by_content_id": {content_id: float, ...}}``
        """
        exclude_src = set(exclude_sources) if exclude_sources else set()
        exclude_st = set(exclude_source_types) if exclude_source_types else set()
        weights = weights or {}

        groups: dict[str, float] = {}  # content_id → max weighted confidence
        claims_processed = 0
        claims_excluded = 0

        for claim in self.iter_claims():
            if claim.provenance.source_id in exclude_src or claim.provenance.source_type in exclude_st:
                claims_excluded += 1
                continue
            claims_processed += 1
            w = weights.get(claim.provenance.source_type, 1.0)
            weighted = min(claim.confidence * w, 1.0)
            cur = groups.get(claim.content_id)
            if cur is None or weighted > cur:
                groups[claim.content_id] = weighted

        return {
            "claims_processed": claims_processed,
            "claims_excluded": claims_excluded,
            "groups_recomputed": len(groups),
            "confidence_by_content_id": groups,
        }

    # --- Inquiry tracking ---

    def ingest_inquiry(
        self,
        question: str,
        subject: tuple[str, str],
        object: tuple[str, str],
        predicate_hint: str = "",
    ) -> str:
        """Register an inquiry (research question). Returns the inquiry claim_id."""
        claim_id = self.ingest(
            subject=subject,
            predicate=("inquiry", "inquiry"),
            object=object,
            provenance={"source_type": "human_annotation", "source_id": "inquiry"},
            payload={
                "schema": "",
                "data": {
                    "question": question,
                    "predicate_hint": predicate_hint,
                },
            },
        )
        self._fire(EventType.INQUIRY_CREATED, claim_id=claim_id, question=question)
        return claim_id

    def open_inquiries(self) -> list[Claim]:
        """Return all active inquiry claims."""
        return [
            c for c in self.claims_for_predicate("inquiry")
            if c.status == ClaimStatus.ACTIVE
        ]

    def check_inquiry_matches(
        self,
        subject_id: str | None = None,
        object_id: str | None = None,
        predicate_id: str | None = None,
    ) -> list[str]:
        """Check if any open inquiries match the given entity pair or predicate.

        Entity IDs are normalized before comparison to match ingestion behavior.
        Returns list of matching inquiry claim_ids.
        """
        # Normalize inputs to match how inquiries are stored
        norm_subj = normalize_entity_id(subject_id) if subject_id else None
        norm_obj = normalize_entity_id(object_id) if object_id else None

        matches = []
        for inquiry in self.open_inquiries():
            subj = inquiry.subject.id
            obj = inquiry.object.id
            hint = ""
            if inquiry.payload and inquiry.payload.data:
                hint = inquiry.payload.data.get("predicate_hint", "")

            matched = False
            # Match by subject+object pair (either direction)
            if norm_subj and norm_obj:
                if (subj == norm_subj and obj == norm_obj) or \
                   (subj == norm_obj and obj == norm_subj):
                    matched = True

            # Match by predicate hint
            if predicate_id and hint and predicate_id == hint:
                matched = True

            if matched:
                matches.append(inquiry.claim_id)

        return matches

    # --- Graph operations ---

    def path_exists(self, entity_a: str, entity_b: str, max_depth: int = 3) -> bool:
        """Check if a path exists between two entities in the claim graph."""
        a = normalize_entity_id(entity_a)
        b = normalize_entity_id(entity_b)
        return self._store.path_exists(a, b, max_depth)

    def find_paths(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
        top_k: int = 5,
    ) -> list[PathResult]:
        """Find top-k paths between two entities with per-hop metadata.

        Returns paths sorted by total_confidence (product of hop confidences).
        """
        return self._query_engine.find_paths(entity_a, entity_b, max_depth, top_k)

    def get_adjacency_list(self) -> dict[str, set[str]]:
        """Build in-memory adjacency list from all claim edges."""
        return self._store.get_adjacency_list()

    def get_weighted_adjacency(self) -> dict[tuple[str, str], dict]:
        """Build weighted adjacency with per-edge confidence, sources, predicates.

        Returns {(entity_a, entity_b): {"max_confidence": float,
        "source_types": set, "claim_count": int, "predicates": set}}
        where entity_a < entity_b (canonical ordering).
        """
        adj: dict[tuple[str, str], dict] = {}
        for claim in self.iter_claims():
            pair = (
                min(claim.subject.id, claim.object.id),
                max(claim.subject.id, claim.object.id),
            )
            if pair not in adj:
                adj[pair] = {
                    "max_confidence": claim.confidence,
                    "source_types": {claim.provenance.source_type},
                    "claim_count": 1,
                    "predicates": {claim.predicate.id},
                }
            else:
                info = adj[pair]
                info["max_confidence"] = max(info["max_confidence"], claim.confidence)
                info["source_types"].add(claim.provenance.source_type)
                info["claim_count"] += 1
                info["predicates"].add(claim.predicate.id)
        return adj

    # --- Knowledge topology ---

    def compute_topology(
        self,
        resolutions: list[float] | None = None,
        min_community_size: int = 3,
    ) -> None:
        """Run Leiden community detection at multiple resolutions."""
        try:
            from attestdb_enterprise.topology import KnowledgeTopology
        except ImportError:
            raise ImportError(
                "Knowledge topology requires attestdb-enterprise: "
                "pip install attestdb-enterprise"
            )

        adj = self.get_adjacency_list()
        entity_types = {}
        for eid in adj:
            raw = self._store.get_entity(eid)
            if raw:
                entity_types[eid] = entity_summary_from_dict(raw).entity_type

        self._topology = KnowledgeTopology(adj, entity_types)
        self._topology.compute(resolutions, min_community_size)

    def topics(self, level: int | None = None) -> list[TopicNode]:
        """Return topic nodes from computed topology."""
        if not hasattr(self, "_topology") or self._topology is None:
            raise RuntimeError("Call compute_topology() first")
        return self._topology.topics(level)

    def density_map(self) -> list[DensityMapEntry]:
        """Return density map from computed topology."""
        if not hasattr(self, "_topology") or self._topology is None:
            raise RuntimeError("Call compute_topology() first")
        return self._topology.density_map()

    def cross_domain_bridges(self, top_k: int = 20) -> list[CrossDomainBridge]:
        """Return cross-domain bridge entities from computed topology."""
        if not hasattr(self, "_topology") or self._topology is None:
            raise RuntimeError("Call compute_topology() first")
        return self._topology.cross_domain_bridges(top_k)

    # --- Quality report ---

    def quality_report(self, stale_threshold: int = 0, expected_patterns=None):
        """Generate a quality report over the entire knowledge graph."""
        return self._analytics.quality_report(stale_threshold, expected_patterns)

    # --- Confidence calibration ---

    def calibration_report(self, labeled_claims, n_bins: int = 5):
        """Compare predicted confidence scores against ground truth."""
        return self._analytics.calibration_report(labeled_claims, n_bins)

    # --- Knowledge health ---

    def knowledge_health(self):
        """Compute quantified health metrics for the knowledge graph."""
        return self._analytics.knowledge_health()

    # --- Topic query ---

    def query_topic(
        self,
        topic_id: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        max_claims: int = DEFAULT_MAX_CLAIMS,
    ) -> list[ContextFrame]:
        """Query all entities in a topic, returning a ContextFrame per member."""
        if not hasattr(self, "_topology") or self._topology is None:
            raise RuntimeError("Call compute_topology() first")

        # Find the topic node
        topic_node = None
        for node in self._topology.topics():
            if node.id == topic_id:
                topic_node = node
                break

        if topic_node is None:
            raise ValueError(f"Unknown topic: {topic_id!r}")

        frames = []
        for entity_id in topic_node.entities:
            try:
                frame = self.query(
                    entity_id, depth=depth,
                    min_confidence=min_confidence, max_claims=max_claims,
                )
                frames.append(frame)
            except Exception:
                continue

        return frames

    # --- Curation API ---

    def configure_curator(self, model: str = "heuristic", api_key: str | None = None, env_path: str | None = None) -> None:
        """Configure the curator model. Resets cached intelligence instances."""
        self._intel.configure_curator(model, api_key, env_path)

    def set_domain_context(self, context: str) -> None:
        """Set domain context for LLM extraction prioritization."""
        self._intel.set_domain_context(context)

    def _get_curator(self):
        return self._intel._get_curator()

    def _get_text_extractor(self):
        return self._intel._get_text_extractor()

    def _get_insight_engine(self):
        return self._intel._get_insight_engine()

    def _get_temporal_engine(self):
        return self._intel._get_temporal_engine()

    def _get_researcher(self):
        return self._intel._get_researcher()

    def curate(self, claims: list[ClaimInput], agent_id: str = "default"):
        """Triage and ingest claims through the curator."""
        return self._intel.curate(claims, agent_id)

    def ingest_text(self, text: str, source_id: str = "", use_curator: bool = True):
        """Extract claims from text and ingest. Optional curator triage."""
        return self._intel.ingest_text(text, source_id=source_id, use_curator=use_curator)

    def ingest_texts(self, texts: list[dict], use_curator: bool = True) -> dict:
        """Batch wrapper around ingest_text()."""
        return self._intel.ingest_texts(texts, use_curator=use_curator)

    def _get_extractor(self, mode: str = "llm"):
        return self._intel._get_extractor(mode)

    def ingest_chat(self, messages: list[dict], conversation_id: str = "", platform: str = "generic", use_curator: bool = True, extraction: str = "llm"):
        """Extract and ingest claims from a chat conversation."""
        return self._intel.ingest_chat(messages, conversation_id=conversation_id, platform=platform, use_curator=use_curator, extraction=extraction)

    def ingest_chat_file(self, path: str, platform: str = "auto", use_curator: bool = True, extraction: str = "llm"):
        """Extract and ingest claims from a chat log file."""
        return self._intel.ingest_chat_file(path, platform=platform, use_curator=use_curator, extraction=extraction)

    def connect(self, name: str, *, save: bool = False, **kwargs) -> "Connector":
        """Create a connector for an external data source."""
        return self._intel.connect(name, save=save, **kwargs)

    def _get_scheduler(self):
        return self._intel._get_scheduler()

    def _post_sync_check(self, connector_name: str, claims_ingested: int, **kwargs):
        return self._intel._post_sync_check(connector_name, claims_ingested, **kwargs)

    def sync(self, name: str, interval: float = 300.0, *, save: bool = False, run_immediately: bool = True, jitter: float = 0.1, **connector_kwargs):
        """Start continuous sync for a connector."""
        return self._intel.sync(name, interval, save=save, run_immediately=run_immediately, jitter=jitter, **connector_kwargs)

    def sync_status(self) -> list[dict]:
        """Status of all scheduled connectors."""
        return self._intel.sync_status()

    def sync_stop(self, name: str) -> None:
        """Stop a scheduled connector."""
        self._intel.sync_stop(name)

    def sync_stop_all(self) -> None:
        """Stop all scheduled connectors."""
        self._intel.sync_stop_all()

    def sync_pause(self, name: str) -> None:
        """Pause a connector (thread stays alive but skips execution)."""
        self._intel.sync_pause(name)

    def sync_resume(self, name: str) -> None:
        """Resume a paused connector."""
        self._intel.sync_resume(name)

    def sync_run_now(self, name: str) -> None:
        """Trigger an immediate run of a connector."""
        self._intel.sync_run_now(name)

    def ingest_slack(self, path: str, bot_ids: set[str] | None = None, channels: list[str] | None = None, use_curator: bool = True, extraction: str = "llm"):
        """Extract and ingest claims from a Slack workspace export ZIP."""
        return self._intel.ingest_slack(path, bot_ids=bot_ids, channels=channels, use_curator=use_curator, extraction=extraction)

    def find_bridges(self, **kwargs) -> list:
        return self._intel.find_bridges(**kwargs)

    def find_gaps(self, expected_patterns, **kwargs) -> list:
        return self._intel.find_gaps(expected_patterns, **kwargs)

    def find_confidence_alerts(self, **kwargs) -> list:
        return self._intel.find_confidence_alerts(**kwargs)

    # --- Temporal analysis ---

    def temporal_analyze(self, entity_id: str, analysis_type: str = "regime_shifts", metric: str = "claim_count", bucket: str | None = None, **kwargs) -> "TemporalResult":
        """Analyze temporal patterns for an entity's claims."""
        return self._intel.temporal_analyze(entity_id, analysis_type=analysis_type, metric=metric, bucket=bucket, **kwargs)

    def temporal_summary(self, entity_id: str, metric: str = "claim_count", bucket: str | None = None) -> "TemporalResult":
        """Run all temporal analyses (shifts, velocity, cycles) at once."""
        return self._intel.temporal_summary(entity_id, metric=metric, bucket=bucket)

    # --- Autonomous research ---

    def investigate(self, max_questions: int = 20, use_curator: bool = True, search_fn=None) -> "InvestigationReport":
        """Autonomous gap-closing loop: detect blindspots, research, ingest."""
        return self._intel.investigate(max_questions=max_questions, use_curator=use_curator, search_fn=search_fn)

    def research_question(self, question: str, entity_id: str | None = None, entity_type: str = "", predicate_hint: str = "") -> "ResearchResult":
        """Research a single question and ingest discovered claims."""
        return self._intel.research_question(question, entity_id=entity_id, entity_type=entity_type, predicate_hint=predicate_hint)

    # --- Structural embeddings ---

    def _structural_embedding_target(self):
        """Return the embedding index to use for structural embeddings."""
        if self._multi_embedding_index and "structural" in self._multi_embedding_index._spaces:
            return self._multi_embedding_index._spaces["structural"]
        return self._embedding_index

    def generate_structural_embeddings(self, dim: int = 64) -> int:
        """Compute SVD-based graph-structural embeddings and add to the embedding index.

        Returns the number of entities embedded.
        """
        from attestdb.intelligence.graph_embeddings import compute_graph_embeddings

        target = self._structural_embedding_target()
        if target is None:
            return 0

        adj = self.get_adjacency_list()
        entity_embeddings = compute_graph_embeddings(adj, dim)

        count = 0
        for entity_id, embedding in entity_embeddings.items():
            # Use synthetic key to avoid collisions (claims span two entities)
            key = f"_struct_{entity_id}"
            target.add(key, embedding)
            count += 1

        return count

    def generate_weighted_structural_embeddings(self, dim: int = 64) -> int:
        """Compute confidence-weighted SVD embeddings and add to the embedding index.

        Uses per-edge confidence and source diversity for edge weights,
        producing better-separated embeddings than uniform-weight SVD.

        Returns the number of entities embedded.
        """
        from attestdb.intelligence.graph_embeddings import compute_weighted_graph_embeddings

        target = self._structural_embedding_target()
        if target is None:
            return 0

        weighted_adj = self.get_weighted_adjacency()
        entity_embeddings = compute_weighted_graph_embeddings(weighted_adj, dim)

        count = 0
        for entity_id, embedding in entity_embeddings.items():
            key = f"_struct_{entity_id}"
            target.add(key, embedding)
            count += 1

        return count

    def pagerank(self, damping: float = 0.85, top_k: int = 0) -> dict[str, float]:
        """Compute PageRank centrality over the knowledge graph.

        Args:
            damping: Damping factor (0.85 standard).
            top_k: If > 0, return only the top-K entities by rank. 0 = all.

        Returns:
            {entity_id: pagerank_score} sorted by score descending.
        """
        from attestdb.intelligence.graph_embeddings import compute_pagerank

        adj = self.get_adjacency_list()
        scores = compute_pagerank(adj, damping=damping)
        ranked = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        if top_k > 0:
            return dict(list(ranked.items())[:top_k])
        return ranked

    def betweenness(self, sample_size: int = 200, top_k: int = 0, *,
                    adaptive: bool = True, max_nodes: int = 10_000) -> dict[str, float]:
        """Approximate betweenness centrality (identifies bridge entities).

        Args:
            sample_size: Number of source nodes for approximation (200 = fast).
            top_k: If > 0, return only the top-K entities. 0 = all.
            adaptive: Auto-reduce sample_size and extract subgraph for large graphs.
            max_nodes: Max nodes for subgraph extraction (0 = disable).

        Returns:
            {entity_id: betweenness_score} sorted descending.
        """
        from attestdb.intelligence.graph_embeddings import compute_betweenness_centrality

        adj = self.get_adjacency_list()
        scores = compute_betweenness_centrality(
            adj, sample_size=sample_size, adaptive=adaptive, max_nodes=max_nodes,
        )
        ranked = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        if top_k > 0:
            return dict(list(ranked.items())[:top_k])
        return ranked

    # --- Schema ---

    def schema(self) -> SchemaDescriptor:
        """Return a metadata descriptor of the knowledge graph.

        Shows entity types, predicates, relationship patterns (subject_type →
        predicate → object_type), source types, and counts. Similar to Neo4j's
        db.schema.visualization().
        """
        desc = SchemaDescriptor()

        # Entity types + totals from Rust stats (no Python iteration)
        rust_stats = self._store.stats()
        desc.total_entities = rust_stats.get("entity_count", 0)
        desc.entity_types = dict(rust_stats.get("entity_types", {}))

        # Predicate counts from Rust index (no Python iteration)
        desc.predicate_types = dict(self._store.predicate_counts())

        desc.total_claims = rust_stats.get("total_claims", 0)

        # Single claim scan only for relationship patterns + source types
        # (no Rust index for these yet)
        pattern_counts: dict[tuple[str, str, str], int] = {}
        for claim in self.iter_claims():
            st = claim.provenance.source_type
            desc.source_types[st] = desc.source_types.get(st, 0) + 1

            subj_type = claim.subject.entity_type
            pred = claim.predicate.id
            obj_type = claim.object.entity_type
            key = (subj_type, pred, obj_type)
            pattern_counts[key] = pattern_counts.get(key, 0) + 1

        # Build sorted relationship patterns
        desc.relationship_patterns = [
            RelationshipPattern(
                subject_type=k[0], predicate=k[1], object_type=k[2], count=v,
            )
            for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])
        ]

        # Registered vocabularies
        try:
            desc.registered_vocabularies = list(self._store.get_registered_vocabularies().keys())
        except Exception:
            pass

        return desc

    # --- Stats ---

    def stats(self) -> dict:
        s = self._store.stats()
        s["embedding_index_size"] = len(self._embedding_index) if self._embedding_index else 0
        return s

    # --- Provenance ---

    @staticmethod
    def resolve_source_url(source_id: str, source_type: str = "") -> str | None:
        """Resolve a source_id to a clickable URL for the original source."""
        from attestdb.core.provenance import resolve_source_url
        return resolve_source_url(source_id, source_type)

    # --- Build manifest & source health ---

    def build_manifest(self) -> "BuildManifest":
        """Access the build manifest for this database."""
        from attestdb.infrastructure.build_manifest import BuildManifest
        return BuildManifest(self._db_path)

    def source_health(self) -> list[dict]:
        """Per-source health: live LMDB counts merged with latest build info.

        Returns a list of dicts, one per source_id, with keys:
        ``source_id``, ``live_claims``, ``build_status``, ``build_ingested``,
        ``build_elapsed_sec``, ``build_errors``.
        """
        # Live counts from Rust index
        live_counts: dict[str, int] = {}
        if hasattr(self._store, "source_id_counts"):
            live_counts = self._store.source_id_counts()

        # Build history (may not exist yet)
        from attestdb.infrastructure.build_manifest import BuildManifest
        manifest = BuildManifest(self._db_path)
        report = manifest.latest_build()
        build_sources = report.sources if report else {}

        # Merge: all source_ids from either source
        all_ids = set(live_counts) | set(build_sources)
        results = []
        for sid in sorted(all_ids):
            entry: dict = {"source_id": sid, "live_claims": live_counts.get(sid, 0)}
            if sid in build_sources:
                sr = build_sources[sid]
                entry["build_status"] = sr.status
                entry["build_ingested"] = sr.ingested
                entry["build_elapsed_sec"] = sr.elapsed_sec
                entry["build_errors"] = sr.errors
            else:
                entry["build_status"] = None
            results.append(entry)
        return results

    # --- New API methods (Stage 7) ---

    def impact(self, source_id):
        """Analyze the impact of a source: how many claims depend on it."""
        return self._analytics.impact(source_id)

    def blindspots(self, min_claims: int = 5):
        """Find knowledge blindspots: single-source entities and gaps."""
        return self._analytics.blindspots(min_claims)

    def consensus(self, topic):
        """Analyze consensus around a topic (entity)."""
        return self._analytics.consensus(topic)

    def fragile(self, max_sources: int = 1, min_age_days: int = 0):
        """Find fragile claims: backed by few sources, optionally filtered by age."""
        return self._analytics.fragile(max_sources, min_age_days)

    def stale(self, days: int = 90, limit: int = 0):
        """Find stale claims: not updated within the given number of days."""
        return self._analytics.stale(days, limit)

    def audit(self, claim_id):
        """Build a full audit trail for a claim."""
        return self._analytics.audit(claim_id)

    def drift(self, days: int = 30):
        """Measure knowledge drift over the given time period."""
        return self._analytics.drift(days)

    def source_reliability(self, source_id=None):
        """Compute per-source reliability metrics."""
        return self._analytics.source_reliability(source_id)

    def hypothetical(self, claim):
        """What-if analysis: what would happen if this claim were ingested."""
        return self._analytics.hypothetical(claim)

    def what_if(self, subject, predicate, object, confidence=0.6, extra_causal_predicates=None):
        """One-call hypothetical reasoning -- queries the knowledge graph directly."""
        return self._analytics.what_if(subject, predicate, object, confidence, extra_causal_predicates)

    def build_entity_aliases(self, entity_type: str = "gene", batch_size: int = 50000):
        """Build an alias map by matching display names across entity ID formats."""
        return self._analytics.build_entity_aliases(entity_type, batch_size)

    def predict(self, entity_id, max_intermediaries=100, min_paths=3, min_consensus=0.65, directional_only=False, entity_aliases=None, extra_causal_predicates=None):
        """Discover novel regulatory predictions via causal composition."""
        return self._analytics.predict(entity_id, max_intermediaries, min_paths, min_consensus, directional_only, entity_aliases, extra_causal_predicates)

    # --- Verification (delegated to VerificationEngine) ---

    def configure_verification_budget(self, max_usd: float) -> None:
        """Set a per-session verification budget."""
        return self._verification.configure_verification_budget(max_usd)

    def verification_budget_status(self) -> dict:
        """Return current verification budget and spending."""
        return self._verification.verification_budget_status()

    def verify_claim(self, claim, tier=None, checks=None):
        """Run the verification pipeline on a claim."""
        return self._verification.verify_claim(claim, tier, checks)

    def check_freshness(self, source_type=None, entity_filter=None):
        """Check freshness of claims, optionally filtered by source type or entity."""
        return self._verification.check_freshness(source_type, entity_filter)

    def sweep_stale(self, dry_run=True, source_type=None, decay_factor=0.9, confidence_floor=0.2):
        """Sweep stale claims and optionally apply confidence decay."""
        return self._verification.sweep_stale(dry_run, source_type, decay_factor, confidence_floor)

    # --- Paper Audit ---

    def audit_paper(
        self,
        source: str,
        extraction_method: str = "llm",
        create_thread: bool = True,
        tier: int | None = None,
    ) -> "PaperAuditReport":
        """Ingest a scientific paper and verify its claims.

        source: DOI, PubMed ID, URL, or file path.
        Returns PaperAuditReport with per-claim verdicts.
        """
        from attestdb.intelligence.paper_auditor import audit_paper as _audit
        return _audit(self, source, extraction_method, create_thread, tier)

    def bulk_audit(
        self,
        sources: list[str],
        create_threads: bool = True,
        tier: int | None = None,
    ) -> list["PaperAuditReport"]:
        """Audit multiple papers. Budget-aware."""
        from attestdb.intelligence.paper_auditor import bulk_audit as _bulk
        return _bulk(self, sources, create_threads=create_threads, tier=tier)

    def configure_compute_backend(self, config: dict) -> None:
        """Configure the compute backend for reproducibility checks.

        config:
            backend: "local_docker" | "vibesci" | None
            local_docker: {memory_limit, cpu_limit, timeout_seconds, base_images}
            vibesci: {api_endpoint, api_token}
        """
        from attestdb.intelligence.compute_backend import create_compute_backend
        self._compute_backend = create_compute_backend(config)

    # --- Topic Threads (delegated to TopicThreads) ---

    @staticmethod
    def _format_claim_oneliner(c):
        """Format a claim as a single-line string."""
        from attestdb.infrastructure.topic_threads import TopicThreads
        return TopicThreads._format_claim_oneliner(c)

    @staticmethod
    def _select_key_findings(claims, target=3, max_count=10, initial_threshold=0.8, floor=0.4):
        """Select key findings with adaptive confidence threshold."""
        from attestdb.infrastructure.topic_threads import TopicThreads
        return TopicThreads._select_key_findings(claims, target, max_count, initial_threshold, floor)

    def _build_contradiction_details(self, contradictions, claims_dict):
        """Build structured contradiction details with evidence counts and confidence."""
        return self._threads._build_contradiction_details(contradictions, claims_dict)

    def _generate_thread_synthesis(self, key_findings, contradiction_details, seed_entities, claim_count, entity_count):
        """Generate a narrative synthesis of a thread via LLM."""
        return self._threads._generate_thread_synthesis(key_findings, contradiction_details, seed_entities, claim_count, entity_count)

    @staticmethod
    def _structural_thread_summary(key_findings, contradiction_details, seed_entities, claim_count, entity_count):
        """Structural metadata summary (no LLM)."""
        from attestdb.infrastructure.topic_threads import TopicThreads
        return TopicThreads._structural_thread_summary(key_findings, contradiction_details, seed_entities, claim_count, entity_count)

    def create_thread(self, seed, depth=2, min_confidence=0.5, source_filter=None, exclude_source_types=None):
        """Create a new topic thread from a seed."""
        return self._threads.create_thread(seed, depth, min_confidence, source_filter, exclude_source_types)

    def resume_thread(self, thread_id):
        """Resume an existing thread. Re-executes traversal, diffs against previous state."""
        return self._threads.resume_thread(thread_id)

    def extend_thread(self, thread_id, direction=None, additional_depth=1):
        """Extend a thread deeper into the graph from frontier entities."""
        return self._threads.extend_thread(thread_id, direction, additional_depth)

    def branch_thread(self, parent_thread_id, new_seed, keep_parent_claims=True):
        """Branch a thread to explore a tangent while preserving the parent."""
        return self._threads.branch_thread(parent_thread_id, new_seed, keep_parent_claims)

    def thread_context(self, thread_id, max_tokens=4000, focus=None):
        """Serialize a thread into a context string for an agent's context window."""
        return self._threads.thread_context(thread_id, max_tokens, focus)

    def list_threads(self, entity_filter=None, limit=20):
        """List existing threads, optionally filtered to threads touching an entity."""
        return self._threads.list_threads(entity_filter, limit)

    def _resolve_thread_seed(self, seed):
        """Resolve seed input to a list of entity IDs."""
        return self._threads._resolve_thread_seed(seed)

    def _persist_thread_state(self, state, seed_entities, overwrite_id=None):
        """Persist a thread state as claims in the graph."""
        return self._threads._persist_thread_state(state, seed_entities, overwrite_id)

    def _load_thread_state(self, thread_id):
        """Load a thread's state from its most recent claim payload."""
        return self._threads._load_thread_state(thread_id)

    def _mark_threads_stale(self, entity_id):
        """Mark threads touching this entity as stale (narrative needs regeneration)."""
        return self._threads._mark_threads_stale(entity_id)

    # --- Knowledge-Intelligence APIs ---

    def evolution(self, entity_id, since=None):
        """Entity-specific knowledge evolution over time."""
        return self._analytics.evolution(entity_id, since)

    def trace(self, entity_a, entity_b, max_depth=3, top_k=5):
        """Trace reasoning chains between two entities."""
        return self._analytics.trace(entity_a, entity_b, max_depth, top_k)

    def suggest_investigations(self, top_k: int = 10):
        """Suggest high-value investigation targets."""
        return self._analytics.suggest_investigations(top_k)

    def close_gaps(self, hypothesis=None, top_k=5, search_fn=None, use_curator=True):
        """Search for evidence to close knowledge gaps."""
        return self._analytics.close_gaps(hypothesis, top_k, search_fn, use_curator)

    def test_hypothesis(self, hypothesis):
        """Test a hypothesis against existing evidence."""
        return self._analytics.test_hypothesis(hypothesis)

    def discover(self, top_k: int = 10):
        """Discover novel insights from structural patterns."""
        return self._analytics.discover(top_k)

    def analogies(self, *args, **kwargs):
        """Find structural analogies for an entity."""
        return self._analytics.analogies(*args, **kwargs)

    def _parse_timestamp(self, ts):
        """Parse a timestamp string/int/float to nanoseconds."""
        return self._analytics._parse_timestamp(ts)

    def diff(self, since, until=None):
        """Compute knowledge diff between two time points."""
        return self._analytics.diff(since, until)

    def resolve_contradictions(self, *args, **kwargs):
        """Detect and optionally resolve contradictions in the knowledge graph."""
        return self._analytics.resolve_contradictions(*args, **kwargs)

    def simulate(self, *args, **kwargs):
        """Simulate a hypothetical action and predict its impact."""
        return self._analytics.simulate(*args, **kwargs)

    def _simulate_retract_source(self, source_id):
        """Simulate retracting a source."""
        return self._analytics._simulate_retract_source(source_id)

    def _simulate_add_claim(self, claim_input):
        """Simulate adding a claim."""
        return self._analytics._simulate_add_claim(claim_input)

    def _simulate_remove_entity(self, entity_id):
        """Simulate removing an entity."""
        return self._analytics._simulate_remove_entity(entity_id)

    def compile(self, topic, max_entities=50, use_llm=False):
        """Compile a knowledge brief on a topic."""
        return self._analytics.compile(topic, max_entities, use_llm)

    def _template_summary(self, *args, **kwargs):
        """Build a template-based summary."""
        return self._analytics._template_summary(*args, **kwargs)

    def explain_why(self, *args, **kwargs):
        """Explain the relationship between two entities."""
        return self._analytics.explain_why(*args, **kwargs)

    def forecast(self, entity_id, top_k: int = 10):
        """Forecast future connections for an entity."""
        return self._analytics.forecast(entity_id, top_k)

    def merge_report(self, other):
        """Compare this database with another and generate a merge report."""
        return self._analytics.merge_report(other)

    def _gather_consensus_evidence(self, question, max_claims=50):
        """Gather evidence for consensus from the knowledge graph."""
        return self._analytics._gather_consensus_evidence(question, max_claims)

    def agent_consensus(self, *args, **kwargs):
        """Multi-LLM consensus with cross-pollination."""
        return self._analytics.agent_consensus(*args, **kwargs)

    # --- Eval engine (delegated to EvalEngine) ---

    def generate_eval(self, **kwargs):
        """Generate a domain evaluation set from the knowledge graph."""
        return self._eval_engine.generate_eval(**kwargs)

    def score_eval(self, eval_set, agent_answers, agent_id="unknown"):
        """Score agent answers against an eval set."""
        return self._eval_engine.score_eval(eval_set, agent_answers, agent_id)

    def eval_history(self, agent_id=None):
        """Get eval result history from the graph."""
        return self._eval_engine.eval_history(agent_id)

    # --- Agent registry (delegated to AgentRegistry) ---

    def register_agent(self, agent_id, **kwargs):
        """Register an agent and its domain expertise."""
        return self._agent_registry.register(agent_id, **kwargs)

    def list_agents(self):
        """List all registered agents with their domains and eval scores."""
        return self._agent_registry.list_agents()

    def agent_profile(self, agent_id):
        """Get full agent profile: domains, capabilities, eval history."""
        return self._agent_registry.agent_profile(agent_id)

    def agent_leaderboard(self, domain=None):
        """Rank agents by eval score, optionally filtered by domain."""
        return self._agent_registry.leaderboard(domain)


# Backward-compat alias
SubstrateDB = AttestDB


def open(
    path: str,
    embedding_dim: int | None = 768,
    strict: bool = False,
    embedding_spaces: dict[str, int] | None = None,
) -> AttestDB:
    """Open or create an Attest database."""
    return AttestDB(path, embedding_dim=embedding_dim, strict=strict, embedding_spaces=embedding_spaces)
