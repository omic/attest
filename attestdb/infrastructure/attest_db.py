"""Top-level AttestDB class — the user-facing API."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
DEFAULT_MAX_RELATIONSHIPS = 400
ENTITY_BUDGET_FLOOR = 60
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
        instance._query_engine = QueryEngine(
            instance._store,
            claim_converter=lambda d: claim_from_dict(d),
        )
        instance._event_hooks = {}
        instance._scheduler = None
        instance._purge_timer = None
        instance._purge_interval = 0
        instance._webhooks = []
        instance._webhooks_path = None
        instance._webhook_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="attest-webhook")
        instance._chain_log = []
        instance._last_chain_hash = "genesis"
        instance._actor = ""
        instance._audit_path = None
        instance._rbac = {}
        instance._rbac_path = None
        instance._rbac_enabled = False
        instance._autodidact = None
        instance._decay_config = None
        instance._curator = None
        instance._text_extractor = None
        instance._insight_engine = None
        instance._temporal_engine = None
        instance._researcher = None
        instance._curator_model = "heuristic"
        instance._curator_api_key = None
        instance._curator_env_path = None
        instance._domain_context = None
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

        # Query engine — Rust engine handles status filtering natively
        self._query_engine = QueryEngine(self._store, claim_converter=claim_from_dict)

        # Event hooks
        self._event_hooks: dict[str, list] = {}

        # Continuous ingestion scheduler (lazy-init)
        self._scheduler = None

        # Auto-purge timer for TTL expiry
        self._purge_timer: threading.Timer | None = None
        self._purge_interval: float = 0

        # Webhook registrations
        self._webhooks: list[dict] = []
        self._webhooks_path: str | None = None
        self._webhook_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="attest-webhook")
        if not self._is_memory_db:
            self._webhooks_path = self._db_path + ".webhooks.json"
            self._load_webhooks()

        # Tamper-evident audit chain (Merkle hash chain over claim IDs)
        self._chain_log: list[tuple[str, str]] = []  # [(claim_id, chain_hash), ...]
        self._last_chain_hash: str = "genesis"
        self._load_chain_log()

        # Audit log (append-only JSONL)
        self._actor: str = ""
        self._audit_path: str | None = None
        if not self._is_memory_db:
            self._audit_path = self._db_path + ".audit.jsonl"

        # RBAC (per-namespace role-based access control)
        self._rbac: dict[str, dict[str, str]] = {}  # {principal: {namespace: role}}
        self._rbac_path: str | None = None
        self._rbac_enabled: bool = False
        if not self._is_memory_db:
            self._rbac_path = self._db_path + ".rbac.json"
            self._load_rbac()

        # Autodidact daemon (lazy-init)
        self._autodidact: "AutodidactDaemon | None" = None

        # Confidence decay config (query-time only)
        self._decay_config: "DecayConfig | None" = None

        # Lazy-init intelligence components
        self._curator = None
        self._text_extractor = None
        self._insight_engine = None
        self._temporal_engine = None
        self._researcher = None
        self._curator_model = "heuristic"
        self._curator_api_key = None
        self._curator_env_path = None
        self._domain_context: str | None = None

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
        return claim_from_dict(d)

    def _all_claims(self) -> list[Claim]:
        """Return all claims via Rust store."""
        return [claim_from_dict(d) for d in self._store.all_claims()]

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
                if self._autodidact:
                    self._autodidact.stop()
                    self._autodidact = None
                if self._purge_timer:
                    self._purge_timer.cancel()
                    self._purge_timer = None
                if self._scheduler:
                    self._scheduler.stop_all()
                    self._scheduler = None
                self._webhook_executor.shutdown(wait=False)
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
        self._actor = actor

    def _audit_write(self, event: str, **kwargs) -> None:
        """Append an audit event to the JSONL log."""
        if not self._audit_path:
            return

        entry = {
            "event": event,
            "timestamp": int(time.time() * 1_000_000_000),
            "actor": self._actor,
            **kwargs,
        }
        try:
            with builtins.open(self._audit_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write audit event: %s", exc)

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
        if not self._audit_path or not os.path.exists(self._audit_path):
            return []
        events = []
        with builtins.open(self._audit_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = d.get("timestamp", 0)
                if ts <= since:
                    continue
                if event_type and d.get("event") != event_type:
                    continue
                if actor and d.get("actor") != actor:
                    continue
                evt = d.get("event", "")
                events.append(AuditEvent(
                    event=evt,
                    timestamp=ts,
                    actor=d.get("actor", ""),
                    claim_id=d.get("claim_id", ""),
                    source_id=d.get("source_id", ""),
                    namespace=d.get("namespace", ""),
                    details={k: v for k, v in d.items()
                             if k not in ("event", "timestamp", "actor",
                                          "claim_id", "source_id", "namespace")},
                ))
                if len(events) >= limit:
                    break
        return events

    # --- RBAC (role-based access control) ---

    def _load_rbac(self) -> None:
        """Load RBAC config from sidecar file."""
        if self._rbac_path and os.path.exists(self._rbac_path):
            try:
                with builtins.open(self._rbac_path) as f:
                    data = json.load(f)
                self._rbac = data.get("grants", {})
                self._rbac_enabled = data.get("enabled", False)
            except Exception as exc:
                logger.warning("Failed to load RBAC config: %s", exc)

    def _save_rbac(self) -> None:
        """Persist RBAC config to sidecar file."""
        if not self._rbac_path:
            return
        try:
            with builtins.open(self._rbac_path, "w") as f:
                json.dump({"enabled": self._rbac_enabled, "grants": self._rbac}, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save RBAC config: %s", exc)

    def enable_rbac(self) -> None:
        """Enable RBAC enforcement. Once enabled, all mutations require a
        matching grant for the current actor + active namespace."""
        self._rbac_enabled = True
        self._save_rbac()

    def disable_rbac(self) -> None:
        """Disable RBAC enforcement (all operations permitted)."""
        self._rbac_enabled = False
        self._save_rbac()

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
        if isinstance(role, Role):
            role = role.value
        if role not in ("admin", "writer", "reader"):
            raise ValueError(f"Invalid role: {role}. Must be admin, writer, or reader.")
        self._check_permission("admin")
        grants = self._rbac.setdefault(principal, {})
        grants[namespace] = role
        self._save_rbac()
        self._audit_write("rbac_grant", principal=principal, namespace=namespace, role=role)

    def revoke(self, principal: str, namespace: str) -> None:
        """Revoke a principal's access to a namespace.

        Args:
            principal: User or service identity.
            namespace: Namespace to revoke. Use "*" to revoke global access.
        """
        self._check_permission("admin")
        grants = self._rbac.get(principal, {})
        grants.pop(namespace, None)
        if not grants:
            self._rbac.pop(principal, None)
        self._save_rbac()
        self._audit_write("rbac_revoke", principal=principal, namespace=namespace)

    def list_grants(self, principal: str | None = None) -> dict:
        """List all RBAC grants, optionally filtered by principal.

        Returns dict: {principal: {namespace: role}}.
        """
        if principal:
            grants = self._rbac.get(principal, {})
            return {principal: grants} if grants else {}
        return dict(self._rbac)

    def _get_actor_role(self, namespace: str = "") -> str | None:
        """Get the current actor's effective role for a namespace."""
        if not self._actor:
            return None
        grants = self._rbac.get(self._actor, {})
        # Check specific namespace first, then wildcard
        role = grants.get(namespace) or grants.get("*")
        return role

    def _check_permission(self, required_role: str) -> None:
        """Raise PermissionError if RBAC is enabled and actor lacks permission.

        Role hierarchy: admin > writer > reader.
        """
        if not self._rbac_enabled:
            return
        if not self._actor:
            raise PermissionError("RBAC is enabled but no actor set. Call db.set_actor() first.")

        # Determine effective namespace
        ns_filter = self._store.get_namespace_filter()
        namespace = ns_filter[0] if len(ns_filter) == 1 else ""

        role = self._get_actor_role(namespace)
        if role is None:
            raise PermissionError(
                f"Actor '{self._actor}' has no access to namespace '{namespace or '*'}'."
            )

        hierarchy = {"admin": 3, "writer": 2, "reader": 1}
        required_level = hierarchy.get(required_role, 0)
        actual_level = hierarchy.get(role, 0)
        if actual_level < required_level:
            raise PermissionError(
                f"Actor '{self._actor}' has role '{role}' but '{required_role}' is required."
            )

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

    # --- Event hooks ---

    def on(self, event: str, callback) -> None:
        """Register a callback for a lifecycle event.

        Events: "claim_ingested", "source_retracted", "claim_corroborated",
        "inquiry_matched", "snapshot_created", "inquiry_created",
        "sync_completed", "insight_alerts"
        """
        self._event_hooks.setdefault(event, []).append(callback)

    def off(self, event: str, callback) -> None:
        """Remove a registered callback."""
        hooks = self._event_hooks.get(event, [])
        if callback in hooks:
            hooks.remove(callback)

    def _fire(self, event: str, **kwargs) -> None:
        """Fire all callbacks for an event and POST to registered webhooks."""
        for cb in self._event_hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as exc:
                logger.warning("Event hook %s failed: %s", event, exc)
        self._fire_webhooks(event, kwargs)

    # --- Webhooks ---

    def _load_webhooks(self) -> None:
        if self._webhooks_path and os.path.exists(self._webhooks_path):
            try:
                with builtins.open(self._webhooks_path) as f:
                    self._webhooks = json.load(f)
            except Exception:
                self._webhooks = []

    def _save_webhooks(self) -> None:
        if self._webhooks_path:
            with builtins.open(self._webhooks_path, "w") as f:
                json.dump(self._webhooks, f)

    def register_webhook(
        self, url: str, events: list[str] | None = None, secret: str | None = None,
    ) -> None:
        """Register an HTTP endpoint to receive event notifications.

        Args:
            url: The URL to POST event payloads to.
            events: List of event names to subscribe to. None = all events.
            secret: Optional HMAC-SHA256 secret for payload signing.
        """
        # Remove existing registration for same URL
        self._webhooks = [w for w in self._webhooks if w["url"] != url]
        self._webhooks.append({
            "url": url,
            "events": events,
            "secret": secret,
        })
        self._save_webhooks()

    def remove_webhook(self, url: str) -> bool:
        """Remove a registered webhook by URL. Returns True if found."""
        before = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w["url"] != url]
        if len(self._webhooks) < before:
            self._save_webhooks()
            return True
        return False

    def list_webhooks(self) -> list[dict]:
        """Return registered webhooks (secrets are masked)."""
        return [
            {"url": w["url"], "events": w["events"], "has_secret": bool(w.get("secret"))}
            for w in self._webhooks
        ]

    def _fire_webhooks(self, event: str, data: dict) -> None:
        """POST event payload to matching webhook URLs (fire-and-forget, non-blocking)."""
        if not self._webhooks:
            return
        try:
            import requests as _requests
        except ImportError:
            return
        payload = json.dumps({"event": event, "data": {k: str(v) for k, v in data.items()}})
        for wh in self._webhooks:
            if wh["events"] is not None and event not in wh["events"]:
                continue
            headers = {"Content-Type": "application/json"}
            if wh.get("secret"):
                sig = hmac.new(
                    wh["secret"].encode(), payload.encode(), hashlib.sha256,
                ).hexdigest()
                headers["X-Attest-Signature"] = sig
            self._webhook_executor.submit(
                self._post_webhook, _requests, wh["url"], payload, headers,
            )

    @staticmethod
    def _post_webhook(requests_mod, url: str, payload: str, headers: dict) -> None:
        """Execute a single webhook POST (runs in thread pool)."""
        try:
            requests_mod.post(url, data=payload, headers=headers, timeout=5)
        except Exception as exc:
            logger.warning("Webhook POST to %s failed: %s", url, exc)

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

        self._fire("snapshot_created", dest_path=dest_path)
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
        self._text_extractor = None

    def register_predicate(self, predicate_id: str, constraints: dict) -> None:
        self._store.register_predicate(predicate_id, constraints)
        self._text_extractor = None

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
    ) -> str:
        """Ingest a single claim. Returns claim_id.

        Accepts either a ClaimInput object or individual fields::

            db.ingest(claim_input)
            db.ingest(subject=(...), predicate=(...), object=(...), provenance={...})
            db.ingest(subject=(...), ..., namespace="team_a", ttl_seconds=3600)
        """
        self._check_permission("writer")
        if isinstance(subject, ClaimInput):
            ci = subject
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
        self._audit_write(
            "claim_ingested",
            claim_id=claim_id,
            namespace=ci.namespace if hasattr(ci, "namespace") else "",
            subject=ci.subject[0],
            predicate=ci.predicate[0],
            object=ci.object[0],
        )
        self._fire("claim_ingested", claim_id=claim_id, claim_input=ci)
        # Check corroboration
        content_id = compute_content_id(
            normalize_entity_id(ci.subject[0]),
            ci.predicate[0],
            normalize_entity_id(ci.object[0]),
        )
        existing = self.claims_by_content_id(content_id)
        if len(existing) > 1:
            self._fire("claim_corroborated", content_id=content_id, count=len(existing))
        # Check inquiry matches
        matches = self.check_inquiry_matches(
            subject_id=ci.subject[0], object_id=ci.object[0], predicate_id=ci.predicate[0],
        )
        for m in matches:
            self._fire("inquiry_matched", inquiry_id=m, claim_id=claim_id)
        return claim_id

    def ingest_batch(self, claims: list[ClaimInput]) -> BatchResult:
        self._check_permission("writer")
        with _span("attestdb.ingest_batch", {"batch_size": len(claims)}):
            result = self._pipeline.ingest_batch(claims, on_ingested=self._append_chain)
            self._flush_chain_log()
            _set_attr("ingested", result.ingested)
            _set_attr("duplicates", result.duplicates)
        if result.ingested > 0:
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
    ) -> ContextFrame:
        with _span("attestdb.query", {"focal_entity": focal_entity, "depth": depth}):
            frame = self._query_engine.query(
                focal_entity, depth, min_confidence, exclude_source_types,
                max_claims, max_tokens,
                confidence_threshold=confidence_threshold,
                predicate_types=predicate_types,
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
                model=self._curator_model,
                api_key=self._curator_api_key,
                env_path=self._curator_env_path,
            )
        elif not frame.narrative:
            try:
                from attestdb.intelligence.narrative import generate_narrative
                frame.narrative = generate_narrative(frame)
            except ImportError:
                pass  # narrative generation requires attestdb-intelligence
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

    # --- Entity operations ---

    def resolve(self, entity_id: str) -> str:
        """Resolve an entity ID to its canonical form via union-find."""
        return self._store.resolve(normalize_entity_id(entity_id))

    def get_entity(self, entity_id: str) -> EntitySummary | None:
        canonical = self._store.resolve(normalize_entity_id(entity_id))
        raw = self._store.get_entity(canonical)
        return entity_summary_from_dict(raw) if raw else None

    def list_entities(
        self,
        entity_type: str | None = None,
        min_claims: int = 0,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[EntitySummary]:
        # Rust uses limit=0 to mean "no limit"; Python uses None.
        rust_limit = 0 if limit is None else limit
        raw = self._store.list_entities(entity_type, min_claims, offset, rust_limit)
        return [entity_summary_from_dict(d) for d in raw]

    def count_entities(self, entity_type: str | None = None, min_claims: int = 0) -> int:
        """Return the number of entities without materializing them."""
        return self._store.count_entities(entity_type, min_claims)

    # --- Entity resolution ---

    def enable_entity_resolution(self, mode: str = "external_ids") -> None:
        """Enable entity resolution during ingestion.

        Modes: "external_ids" (exact + ext ID), "fuzzy" (+ text search), "full" (reserved).
        Must be called explicitly — resolution is opt-in.
        """
        from attestdb.infrastructure.entity_resolver import EntityResolver

        resolver = EntityResolver(self._store, mode=mode)
        resolver.build_index()
        self._pipeline._resolver = resolver
        self._entity_resolver = resolver

    def resolve_entity(
        self,
        name: str,
        entity_type: str = "",
        external_ids: dict[str, str] | None = None,
    ) -> tuple[str | None, float]:
        """Manually resolve an entity name against existing entities.

        Returns (existing_entity_id, confidence) or (None, 0.0).
        """
        if not hasattr(self, "_entity_resolver") or self._entity_resolver is None:
            from attestdb.infrastructure.entity_resolver import EntityResolver
            resolver = EntityResolver(self._store, mode="external_ids")
            resolver.build_index()
            return resolver.resolve(name, entity_type, external_ids)
        return self._entity_resolver.resolve(name, entity_type, external_ids)

    def merge_entities(self, entity_a: str, entity_b: str, reason: str = "") -> str:
        """Merge two entities via same_as claim. Returns claim_id.

        The existing union-find handles downstream alias resolution.
        """
        return self.ingest(
            subject=(entity_a, "entity"),
            predicate=("same_as", "alias"),
            object=(entity_b, "entity"),
            provenance={
                "source_type": "human_annotation",
                "source_id": "entity_resolution",
            },
            payload={"schema": "", "data": {"reason": reason}} if reason else None,
        )

    def find_duplicate_entities(
        self, min_confidence: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Scan for potential duplicate entities.

        Returns list of (entity_a, entity_b, confidence) sorted by confidence desc.
        """
        from attestdb.infrastructure.entity_resolver import EntityResolver

        if hasattr(self, "_entity_resolver") and self._entity_resolver is not None:
            return self._entity_resolver.find_duplicates(min_confidence)
        resolver = EntityResolver(self._store, mode="external_ids")
        resolver.build_index()
        return resolver.find_duplicates(min_confidence)

    def auto_merge_duplicates(
        self, min_confidence: float = 0.95,
    ) -> dict:
        """Auto-merge detected duplicate entities above confidence threshold.

        Calls find_duplicate_entities(), ingests same_as claims for each pair.
        Returns dict with merged_count, skipped_count, claim_ids, merged_pairs.
        """
        dupes = self.find_duplicate_entities(min_confidence=min_confidence)
        merged_count = 0
        skipped_count = 0
        claim_ids: list[str] = []
        merged_pairs: list[tuple[str, str, float]] = []

        for entity_a, entity_b, confidence in dupes:
            try:
                claim_id = self.merge_entities(
                    entity_a, entity_b,
                    reason=f"auto-merged (confidence={confidence:.3f})",
                )
                merged_count += 1
                claim_ids.append(claim_id)
                merged_pairs.append((entity_a, entity_b, confidence))
            except Exception:
                skipped_count += 1

        return {
            "merged_count": merged_count,
            "skipped_count": skipped_count,
            "claim_ids": claim_ids,
            "merged_pairs": merged_pairs,
        }

    def search_entities(self, query: str, top_k: int = 10) -> list[EntitySummary]:
        """Search entities by text matching on id and display_name."""
        return [entity_summary_from_dict(d) for d in self._store.search_entities(query, top_k)]

    # --- Ask (question answering) ---

    def _get_llm_client(self):
        """Return (client, model) from the text extractor, or (None, None)."""
        try:
            ext = self._get_text_extractor()
            if ext._client and ext._llm_model:
                return ext._client, ext._llm_model
        except Exception:
            pass
        return None, None

    def _llm_call(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str | None:
        """Single LLM call. Returns response text or None."""
        client, model = self._get_llm_client()
        if not client:
            return None
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return r.choices[0].message.content.strip()
            except Exception:
                return None

    # --- Graph-neighborhood clustering for ask() ---

    def _cluster_entities(self, entity_ids: list[str]) -> list[list[str]]:
        """Cluster candidate entities by 2-hop graph connectivity.

        Two candidates are "topic-connected" if directly adjacent OR share
        at least one graph neighbor.  Returns connected components sorted
        by size descending.
        """
        if len(entity_ids) <= 1:
            return [list(entity_ids)] if entity_ids else []

        adj = self.get_adjacency_list()

        # Build neighbor sets for each candidate (from full graph adjacency)
        neighbors: dict[str, set[str]] = {}
        for eid in entity_ids:
            neighbors[eid] = adj.get(eid, set())

        # Build candidate-level adjacency: two candidates are connected if
        # they share a direct edge OR share >= 1 common neighbor (2-hop)
        cand_adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        ids = list(entity_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                # Direct edge?
                if b in neighbors[a]:
                    cand_adj[a].add(b)
                    cand_adj[b].add(a)
                # Shared neighbor (2-hop)?
                elif neighbors[a] & neighbors[b]:
                    cand_adj[a].add(b)
                    cand_adj[b].add(a)

        # BFS connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for eid in entity_ids:
            if eid in visited:
                continue
            component: list[str] = []
            queue = deque([eid])
            visited.add(eid)
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in cand_adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            clusters.append(component)

        clusters.sort(key=lambda c: -len(c))
        return clusters

    def _label_cluster(self, cluster: list[str], entity_map: dict[str, "EntitySummary"]) -> str:
        """Generate a human-readable label for a cluster of entities.

        Uses topology community labels if computed, otherwise falls back
        to dominant entity type + top 2 entity names.
        """
        # Try topology labels (opportunistic — only if already computed)
        if hasattr(self, "_topology") and self._topology is not None:
            # Find the community that contains the most cluster members
            best_label = ""
            best_overlap = 0
            cluster_set = set(cluster)
            for _res, communities in self._topology.communities.items():
                for comm in communities:
                    overlap = len(cluster_set & set(comm.members))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_label = comm.label
            if best_label:
                return best_label

        # Fallback: dominant entity type + top 2 names
        type_counts: dict[str, int] = {}
        for eid in cluster:
            e = entity_map.get(eid)
            if e:
                etype = e.entity_type or "unknown"
                type_counts[etype] = type_counts.get(etype, 0) + 1

        dominant = max(type_counts, key=type_counts.get) if type_counts else "unknown"

        # Top 2 entities by claim count
        ranked = sorted(
            (entity_map[eid] for eid in cluster if eid in entity_map),
            key=lambda e: -e.claim_count,
        )
        names = [e.name or e.id for e in ranked[:2]]
        if names:
            return f"{dominant} ({', '.join(names)})"
        return dominant

    def _gather_clustered_evidence(
        self,
        clusters: list[list[str]],
        entity_map: dict[str, "EntitySummary"],
        max_rels: int = DEFAULT_MAX_RELATIONSHIPS,
    ) -> str:
        """Build evidence organized by topic cluster.

        Each cluster gets a ``# Topic: {label}`` section header.
        Budget is allocated proportionally (at least 60 rels per cluster).
        Capped at 5 topic sections.
        """
        clusters = clusters[:5]
        total_entities = sum(len(c) for c in clusters)

        sections: list[str] = []
        for cluster in clusters:
            label = self._label_cluster(cluster, entity_map)
            # Proportional budget, floor of 60
            budget = max(ENTITY_BUDGET_FLOOR, int(max_rels * len(cluster) / max(total_entities, 1)))
            evidence = self._gather_evidence(cluster, max_rels=budget)
            if evidence.strip():
                sections.append(f"# Topic: {label}\n{evidence}")

        return "\n\n".join(sections)

    def ask(self, question: str, top_k: int = 10) -> AskResult:
        """Answer a natural-language question using the knowledge graph.

        Two lightweight LLM passes:
        1. Select relevant entity types from a compact catalog
        2. Gather deep evidence for matching entities and synthesize

        Returns AskResult with structured citations, contradictions, and gaps.
        Dict-compatible for backward compat (r["answer"] works).
        """
        all_entities = self.list_entities()
        if not all_entities:
            return AskResult(meta={"n_searched": 0, "n_search_hits": 0, "selected_types": []})

        # Group entities by type
        by_type: dict[str, list] = {}
        for e in all_entities:
            by_type.setdefault(e.entity_type, []).append(e)
        for t in by_type:
            by_type[t].sort(key=lambda e: -e.claim_count)

        # Step 1: Search for entities whose names match the question
        _stop = frozenset({
            "what", "who", "how", "why", "when", "where", "which", "does",
            "the", "are", "for", "and", "that", "this", "with", "from",
            "about", "has", "have", "been", "is", "was", "were", "tell",
            "me", "show", "find", "get", "list", "of", "in", "on", "at",
            "by", "an", "a", "do", "did", "our", "we", "us", "made",
            "being", "last", "recent", "done", "any", "all", "some",
            "much", "many", "can", "could", "would", "should", "will",
            "to", "it", "its", "be", "not", "no", "or", "but", "if",
            "than", "them", "they", "their", "there", "then", "top",
            "best", "most", "also", "just", "only", "very", "more",
        })
        words = [w.strip("?.,!\"'()[]{}:;") for w in question.lower().split()]
        query_terms = " ".join(w for w in words if w and len(w) >= 2 and w not in _stop)

        search_hits: list = []
        if query_terms:
            # Get candidates from text index, then rank by claim count
            # so knowledge-rich entities surface above name-only matches
            search_hits = self.search_entities(query_terms, top_k=30)

            # Also search individual content words and progressively
            # shorter prefixes — the text index does exact token matching,
            # so "immunotoxicity" won't match "immunotox". Trying prefixes
            # from len-2 down to 5 chars catches stem variations.
            hit_ids = {e.id for e in search_hits}
            content_words = [w for w in query_terms.split() if len(w) >= 5]
            for word in content_words:
                for hit in self.search_entities(word, top_k=10):
                    if hit.id not in hit_ids:
                        search_hits.append(hit)
                        hit_ids.add(hit.id)
                # Try progressively shorter prefixes
                if len(word) >= 7:
                    for plen in range(len(word) - 2, 4, -1):
                        prefix = word[:plen]
                        before = len(hit_ids)
                        for hit in self.search_entities(prefix, top_k=10):
                            if hit.id not in hit_ids:
                                search_hits.append(hit)
                                hit_ids.add(hit.id)
                        if len(hit_ids) > before:
                            break  # found new hits at this prefix length

            search_hits.sort(key=lambda e: -e.claim_count)
            search_hits = search_hits[:10]

        # Step 2: Ask LLM which entity types are relevant to the question
        # Include types with >= 3 entities (singletons/pairs are noise)
        min_type_size = 3
        type_catalog = ", ".join(
            f"{t} ({len(ents)}: "
            f"{', '.join((e.name or e.id) for e in ents[:2])})"
            for t, ents in sorted(by_type.items(), key=lambda x: -len(x[1]))
            if len(ents) >= min_type_size
        )
        type_prompt = (
            f"Entity types in a knowledge graph: {type_catalog}\n\n"
            f"Question: {question}\n\n"
            "Which 5-8 entity types contain the most interesting "
            "domain-specific knowledge for answering this question? "
            "Avoid generic types like process, concept, information, "
            "activity, task, or change. "
            "Return ONLY type names, comma-separated."
        )
        type_response = self._llm_call(type_prompt, max_tokens=100, temperature=0.0)

        selected_types: set[str] = set()
        if type_response:
            for part in type_response.split(","):
                t = part.strip().lower()
                if t in by_type:
                    selected_types.add(t)

        # Step 3: Build entity set — search hits + top entities from selected types
        seen_ids: set[str] = set()
        selected: list = []

        for e in search_hits:
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                selected.append(e)

        for t in selected_types:
            for e in by_type[t][:3]:
                if e.id not in seen_ids and len(selected) < 20:
                    seen_ids.add(e.id)
                    selected.append(e)

        # Step 4: Cluster candidates by graph connectivity
        selected_ids = [e.id for e in selected[:20]]
        entity_map = {e.id: e for e in selected[:20]}
        clusters = self._cluster_entities(selected_ids)

        search_hit_ids = {e.id for e in search_hits}
        multi_topic = len(clusters) > 1

        if multi_topic and search_hit_ids:
            # Focused question — keep only cluster(s) containing search hits
            relevant = [c for c in clusters if search_hit_ids & set(c)]
            if relevant:
                clusters = relevant
                # Narrow selected entities to relevant clusters only
                kept = set()
                for c in clusters:
                    kept.update(c)
                selected = [e for e in selected if e.id in kept]
                selected_ids = [e.id for e in selected[:20]]
                multi_topic = len(clusters) > 1

        # Step 5: Gather evidence (clustered or flat)
        if multi_topic:
            evidence = self._gather_clustered_evidence(
                clusters, entity_map, max_rels=DEFAULT_MAX_RELATIONSHIPS,
            )
        else:
            evidence = self._gather_evidence(selected_ids, max_rels=DEFAULT_MAX_RELATIONSHIPS)

        # Step 6: LLM synthesis
        base_instructions = (
            "- Trace multi-hop reasoning chains explicitly (A → B → C) when they "
            "answer the question.\n"
            "- Distinguish well-supported facts (high confidence, multiple sources) "
            "from weaker inferences.\n"
            "- Note contradictions when present — don't hide conflicting evidence.\n"
            "- Items tagged with ⚠ WARNING, BUG, VULNERABILITY, or PATTERN are "
            "knowledge annotations — surface these prominently as they represent "
            "critical operational learnings.\n"
            "- Quote evidence text when available and relevant.\n"
            "- Answer in 3-8 sentences using specific names, facts, and "
            "relationships from the evidence. Synthesize directly from the "
            "evidence — do not hedge or say information is unavailable if the "
            "evidence contains relevant facts."
        )
        if multi_topic:
            base_instructions += (
                "\n- The evidence is organized by topic area. Address each topic "
                "area separately.\n"
                "- Do not mix evidence from different topic areas in the same "
                "paragraph.\n"
                "- Use the topic labels as section headers in your answer."
            )
        prompt = (
            "You are answering questions about an organization's knowledge "
            "graph. Below is detailed evidence — entities, their relationships, "
            "multi-hop chains (marked [2-hop]), confidence scores, source counts, "
            "and evidence quotes from real conversations and documents.\n\n"
            f"{evidence}\n\n"
            "---\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n{base_instructions}"
        )
        answer = self._llm_call(prompt, max_tokens=1024)

        # Build cluster diagnostics
        cluster_labels = [
            self._label_cluster(c, entity_map) for c in clusters
        ] if multi_topic else []

        # Sort entities by claim count so richest show first in UI
        selected.sort(key=lambda e: -e.claim_count)

        # Collect structured citations, contradictions, and gaps from entities
        citations: list[Citation] = []
        all_contradictions: list[dict] = []
        gaps: list[str] = []
        seen_claim_ids: set[str] = set()
        for e in selected[:top_k]:
            try:
                frame = self.query(e.id, depth=1)
            except Exception:
                continue
            # Citations from direct relationships
            for rel in frame.direct_relationships:
                for claim in self.claims_for(e.id, predicate_type=rel.predicate):
                    if claim.claim_id in seen_claim_ids:
                        continue
                    seen_claim_ids.add(claim.claim_id)
                    citations.append(Citation(
                        claim_id=claim.claim_id,
                        subject=claim.subject.id,
                        predicate=claim.predicate.id,
                        object=claim.object.id,
                        confidence=claim.confidence,
                        source_id=claim.provenance.source_id,
                        source_type=claim.provenance.source_type,
                    ))
                    if len(citations) >= 50:
                        break
                if len(citations) >= 50:
                    break
            # Contradictions
            for c in frame.contradictions:
                all_contradictions.append({
                    "claim_a": c.claim_a, "claim_b": c.claim_b,
                    "description": c.description, "status": c.status,
                })
            # Gaps
            for g in frame.knowledge_gaps:
                if g not in gaps:
                    gaps.append(g)

        return AskResult(
            answer=answer,
            citations=citations,
            contradictions=all_contradictions,
            gaps=gaps,
            entities=selected[:top_k],
            evidence=evidence,
            meta={
                "n_searched": len(selected),
                "n_search_hits": len(search_hits),
                "selected_types": sorted(selected_types),
                "n_clusters": len(clusters),
                "cluster_sizes": [len(c) for c in clusters],
                "cluster_labels": cluster_labels,
            },
        )

    def _gather_evidence(self, entity_ids: list[str], max_rels: int = 60) -> str:
        """Build rich evidence from relationships for a set of entities.

        Each relationship is annotated with confidence and source count.
        Evidence text from payloads is included when available.
        Contradictions are surfaced from ContextFrame.
        Hop-2 queries are issued only for budgeted relationships (not all),
        keeping latency bounded regardless of entity degree.
        Adaptive budget: entities with fewer relationships yield unused slots
        to knowledge-rich entities.
        """
        # First pass: collect frames and rels per entity (depth=1 — fast)
        entity_data: list[tuple] = []  # (eid, entity, rels_sorted, contradictions)
        for eid in entity_ids:
            try:
                frame = self.query(eid, depth=1)
            except Exception:
                continue
            entity = frame.focal_entity
            direct = sorted(frame.direct_relationships, key=lambda r: -r.confidence)
            entity_data.append((eid, entity, direct, frame.contradictions))

        if not entity_data:
            return ""

        # Adaptive budget: start with equal share, reallocate unused
        n = len(entity_data)
        per_entity = max(max_rels // n, 3)
        available = [len(rels) for _, _, rels, _ in entity_data]
        allocated = [min(per_entity, a) for a in available]
        surplus = max_rels - sum(allocated)

        if surplus > 0:
            for i in range(n):
                extra = min(surplus, available[i] - allocated[i])
                if extra > 0:
                    allocated[i] += extra
                    surplus -= extra
                if surplus <= 0:
                    break

        # Second pass: build hop-2 map only for budgeted relationships.
        # Cap at 5 hop-2 queries per entity to keep latency bounded.
        _HOP2_PER_ENTITY = 5
        hop2_maps: list[dict[str, list]] = []
        for idx, (eid, entity, rels, _) in enumerate(entity_data):
            hop2_map: dict[str, list] = {}
            hop2_count = 0
            for rel in rels[:allocated[idx]]:
                if hop2_count >= _HOP2_PER_ENTITY:
                    break
                tid = rel.target.id
                if tid in hop2_map or tid == eid:
                    continue
                try:
                    hop2_frame = self.query(tid, depth=1)
                except Exception:
                    continue
                hop2_rels = [
                    r for r in hop2_frame.direct_relationships
                    if r.target.id != eid
                ]
                if hop2_rels:
                    hop2_map[tid] = sorted(hop2_rels, key=lambda r: -r.confidence)[:3]
                    hop2_count += 1
            hop2_maps.append(hop2_map)

        lines: list[str] = []
        seen: set[str] = set()
        for idx, (eid, entity, rels, contradictions) in enumerate(entity_data):
            ename = entity.name or entity.id
            hop2_map = hop2_maps[idx]
            lines.append(f"\n## {ename} ({entity.entity_type})")

            for rel in rels[:allocated[idx]]:
                target = rel.target.name or rel.target.id
                triple = f"{ename} → {rel.predicate} → {target}"
                if triple in seen:
                    continue
                seen.add(triple)
                src_info = f", from: {', '.join(rel.source_types)}" if rel.source_types else ""
                n_src = rel.n_independent_sources
                s_sfx = "s" if n_src != 1 else ""
                # Tag knowledge claims so the LLM treats them appropriately
                from attestdb.core.vocabulary import KNOWLEDGE_PRIORITY, knowledge_label
                tag = ""
                if rel.predicate in KNOWLEDGE_PRIORITY:
                    tag = f" ⚠ {knowledge_label(rel.predicate).upper()}"
                ann = (
                    f"[conf={rel.confidence:.2f}, "
                    f"{n_src} source{s_sfx}{src_info}]{tag}"
                )
                lines.append(f"- {triple} {ann}")

                # Evidence text from payload
                if rel.payload and isinstance(rel.payload, dict):
                    ev = rel.payload.get("evidence_text", "")
                    if ev:
                        lines.append(f'    Evidence: "{ev}"')

                # 2-hop extensions
                tid = rel.target.id
                if tid in hop2_map:
                    for hop2 in hop2_map[tid]:
                        h2_target = hop2.target.name or hop2.target.id
                        h2_triple = f"{target} → {hop2.predicate} → {h2_target}"
                        if h2_triple not in seen:
                            seen.add(h2_triple)
                            h2_n = hop2.n_independent_sources
                            h2_s = "s" if h2_n != 1 else ""
                            h2_ann = (
                                f"[conf={hop2.confidence:.2f}, "
                                f"{h2_n} source{h2_s}]"
                            )
                            lines.append(f"  [2-hop] {h2_triple} {h2_ann}")

            # Surface contradictions
            if contradictions:
                c_sfx = "s" if len(contradictions) != 1 else ""
                lines.append(
                    f"  ⚠ {len(contradictions)} "
                    f"contradiction{c_sfx} detected"
                )
                for c in contradictions[:3]:
                    if c.description:
                        lines.append(f"    - {c.description}")

        return "\n".join(lines)

    # --- Claim operations ---

    def claims_for(
        self,
        entity_id: str,
        predicate_type: str | None = None,
        source_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[Claim]:
        return [
            claim_from_dict(d)
            for d in self._store.claims_for(entity_id, predicate_type, source_type, min_confidence)
        ]

    def claims_by_content_id(self, content_id: str) -> list[Claim]:
        return [claim_from_dict(d) for d in self._store.claims_by_content_id(content_id)]

    def claims_by_source_id(self, source_id: str) -> list[Claim]:
        """Return all claims with the given source_id via Rust index (O(k))."""
        return [claim_from_dict(d) for d in self._store.claims_by_source_id(source_id)]

    def corroboration_report(self, min_sources: int = 2) -> dict:
        """Report on corroboration status across the knowledge graph.

        Returns corroborated claims (grouped by content_id) and single-source
        claims that need independent confirmation.
        """
        from attestdb.core.confidence import count_independent_sources, corroboration_boost

        content_id_claims: dict[str, list] = {}

        for claim in self.iter_claims():
            cid = claim.content_id
            if cid not in content_id_claims:
                content_id_claims[cid] = []
            content_id_claims[cid].append(claim)

        corroborated = []
        single_source = []

        for cid, claims in content_id_claims.items():
            n_indep = count_independent_sources(claims)
            if n_indep >= min_sources:
                boost = corroboration_boost(n_indep)
                corroborated.append({
                    "content_id": cid,
                    "subject": claims[0].subject.id,
                    "predicate": claims[0].predicate.id,
                    "object": claims[0].object.id,
                    "n_claims": len(claims),
                    "n_independent_sources": n_indep,
                    "confidence_boost": round(boost, 3),
                    "source_types": list({c.provenance.source_type for c in claims}),
                })
            elif len(claims) == 1:
                single_source.append({
                    "content_id": cid,
                    "subject": claims[0].subject.id,
                    "predicate": claims[0].predicate.id,
                    "object": claims[0].object.id,
                    "source_type": claims[0].provenance.source_type,
                    "confidence": claims[0].confidence,
                })

        corroborated.sort(key=lambda x: x["n_independent_sources"], reverse=True)
        single_source.sort(key=lambda x: x["confidence"])

        total = len(content_id_claims)
        return {
            "total_content_ids": total,
            "corroborated_count": len(corroborated),
            "single_source_count": len(single_source),
            "corroboration_ratio": len(corroborated) / total if total else 0,
            "corroborated": corroborated,
            "needs_corroboration": single_source[:50],
        }

    def claims_for_predicate(self, predicate_id: str) -> list[Claim]:
        """Return all claims with the given predicate id via Rust index (O(k))."""
        return [claim_from_dict(d) for d in self._store.claims_by_predicate_id(predicate_id)]

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
        result = RetractResult(source_id, reason, len(claim_ids), claim_ids)
        self._audit_write(
            "source_retracted",
            source_id=source_id,
            reason=reason,
            retracted_count=len(claim_ids),
        )
        self._fire("source_retracted", source_id=source_id, reason=reason, claim_ids=claim_ids)
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
        self._fire("inquiry_created", claim_id=claim_id, question=question)
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

    def quality_report(
        self,
        stale_threshold: int = 0,
        expected_patterns: dict[str, set[str]] | None = None,
    ) -> QualityReport:
        """Generate a quality report over the entire knowledge graph.

        Uses Rust-native analytics indexes where available:
        - stats() for entity type counts and totals
        - predicate_counts() for predicate distribution
        - entity_source_counts() for single-source detection
        - Single-pass iter_claims for source_type distribution and staleness
        """
        report = QualityReport()
        rust_stats = self._store.stats()
        report.total_entities = rust_stats.get("entity_count", 0)
        report.total_claims = rust_stats.get("total_claims", 0)

        # Entity type counts from Rust (O(1) — pre-indexed)
        report.entity_type_counts = {
            k: v for k, v in rust_stats.get("entity_types", {}).items()
        }

        # Predicate distribution from Rust (O(1) — counter table)
        pred_counts = self._store.predicate_counts()
        report.predicate_distribution = {k: int(v) for k, v in pred_counts.items()}

        # Source type distribution + stale entity detection via single claim pass.
        # Single-source entity detection via Rust entity_source_counts().
        source_type_dist: dict[str, int] = {}
        # Use source_types from stats for distribution (count per source_type)
        source_types_from_stats = rust_stats.get("source_types", {})
        if source_types_from_stats:
            source_type_dist = {k: v for k, v in source_types_from_stats.items()}

        # Single-source detection: use entity_source_counts per entity
        # which returns (source_id, count, avg_conf) without materializing claims
        single_source_count = 0
        stale_count = 0
        entities = self.list_entities()

        for entity in entities:
            src_counts = self._store.entity_source_counts(entity.id)
            if len(src_counts) == 1:
                single_source_count += 1

        report.single_source_entity_count = single_source_count
        report.source_type_distribution = source_type_dist

        # Stale detection requires claim timestamps — only scan if requested
        if stale_threshold > 0:
            entity_latest: dict[str, int] = {}
            for claim in self.iter_claims():
                for eid in (claim.subject.id, claim.object.id):
                    if claim.timestamp > entity_latest.get(eid, 0):
                        entity_latest[eid] = claim.timestamp
            stale_count = sum(
                1 for ts in entity_latest.values() if ts < stale_threshold
            )
        report.stale_entity_count = stale_count

        if report.total_entities > 0:
            report.avg_claims_per_entity = report.total_claims / report.total_entities

        # Gap and confidence alert counts via insight engine
        if expected_patterns:
            gaps = self.find_gaps(expected_patterns, min_claims=1)
            report.gap_count = len(gaps)

        alerts = self.find_confidence_alerts(min_claims=2)
        report.confidence_alert_count = len(alerts)

        return report

    # --- Confidence calibration ---

    def calibration_report(
        self,
        labeled_claims: list[tuple[dict, bool]],
        n_bins: int = 5,
    ) -> dict:
        """Compare predicted confidence scores against ground truth.

        See :meth:`EvalHarness.calibration_report` for full documentation.
        """
        from attestdb.intelligence.eval_harness import EvalHarness
        harness = EvalHarness(self)
        return harness.calibration_report(labeled_claims, n_bins=n_bins)

    # --- Knowledge health ---

    def knowledge_health(self) -> KnowledgeHealth:
        """Compute quantified health metrics for the knowledge graph.

        Uses two efficient passes:
        1. Rust entity_source_counts() for multi-source detection (no claim materialization)
        2. Single iter_claims pass for confidence/timestamp/corroboration stats
        """
        h = KnowledgeHealth()
        rust_stats = self._store.stats()
        h.total_entities = rust_stats.get("entity_count", 0)
        if h.total_entities == 0:
            return h

        # Multi-source detection via Rust index (no claim materialization)
        multi_source_count = 0
        entities = self.list_entities()
        for entity in entities:
            src_counts = self._store.entity_source_counts(entity.id)
            if len(src_counts) > 1:
                multi_source_count += 1

        # Single pass over claims for confidence/timestamp/content_id stats
        content_id_counts: dict[str, int] = {}
        source_types: set[str] = set()
        all_confidences: list[float] = []
        all_timestamps: list[int] = []
        chain_lengths: list[int] = []

        # Session metadata predicates that don't represent real knowledge
        _session_predicates = {
            "awaiting_outcome", "had_outcome", "used_tool",
            "session_metadata", "produced_by", "confidence_updated",
        }

        for claim in self.iter_claims():
            # Skip session bookkeeping claims from health calculations
            if (
                claim.provenance.source_type in ("session_end", "outcome_report")
                or claim.predicate.id in _session_predicates
            ):
                continue

            all_confidences.append(claim.confidence)
            all_timestamps.append(claim.timestamp)
            source_types.add(claim.provenance.source_type)
            chain_lengths.append(len(claim.provenance.chain))
            content_id_counts[claim.content_id] = (
                content_id_counts.get(claim.content_id, 0) + 1
            )

        h.total_claims = len(all_confidences)
        if h.total_claims == 0:
            return h

        # Basic metrics
        h.avg_confidence = sum(all_confidences) / len(all_confidences)
        h.multi_source_ratio = multi_source_count / h.total_entities
        h.source_diversity = len(source_types)
        h.knowledge_density = h.total_claims / h.total_entities
        h.avg_provenance_depth = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0.0

        # Corroboration ratio: fraction of content_ids with >1 claim
        if content_id_counts:
            corroborated = sum(1 for c in content_id_counts.values() if c > 1)
            h.corroboration_ratio = corroborated / len(content_id_counts)

        # Freshness: exponential decay based on age of claims (half-life = 30 days)
        # Timestamps are in nanoseconds (see ingestion.py)
        now_ns = int(time.time() * 1_000_000_000)
        if all_timestamps and max(all_timestamps) > 0:
            half_life_ns = FRESHNESS_HALF_LIFE_DAYS * 86400 * 1_000_000_000
            freshness_sum = 0.0
            for ts in all_timestamps:
                age_ns = max(0, now_ns - ts)
                freshness_sum += 2.0 ** (-age_ns / half_life_ns)
            h.freshness_score = freshness_sum / len(all_timestamps)

        # Confidence trend: compare avg confidence of recent half vs older half
        if len(all_confidences) >= 2:
            paired = sorted(zip(all_timestamps, all_confidences))
            mid = len(paired) // 2
            old_avg = sum(c for _, c in paired[:mid]) / mid
            new_avg = sum(c for _, c in paired[mid:]) / (len(paired) - mid)
            h.confidence_trend = new_avg - old_avg

        # Composite health score (0-100)
        h.health_score = min(100.0, max(0.0, (
            h.multi_source_ratio * 30.0
            + h.corroboration_ratio * 25.0
            + h.freshness_score * 20.0
            + min(1.0, h.source_diversity / 5.0) * 15.0
            + max(0.0, min(1.0, 0.5 + h.confidence_trend)) * 10.0
        )))

        return h

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

    def configure_curator(
        self,
        model: str = "heuristic",
        api_key: str | None = None,
        env_path: str | None = None,
    ) -> None:
        """Configure the curator model. Resets cached intelligence instances."""
        self._curator_model = model
        self._curator_api_key = api_key
        self._curator_env_path = env_path
        self._curator = None
        self._text_extractor = None
        self._researcher = None

    def set_domain_context(self, context: str) -> None:
        """Set domain context for LLM extraction prioritization."""
        self._domain_context = context
        self._text_extractor = None  # force rebuild

    def _get_curator(self):
        if self._curator is None:
            from attestdb.intelligence.curator import CuratorV1
            self._curator = CuratorV1(
                self, model=self._curator_model,
                api_key=self._curator_api_key, env_path=self._curator_env_path,
            )
        return self._curator

    def _get_text_extractor(self):
        if self._text_extractor is None:
            from attestdb.core.vocabulary import BUILT_IN_ENTITY_TYPES, BUILT_IN_PREDICATE_TYPES
            from attestdb.intelligence.text_extractor import TextExtractor

            # Merge built-in + registered entity types and predicates
            entity_types = set(BUILT_IN_ENTITY_TYPES)
            predicates = set(BUILT_IN_PREDICATE_TYPES)
            try:
                for vocab in self._store.get_registered_vocabularies().values():
                    entity_types.update(vocab.get("entity_types", []))
                    predicates.update(vocab.get("predicate_types", []))
            except Exception:
                pass

            # Get predicate constraints
            predicate_constraints = {}
            try:
                predicate_constraints = self._store.get_predicate_constraints()
            except Exception:
                pass

            # Use "auto" fallback chain unless a specific LLM provider was configured.
            extractor_model = self._curator_model if self._curator_model != "heuristic" else "auto"
            self._text_extractor = TextExtractor(
                model=extractor_model,
                api_key=self._curator_api_key,
                env_path=self._curator_env_path,
                entity_types=list(entity_types),
                predicates=list(predicates),
                predicate_constraints=predicate_constraints,
                discovery_mode=True,
                domain_context=self._domain_context,
            )
        return self._text_extractor

    def _get_insight_engine(self):
        if self._insight_engine is None:
            from attestdb.intelligence.insight_engine import InsightEngineV1
            self._insight_engine = InsightEngineV1(self)
        return self._insight_engine

    def _get_temporal_engine(self):
        if self._temporal_engine is None:
            from attestdb.intelligence.temporal import TemporalEngine
            self._temporal_engine = TemporalEngine(self)
        return self._temporal_engine

    def _get_researcher(self):
        if self._researcher is None:
            try:
                from attestdb_enterprise.researcher import Researcher
            except ImportError:
                raise ImportError(
                    "Researcher requires attestdb-enterprise: "
                    "pip install attestdb-enterprise"
                )
            self._researcher = Researcher(
                self, model=self._curator_model,
                api_key=self._curator_api_key, env_path=self._curator_env_path,
            )
        return self._researcher

    def curate(self, claims: list[ClaimInput], agent_id: str = "default"):
        """Triage and ingest claims through the curator."""
        return self._get_curator().process_agent_output(agent_id, claims)

    def ingest_text(self, text: str, source_id: str = "", use_curator: bool = True):
        """Extract claims from text and ingest. Optional curator triage."""
        curator = self._get_curator() if use_curator else None
        return self._get_text_extractor().extract_and_ingest(
            text, self, source_id=source_id, curator=curator,
        )

    def ingest_texts(
        self,
        texts: list[dict],
        use_curator: bool = True,
    ) -> dict:
        """Batch wrapper around ingest_text().

        Args:
            texts: List of dicts with keys: text, source_id, and optionally source_type.
            use_curator: Whether to triage claims through the curator.

        Returns:
            Summary dict with total_extracted, total_stored, total_skipped, and per-text results.
        """
        total_extracted = 0
        total_stored = 0
        total_skipped = 0
        results = []

        for item in texts:
            text = item["text"]
            source_id = item.get("source_id", "")
            result = self.ingest_text(text, source_id=source_id, use_curator=use_curator)
            n_extracted = result.raw_count
            n_stored = result.n_valid
            n_skipped = n_extracted - n_stored
            total_extracted += n_extracted
            total_stored += n_stored
            total_skipped += n_skipped
            results.append({
                "source_id": source_id,
                "extracted": n_extracted,
                "stored": n_stored,
                "skipped": n_skipped,
            })

        return {
            "total_extracted": total_extracted,
            "total_stored": total_stored,
            "total_skipped": total_skipped,
            "results": results,
        }

    def _get_extractor(self, mode: str = "llm"):
        """Get the appropriate extractor for the given mode.

        Modes:
            "llm" — Full LLM extraction (requires API key).
            "heuristic" — Pattern-based, no API key needed.
            "smart" — Heuristic pre-scan → novelty check → LLM only for novel.
        """
        if mode in ("heuristic", "smart"):
            from attestdb.intelligence.heuristic_extractor import HeuristicExtractor
            constraints = {}
            try:
                constraints = self._store.get_predicate_constraints()
            except Exception:
                pass
            # Build known entity dictionary from existing entities
            known = {}
            try:
                for es in self.list_entities():
                    known[es.id] = es.entity_type
            except Exception:
                pass
            heuristic = HeuristicExtractor(
                predicate_constraints=constraints,
                known_entities=known if known else None,
            )
            if mode == "smart":
                from attestdb.intelligence.smart_extractor import SmartExtractor
                return SmartExtractor(self._get_text_extractor(), heuristic, self)
            return heuristic
        return self._get_text_extractor()

    def ingest_chat(
        self,
        messages: list[dict],
        conversation_id: str = "",
        platform: str = "generic",
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a chat conversation.

        Args:
            messages: OpenAI-format messages [{role, content}, ...].
            conversation_id: Optional ID for the conversation.
            platform: Platform hint ("generic", "chatgpt", "claude").
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            ChatIngestionResult with per-turn breakdown.
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self, curator)
        return ingestor.ingest_messages(messages, conversation_id, platform)

    def ingest_chat_file(
        self,
        path: str,
        platform: str = "auto",
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a chat log file (.zip, .json, .txt, .md).

        Args:
            path: Path to the chat log file.
            platform: Format hint ("auto", "chatgpt", "openai", "generic").
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            List of ChatIngestionResult (one per conversation in file).
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self, curator)
        return ingestor.ingest_file(path, platform)

    def connect(self, name: str, *, save: bool = False, **kwargs) -> "Connector":
        """Create a connector for an external data source.

        Args:
            name: Connector name (e.g. "slack", "postgres", "notion").
            save: Persist credentials to the encrypted token store.
            **kwargs: Passed to the connector constructor.

        Returns:
            A :class:`~attestdb.connectors.base.Connector` instance.
            Call ``.run(db)`` to execute.

        Example::

            conn = db.connect("slack", token="xoxb-...")
            result = conn.run(db)
        """
        from attestdb.connectors import connect as _connect

        conn = _connect(name, db_path=self._db_path, **kwargs)
        if save and kwargs:
            from attestdb.connectors.token_store import TokenStore

            ts = TokenStore(self._db_path)
            ts.save_token(name, dict(kwargs))
        return conn

    def _get_scheduler(self):
        """Lazy-init the connector scheduler."""
        if self._scheduler is None:
            from attestdb.connectors.scheduler import ConnectorScheduler
            self._scheduler = ConnectorScheduler()
        return self._scheduler

    def _post_sync_check(self, connector_name: str, claims_ingested: int, **kwargs):
        """Lightweight insight check after each sync cycle. Fires insight events."""
        if claims_ingested == 0:
            return
        try:
            engine = self._get_insight_engine()
            alerts = engine.find_confidence_alerts(min_claims=3)
            if alerts:
                self._fire(
                    "insight_alerts", alerts=alerts,
                    trigger="sync", connector=connector_name,
                )
        except Exception as e:
            logger.warning("Post-sync insight check failed: %s", e)

    def sync(
        self,
        name: str,
        interval: float = 300.0,
        *,
        save: bool = False,
        run_immediately: bool = True,
        jitter: float = 0.1,
        **connector_kwargs,
    ):
        """Start continuous sync for a connector.

        Creates the connector, schedules it for periodic execution in a
        background thread, and returns a SyncHandle for monitoring.

        Args:
            name: Connector name (e.g. "slack", "github", "jira").
            interval: Seconds between runs (default: 300 = 5 min).
            save: Persist connector credentials to encrypted token store.
            run_immediately: Run once before waiting for the first interval.
            jitter: Random jitter fraction to prevent thundering herd.
            **connector_kwargs: Passed to the connector constructor.

        Returns:
            SyncHandle with status, last_run, total_claims, etc.

        Example::

            db.sync("slack", interval=300, token="xoxb-...")
            db.sync("github", interval=600, token="ghp_...")
            print(db.sync_status())
        """
        # Register post-sync insight hook once
        if not getattr(self, '_sync_hook_registered', False):
            self.on("sync_completed", self._post_sync_check)
            self._sync_hook_registered = True

        conn = self.connect(name, save=save, **connector_kwargs)
        scheduler = self._get_scheduler()
        return scheduler.schedule(
            conn, self, interval=interval, jitter=jitter,
            run_immediately=run_immediately,
        )

    def sync_status(self) -> list[dict]:
        """Status of all scheduled connectors.

        Returns:
            List of dicts with name, interval, status, last_run, next_run,
            total_runs, total_claims, error_count, last_error.
        """
        if self._scheduler is None:
            return []
        return self._scheduler.status()

    def sync_stop(self, name: str) -> None:
        """Stop a scheduled connector."""
        if self._scheduler:
            self._scheduler.stop(name)

    def sync_stop_all(self) -> None:
        """Stop all scheduled connectors."""
        if self._scheduler:
            self._scheduler.stop_all()

    def sync_pause(self, name: str) -> None:
        """Pause a connector (thread stays alive but skips execution)."""
        if self._scheduler:
            self._scheduler.pause(name)

    def sync_resume(self, name: str) -> None:
        """Resume a paused connector."""
        if self._scheduler:
            self._scheduler.resume(name)

    def sync_run_now(self, name: str) -> None:
        """Trigger an immediate run of a connector."""
        if self._scheduler:
            self._scheduler.run_now(name)

    def ingest_slack(
        self,
        path: str,
        bot_ids: set[str] | None = None,
        channels: list[str] | None = None,
        use_curator: bool = True,
        extraction: str = "llm",
    ):
        """Extract and ingest claims from a Slack workspace export ZIP.

        Args:
            path: Path to the Slack export ZIP file.
            bot_ids: Only treat these bot_ids as "assistant". None = all bots.
            channels: Only process these channel names. None = all.
            use_curator: Whether to triage claims through the curator.
            extraction: "llm", "heuristic", or "smart" (heuristic + LLM for novel).

        Returns:
            List of ChatIngestionResult (one per channel/thread with bot interaction).
        """
        from attestdb.intelligence.chat_ingestor import ChatIngestor
        curator = self._get_curator() if use_curator else None
        extractor = self._get_extractor(extraction)
        ingestor = ChatIngestor(extractor, self, curator)
        return ingestor.ingest_slack_export(path, bot_ids=bot_ids, channels=channels)

    def find_bridges(self, **kwargs) -> list:
        return self._get_insight_engine().find_bridges(**kwargs)

    def find_gaps(self, expected_patterns, **kwargs) -> list:
        return self._get_insight_engine().find_gaps(expected_patterns, **kwargs)

    def find_confidence_alerts(self, **kwargs) -> list:
        return self._get_insight_engine().find_confidence_alerts(**kwargs)

    # --- Temporal analysis ---

    def temporal_analyze(
        self,
        entity_id: str,
        analysis_type: str = "regime_shifts",
        metric: str = "claim_count",
        bucket: str | None = None,
        **kwargs,
    ) -> "TemporalResult":
        """Analyze temporal patterns for an entity's claims.

        Builds a time series from the entity's claims (bucketed by
        day/week/month) and runs the specified analysis.

        Args:
            entity_id: Entity to analyze.
            analysis_type: ``"regime_shifts"``, ``"velocity"``, or ``"cycles"``.
            metric: ``"claim_count"`` (default) or ``"avg_confidence"``.
            bucket: ``"day"``, ``"week"``, ``"month"``, or ``None`` (auto).
            **kwargs: Analysis-specific params (window_size, threshold, etc.).

        Returns:
            :class:`TemporalResult` with detected patterns.

        Example::

            result = db.temporal_analyze("brca1", "regime_shifts")
            print(f"Found {result.num_shifts} regime shifts")

            result = db.temporal_analyze("brca1", "velocity")
            print(f"Max velocity: {result.velocity.max_velocity}")
        """
        return self._get_temporal_engine().analyze(
            entity_id, analysis_type=analysis_type,
            metric=metric, bucket=bucket, **kwargs,
        )

    def temporal_summary(
        self,
        entity_id: str,
        metric: str = "claim_count",
        bucket: str | None = None,
    ) -> "TemporalResult":
        """Run all temporal analyses (shifts, velocity, cycles) at once.

        Args:
            entity_id: Entity to analyze.
            metric: ``"claim_count"`` or ``"avg_confidence"``.
            bucket: ``"day"``, ``"week"``, ``"month"``, or ``None`` (auto).

        Returns:
            :class:`TemporalResult` with all three analyses populated.

        Example::

            result = db.temporal_summary("brca1")
            print(f"Shifts: {result.num_shifts}")
            print(f"Velocity: {result.velocity.mean_velocity}")
            print(f"Dominant cycle: {result.dominant_period}")
        """
        return self._get_temporal_engine().summary(
            entity_id, metric=metric, bucket=bucket,
        )

    # --- Autonomous research ---

    def investigate(
        self,
        max_questions: int = 20,
        use_curator: bool = True,
        search_fn=None,
    ) -> "InvestigationReport":
        """Autonomous gap-closing loop: detect blindspots, research, ingest.

        Args:
            max_questions: Max questions to generate and research.
            use_curator: Whether to triage discovered claims through the curator.
            search_fn: Optional callback(question_text) -> text.

        Returns:
            InvestigationReport with before/after blindspot counts.
        """
        return self._get_researcher().investigate(
            max_questions=max_questions,
            use_curator=use_curator,
            search_fn=search_fn,
        )

    def research_question(
        self,
        question: str,
        entity_id: str | None = None,
        entity_type: str = "",
        predicate_hint: str = "",
    ) -> "ResearchResult":
        """Research a single question and ingest discovered claims.

        Args:
            question: Natural-language research question.
            entity_id: Optional focal entity.
            entity_type: Optional entity type.
            predicate_hint: Optional predicate to hint at.

        Returns:
            ResearchResult with ingestion counts.
        """
        from attestdb.core.types import ResearchQuestion

        rq = ResearchQuestion(
            entity_id=entity_id or "",
            entity_type=entity_type,
            gap_type="manual",
            question=question,
            predicate_hint=predicate_hint,
        )
        # Register as inquiry if entity_id provided
        if entity_id:
            try:
                claim_id = self.ingest_inquiry(
                    question=question,
                    subject=(entity_id, entity_type),
                    object=(entity_id, entity_type),
                    predicate_hint=predicate_hint,
                )
                rq.inquiry_claim_id = claim_id
            except Exception:
                pass

        return self._get_researcher().research(rq)

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

    def impact(self, source_id: str) -> "ImpactReport":
        """Analyze the impact of a source: how many claims depend on it."""
        from attestdb.core.types import ImpactReport

        claims = self.claims_by_source_id(source_id)
        claim_ids = [c.claim_id for c in claims]

        # Find affected entities
        affected = set()
        for c in claims:
            affected.add(c.subject.id)
            affected.add(c.object.id)

        # Count downstream dependents
        downstream_count = 0
        for cid in claim_ids:
            tree = self.trace_downstream(cid)
            downstream_count += len(tree.dependents)

        return ImpactReport(
            source_id=source_id,
            direct_claims=len(claims),
            downstream_claims=downstream_count,
            affected_entities=sorted(affected),
            claim_ids=claim_ids,
        )

    def blindspots(self, min_claims: int = 5) -> "BlindspotMap":
        """Find knowledge blindspots: single-source entities and gaps.

        Uses Rust entity_source_counts() for single-source detection (no claim
        materialization), and a single iter_claims pass for unresolved warnings.
        """
        from attestdb.core.types import BlindspotMap
        from attestdb.core.vocabulary import knowledge_sort_key

        # Single-source detection via bulk Rust scan (one call, no per-entity roundtrip).
        from attestdb.infrastructure.agents import is_autodidact_source

        raw_single = self._store.find_single_source_entities(min_claims)
        single_source = []
        for entity_id in raw_single:
            # Filter out entities whose sole source is autodidact
            src_counts = self._store.entity_source_counts(entity_id)
            sole_source_id = src_counts[0][0] if src_counts else ""
            if is_autodidact_source(sole_source_id):
                continue
            single_source.append(entity_id)

        # Single pass over claims for subject→predicate map (unresolved warnings)
        subject_predicates: dict[str, set[str]] = {}
        subject_claims: dict[str, list] = {}

        for c in self.iter_claims():
            subj = c.subject.id
            subject_predicates.setdefault(subj, set()).add(c.predicate.id)
            subject_claims.setdefault(subj, []).append(c)

        # Compute knowledge gaps from predicate constraints
        knowledge_gaps: list[dict] = []
        try:
            constraints = self._store.get_predicate_constraints()
            if constraints:
                from attestdb.intelligence.insight_engine import build_expected_patterns
                patterns = build_expected_patterns(constraints)
                gap_results = self.find_gaps(patterns, min_claims=min_claims)
                for gap in gap_results:
                    knowledge_gaps.append({
                        "entity": gap.entity_id,
                        "entity_type": gap.entity_type,
                        "missing": gap.missing_predicate_types,
                    })
        except Exception:
            logger.debug("Could not compute knowledge gaps", exc_info=True)

        # Get confidence alerts as low-confidence areas
        low_conf: list[dict] = []
        try:
            alerts = self.find_confidence_alerts(min_claims=2)
            for alert in alerts:
                low_conf.append({
                    "entity": alert.entity_id,
                    "entity_type": alert.entity_type,
                    "alert_type": alert.alert_type.value,
                    "explanation": alert.explanation,
                    "source_id": alert.source_id,
                })
        except Exception:
            logger.debug("Could not compute confidence alerts", exc_info=True)

        # Find entities with bugs/issues but no corresponding fix.
        # Uses subject→predicate map built in the single pass above.
        unresolved: list[dict] = []
        problem_predicates = {"had_bug", "has_issue", "has_vulnerability"}
        resolution_predicates = {"has_fix", "resolved", "has_status"}
        try:
            for subj, preds in subject_predicates.items():
                problems = preds & problem_predicates
                resolutions = preds & resolution_predicates
                if problems and not resolutions:
                    problem_list = [
                        c for c in subject_claims[subj]
                        if c.predicate.id in problems
                    ]
                    problem_list.sort(
                        key=lambda c: knowledge_sort_key(c.predicate.id, c.confidence),
                    )
                    top = problem_list[0]
                    unresolved.append({
                        "entity": subj,
                        "entity_type": top.subject.entity_type,
                        "predicate": top.predicate.id,
                        "description": top.object.id,
                        "confidence": top.confidence,
                    })
        except Exception:
            logger.debug("Could not compute unresolved warnings", exc_info=True)

        return BlindspotMap(
            single_source_entities=single_source,
            knowledge_gaps=knowledge_gaps,
            low_confidence_areas=low_conf,
            unresolved_warnings=unresolved,
        )

    def consensus(self, topic: str) -> "ConsensusReport":
        """Analyze consensus around a topic (entity)."""
        from attestdb.core.types import ConsensusReport

        claims = self.claims_for(topic)
        if not claims:
            return ConsensusReport(topic=topic)

        sources = {}
        content_ids = {}
        total_conf = 0.0

        for c in claims:
            sid = c.provenance.source_id
            sources[sid] = sources.get(sid, 0) + 1
            content_ids[c.content_id] = content_ids.get(c.content_id, 0) + 1
            total_conf += c.confidence

        corroborated = [cid for cid, count in content_ids.items() if count > 1]

        return ConsensusReport(
            topic=topic,
            total_claims=len(claims),
            unique_sources=len(sources),
            avg_confidence=total_conf / len(claims),
            agreement_ratio=len(corroborated) / len(content_ids) if content_ids else 0.0,
            claims_by_source=sources,
            corroborated_content_ids=corroborated,
        )

    def fragile(self, max_sources: int = 1, min_age_days: int = 0) -> list[Claim]:
        """Find fragile claims: backed by few sources, optionally filtered by age."""
        now_ns = int(time.time() * 1_000_000_000)
        min_age_ns = min_age_days * 86400 * 1_000_000_000

        # Single-pass: group source_ids by content_id, then filter
        content_sources: dict[str, set[str]] = {}
        content_claims: dict[str, list] = {}
        for claim in self.iter_claims():
            if claim.status != ClaimStatus.ACTIVE:
                continue
            if min_age_ns > 0 and (now_ns - claim.timestamp) < min_age_ns:
                continue
            cid = claim.content_id
            content_sources.setdefault(cid, set()).add(claim.provenance.source_id)
            content_claims.setdefault(cid, []).append(claim)

        fragile_claims = []
        for cid, sources in content_sources.items():
            if len(sources) <= max_sources:
                fragile_claims.extend(content_claims[cid])

        return fragile_claims

    def stale(self, days: int = 90) -> list[Claim]:
        """Find stale claims: not updated within the given number of days."""
        cutoff_ns = int(time.time() * 1_000_000_000) - (days * 86400 * 1_000_000_000)
        stale_claims = []
        for claim in self.iter_claims():
            if claim.status != ClaimStatus.ACTIVE:
                continue
            if claim.timestamp < cutoff_ns:
                stale_claims.append(claim)

        return stale_claims

    def audit(self, claim_id: str) -> "AuditTrail":
        """Build a full audit trail for a claim."""
        from attestdb.core.types import AuditTrail

        found = self.get_claim(claim_id)
        if found is None:
            return AuditTrail(claim_id=claim_id)

        # Get corroborating claims
        corroborating = self.claims_by_content_id(found.content_id)
        corr_ids = [c.claim_id for c in corroborating if c.claim_id != claim_id]

        # Get downstream dependents
        tree = self.trace_downstream(claim_id)
        downstream_count = len(tree.dependents)

        return AuditTrail(
            claim_id=claim_id,
            content_id=found.content_id,
            corroborating_claims=corr_ids,
            provenance_chain=found.provenance.chain,
            downstream_dependents=downstream_count,
            source_type=found.provenance.source_type,
            source_id=found.provenance.source_id,
            confidence=found.confidence,
            timestamp=found.timestamp,
        )

    def drift(self, days: int = 30) -> "DriftReport":
        """Measure knowledge drift over the given time period."""
        from attestdb.core.types import DriftReport

        now_ns = int(time.time() * 1_000_000_000)
        cutoff_ns = now_ns - (days * 86400 * 1_000_000_000)

        # Include retracted claims so we can count tombstones
        self._set_include_retracted(True)
        try:
            # Single-pass aggregation via iter_claims()
            n_before = 0
            n_after_new = 0
            n_total = 0
            retracted = 0
            conf_sum_before = 0.0
            conf_sum_total = 0.0
            entities_before: set[str] = set()
            entities_all: set[str] = set()
            old_source_types: set[str] = set()
            new_source_types: set[str] = set()

            after_source_types: set[str] = set()
            for c in self.iter_claims():
                n_total += 1
                conf_sum_total += c.confidence
                entities_all.add(c.subject.id)
                entities_all.add(c.object.id)
                if c.timestamp <= cutoff_ns:
                    n_before += 1
                    conf_sum_before += c.confidence
                    entities_before.add(c.subject.id)
                    entities_before.add(c.object.id)
                    old_source_types.add(c.provenance.source_type)
                else:
                    n_after_new += 1
                    after_source_types.add(c.provenance.source_type)
                    if c.status == ClaimStatus.TOMBSTONED:
                        retracted += 1

            new_source_types = after_source_types - old_source_types
        finally:
            self._set_include_retracted(False)

        conf_before = conf_sum_before / n_before if n_before else 0.0
        conf_after = conf_sum_total / n_total if n_total else 0.0

        return DriftReport(
            period_days=days,
            new_claims=n_after_new,
            new_entities=len(entities_all - entities_before),
            retracted_claims=retracted,
            confidence_delta=conf_after - conf_before,
            new_source_types=sorted(new_source_types),
            entity_count_before=len(entities_before),
            entity_count_after=len(entities_all),
            claim_count_before=n_before,
            claim_count_after=n_total,
        )

    def source_reliability(self, source_id: str | None = None) -> dict:
        """Compute per-source reliability metrics (corroboration/retraction rates).

        If source_id is given, returns metrics for that source only.
        Otherwise, returns a dict of {source_id: metrics} for all sources.

        Uses a single pass to group claims by source AND by content_id,
        avoiding N+1 claims_by_content_id lookups.
        """
        # Include retracted claims so we can compute retraction rates
        self._set_include_retracted(True)
        try:
            # Single pass: group by source AND build content_id → source_ids map
            by_source: dict[str, list[Claim]] = {}
            content_id_sources: dict[str, set[str]] = {}
            for c in self.iter_claims():
                by_source.setdefault(c.provenance.source_id, []).append(c)
                content_id_sources.setdefault(c.content_id, set()).add(c.provenance.source_id)
        finally:
            self._set_include_retracted(False)

        def _metrics(claims: list[Claim]) -> dict:
            total = len(claims)
            active = sum(1 for c in claims if c.status == ClaimStatus.ACTIVE)
            retracted = sum(1 for c in claims if c.status == ClaimStatus.TOMBSTONED)
            degraded = sum(1 for c in claims if c.status == ClaimStatus.PROVENANCE_DEGRADED)

            # Corroboration: check if content_id has claims from other sources
            corroborated = 0
            for c in claims:
                sources_for_cid = content_id_sources.get(c.content_id, set())
                if len(sources_for_cid) > 1:
                    corroborated += 1

            return {
                "total_claims": total,
                "active": active,
                "retracted": retracted,
                "degraded": degraded,
                "corroboration_rate": corroborated / total if total else 0.0,
                "retraction_rate": retracted / total if total else 0.0,
            }

        if source_id is not None:
            claims = by_source.get(source_id, [])
            return _metrics(claims)

        return {sid: _metrics(claims) for sid, claims in by_source.items()}

    def hypothetical(self, claim: ClaimInput) -> "HypotheticalReport":
        """What-if analysis: what would happen if this claim were ingested."""
        from attestdb.core.types import HypotheticalReport

        content_id = compute_content_id(
            normalize_entity_id(claim.subject[0]),
            claim.predicate[0],
            normalize_entity_id(claim.object[0]),
        )

        existing = self.claims_by_content_id(content_id)

        # Check if it fills a gap by looking at entity connectivity
        subj_exists = self._store.get_entity(normalize_entity_id(claim.subject[0])) is not None
        obj_exists = self._store.get_entity(normalize_entity_id(claim.object[0])) is not None
        fills_gap = subj_exists and obj_exists and not self.path_exists(
            claim.subject[0], claim.object[0], max_depth=2
        )

        related = set()
        if subj_exists:
            for c in self.claims_for(claim.subject[0]):
                related.add(c.object.id)
        if obj_exists:
            for c in self.claims_for(claim.object[0]):
                related.add(c.subject.id)

        return HypotheticalReport(
            would_corroborate=len(existing) > 0,
            existing_corroborations=len(existing),
            fills_gap=fills_gap,
            content_id=content_id,
            related_entities=sorted(related)[:20],
        )

    # --- Knowledge-Intelligence APIs ---

    def evolution(self, entity_id: str, since: str | int | None = None) -> EvolutionReport:
        """Entity-specific knowledge evolution over time.

        Shows how understanding of an entity has changed: new connections,
        confidence shifts, retractions, source diversification.

        Args:
            entity_id: The entity to analyze.
            since: Cutoff — ISO string, nanosecond int, or None (full history).

        Returns:
            EvolutionReport with trajectory and change details.
        """
        eid = normalize_entity_id(entity_id)

        # Timestamp resolution
        if since is None:
            cutoff = 0
        elif isinstance(since, int):
            cutoff = since
        else:
            from datetime import datetime
            dt = datetime.fromisoformat(str(since))
            cutoff = int(dt.timestamp() * 1_000_000_000)

        # Include retracted claims to track retractions in evolution
        self._set_include_retracted(True)
        try:
            all_claims = self.claims_for(eid)
        finally:
            self._set_include_retracted(False)
        if not all_claims:
            return EvolutionReport(entity_id=eid, since_timestamp=cutoff)

        before = [c for c in all_claims if c.timestamp <= cutoff]
        after = [c for c in all_claims if c.timestamp > cutoff]

        # Neighbors before and after
        def _neighbors(claims: list[Claim]) -> set[str]:
            n = set()
            for c in claims:
                if c.subject.id == eid:
                    n.add(c.object.id)
                else:
                    n.add(c.subject.id)
            return n

        neighbors_before = _neighbors(before)
        neighbors_after = _neighbors(after)
        new_connections = sorted(neighbors_after - neighbors_before)

        # Retracted claims in the after period
        retracted = sum(1 for c in after if c.status == ClaimStatus.TOMBSTONED)

        # Source diversification
        src_before = sorted({c.provenance.source_type for c in before})
        src_after = sorted({c.provenance.source_type for c in all_claims})
        new_src = sorted(set(src_after) - set(src_before))

        # Confidence changes: compare corroboration counts by content_id
        content_before: dict[str, list[Claim]] = {}
        for c in before:
            content_before.setdefault(c.content_id, []).append(c)
        content_after: dict[str, list[Claim]] = {}
        for c in all_claims:
            content_after.setdefault(c.content_id, []).append(c)

        confidence_changes: list[ConfidenceChange] = []
        new_corroborations = 0
        for cid, after_claims in content_after.items():
            before_claims = content_before.get(cid, [])
            if before_claims and len(after_claims) != len(before_claims):
                conf_b = max(c.confidence for c in before_claims)
                conf_a = max(c.confidence for c in after_claims)
                delta = conf_a - conf_b
                # Determine target entity from the first claim
                fc = after_claims[0]
                target = fc.object.id if fc.subject.id == eid else fc.subject.id
                reason = (
                    "new_corroboration"
                    if len(after_claims) > len(before_claims)
                    else "source_retracted"
                )
                confidence_changes.append(ConfidenceChange(
                    content_id=cid,
                    predicate=fc.predicate.id,
                    target_entity=target,
                    confidence_before=conf_b,
                    confidence_after=conf_a,
                    delta=delta,
                    reason=reason,
                ))
            if len(before_claims) == 1 and len(after_claims) > 1:
                new_corroborations += 1

        # Trajectory
        new_claim_count = len(after)
        if new_claim_count > retracted and (new_connections or new_corroborations > 0):
            trajectory = "growing"
        elif retracted > new_claim_count:
            trajectory = "declining"
        else:
            trajectory = "stable"

        return EvolutionReport(
            entity_id=eid,
            since_timestamp=cutoff,
            new_connections=new_connections,
            new_claims=new_claim_count,
            retracted_claims=retracted,
            confidence_changes=confidence_changes,
            new_corroborations=new_corroborations,
            source_types_before=src_before,
            source_types_after=src_after,
            new_source_types=new_src,
            trajectory=trajectory,
            total_claims_before=len(before),
            total_claims_after=len(all_claims),
        )

    def trace(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
        top_k: int = 5,
    ) -> list[ReasoningChain]:
        """Reasoning chains with source-overlap-discounted confidence.

        Unlike find_paths(), accounts for source overlap (shared sources
        reduce effective confidence), includes evidence quotes from payloads,
        flags contradictions per hop, computes a reliability score.

        Args:
            entity_a: Start entity.
            entity_b: End entity.
            max_depth: Maximum path length.
            top_k: Maximum chains to return.

        Returns:
            List of ReasoningChain sorted by chain_confidence desc.
        """
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        paths = self.find_paths(entity_a, entity_b, max_depth=max_depth, top_k=top_k)
        if not paths:
            return []

        b_norm = normalize_entity_id(entity_b)
        chains: list[ReasoningChain] = []

        for path in paths:
            hops: list[ReasoningHop] = []
            # PathStep.entity_id is the SOURCE of each hop.
            # Destination = next step's entity_id, or entity_b for last step.
            for i, step in enumerate(path.steps):
                from_eid = step.entity_id
                if i + 1 < len(path.steps):
                    to_eid = path.steps[i + 1].entity_id
                else:
                    to_eid = b_norm

                # Get all claims between from_eid and to_eid
                from_claims = self.claims_for(from_eid)
                hop_claims = [
                    c for c in from_claims
                    if (c.subject.id == from_eid and c.object.id == to_eid)
                    or (c.object.id == from_eid and c.subject.id == to_eid)
                ]

                source_types = list({c.provenance.source_type for c in hop_claims})
                source_ids = list({c.provenance.source_id for c in hop_claims})

                # Evidence text from payload
                evidence = ""
                for c in hop_claims:
                    if c.payload and c.payload.data:
                        text = c.payload.data.get("text", c.payload.data.get("evidence", ""))
                        if text:
                            evidence = str(text)
                            break

                # Contradiction check
                has_contradiction = False
                contradiction_pred = ""
                predicates = {c.predicate.id for c in hop_claims}
                for pred in predicates:
                    opp = OPPOSITE_PREDICATES.get(pred)
                    if opp and opp in predicates:
                        has_contradiction = True
                        contradiction_pred = opp
                        break

                best_conf = max((c.confidence for c in hop_claims), default=step.confidence)

                hops.append(ReasoningHop(
                    from_entity=from_eid,
                    to_entity=to_eid,
                    predicate=step.predicate,
                    confidence=best_conf,
                    source_types=source_types,
                    source_ids=source_ids,
                    evidence_text=evidence,
                    has_contradiction=has_contradiction,
                    contradiction_predicate=contradiction_pred,
                ))

            # Source overlap detection
            raw_conf = 1.0
            for h in hops:
                raw_conf *= h.confidence

            overlaps: list[SourceOverlap] = []
            for i in range(len(hops)):
                for j in range(i + 1, len(hops)):
                    shared = sorted(set(hops[i].source_ids) & set(hops[j].source_ids))
                    if shared:
                        discount = 1.0 / (1.0 + 0.3 * len(shared))
                        overlaps.append(SourceOverlap(
                            hop_a_index=i,
                            hop_b_index=j,
                            shared_sources=shared,
                            discount_factor=discount,
                        ))

            discount_product = 1.0
            for o in overlaps:
                discount_product *= o.discount_factor
            chain_conf = raw_conf * discount_product

            # Reliability score
            total_sources = len({sid for h in hops for sid in h.source_ids})
            source_diversity = total_sources / (len(hops) * 2) if hops else 0.0
            contradiction_ratio = (
                (sum(1 for h in hops if h.has_contradiction)
                 / len(hops))
                if hops else 0.0
            )
            reliability = 0.6 * chain_conf + 0.3 * source_diversity - 0.1 * contradiction_ratio
            reliability = max(0.0, min(1.0, reliability))

            chains.append(ReasoningChain(
                hops=hops,
                raw_confidence=raw_conf,
                source_overlaps=overlaps,
                chain_confidence=chain_conf,
                length=len(hops),
                reliability_score=reliability,
            ))

        chains.sort(key=lambda c: c.chain_confidence, reverse=True)
        return chains[:top_k]

    def suggest_investigations(self, top_k: int = 10) -> list[Investigation]:
        """Unified prioritized investigation recommendations.

        Synthesizes all existing insight signals (confidence alerts,
        blindspots, gaps, bridges) into a single ranked list.

        Args:
            top_k: Maximum recommendations to return.

        Returns:
            List of Investigation sorted by priority_score desc.
        """
        investigations: list[Investigation] = []
        seen_entities: set[str] = set()

        # 1. Confidence alerts → single_source / stale
        try:
            alerts = self.find_confidence_alerts(min_claims=2)
            for alert in alerts:
                if alert.entity_id in seen_entities:
                    continue
                seen_entities.add(alert.entity_id)
                # "single_source", "stale_evidence", "mixed_source_quality"
                signal = alert.alert_type.value
                if signal == "stale_evidence":
                    signal = "stale"
                elif signal == "mixed_source_quality":
                    signal = "confidence_gap"
                base = {"single_source": 0.6, "stale": 0.2, "confidence_gap": 0.5}.get(signal, 0.3)
                investigations.append(Investigation(
                    entity_id=alert.entity_id,
                    entity_type=alert.entity_type,
                    reason=alert.explanation,
                    signal_type=signal,
                    priority_score=base,
                    suggested_action=(
                        f"Seek additional sources for {alert.entity_id}"
                        if signal == "single_source"
                        else f"Update evidence for {alert.entity_id}"
                        if signal == "stale"
                        else f"Reconcile confidence spread for {alert.entity_id}"
                    ),
                ))
        except Exception:
            logger.debug("Could not compute confidence alerts for investigations", exc_info=True)

        # 2. Blindspots → single_source (deduplicate with alerts)
        try:
            bs = self.blindspots(min_claims=1)
            for eid in bs.single_source_entities:
                if eid in seen_entities:
                    continue
                seen_entities.add(eid)
                entity = self._store.get_entity(eid)
                etype = entity_summary_from_dict(entity).entity_type if entity else ""
                investigations.append(Investigation(
                    entity_id=eid,
                    entity_type=etype,
                    reason=f"Entity {eid} has only one source",
                    signal_type="single_source",
                    priority_score=0.6,
                    suggested_action=f"Seek additional sources for {eid}",
                ))
        except Exception:
            logger.debug("Could not compute blindspots for investigations", exc_info=True)

        # 3. Knowledge gaps (if vocabulary registered)
        try:
            constraints = self._store.get_predicate_constraints()
            if constraints:
                from attestdb.intelligence.insight_engine import build_expected_patterns
                patterns = build_expected_patterns(constraints)
                gaps = self.find_gaps(patterns, min_claims=1)
                for gap in gaps:
                    if gap.entity_id in seen_entities:
                        continue
                    seen_entities.add(gap.entity_id)
                    missing = ", ".join(gap.missing_predicate_types[:3])
                    investigations.append(Investigation(
                        entity_id=gap.entity_id,
                        entity_type=gap.entity_type,
                        reason=f"Missing predicates: {missing}",
                        signal_type="missing_predicate",
                        priority_score=0.4,
                        suggested_action=f"Add {missing} relationships for {gap.entity_id}",
                    ))
        except Exception:
            logger.debug("Could not compute gaps for investigations", exc_info=True)

        # 4. Predicted links (bridges)
        try:
            bridges = self.find_bridges(top_k=top_k * 2)
            for bridge in bridges:
                eid = bridge.entity_a
                if eid in seen_entities:
                    eid = bridge.entity_b
                if eid in seen_entities:
                    continue
                seen_entities.add(eid)
                entity = self._store.get_entity(eid)
                etype = entity_summary_from_dict(entity).entity_type if entity else ""
                score = 0.3 + bridge.similarity * 0.3
                investigations.append(Investigation(
                    entity_id=eid,
                    entity_type=etype,
                    reason=(
                        f"Predicted link: {bridge.entity_a} "
                        f"↔ {bridge.entity_b} "
                        f"(similarity={bridge.similarity:.2f})"
                    ),
                    signal_type="predicted_link",
                    priority_score=score,
                    suggested_action=(
                        f"Investigate link between "
                        f"{bridge.entity_a} and {bridge.entity_b}"
                    ),
                ))
        except Exception:
            logger.debug("Could not compute bridges for investigations", exc_info=True)

        # Centrality boost + affected downstream
        adj = self.get_adjacency_list()
        for inv in investigations:
            degree = len(adj.get(inv.entity_id, set()))
            boost = min(degree / 20.0, 1.0)
            inv.priority_score *= (1.0 + 0.5 * boost)
            inv.affected_downstream = len(self.claims_for(inv.entity_id))

        investigations.sort(key=lambda x: x.priority_score, reverse=True)
        return investigations[:top_k]

    def close_gaps(self, hypothesis=None, top_k=5, search_fn=None, use_curator=True):
        """Close knowledge gaps by researching and ingesting new evidence.

        Two modes:
        - **Hypothesis mode** (hypothesis is not None): test_hypothesis → extract
          gaps → research each → re-test → return before/after.
        - **Blindspot mode** (hypothesis is None): suggest_investigations → research
          each → measure blindspot improvement.

        Args:
            hypothesis: Natural-language hypothesis to close gaps for.
            top_k: Max gaps/investigations to research.
            search_fn: Optional callback(question_text) -> text.
            use_curator: Whether to triage discovered claims through curator.

        Returns:
            CloseGapsReport with before/after metrics.
        """
        return self._get_researcher().close_gaps(
            hypothesis=hypothesis, top_k=top_k,
            search_fn=search_fn, use_curator=use_curator,
        )

    def test_hypothesis(self, hypothesis: str) -> HypothesisVerdict:
        """Evaluate a natural-language hypothesis against the knowledge graph.

        Takes a hypothesis like "aspirin reduces inflammation via COX-2",
        uses 1 LLM call to parse entity pairs, then evaluates multi-hop
        chains against the graph.

        Args:
            hypothesis: Natural language hypothesis to test.

        Returns:
            HypothesisVerdict with supporting/contradicting evidence chains.
        """
        import math

        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        # Step 1: Parse hypothesis — LLM or fallback
        parsed_entities: list[dict] = []
        parsed_rels: list[dict] = []
        fallback_missing: list[str] = []

        llm_response = self._llm_call(
            f"Extract entities and relationships from this hypothesis. "
            f"Return ONLY valid JSON, no markdown:\n"
            f'{{"entities": [{{"name": "...", "type": "..."}}], '
            f'"relationships": [{{"from": "...", "to": "...", "predicate": "..."}}]}}\n\n'
            f"Hypothesis: {hypothesis}",
            max_tokens=512,
            temperature=0.0,
        )

        if llm_response:
            try:
                # Strip markdown fences if present
                text = llm_response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                parsed = json.loads(text.strip())
                parsed_entities = parsed.get("entities", [])
                parsed_rels = parsed.get("relationships", [])
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: extract words and search
        if not parsed_entities:
            words = [w.strip(".,;:!?()\"'") for w in hypothesis.split() if len(w) >= 2]
            stop = {"is", "an", "at", "or", "in", "on", "to", "of", "by", "as", "if", "so",
                    "up", "do", "no", "my", "we", "he", "it", "am", "be",
                    "the", "and", "for", "are", "but", "not", "can", "had", "has", "was",
                    "that", "this", "with", "from", "into", "than", "also", "have", "been",
                    "does", "will", "would", "could", "should", "through", "between", "which",
                    "their", "there", "about", "after", "before", "over", "under", "very",
                    "much", "more", "some", "such", "most", "each", "only", "both", "then",
                    "just", "like", "when", "what", "where", "them", "they", "your",
                    "may", "via", "how", "its", "out", "any", "all", "our", "who", "new",
                    "affects", "causes", "reduces", "increases", "leads", "involves",
                    "through", "because", "since", "using", "during"}
            words = [w for w in words if w.lower() not in stop]
            seen_ids: set[str] = set()
            fallback_missing: list[str] = []
            for word in words:
                results = self.search_entities(word, top_k=3)
                if results:
                    best = max(results, key=lambda e: e.claim_count)
                    if best.id not in seen_ids:
                        seen_ids.add(best.id)
                        parsed_entities.append({"name": best.id, "type": best.entity_type})
                else:
                    fallback_missing.append(word)

        # Step 2: Entity resolution
        entities_found: list[str] = []
        entities_missing: list[str] = list(fallback_missing)
        resolved: dict[str, str] = {}  # parsed name → resolved entity_id

        for ent in parsed_entities:
            name = ent.get("name", "")
            if not name:
                continue
            # Search with both original and lowercase — text index is case-sensitive
            results = self.search_entities(name, top_k=5)
            if name != name.lower():
                results += self.search_entities(name.lower(), top_k=5)
            # Deduplicate
            seen_res: dict[str, object] = {}
            for r in results:
                if r.id not in seen_res or r.claim_count > seen_res[r.id].claim_count:
                    seen_res[r.id] = r
            results = list(seen_res.values())
            if results:
                best = max(results, key=lambda e: e.claim_count)
                resolved[name.lower()] = best.id
                entities_found.append(best.id)
            else:
                entities_missing.append(name)

        # Insufficient data check
        if len(entities_found) < 2:
            return HypothesisVerdict(
                hypothesis=hypothesis,
                verdict="insufficient_data",
                entities_found=sorted(set(entities_found)),
                entities_missing=entities_missing,
                suggested_next_steps=(
                    [f"Add data about {e}" for e in entities_missing]
                    or ["Add more entities to the knowledge graph"]
                ),
            )

        # Build relationships to evaluate — from LLM or from entity pairs
        relationships: list[tuple[str, str, str]] = []  # (from_id, to_id, predicate)
        if parsed_rels:
            for rel in parsed_rels:
                from_name = rel.get("from", "").lower()
                to_name = rel.get("to", "").lower()
                pred = rel.get("predicate", "relates_to")
                from_id = resolved.get(from_name)
                to_id = resolved.get(to_name)
                if from_id and to_id:
                    relationships.append((from_id, to_id, pred))

        # Fallback: pair consecutive resolved entities
        if not relationships:
            found_ids = sorted(set(entities_found))
            for i in range(len(found_ids) - 1):
                relationships.append((found_ids[i], found_ids[i + 1], "relates_to"))

        # Step 3: Evidence gathering per relationship
        supporting_chains: list[EvidenceChain] = []
        contradicting_chains: list[EvidenceChain] = []
        confidence_gaps: list[ConfidenceGap] = []
        best_confidences: list[float] = []

        for from_id, to_id, expected_pred in relationships:
            paths = self.find_paths(from_id, to_id, max_depth=3, top_k=5)

            if not paths:
                confidence_gaps.append(ConfidenceGap(
                    from_entity=from_id,
                    to_entity=to_id,
                    issue="missing",
                    explanation=f"No path found between {from_id} and {to_id}",
                ))
                continue

            rel_best = 0.0
            for path in paths:
                direction = "supporting"
                # Check if path predicates oppose the expected predicate
                for step in path.steps:
                    opp = OPPOSITE_PREDICATES.get(expected_pred)
                    if opp and step.predicate == opp:
                        direction = "contradicting"
                        break
                    opp2 = OPPOSITE_PREDICATES.get(step.predicate)
                    if opp2 and opp2 == expected_pred:
                        direction = "contradicting"
                        break

                chain = EvidenceChain(
                    steps=list(path.steps),
                    propagated_confidence=path.total_confidence,
                    direction=direction,
                )
                if direction == "supporting":
                    supporting_chains.append(chain)
                    rel_best = max(rel_best, path.total_confidence)
                else:
                    contradicting_chains.append(chain)

                # Weak hop detection
                for step in path.steps:
                    if step.confidence < 0.3:
                        confidence_gaps.append(ConfidenceGap(
                            from_entity=step.entity_id,
                            to_entity=to_id,
                            issue="weak",
                            current_confidence=step.confidence,
                            explanation=(
                                f"Weak evidence for {step.predicate} "
                                f"(confidence={step.confidence:.2f})"
                            ),
                        ))

            if rel_best > 0:
                best_confidences.append(rel_best)

        # Step 5: Verdict
        has_supporting = len(supporting_chains) > 0
        has_contradicting = len(contradicting_chains) > 0
        all_rels_have_support = all(
            any(
                any(s.entity_id == to_id for s in chain.steps)
                for chain in supporting_chains
                if chain.propagated_confidence >= 0.3
            )
            for _, to_id, _ in relationships
        ) if relationships else False

        if all_rels_have_support and not has_contradicting:
            verdict = "supported"
        elif has_contradicting and not has_supporting:
            verdict = "contradicted"
        elif has_supporting and has_contradicting:
            verdict = "partial"
        elif has_supporting:
            verdict = "partial"
        else:
            verdict = "unsupported"

        # Verdict confidence = geometric mean of best chain confidences
        if best_confidences:
            log_sum = sum(math.log(c) for c in best_confidences)
            verdict_confidence = math.exp(
                log_sum / len(best_confidences)
            )
        else:
            verdict_confidence = 0.0

        # Step 6: Suggested next steps
        next_steps: list[str] = []
        for gap in confidence_gaps:
            if gap.issue == "missing":
                next_steps.append(f"Investigate link between {gap.from_entity} and {gap.to_entity}")
            elif gap.issue == "weak":
                next_steps.append(f"Seek corroboration for {gap.from_entity} ({gap.explanation})")
        for e in entities_missing:
            next_steps.append(f"Add data about {e}")

        return HypothesisVerdict(
            hypothesis=hypothesis,
            verdict=verdict,
            verdict_confidence=verdict_confidence,
            supporting_chains=supporting_chains,
            contradicting_chains=contradicting_chains,
            confidence_gaps=confidence_gaps,
            entities_found=sorted(set(entities_found)),
            entities_missing=entities_missing,
            suggested_next_steps=next_steps[:10],
        )


    # --- Discovery engine ---

    def discover(self, top_k: int = 10) -> list[Discovery]:
        """Proactive hypothesis generation from graph structure.

        Three signals (all pure computation, no LLM):
        1. Bridge predictions — ensemble-scored pairs with composed predicates
        2. Cross-domain insights — topology bridge entities with differing predicates
        3. Chain completion — 2-hop pairs missing a direct connection

        Returns list of Discovery sorted by confidence * novelty, truncated to top_k.
        """
        from attestdb.core.vocabulary import predict_predicate_from_paths

        entities = self.list_entities()
        if len(entities) < 3:
            return []

        discoveries: list[Discovery] = []
        entity_type_map = {e.id: e.entity_type for e in entities}
        entity_name_map = {e.id: e.name or e.id for e in entities}

        # Degree map for novelty scoring
        adj = self.get_adjacency_list()
        degree = {eid: len(neighbors) for eid, neighbors in adj.items()}
        max_degree = max(degree.values()) if degree else 1

        # --- Signal 1: Bridge predictions via EnsembleScorer ---
        try:
            from attestdb_enterprise.ensemble_scorer import EnsembleScorer

            from attestdb.intelligence.graph_embeddings import (
                compute_graph_embeddings,
                compute_weighted_graph_embeddings,
            )

            weighted_adj = self.get_weighted_adjacency()
            embeddings = compute_graph_embeddings(adj, dim=min(64, len(adj)))
            w_embeddings = compute_weighted_graph_embeddings(weighted_adj, dim=min(64, len(adj)))
            # Merge embeddings (weighted wins if available)
            for k, v in w_embeddings.items():
                embeddings[k] = v

            # Build community map
            communities: dict[str, set[str]] = {}
            if hasattr(self, "_topology") and self._topology is not None:
                for _res, comms in self._topology.communities.items():
                    for comm in comms:
                        communities[comm.id] = set(comm.members)

            scorer = EnsembleScorer(
                weighted_adj=weighted_adj,
                adj=adj,
                embeddings=embeddings,
                communities=communities,
                entity_names=entity_name_map,
                entity_types=entity_type_map,
            )
            predictions = scorer.score_candidates(top_k=top_k * 3, seed=42)

            # Entity to community mapping for inter-community detection
            entity_to_comm: dict[str, str] = {}
            for cid, members in communities.items():
                for m in members:
                    entity_to_comm[m] = cid

            for pred in predictions:
                if pred.already_connected if hasattr(pred, "already_connected") else False:
                    continue

                # Compose predicates from evidence paths
                path_dicts = []
                bridge_names = []
                for ep in pred.evidence_paths:
                    path_dicts.append({
                        "ac_predicates": ep.ac_predicates,
                        "cb_predicates": ep.cb_predicates,
                        "path_weight": ep.path_weight,
                    })
                    bridge_names.append(ep.bridge_name or ep.bridge_entity)

                predicted_pred, vote_frac = predict_predicate_from_paths(path_dicts)
                confidence = pred.composite_score

                # Novelty: 1 - shared_neighbors/max_degree, boosted for inter-community
                shared = len(
                    adj.get(pred.entity_a, set())
                    & adj.get(pred.entity_b, set())
                )
                novelty = 1.0 - (shared / max_degree) if max_degree > 0 else 1.0

                is_inter = pred.bridge_type == "inter-community"
                if is_inter:
                    novelty = min(1.0, novelty * 1.2)

                # Penalize hub entities (high-degree = less surprising)
                hub_penalty = 1.0
                for eid in [pred.entity_a, pred.entity_b]:
                    if degree.get(eid, 0) > max_degree * 0.5:
                        hub_penalty *= 0.8

                novelty *= hub_penalty

                name_a = entity_name_map.get(pred.entity_a, pred.entity_a)
                name_b = entity_name_map.get(pred.entity_b, pred.entity_b)
                bridge_str = ", ".join(bridge_names[:3]) if bridge_names else "shared neighbors"

                discoveries.append(Discovery(
                    hypothesis=f"{name_a} may {predicted_pred} {name_b} via {bridge_str}",
                    evidence_summary=(
                        f"{len(pred.evidence_paths)} bridging paths, "
                        f"vote={vote_frac:.0%} for {predicted_pred}"
                    ),
                    discovery_type="bridge_prediction",
                    confidence=confidence,
                    novelty_score=novelty,
                    entities=[pred.entity_a, pred.entity_b],
                    supporting_paths=path_dicts,
                    suggested_action=f"Search for evidence of {name_a} {predicted_pred} {name_b}",
                    predicted_predicate=predicted_pred,
                    entity_types={
                        pred.entity_a: entity_type_map.get(pred.entity_a, "entity"),
                        pred.entity_b: entity_type_map.get(pred.entity_b, "entity"),
                    },
                ))
        except Exception as exc:
            logger.debug("Bridge prediction signal skipped: %s", exc)

        # --- Signal 2: Cross-domain insights ---
        try:
            if hasattr(self, "_topology") and self._topology is not None:
                bridges = self.cross_domain_bridges(
                    top_k=top_k * 2,
                )
                for bridge in bridges:
                    if len(bridge.communities) < 2:
                        continue
                    # Compare predicates used in each community
                    bridge_claims = self.claims_for(bridge.entity_id)
                    comm_predicates: dict[str, set[str]] = {}
                    for claim in bridge_claims:
                        for cid in bridge.communities:
                            members = communities.get(cid, set())
                            if (
                                claim.object.id in members
                                or claim.subject.id in members
                            ):
                                comm_predicates.setdefault(cid, set()).add(claim.predicate.id)

                    comm_ids = list(comm_predicates.keys())
                    for i in range(len(comm_ids)):
                        for j in range(i + 1, len(comm_ids)):
                            preds_a = comm_predicates.get(comm_ids[i], set())
                            preds_b = comm_predicates.get(comm_ids[j], set())
                            diff_preds = preds_a - preds_b
                            if diff_preds:
                                name = entity_name_map.get(bridge.entity_id, bridge.entity_id)
                                for dp in diff_preds:
                                    discoveries.append(Discovery(
                                        hypothesis=(
                                            f"{name}'s {dp} pattern in "
                                            f"{comm_ids[i]} may extend "
                                            f"to {comm_ids[j]}"
                                        ),
                                        evidence_summary=(
                                            f"Bridge entity spans "
                                            f"{len(bridge.communities)} "
                                            f"communities, bridge_score="
                                            f"{bridge.bridge_score:.2f}"
                                        ),
                                        discovery_type="cross_domain",
                                        confidence=bridge.bridge_score * 0.5,
                                        novelty_score=0.8,
                                        entities=[bridge.entity_id],
                                        suggested_action=(
                                            f"Investigate {dp} "
                                            f"relationships for {name} "
                                            f"in community {comm_ids[j]}"
                                        ),
                                        predicted_predicate=dp,
                                        entity_types={bridge.entity_id: bridge.entity_type},
                                    ))
        except Exception as exc:
            logger.debug("Cross-domain signal skipped: %s", exc)

        # --- Signal 3: Chain completion ---
        try:
            from attestdb.core.vocabulary import compose_predicates

            # Find 2-hop pairs not directly connected
            entity_subset = [e for e in entities if degree.get(e.id, 0) >= 2]
            entity_subset.sort(key=lambda e: degree.get(e.id, 0))
            entity_subset = entity_subset[:50]  # Cap for performance

            direct_pairs: set[tuple[str, str]] = set()
            for (a, b) in self.get_weighted_adjacency():
                direct_pairs.add((a, b))
                direct_pairs.add((b, a))

            for entity in entity_subset:
                neighbors = adj.get(entity.id, set())
                for n1 in neighbors:
                    n1_neighbors = adj.get(n1, set())
                    for n2 in n1_neighbors:
                        if n2 == entity.id or n2 == n1:
                            continue
                        pair = (min(entity.id, n2), max(entity.id, n2))
                        if pair in direct_pairs:
                            continue
                        # Get predicates along the path
                        claims_a_n1 = [
                            c for c in self.claims_for(entity.id)
                            if c.object.id == n1 or c.subject.id == n1
                        ]
                        claims_n1_n2 = [
                            c for c in self.claims_for(n1)
                            if c.object.id == n2 or c.subject.id == n2
                        ]
                        if not claims_a_n1 or not claims_n1_n2:
                            continue

                        ac_preds = {c.predicate.id for c in claims_a_n1}
                        cb_preds = {c.predicate.id for c in claims_n1_n2}
                        avg_conf = (
                            sum(c.confidence for c in claims_a_n1) / len(claims_a_n1)
                            + sum(c.confidence for c in claims_n1_n2) / len(claims_n1_n2)
                        ) / 2

                        # Pick the best composed predicate
                        best_composed = "associated_with"
                        for ap in ac_preds:
                            for cp in cb_preds:
                                result = compose_predicates(ap, cp)
                                if result != "associated_with":
                                    best_composed = result
                                    break
                            if best_composed != "associated_with":
                                break

                        name_a = entity_name_map.get(entity.id, entity.id)
                        name_n1 = entity_name_map.get(n1, n1)
                        name_b = entity_name_map.get(n2, n2)

                        novelty = (
                            1.0 - (len(neighbors & n1_neighbors) / max_degree)
                            if max_degree > 0 else 1.0
                        )

                        discoveries.append(Discovery(
                            hypothesis=f"{name_a} may {best_composed} {name_b} via {name_n1}",
                            evidence_summary=(
                                f"2-hop path: {name_a} "
                                f"→[{','.join(ac_preds)}]→ {name_n1} "
                                f"→[{','.join(cb_preds)}]→ {name_b}"
                            ),
                            discovery_type="chain_completion",
                            confidence=avg_conf * 0.7,
                            novelty_score=novelty,
                            entities=[entity.id, n1, n2],
                            supporting_paths=[{
                                "ac_predicates": ac_preds,
                                "cb_predicates": cb_preds,
                                "path_weight": avg_conf,
                            }],
                            suggested_action=(
                                f"Search for direct evidence of "
                                f"{name_a} {best_composed} {name_b}"
                            ),
                            predicted_predicate=best_composed,
                            entity_types={
                                entity.id: entity_type_map.get(entity.id, "entity"),
                                n1: entity_type_map.get(n1, "entity"),
                                n2: entity_type_map.get(n2, "entity"),
                            },
                        ))
                        direct_pairs.add(pair)  # Don't re-discover same pair
        except Exception as exc:
            logger.debug("Chain completion signal skipped: %s", exc)

        # Deduplicate by entity pair
        seen_pairs: set[tuple[str, ...]] = set()
        deduped: list[Discovery] = []
        for d in discoveries:
            key = tuple(sorted(d.entities[:2]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                deduped.append(d)

        # Final ranking: confidence * (0.6 + 0.4 * novelty)
        deduped.sort(key=lambda d: d.confidence * (0.6 + 0.4 * d.novelty_score), reverse=True)
        return deduped[:top_k]

    def analogies(
        self,
        entity_a: str,
        entity_b: str,
        top_k: int = 5,
    ) -> list[Analogy]:
        """Find structural analogies: A:B :: C:D.

        Uses structural embeddings to find entities similar to A and B,
        then predicts the C:D pair and relationship.

        Returns empty list gracefully if no structural embeddings exist.
        """
        from attestdb.core.normalization import normalize_entity_id as _norm

        entity_a = _norm(entity_a)
        entity_b = _norm(entity_b)

        # Get A:B relationship predicates
        ab_claims = [
            c for c in self.claims_for(entity_a)
            if c.object.id == entity_b or c.subject.id == entity_b
        ]
        if not ab_claims:
            return []

        source_predicates = list({c.predicate.id for c in ab_claims})

        # Get structural embeddings
        emb_a = self.get_embedding(f"_struct_{entity_a}")
        emb_b = self.get_embedding(f"_struct_{entity_b}")
        if emb_a is None or emb_b is None:
            return []

        import numpy as np

        emb_a = np.array(emb_a, dtype=np.float32)
        emb_b = np.array(emb_b, dtype=np.float32)

        # Get all entities with structural embeddings
        entities = self.list_entities()
        entity_type_map = {e.id: e.entity_type for e in entities}
        entity_name_map = {e.id: e.name or e.id for e in entities}

        entity_embeddings: dict[str, np.ndarray] = {}
        for e in entities:
            emb = self.get_embedding(f"_struct_{e.id}")
            if emb is not None:
                entity_embeddings[e.id] = np.array(emb, dtype=np.float32)

        if len(entity_embeddings) < 4:
            return []

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return dot / (na * nb)

        # Find top 20 entities similar to A, top 20 similar to B
        sims_a = []
        sims_b = []
        for eid, emb in entity_embeddings.items():
            if eid in (entity_a, entity_b):
                continue
            sims_a.append((eid, _cosine(emb_a, emb)))
            sims_b.append((eid, _cosine(emb_b, emb)))

        sims_a.sort(key=lambda x: -x[1])
        sims_b.sort(key=lambda x: -x[1])
        top_c = sims_a[:20]
        top_d = sims_b[:20]

        # Check which pairs are already connected
        adj = self.get_adjacency_list()

        # Score (C, D) pairs
        candidates: list[Analogy] = []
        for c_id, sim_ac in top_c:
            for d_id, sim_bd in top_d:
                if c_id == d_id:
                    continue
                # Skip if C and D are already connected
                if d_id in adj.get(c_id, set()):
                    continue

                score = sim_ac * sim_bd

                # Type pattern boost: if type(C)==type(A) and type(D)==type(B)
                type_a = entity_type_map.get(entity_a, "")
                type_b = entity_type_map.get(entity_b, "")
                type_c = entity_type_map.get(c_id, "")
                type_d = entity_type_map.get(d_id, "")
                if (
                    type_a and type_c and type_a == type_c
                    and type_b and type_d and type_b == type_d
                ):
                    score *= 1.5

                name_a = entity_name_map.get(entity_a, entity_a)
                name_b = entity_name_map.get(entity_b, entity_b)
                name_c = entity_name_map.get(c_id, c_id)
                name_d = entity_name_map.get(d_id, d_id)

                candidates.append(Analogy(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    entity_c=c_id,
                    entity_d=d_id,
                    predicted_predicate=(
                        source_predicates[0]
                        if source_predicates
                        else "associated_with"
                    ),
                    score=score,
                    explanation=(
                        f"{name_a}:{name_b} :: {name_c}:{name_d}"
                        f" — sim(A,C)={sim_ac:.2f},"
                        f" sim(B,D)={sim_bd:.2f}"
                    ),
                    source_predicates=source_predicates,
                    entity_types={
                        entity_a: type_a or "entity",
                        entity_b: type_b or "entity",
                        c_id: type_c or "entity",
                        d_id: type_d or "entity",
                    },
                ))

        candidates.sort(key=lambda a: -a.score)
        return candidates[:top_k]

    # --- Crown jewels ---

    def _parse_timestamp(self, ts: str | int | float | None) -> int:
        """Convert ISO string, int, or float to nanosecond timestamp."""
        if ts is None:
            return 0
        if isinstance(ts, (int, float)):
            # int: already nanoseconds. float: assume seconds (time.time() style).
            if isinstance(ts, float):
                return int(ts * 1_000_000_000)
            return ts
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts))
        # Treat naive datetimes as UTC to match Rust store convention
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)

    def diff(self, since: str | int, until: str | int | None = None) -> KnowledgeDiff:
        """Knowledge diff — what changed between two time periods.

        Like git diff for knowledge: shows new beliefs, strengthened or
        weakened claims, and new contradictions.

        Args:
            since: Start of period — ISO string or nanosecond int.
            until: End of period — ISO string, nanosecond int, or None (now).

        Returns:
            KnowledgeDiff with categorized belief changes.
        """
        from attestdb.core.confidence import tier1_confidence, tier2_confidence
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        since_ns = self._parse_timestamp(since)
        until_ns = self._parse_timestamp(until) if until is not None else time.time_ns()

        # Include retracted claims so we can count tombstones and detect weakened beliefs
        self._set_include_retracted(True)
        try:
            all_claims = self._all_claims()
        finally:
            self._set_include_retracted(False)
        before_claims = [c for c in all_claims if c.timestamp < since_ns]
        period_claims = [c for c in all_claims if since_ns <= c.timestamp < until_ns]

        # Group by content_id
        before_cids: dict[str, list[Claim]] = {}
        for c in before_claims:
            before_cids.setdefault(c.content_id, []).append(c)

        period_cids: dict[str, list[Claim]] = {}
        for c in period_claims:
            period_cids.setdefault(c.content_id, []).append(c)

        new_beliefs: list[BeliefChange] = []
        strengthened: list[BeliefChange] = []
        weakened: list[BeliefChange] = []
        new_contradictions: list[BeliefChange] = []

        # Collect entities and sources
        before_entities = (
            {c.subject.id for c in before_claims}
            | {c.object.id for c in before_claims}
        )
        period_entities = (
            {c.subject.id for c in period_claims}
            | {c.object.id for c in period_claims}
        )
        before_sources = {c.provenance.source_id for c in before_claims}
        period_sources = {c.provenance.source_id for c in period_claims}

        for cid, p_claims in period_cids.items():
            rep = p_claims[0]  # representative claim
            new_src = sorted({c.provenance.source_id for c in p_claims} - before_sources)

            if cid not in before_cids:
                # New belief
                conf_after = tier2_confidence(rep, p_claims)
                new_beliefs.append(BeliefChange(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    change_type="new",
                    claims_before=0,
                    claims_after=len(p_claims),
                    confidence_before=0.0,
                    confidence_after=conf_after,
                    new_sources=new_src,
                ))
            else:
                # Existing — check strengthened (only active claims for confidence)
                b_claims = before_cids[cid]
                active_b = [c for c in b_claims if c.status == ClaimStatus.ACTIVE]
                all_cid_claims = active_b + p_claims
                conf_before = tier2_confidence(b_claims[0], active_b) if active_b else 0.0
                conf_after = tier2_confidence(rep, all_cid_claims)
                if len(all_cid_claims) > len(b_claims):
                    strengthened.append(BeliefChange(
                        content_id=cid,
                        subject=rep.subject.id,
                        predicate=rep.predicate.id,
                        object=rep.object.id,
                        change_type="strengthened",
                        claims_before=len(b_claims),
                        claims_after=len(all_cid_claims),
                        confidence_before=conf_before,
                        confidence_after=conf_after,
                        new_sources=new_src,
                    ))

        # Check for weakened — claims that existed before but have fewer active
        # claims when considering the state at `until` (not current live state).
        # Combine before + period claims for the time-bounded view.
        until_cids: dict[str, list[Claim]] = {}
        for c in before_claims + period_claims:
            until_cids.setdefault(c.content_id, []).append(c)
        for cid, b_claims in before_cids.items():
            active_before = [c for c in b_claims if c.status == ClaimStatus.ACTIVE]
            all_cid_at_until = until_cids.get(cid, [])
            active_now = [c for c in all_cid_at_until if c.status == ClaimStatus.ACTIVE]
            if len(active_now) < len(active_before):
                rep = b_claims[0]
                conf_before = tier2_confidence(rep, active_before) if active_before else 0.0
                conf_after = tier2_confidence(rep, active_now) if active_now else 0.0
                weakened.append(BeliefChange(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    change_type="weakened",
                    claims_before=len(active_before),
                    claims_after=len(active_now),
                    confidence_before=conf_before,
                    confidence_after=conf_after,
                    new_sources=[],
                ))

        # Check for new contradictions: period claims whose (subj, obj) pair
        # has an opposite predicate — but only if the contradiction didn't
        # already exist in the before period.
        before_preds: dict[tuple[str, str], set[str]] = {}
        for c in before_claims:
            key = (c.subject.id, c.object.id)
            before_preds.setdefault(key, set()).add(c.predicate.id)

        all_preds: dict[tuple[str, str], set[str]] = {}
        for c in before_claims + period_claims:
            key = (c.subject.id, c.object.id)
            all_preds.setdefault(key, set()).add(c.predicate.id)

        seen_contradictions: set[str] = set()
        for c in period_claims:
            pred = c.predicate.id
            opp = OPPOSITE_PREDICATES.get(pred)
            if not opp:
                continue
            key = (c.subject.id, c.object.id)
            if opp not in all_preds.get(key, set()):
                continue
            # Skip if this contradiction already existed before the period
            bp = before_preds.get(key, set())
            if pred in bp and opp in bp:
                continue
            contra_key = f"{c.subject.id}|{c.object.id}|{min(pred,opp)}|{max(pred,opp)}"
            if contra_key not in seen_contradictions:
                seen_contradictions.add(contra_key)
                new_contradictions.append(BeliefChange(
                    content_id=c.content_id,
                    subject=c.subject.id,
                    predicate=pred,
                    object=c.object.id,
                    change_type="contradicted",
                    claims_before=0,
                    claims_after=1,
                    confidence_before=0.0,
                    confidence_after=tier1_confidence(c.provenance.source_type),
                    new_sources=[c.provenance.source_id],
                ))

        total_retracted = sum(1 for c in period_claims if c.status == ClaimStatus.TOMBSTONED)
        new_entity_list = sorted(period_entities - before_entities)
        new_source_list = sorted(period_sources - before_sources)

        # Net confidence: average confidence of period claims
        net_conf = 0.0
        if period_claims:
            net_conf = sum(c.confidence for c in period_claims) / len(period_claims)

        # Summary
        parts = []
        if new_beliefs:
            parts.append(f"{len(new_beliefs)} new beliefs")
        if strengthened:
            parts.append(f"{len(strengthened)} strengthened")
        if weakened:
            parts.append(f"{len(weakened)} weakened")
        if new_contradictions:
            parts.append(f"{len(new_contradictions)} new contradictions")
        if new_entity_list:
            parts.append(f"{len(new_entity_list)} new entities")
        if new_source_list:
            parts.append(f"{len(new_source_list)} new sources")
        summary = "; ".join(parts) if parts else "No changes"

        return KnowledgeDiff(
            since=since_ns,
            until=until_ns,
            new_beliefs=new_beliefs,
            strengthened=strengthened,
            weakened=weakened,
            new_contradictions=new_contradictions,
            new_entities=new_entity_list,
            new_sources=new_source_list,
            total_new_claims=len(period_claims),
            total_retracted=total_retracted,
            net_confidence=net_conf,
            summary=summary,
        )

    def resolve_contradictions(
        self, top_k: int = 10, auto_resolve: bool = False, use_llm: bool = False,
    ) -> ContradictionReport:
        """Find and score all contradictions in the knowledge graph.

        Uses OPPOSITE_PREDICATES to detect contradicting claims on the
        same (subject, object) pair, scores evidence quality on each side,
        and optionally resolves clear winners.

        Args:
            top_k: Maximum contradictions to return.
            auto_resolve: If True, ingest meta-claims for resolved contradictions.
            use_llm: If True and ambiguous, use LLM for adjudication.

        Returns:
            ContradictionReport with scored analyses.
        """
        from attestdb.core.confidence import (
            count_independent_sources,
            recency_factor,
            tier1_confidence,
        )
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        all_claims = self._all_claims()

        # Group by (subject.id, object.id) → {predicate: [claims]}
        pair_preds: dict[tuple[str, str], dict[str, list[Claim]]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            key = (c.subject.id, c.object.id)
            pair_preds.setdefault(key, {}).setdefault(c.predicate.id, []).append(c)

        analyses: list[ContradictionAnalysis] = []

        for (subj, obj), pred_map in pair_preds.items():
            predicates = list(pred_map.keys())
            checked: set[tuple[str, str]] = set()
            for pred_a in predicates:
                opp = OPPOSITE_PREDICATES.get(pred_a)
                if opp and opp in pred_map:
                    pair_key = tuple(sorted([pred_a, opp]))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    claims_a = pred_map[pred_a]
                    claims_b = pred_map[opp]

                    def _build_side(claims: list[Claim], pred: str) -> ContradictionSide:
                        sources = {c.provenance.source_id for c in claims}
                        source_types = sorted({c.provenance.source_type for c in claims})
                        corr = count_independent_sources(claims)
                        avg_conf = (
                            sum(tier1_confidence(c.provenance.source_type) for c in claims)
                            / len(claims)
                        )
                        newest = max(c.timestamp for c in claims)
                        rec = recency_factor(newest)
                        weight = (
                            0.35 * min(corr / 5, 1.0)
                            + 0.25 * avg_conf
                            + 0.20 * min(len(sources) / 3, 1.0)
                            + 0.20 * rec
                        )
                        return ContradictionSide(
                            predicate=pred,
                            claim_count=len(claims),
                            source_count=len(sources),
                            source_types=source_types,
                            avg_confidence=avg_conf,
                            corroboration_count=corr,
                            newest_timestamp=newest,
                            evidence_weight=weight,
                        )

                    side_a = _build_side(claims_a, pred_a)
                    side_b = _build_side(claims_b, opp)

                    margin = abs(side_a.evidence_weight - side_b.evidence_weight)
                    if margin >= 0.15:
                        winner = (
                            "side_a"
                            if side_a.evidence_weight > side_b.evidence_weight
                            else "side_b"
                        )
                        resolution = "a_preferred" if winner == "side_a" else "b_preferred"
                    else:
                        winner = "ambiguous"
                        resolution = "unresolved"

                    explanation = (
                        f"{pred_a} ({side_a.claim_count} claims,"
                        f" weight={side_a.evidence_weight:.2f})"
                        f" vs {opp} ({side_b.claim_count} claims,"
                        f" weight={side_b.evidence_weight:.2f})"
                    )

                    # LLM adjudication for ambiguous cases
                    if winner == "ambiguous" and use_llm:
                        prompt = (
                            f"Two contradicting claims about"
                            f" {subj} and {obj}:\n"
                            f"Side A: '{pred_a}' — "
                            f"{side_a.claim_count} claims from "
                            f"{side_a.source_count} sources\n"
                            f"Side B: '{opp}' — "
                            f"{side_b.claim_count} claims from "
                            f"{side_b.source_count} sources\n"
                            f"Which is more likely correct?"
                            f" Reply with just 'A' or 'B'"
                            f" or 'ambiguous'."
                        )
                        llm_resp = self._llm_call(prompt, max_tokens=32)
                        if llm_resp:
                            resp = llm_resp.strip().lower()
                            # Match explicit "a"/"b"/"side a"/"side b" tokens
                            import re
                            a_match = bool(re.search(r'\bside\s*a\b|\ba\b', resp))
                            b_match = bool(re.search(r'\bside\s*b\b|\bb\b', resp))
                            if a_match and not b_match:
                                winner = "side_a"
                                resolution = "a_preferred"
                                explanation += f" — LLM chose {pred_a}"
                            elif b_match and not a_match:
                                winner = "side_b"
                                resolution = "b_preferred"
                                explanation += f" — LLM chose {opp}"

                    analyses.append(ContradictionAnalysis(
                        subject=subj,
                        object=obj,
                        side_a=side_a,
                        side_b=side_b,
                        winner=winner,
                        margin=margin,
                        resolution=resolution,
                        explanation=explanation,
                    ))

        # Sort by margin descending (clearest resolutions first)
        analyses.sort(key=lambda a: -a.margin)
        analyses = analyses[:top_k]

        resolved = sum(1 for a in analyses if a.resolution != "unresolved")
        ambiguous = sum(1 for a in analyses if a.resolution == "unresolved")

        # Auto-resolve: ingest meta-claims for resolved contradictions
        claims_added = 0
        if auto_resolve:
            for a in analyses:
                if a.resolution == "unresolved":
                    continue
                winning_pred = (
                    a.side_a.predicate
                    if a.resolution == "a_preferred"
                    else a.side_b.predicate
                )
                self.ingest(
                    subject=(a.subject, "entity"),
                    predicate=("contradiction_resolved", "contradiction_resolved"),
                    object=(a.object, "entity"),
                    provenance={
                        "source_type": "computation",
                        "source_id": "attest:resolve_contradictions",
                    },
                    payload={
                        "schema": "contradiction_resolution",
                        "data": {
                            "winner": winning_pred,
                            "margin": a.margin,
                            "explanation": a.explanation,
                        },
                    },
                )
                claims_added += 1

        return ContradictionReport(
            total_found=len(analyses),
            resolved=resolved,
            ambiguous=ambiguous,
            analyses=analyses,
            claims_added=claims_added,
        )

    def simulate(
        self,
        retract_source: str | None = None,
        add_claim: ClaimInput | None = None,
        remove_entity: str | None = None,
    ) -> SimulationReport:
        """Counterfactual simulation — what-if analysis without modifying the DB.

        Computes cascading effects of hypothetical changes: source retractions,
        new claims, or entity removal.

        Args:
            retract_source: Source ID to simulate retracting.
            add_claim: A ClaimInput to simulate adding.
            remove_entity: Entity ID to simulate removing.

        Returns:
            SimulationReport with impact analysis.
        """
        if retract_source:
            return self._simulate_retract_source(retract_source)
        elif add_claim:
            return self._simulate_add_claim(add_claim)
        elif remove_entity:
            return self._simulate_remove_entity(remove_entity)
        else:
            return SimulationReport(scenario="no-op", summary="No scenario specified")

    def _simulate_retract_source(self, source_id: str) -> SimulationReport:
        """Simulate retracting a source — read-only analysis."""
        from attestdb.core.confidence import tier2_confidence

        # Direct claims from this source
        direct_claims = self.claims_by_source_id(source_id)
        direct_ids = {c.claim_id for c in direct_claims}

        if not direct_claims:
            return SimulationReport(
                scenario=f"retract_source:{source_id}",
                summary=f"Source '{source_id}' has no claims — no impact.",
            )

        # Downstream cascade (read-only BFS)
        reverse_index = self._build_reverse_provenance_index()
        degraded_ids: set[str] = set()
        queue = deque(direct_ids)
        while queue:
            cid = queue.popleft()
            for dep_id in reverse_index.get(cid, []):
                if dep_id not in direct_ids and dep_id not in degraded_ids:
                    degraded_ids.add(dep_id)
                    queue.append(dep_id)

        # All affected claim IDs
        all_affected_ids = direct_ids | degraded_ids
        all_claims = self._all_claims()

        # Entities affected
        entities_affected: set[str] = set()
        for c in direct_claims:
            entities_affected.add(c.subject.id)
            entities_affected.add(c.object.id)

        # Build entity → sources mapping (before and after)
        entity_sources_before: dict[str, set[str]] = {}
        entity_sources_after: dict[str, set[str]] = {}
        entity_claim_count_before: dict[str, int] = {}
        entity_claim_count_after: dict[str, int] = {}

        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            for eid in (c.subject.id, c.object.id):
                entity_sources_before.setdefault(eid, set()).add(c.provenance.source_id)
                entity_claim_count_before[eid] = entity_claim_count_before.get(eid, 0) + 1
                if c.claim_id not in all_affected_ids:
                    entity_sources_after.setdefault(eid, set()).add(c.provenance.source_id)
                    entity_claim_count_after[eid] = entity_claim_count_after.get(eid, 0) + 1

        # Orphaned entities: those that lose ALL claims
        orphaned = sum(
            1 for eid in entities_affected
            if entity_claim_count_before.get(eid, 0) > 0
            and entity_claim_count_after.get(eid, 0) == 0
        )

        # Entities going from multi-source to single-source
        single_source = sum(
            1 for eid in entities_affected
            if len(entity_sources_before.get(eid, set())) > 1
            and len(entity_sources_after.get(eid, set())) <= 1
        )

        # Confidence shifts for affected content_ids
        confidence_shifts: list[ConfidenceShift] = []
        affected_content_ids: dict[str, list[Claim]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            if c.claim_id in all_affected_ids:
                affected_content_ids.setdefault(c.content_id, [])

        for cid in affected_content_ids:
            all_cid_claims = self.claims_by_content_id(cid)
            active_claims = [c for c in all_cid_claims if c.status == ClaimStatus.ACTIVE]
            remaining = [c for c in active_claims if c.claim_id not in all_affected_ids]
            if active_claims:
                conf_before = tier2_confidence(active_claims[0], active_claims)
                conf_after = tier2_confidence(remaining[0], remaining) if remaining else 0.0
                rep = active_claims[0]
                confidence_shifts.append(ConfidenceShift(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    confidence_before=conf_before,
                    confidence_after=conf_after,
                ))

        # Connection losses: entity pairs losing all edges
        connection_losses: list[ConnectionLoss] = []
        edge_before: dict[tuple[str, str], set[str]] = {}
        edge_after: dict[tuple[str, str], set[str]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pair = (c.subject.id, c.object.id)
            edge_before.setdefault(pair, set()).add(c.predicate.id)
            if c.claim_id not in all_affected_ids:
                edge_after.setdefault(pair, set()).add(c.predicate.id)

        for pair, preds in edge_before.items():
            remaining_preds = edge_after.get(pair, set())
            lost = preds - remaining_preds
            if lost:
                connection_losses.append(ConnectionLoss(
                    entity_a=pair[0],
                    entity_b=pair[1],
                    lost_predicates=sorted(lost),
                ))

        # Risk score
        total_active = sum(1 for c in all_claims if c.status == ClaimStatus.ACTIVE)
        claims_fraction = len(direct_ids) / max(total_active, 1)
        total_pairs = len(edge_before)
        conn_loss_fraction = len(connection_losses) / max(total_pairs, 1)
        degraded_fraction = len(degraded_ids) / max(total_active, 1)
        avg_conf_loss = 0.0
        if confidence_shifts:
            avg_conf_loss = sum(
                cs.confidence_before - cs.confidence_after for cs in confidence_shifts
            ) / len(confidence_shifts)
        risk_score = (
            0.3 * claims_fraction
            + 0.3 * conn_loss_fraction
            + 0.2 * degraded_fraction
            + 0.2 * min(avg_conf_loss, 1.0)
        )
        risk_score = min(risk_score, 1.0)
        if risk_score > 0.5:
            risk_level = "critical"
        elif risk_score > 0.3:
            risk_level = "high"
        elif risk_score > 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        summary = (
            f"Retracting source '{source_id}': {len(direct_ids)} claims removed, "
            f"{len(degraded_ids)} degraded, {orphaned} entities orphaned, "
            f"{len(connection_losses)} connections lost. Risk: {risk_level}."
        )

        return SimulationReport(
            scenario=f"retract_source:{source_id}",
            claims_affected=len(all_affected_ids),
            claims_removed=len(direct_ids),
            claims_degraded=len(degraded_ids),
            entities_affected=sorted(entities_affected),
            entities_now_orphaned=orphaned,
            entities_now_single_source=single_source,
            connection_losses=connection_losses,
            confidence_shifts=confidence_shifts,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
        )

    def _simulate_add_claim(self, claim_input: ClaimInput) -> SimulationReport:
        """Simulate adding a claim — check corroborations, contradictions, gaps."""
        from attestdb.core.hashing import compute_content_id
        from attestdb.core.normalization import normalize_entity_id
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        subj_id = normalize_entity_id(claim_input.subject[0])
        obj_id = normalize_entity_id(claim_input.object[0])
        pred_id = claim_input.predicate[0]

        content_id = compute_content_id(subj_id, pred_id, obj_id)

        # Check existing corroborations
        existing = self.claims_by_content_id(content_id)
        new_corroborations = len(existing)

        # Check contradictions
        new_contradictions: list[str] = []
        opp = OPPOSITE_PREDICATES.get(pred_id)
        if opp:
            # Look for opposite predicate on same (subj, obj)
            all_subj_claims = self.claims_for(subj_id)
            for c in all_subj_claims:
                if c.predicate.id == opp and c.object.id == obj_id:
                    new_contradictions.append(
                        f"{subj_id} {pred_id} {obj_id} contradicts"
                        f" existing {subj_id} {opp} {obj_id}"
                    )
                    break

        # Check if this closes a gap (entities exist but unconnected)
        gaps_closed = 0
        subj_entity = self.get_entity(subj_id)
        obj_entity = self.get_entity(obj_id)
        if subj_entity and obj_entity:
            if not self.path_exists(subj_id, obj_id, max_depth=2):
                gaps_closed = 1

        summary = f"Adding {subj_id} {pred_id} {obj_id}: "
        parts = []
        if new_corroborations:
            parts.append(f"corroborates {new_corroborations} existing claims")
        if new_contradictions:
            parts.append(f"{len(new_contradictions)} contradictions")
        if gaps_closed:
            parts.append("closes a knowledge gap")
        summary += "; ".join(parts) if parts else "new isolated claim"

        return SimulationReport(
            scenario=f"add_claim:{subj_id}_{pred_id}_{obj_id}",
            claims_affected=0,
            new_contradictions=new_contradictions,
            new_corroborations=new_corroborations,
            gaps_closed=gaps_closed,
            risk_score=0.0,
            risk_level="low",
            summary=summary,
        )

    def _simulate_remove_entity(self, entity_id: str) -> SimulationReport:
        """Simulate removing an entity — analyze downstream cascade."""
        from attestdb.core.normalization import normalize_entity_id

        eid = normalize_entity_id(entity_id)
        entity_claims = self.claims_for(eid)

        if not entity_claims:
            return SimulationReport(
                scenario=f"remove_entity:{eid}",
                summary=f"Entity '{eid}' has no claims — no impact.",
            )

        # Treat all entity claims as removed
        removed_ids = {c.claim_id for c in entity_claims}

        # Downstream cascade
        reverse_index = self._build_reverse_provenance_index()
        degraded_ids: set[str] = set()
        queue = deque(removed_ids)
        while queue:
            cid = queue.popleft()
            for dep_id in reverse_index.get(cid, []):
                if dep_id not in removed_ids and dep_id not in degraded_ids:
                    degraded_ids.add(dep_id)
                    queue.append(dep_id)

        # Entities affected
        entities_affected: set[str] = set()
        for c in entity_claims:
            entities_affected.add(c.subject.id)
            entities_affected.add(c.object.id)

        # Connection losses
        connection_losses: list[ConnectionLoss] = []
        all_affected_ids = removed_ids | degraded_ids
        all_claims = self._all_claims()

        edge_before: dict[tuple[str, str], set[str]] = {}
        edge_after: dict[tuple[str, str], set[str]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pair = (c.subject.id, c.object.id)
            edge_before.setdefault(pair, set()).add(c.predicate.id)
            if c.claim_id not in all_affected_ids:
                edge_after.setdefault(pair, set()).add(c.predicate.id)

        for pair, preds in edge_before.items():
            remaining_preds = edge_after.get(pair, set())
            lost = preds - remaining_preds
            if lost:
                connection_losses.append(ConnectionLoss(
                    entity_a=pair[0],
                    entity_b=pair[1],
                    lost_predicates=sorted(lost),
                ))

        # Risk score based on entity degree and downstream
        total_active = sum(1 for c in all_claims if c.status == ClaimStatus.ACTIVE)
        claims_fraction = len(removed_ids) / max(total_active, 1)
        total_pairs = len(edge_before)
        conn_loss_fraction = len(connection_losses) / max(total_pairs, 1)
        risk_score = min(
            0.4 * claims_fraction
            + 0.4 * conn_loss_fraction
            + 0.2 * (len(degraded_ids) / max(total_active, 1)),
            1.0,
        )

        if risk_score > 0.5:
            risk_level = "critical"
        elif risk_score > 0.3:
            risk_level = "high"
        elif risk_score > 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        summary = (
            f"Removing entity '{eid}': {len(removed_ids)} claims removed, "
            f"{len(degraded_ids)} degraded, {len(connection_losses)} connections lost. "
            f"Risk: {risk_level}."
        )

        return SimulationReport(
            scenario=f"remove_entity:{eid}",
            claims_affected=len(all_affected_ids),
            claims_removed=len(removed_ids),
            claims_degraded=len(degraded_ids),
            entities_affected=sorted(entities_affected),
            connection_losses=connection_losses,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
        )

    def compile(self, topic: str, max_entities: int = 50, use_llm: bool = False) -> KnowledgeBrief:
        """Compile a structured research brief on a topic.

        Searches for relevant entities, clusters them by graph proximity,
        and generates sections with citations, contradictions, and gaps.

        Args:
            topic: The topic to compile a brief for.
            max_entities: Maximum entities to include.
            use_llm: If True, use LLM for narrative generation.

        Returns:
            KnowledgeBrief with structured sections.
        """
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        # Search for entities matching the topic
        entities: list[EntitySummary] = []
        seen_ids: set[str] = set()

        # Search full topic
        for e in self.search_entities(topic, top_k=max_entities):
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                entities.append(e)

        # Also search individual words for broader coverage
        words = topic.split()
        if len(words) > 1:
            for word in words:
                if len(word) < 3:
                    continue
                for e in self.search_entities(word, top_k=max_entities):
                    if e.id not in seen_ids and len(entities) < max_entities:
                        seen_ids.add(e.id)
                        entities.append(e)

        if not entities:
            return KnowledgeBrief(
                topic=topic,
                executive_summary=f"No entities found for topic '{topic}'.",
            )

        entity_ids = [e.id for e in entities]
        entity_map = {e.id: e for e in entities}

        # Cluster entities by graph proximity
        clusters = self._cluster_entities(entity_ids)

        sections: list[BriefSection] = []
        all_citations: list[Citation] = []
        all_sources: set[str] = set()
        all_contradictions: list[str] = []
        all_gaps: list[str] = []
        total_claims_cited = 0

        for cluster in clusters:
            # Label the cluster
            title = self._label_cluster(cluster, entity_map)

            # Gather all claims for entities in this cluster
            cluster_claims: list[Claim] = []
            claim_ids_seen: set[str] = set()
            for eid in cluster:
                for c in self.claims_for(eid):
                    if c.claim_id not in claim_ids_seen and c.status == ClaimStatus.ACTIVE:
                        claim_ids_seen.add(c.claim_id)
                        cluster_claims.append(c)

            # Build citations
            citations: list[Citation] = []
            content_id_counts: dict[str, int] = {}
            for c in cluster_claims:
                content_id_counts[c.content_id] = content_id_counts.get(c.content_id, 0) + 1

            # Dedupe citations by content_id, keep highest confidence
            content_id_best: dict[str, Claim] = {}
            for c in cluster_claims:
                if (
                    c.content_id not in content_id_best
                    or c.confidence > content_id_best[c.content_id].confidence
                ):
                    content_id_best[c.content_id] = c

            for cid, c in content_id_best.items():
                all_sources.add(c.provenance.source_id)
                citations.append(Citation(
                    claim_id=c.claim_id,
                    subject=c.subject.id,
                    predicate=c.predicate.id,
                    object=c.object.id,
                    confidence=c.confidence,
                    source_id=c.provenance.source_id,
                    source_type=c.provenance.source_type,
                    corroboration_count=content_id_counts.get(cid, 0),
                ))

            citations.sort(key=lambda ct: -ct.confidence)
            total_claims_cited += len(citations)
            all_citations.extend(citations)

            # Key findings: top 5 by confidence
            key_findings: list[str] = []
            for ct in citations[:5]:
                corr_str = (
                    f", {ct.corroboration_count} sources"
                    if ct.corroboration_count > 1 else ""
                )
                key_findings.append(
                    f"{ct.subject} {ct.predicate} {ct.object} "
                    f"(confidence: {ct.confidence:.2f}{corr_str})"
                )

            # Detect contradictions within section
            section_contradictions: list[str] = []
            seen_pairs: set[tuple[str, str, str]] = set()
            subj_obj_preds: dict[tuple[str, str], set[str]] = {}
            for c in cluster_claims:
                key = (c.subject.id, c.object.id)
                subj_obj_preds.setdefault(key, set()).add(c.predicate.id)
            for (subj, obj), preds in subj_obj_preds.items():
                for p in preds:
                    opp = OPPOSITE_PREDICATES.get(p)
                    if opp and opp in preds:
                        dedup = tuple(sorted([p, opp]))
                        pair_key = (subj, dedup[0], dedup[1])
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            section_contradictions.append(f"{subj}: {p} vs {opp} {obj}")
            all_contradictions.extend(section_contradictions)

            # Detect gaps: single-source entities in cluster
            section_gaps: list[str] = []
            for eid in cluster:
                eid_claims = [
                    c for c in cluster_claims
                    if c.subject.id == eid or c.object.id == eid
                ]
                sources = {c.provenance.source_id for c in eid_claims}
                if len(sources) == 1 and len(eid_claims) > 0:
                    section_gaps.append(
                        f"{eid}: single source"
                        f" ({next(iter(sources))})"
                    )
            all_gaps.extend(section_gaps)

            # Section confidence
            avg_conf = (
                (sum(c.confidence for c in cluster_claims)
                 / len(cluster_claims))
                if cluster_claims else 0.0
            )

            sections.append(BriefSection(
                title=title,
                entities=cluster,
                key_findings=key_findings,
                citations=citations,
                contradictions=section_contradictions,
                gaps=section_gaps,
                avg_confidence=avg_conf,
            ))

        # Sort sections by citation count descending
        sections.sort(key=lambda s: -len(s.citations))

        # Overall metrics
        avg_confidence = (
            (sum(s.avg_confidence for s in sections) / len(sections))
            if sections else 0.0
        )

        # Strongest findings: top 3 by confidence * corroboration
        all_citations.sort(key=lambda ct: -(ct.confidence * ct.corroboration_count))
        strongest = [
            f"{ct.subject} {ct.predicate} {ct.object}"
            f" (conf={ct.confidence:.2f},"
            f" {ct.corroboration_count} sources)"
            for ct in all_citations[:3]
        ]

        # Weakest areas: sections with lowest avg_confidence or most gaps
        weak_sections = sorted(sections, key=lambda s: s.avg_confidence)
        weakest = [
            f"{s.title} (avg confidence:"
            f" {s.avg_confidence:.2f}, {len(s.gaps)} gaps)"
            for s in weak_sections[:3]
        ]

        # Executive summary
        if use_llm:
            findings_text = "\n".join(strongest[:5])
            prompt = (
                f"Write a 2-3 sentence executive summary about '{topic}' based on:\n"
                f"- {len(entities)} entities analyzed across {len(sections)} topic clusters\n"
                f"- Key findings:\n{findings_text}\n"
                f"- {len(all_contradictions)} contradictions, {len(all_gaps)} knowledge gaps\n"
                f"Be concise and factual."
            )
            exec_summary = self._llm_call(prompt, max_tokens=256)
            if not exec_summary:
                exec_summary = self._template_summary(
                    topic, entities, sections,
                    all_contradictions, all_gaps, strongest,
                )
        else:
            exec_summary = self._template_summary(
                topic, entities, sections,
                all_contradictions, all_gaps, strongest,
            )

        return KnowledgeBrief(
            topic=topic,
            sections=sections,
            executive_summary=exec_summary,
            total_entities=len(entities),
            total_claims_cited=total_claims_cited,
            total_sources=len(all_sources),
            avg_confidence=avg_confidence,
            total_contradictions=len(all_contradictions),
            total_gaps=len(all_gaps),
            strongest_findings=strongest,
            weakest_areas=weakest,
        )

    def _template_summary(
        self, topic: str, entities: list, sections: list,
        contradictions: list, gaps: list, strongest: list,
    ) -> str:
        """Generate a template-based executive summary."""
        parts = [f"Brief on '{topic}': {len(entities)} entities across {len(sections)} clusters."]
        if strongest:
            parts.append(f"Top finding: {strongest[0]}.")
        if contradictions:
            parts.append(f"{len(contradictions)} contradictions detected.")
        if gaps:
            parts.append(f"{len(gaps)} knowledge gaps identified.")
        return " ".join(parts)

    def explain_why(
        self, entity_a: str, entity_b: str, max_depth: int = 4, use_llm: bool = False,
    ) -> Explanation:
        """Explain how two entities are connected with full provenance.

        Traces the best reasoning chain between two entities and produces
        a human-readable narrative with source citations at every hop.

        Args:
            entity_a: Start entity.
            entity_b: End entity.
            max_depth: Maximum path length to search.
            use_llm: If True, use LLM to generate narrative.

        Returns:
            Explanation with steps, confidence, and narrative.
        """
        a_norm = normalize_entity_id(entity_a)
        b_norm = normalize_entity_id(entity_b)

        chains = self.trace(a_norm, b_norm, max_depth=max_depth, top_k=5)

        if not chains:
            return Explanation(
                entity_a=a_norm,
                entity_b=b_norm,
                connected=False,
                narrative=(
                    f"No connection found between '{a_norm}'"
                    f" and '{b_norm}' within {max_depth} hops."
                ),
            )

        best = chains[0]
        steps: list[ExplanationStep] = []
        all_sources: set[str] = set()

        for hop in best.hops:
            source_ids = hop.source_ids or []
            all_sources.update(source_ids)

            source_summary = ""
            if source_ids:
                source_summary = f"{len(source_ids)} source(s): {', '.join(source_ids[:3])}"
                if len(source_ids) > 3:
                    source_summary += f" (+{len(source_ids) - 3} more)"

            steps.append(ExplanationStep(
                from_entity=hop.from_entity,
                to_entity=hop.to_entity,
                predicate=hop.predicate,
                confidence=hop.confidence,
                source_summary=source_summary,
                evidence_text=hop.evidence_text,
                has_contradiction=hop.has_contradiction,
            ))

        # Build template narrative
        narrative_parts = [
            f"Connection: {a_norm} → {b_norm}"
            f" ({len(steps)} hops,"
            f" confidence={best.chain_confidence:.2f})"
        ]
        for i, step in enumerate(steps, 1):
            line = (
                f"  {i}. {step.from_entity} —[{step.predicate}]→"
                f" {step.to_entity} (conf={step.confidence:.2f})"
            )
            if step.source_summary:
                line += f" [{step.source_summary}]"
            if step.has_contradiction:
                line += " ⚠ contradicted"
            if step.evidence_text:
                line += f'\n     Evidence: "{step.evidence_text[:100]}"'
            narrative_parts.append(line)

        template_narrative = "\n".join(narrative_parts)

        if use_llm:
            prompt = (
                f"Explain in 2-3 sentences how {a_norm} connects to {b_norm}:\n"
                f"{template_narrative}\n"
                f"Be concise and cite sources."
            )
            llm_narrative = self._llm_call(prompt, max_tokens=256)
            narrative = llm_narrative if llm_narrative else template_narrative
        else:
            narrative = template_narrative

        return Explanation(
            entity_a=a_norm,
            entity_b=b_norm,
            connected=True,
            steps=steps,
            chain_confidence=best.chain_confidence,
            narrative=narrative,
            alternative_paths=len(chains) - 1,
            source_count=len(all_sources),
        )

    def forecast(self, entity_id: str, top_k: int = 10) -> Forecast:
        """Predict next connections for an entity.

        Uses graph structure (2-hop neighbors not yet directly connected)
        and historical growth patterns to predict likely future connections.

        Args:
            entity_id: Entity to forecast for.
            top_k: Maximum predictions.

        Returns:
            Forecast with predicted connections and growth trajectory.
        """
        eid = normalize_entity_id(entity_id)
        entity_claims = self.claims_for(eid)

        if not entity_claims:
            return Forecast(entity_id=eid)

        adj = self.get_adjacency_list()
        direct_neighbors = adj.get(eid, set())

        # Entity type and name maps
        entity_map: dict[str, EntitySummary] = {}
        for e in self.list_entities():
            entity_map[e.id] = e

        # 2-hop candidates: neighbors of neighbors not directly connected
        candidates: dict[str, dict] = {}
        for neighbor in direct_neighbors:
            for hop2 in adj.get(neighbor, set()):
                if hop2 != eid and hop2 not in direct_neighbors and hop2 not in candidates:
                    candidates[hop2] = {"bridges": [], "predicates": []}
                if hop2 != eid and hop2 not in direct_neighbors:
                    candidates.setdefault(hop2, {"bridges": [], "predicates": []})
                    candidates[hop2]["bridges"].append(neighbor)
                    # Collect predicates from bridge
                    for c in self.claims_for(neighbor):
                        if c.object.id == hop2 or c.subject.id == hop2:
                            candidates[hop2]["predicates"].append(c.predicate.id)

        # Score candidates
        predictions: list[ForecastConnection] = []
        for cand_id, info in candidates.items():
            bridges = info["bridges"]
            predicates = info["predicates"]

            # Score: more bridges = more likely, weighted by bridge degree
            bridge_count = len(set(bridges))
            bridge_score = min(bridge_count / 3, 1.0)

            # Degree similarity (similar degree entities are more likely to connect)
            cand_degree = len(adj.get(cand_id, set()))
            my_degree = len(direct_neighbors)
            degree_sim = 1.0 - abs(cand_degree - my_degree) / max(cand_degree + my_degree, 1)

            score = 0.6 * bridge_score + 0.4 * degree_sim

            # Predicted predicate: most common among bridge predicates
            pred = "associated_with"
            if predicates:
                from collections import Counter
                pred_counts = Counter(predicates)
                pred = pred_counts.most_common(1)[0][0]

            cand_entity = entity_map.get(cand_id)
            cand_type = cand_entity.entity_type if cand_entity else "entity"

            reason = f"{bridge_count} bridging entities ({', '.join(sorted(set(bridges))[:3])})"

            predictions.append(ForecastConnection(
                target_entity=cand_id,
                target_type=cand_type,
                predicted_predicate=pred,
                score=score,
                reason=reason,
                evidence_entities=sorted(set(bridges))[:5],
            ))

        predictions.sort(key=lambda p: -p.score)
        predictions = predictions[:top_k]

        # Growth trajectory from timestamps
        timestamps = sorted(c.timestamp for c in entity_claims if c.timestamp > 0)
        growth_rate = 0.0
        trajectory = "stable"
        if len(timestamps) >= 2:
            span_days = (timestamps[-1] - timestamps[0]) / 1_000_000_000 / 86400
            if span_days > 0:
                growth_rate = len(timestamps) / max(span_days / 30, 0.1)  # per month
                # Compare first half vs second half
                mid = len(timestamps) // 2
                first_half_span = (timestamps[mid] - timestamps[0]) / 1_000_000_000 / 86400
                second_half_span = (timestamps[-1] - timestamps[mid]) / 1_000_000_000 / 86400
                first_rate = mid / max(first_half_span / 30, 0.1)
                second_rate = (len(timestamps) - mid) / max(second_half_span / 30, 0.1)
                if second_rate > first_rate * 1.3:
                    trajectory = "growing"
                elif second_rate < first_rate * 0.7:
                    trajectory = "declining"

        return Forecast(
            entity_id=eid,
            predictions=predictions,
            growth_rate=growth_rate,
            trajectory=trajectory,
            total_current_connections=len(direct_neighbors),
        )

    def merge_report(self, other: "AttestDB") -> MergeReport:
        """Compare two knowledge bases — what each knows that the other doesn't.

        Args:
            other: Another AttestDB instance to compare against.

        Returns:
            MergeReport with unique/shared beliefs, entity diffs, and conflicts.
        """
        from attestdb.core.confidence import tier2_confidence
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        self_claims = self._all_claims()
        other_claims = other._all_claims()

        # Group by content_id
        self_by_cid: dict[str, list[Claim]] = {}
        for c in self_claims:
            if c.status == ClaimStatus.ACTIVE:
                self_by_cid.setdefault(c.content_id, []).append(c)

        other_by_cid: dict[str, list[Claim]] = {}
        for c in other_claims:
            if c.status == ClaimStatus.ACTIVE:
                other_by_cid.setdefault(c.content_id, []).append(c)

        self_cids = set(self_by_cid.keys())
        other_cids = set(other_by_cid.keys())

        shared = self_cids & other_cids
        self_unique = self_cids - other_cids
        other_unique = other_cids - self_cids

        # Entity comparison
        self_entities = {c.subject.id for c in self_claims} | {c.object.id for c in self_claims}
        other_entities = {c.subject.id for c in other_claims} | {c.object.id for c in other_claims}

        # Find conflicts: opposing predicates on same (subj, obj) across DBs
        conflicts: list[MergeConflict] = []

        # Build self's (subj, obj, pred) index
        self_triples: dict[tuple[str, str], dict[str, list[Claim]]] = {}
        for c in self_claims:
            if c.status == ClaimStatus.ACTIVE:
                key = (c.subject.id, c.object.id)
                self_triples.setdefault(key, {}).setdefault(c.predicate.id, []).append(c)

        for c in other_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pred = c.predicate.id
            opp = OPPOSITE_PREDICATES.get(pred)
            if not opp:
                continue
            key = (c.subject.id, c.object.id)
            if key in self_triples and opp in self_triples[key]:
                # Conflict: other says pred, self says opposite
                self_opp_claims = self_triples[key][opp]
                other_pred_claims = [
                    oc for oc in other_claims
                    if oc.status == ClaimStatus.ACTIVE
                    and oc.subject.id == c.subject.id
                    and oc.object.id == c.object.id
                    and oc.predicate.id == pred
                ]
                if other_pred_claims:
                    # Avoid duplicate conflicts
                    if not any(
                        mc.subject == c.subject.id and mc.object == c.object.id
                        for mc in conflicts
                    ):
                        self_conf = tier2_confidence(self_opp_claims[0], self_opp_claims)
                        other_conf = tier2_confidence(other_pred_claims[0], other_pred_claims)
                        conflicts.append(MergeConflict(
                            content_id=c.content_id,
                            subject=c.subject.id,
                            predicate=f"{opp} (self) vs {pred} (other)",
                            object=c.object.id,
                            self_confidence=self_conf,
                            other_confidence=other_conf,
                            self_sources=len({sc.provenance.source_id for sc in self_opp_claims}),
                            other_sources=len(
                                {oc.provenance.source_id for oc in other_pred_claims}
                            ),
                        ))

        # Summary
        parts = []
        parts.append(f"Self: {len(self_unique)} unique beliefs, {len(self_entities)} entities")
        parts.append(f"Other: {len(other_unique)} unique beliefs, {len(other_entities)} entities")
        shared_ents = len(self_entities & other_entities)
        parts.append(
            f"Shared: {len(shared)} beliefs,"
            f" {shared_ents} entities"
        )
        if conflicts:
            parts.append(f"{len(conflicts)} conflicts")
        summary = "; ".join(parts)

        return MergeReport(
            self_unique_beliefs=len(self_unique),
            other_unique_beliefs=len(other_unique),
            shared_beliefs=len(shared),
            conflicts=conflicts,
            self_unique_entities=sorted(self_entities - other_entities),
            other_unique_entities=sorted(other_entities - self_entities),
            shared_entities=sorted(self_entities & other_entities),
            self_total_claims=len(self_claims),
            other_total_claims=len(other_claims),
            summary=summary,
        )


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
