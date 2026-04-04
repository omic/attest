"""Read-only time-travel view of an AttestDB at a given timestamp."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import Claim, ContextFrame, claim_from_dict
from attestdb.infrastructure.query_engine import QueryEngine

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB


# ---------------------------------------------------------------------------
# Filtered store proxy
# ---------------------------------------------------------------------------

class TemporalStoreView:
    """Wraps a Rust store and filters claim-returning methods by timestamp.

    Non-claim methods (get_entity, stats, list_entities, …) pass through
    unchanged via ``__getattr__``.
    """

    __slots__ = ("_store", "_max_ts")

    def __init__(self, store, max_timestamp: int):
        self._store = store
        self._max_ts = max_timestamp

    # -- helpers --

    def _filter(self, raw_claims: list[dict]) -> list[dict]:
        ts = self._max_ts
        return [c for c in raw_claims if c.get("timestamp", 0) <= ts]

    # -- claim-returning methods (filtered) --

    def claims_for(self, entity, pred=None, source=None, min_conf=0.0, limit=0):
        raw = self._store.claims_for(entity, pred, source, min_conf, limit)
        return self._filter(raw)

    def claims_by_content_id(self, content_id):
        return self._filter(self._store.claims_by_content_id(content_id))

    def claims_by_source_id(self, source_id):
        return self._filter(self._store.claims_by_source_id(source_id))

    def claims_by_predicate_id(self, predicate_id):
        return self._filter(self._store.claims_by_predicate_id(predicate_id))

    def claims_in_range(self, start, end):
        return self._store.claims_in_range(start, min(end, self._max_ts))

    def all_claims(self, offset=0, limit=1000):
        return self._filter(self._store.all_claims(offset, limit))

    def get_claim(self, claim_id):
        d = self._store.get_claim(claim_id)
        if d is None:
            return None
        if d.get("timestamp", 0) > self._max_ts:
            return None
        return d

    # -- pass-through for everything else --

    def __getattr__(self, name):
        return getattr(self._store, name)


# ---------------------------------------------------------------------------
# Snapshot query engine
# ---------------------------------------------------------------------------

class SnapshotQueryEngine(QueryEngine):
    """QueryEngine that filters claims by timestamp, giving a point-in-time view."""

    def __init__(self, store, timestamp: int, claim_converter=None):
        super().__init__(store, claim_converter=claim_converter)
        self._timestamp = timestamp

    def query(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = 500,
        max_tokens: int = 4000,
    ) -> ContextFrame:
        return self._execute_query(
            focal_entity, depth=depth, min_confidence=min_confidence,
            exclude_source_types=exclude_source_types,
            max_claims=max_claims, max_tokens=max_tokens,
            claim_filter=lambda c: c.timestamp <= self._timestamp,
            include_quantitative=False,
            include_contradictions=False,
            include_narrative=False,
        )


# ---------------------------------------------------------------------------
# Snapshot (read-only time-travel view)
# ---------------------------------------------------------------------------

class AttestDBSnapshot:
    """Read-only time-travel view.  Wraps AttestDB, filtering claims by timestamp.

    Exposes the same data-access interface that ``AnalyticsEngine`` uses, so
    ``AnalyticsEngine(snapshot)`` works transparently — every analytics method
    (``what_if``, ``predict``, ``evolution``, etc.) automatically operates on
    the point-in-time state.
    """

    def __init__(self, db: "AttestDB", timestamp: int):
        self._db = db
        self._timestamp = timestamp

        # Filtered store proxy — AnalyticsEngine reads self.db._store
        self._store = TemporalStoreView(db._store, timestamp)

        # Query engine with timestamp filter
        self._query_engine = SnapshotQueryEngine(
            self._store, timestamp,
            claim_converter=db._query_engine._convert_claim,
        )
        decay_cfg = getattr(db, "_decay_config", None)
        if decay_cfg:
            self._query_engine.set_decay_config(decay_cfg)

        # Own cache so temporal queries don't pollute the live db cache
        self._cache: dict[str, object] = {}
        self._cache_ts: dict[str, float] = {}

        # Analytics engine bound to *this* snapshot
        from attestdb.infrastructure.analytics import AnalyticsEngine
        self._analytics = AnalyticsEngine(self)

    # -- timestamp filter helper --

    def _ts_filter(self, claim: Claim) -> bool:
        return claim.timestamp <= self._timestamp

    # ------------------------------------------------------------------ #
    # Claim-returning methods (filtered by timestamp)                      #
    # ------------------------------------------------------------------ #

    def get_claim(self, claim_id: str) -> "Claim | None":
        d = self._store.get_claim(claim_id)
        if d is None:
            return None
        return claim_from_dict(d)

    def _all_claims(self) -> list[Claim]:
        return [claim_from_dict(d) for d in self._store.all_claims()]

    def iter_claims(self, batch_size: int = 5000, exclude_expired: bool = False):
        now_ns = int(time.time() * 1_000_000_000) if exclude_expired else 0
        offset = 0
        while True:
            batch = self._store.all_claims(offset, batch_size)
            if not batch:
                break
            for d in batch:
                claim = claim_from_dict(d)
                if exclude_expired and claim.expires_at and claim.expires_at <= now_ns:
                    continue
                yield claim
            if len(batch) < batch_size:
                break
            offset += batch_size

    def claims_for(
        self,
        entity_id: str,
        predicate_type: str | None = None,
        source_type: str | None = None,
        min_confidence: float = 0.0,
        principal=None,
    ) -> list[Claim]:
        return [
            claim_from_dict(d)
            for d in self._store.claims_for(entity_id, predicate_type, source_type, min_confidence)
        ]

    def claims_by_content_id(self, content_id: str) -> list[Claim]:
        return [claim_from_dict(d) for d in self._store.claims_by_content_id(content_id)]

    def claims_by_source_id(self, source_id: str) -> list[Claim]:
        return [claim_from_dict(d) for d in self._store.claims_by_source_id(source_id)]

    def claims_for_predicate(self, predicate_id: str) -> list[Claim]:
        return [claim_from_dict(d) for d in self._store.claims_by_predicate_id(predicate_id)]

    def text_search(
        self,
        query: str,
        entity_type: str | None = None,
        min_confidence: float = 0.0,
        top_k: int = 20,
    ) -> list[Claim]:
        # Delegate to parent's text_search logic, then filter
        from attestdb.core.types import entity_summary_from_dict
        raw_entities = self._store.search_entities(query, top_k * 3)
        entities = [entity_summary_from_dict(d) for d in raw_entities]
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        seen: set[str] = set()
        results: list[Claim] = []
        for entity in entities:
            raw_claims = self._store.claims_for(entity.id, None, None, 0.0)
            for d in raw_claims[:500]:
                claim = claim_from_dict(d)
                if claim.claim_id in seen:
                    continue
                if claim.confidence < min_confidence:
                    continue
                seen.add(claim.claim_id)
                results.append(claim)
                if len(results) >= top_k:
                    return results
        return results

    # ------------------------------------------------------------------ #
    # Non-claim data methods (delegate to parent db)                       #
    # ------------------------------------------------------------------ #

    def query(self, focal_entity: str, **kwargs) -> ContextFrame:
        return self._query_engine.query(focal_entity, **kwargs)

    def stats(self) -> dict:
        """Point-in-time stats."""
        claims = self._store.claims_in_range(0, self._timestamp)
        entities: set[str] = set()
        for c in claims:
            subj = c.get("subject")
            obj = c.get("object")
            if isinstance(subj, dict):
                entities.add(subj.get("id", ""))
            elif isinstance(subj, str):
                entities.add(subj)
            if isinstance(obj, dict):
                entities.add(obj.get("id", ""))
            elif isinstance(obj, str):
                entities.add(obj)
        entities.discard("")
        return {
            "total_claims": len(claims),
            "entity_count": len(entities),
            "timestamp": self._timestamp,
        }

    def path_exists(self, entity_a: str, entity_b: str, max_depth: int = 3) -> bool:
        return self._db.path_exists(entity_a, entity_b, max_depth)

    def find_paths(self, entity_a, entity_b, max_depth=3, top_k=5):
        return self._query_engine.find_paths(entity_a, entity_b, max_depth, top_k)

    def get_adjacency_list(self):
        return self._store.get_adjacency_list()

    def get_weighted_adjacency(self):
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
                e = adj[pair]
                e["max_confidence"] = max(e["max_confidence"], claim.confidence)
                e["source_types"].add(claim.provenance.source_type)
                e["claim_count"] += 1
                e["predicates"].add(claim.predicate.id)
        return adj

    def search_entities(self, query, top_k=10):
        return self._db.search_entities(query, top_k)

    def list_entities(self, *args, **kwargs):
        return self._db.list_entities(*args, **kwargs)

    def get_entity(self, entity_id):
        return self._db.get_entity(entity_id)

    def get_embedding(self, key):
        return self._db.get_embedding(key)

    def find_bridges(self, **kwargs):
        return self._db.find_bridges(**kwargs)

    def find_gaps(self, expected_patterns, **kwargs):
        return self._db.find_gaps(expected_patterns, **kwargs)

    def find_confidence_alerts(self, **kwargs):
        return self._db.find_confidence_alerts(**kwargs)

    def blindspots(self, **kwargs):
        return self._db.blindspots(**kwargs)

    def trace_downstream(self, claim_id):
        return self._db.trace_downstream(claim_id)

    def cross_domain_bridges(self, *args, **kwargs):
        return getattr(self._db, "cross_domain_bridges", lambda *a, **kw: [])(*args, **kwargs)

    # ------------------------------------------------------------------ #
    # Internal methods used by AnalyticsEngine                             #
    # ------------------------------------------------------------------ #

    def _set_include_retracted(self, include: bool) -> None:
        self._db._set_include_retracted(include)

    def _build_reverse_provenance_index(self):
        return self._db._build_reverse_provenance_index()

    def _llm_call(self, *args, **kwargs):
        return self._db._llm_call(*args, **kwargs)

    def _get_researcher(self):
        return self._db._get_researcher()

    def _cluster_entities(self, *args, **kwargs):
        return self._db._cluster_entities(*args, **kwargs)

    def _label_cluster(self, *args, **kwargs):
        return self._db._label_cluster(*args, **kwargs)

    def _template_summary(self, *args, **kwargs):
        return self._db._template_summary(*args, **kwargs)

    def _gather_consensus_evidence(self, *args, **kwargs):
        return self._db._gather_consensus_evidence(*args, **kwargs)

    def _simulate_retract_source(self, source_id):
        return self._db._simulate_retract_source(source_id)

    def _simulate_add_claim(self, claim_input):
        return self._db._simulate_add_claim(claim_input)

    def _simulate_remove_entity(self, entity_id):
        return self._db._simulate_remove_entity(entity_id)

    @property
    def _db_path(self):
        return self._db._db_path

    @property
    def _topology(self):
        return getattr(self._db, "_topology", None)

    def _parse_timestamp(self, ts):
        return self._analytics._parse_timestamp(ts)

    def trace(self, entity_a, entity_b, max_depth=3, top_k=5):
        return self._db.trace(entity_a, entity_b, max_depth, top_k)

    def ingest(self, *args, **kwargs):
        raise RuntimeError("Cannot ingest into a read-only temporal snapshot")

    def ingest_batch(self, *args, **kwargs):
        raise RuntimeError("Cannot ingest into a read-only temporal snapshot")

    # ------------------------------------------------------------------ #
    # Analytics delegation                                                 #
    # ------------------------------------------------------------------ #

    def what_if(self, subject, predicate, object, confidence=0.6, extra_causal_predicates=None):
        return self._analytics.what_if(subject, predicate, object, confidence, extra_causal_predicates)

    def predict(self, entity_id, *args, **kwargs):
        return self._analytics.predict(entity_id, *args, **kwargs)

    def evolution(self, entity_id, since=None):
        return self._analytics.evolution(entity_id, since)

    def diff(self, since, until=None):
        return self._analytics.diff(since, until)

    def hypothetical(self, claim):
        return self._analytics.hypothetical(claim)

    def simulate(self, *args, **kwargs):
        return self._analytics.simulate(*args, **kwargs)

    def forecast(self, entity_id, top_k=5):
        return self._analytics.forecast(entity_id, top_k)

    def explain_why(self, *args, **kwargs):
        return self._analytics.explain_why(*args, **kwargs)

    def impact(self, source_id):
        return self._analytics.impact(source_id)

    def consensus(self, topic):
        return self._analytics.consensus(topic)

    def fragile(self, *args, **kwargs):
        return self._analytics.fragile(*args, **kwargs)

    def stale(self, *args, **kwargs):
        return self._analytics.stale(*args, **kwargs)

    def audit(self, claim_id):
        return self._analytics.audit(claim_id)

    def drift(self, days=30):
        return self._analytics.drift(days)

    def source_reliability(self, source_id):
        return self._analytics.source_reliability(source_id)

    def quality_report(self, *args, **kwargs):
        return self._analytics.quality_report(*args, **kwargs)

    def knowledge_health(self):
        return self._analytics.knowledge_health()

    def suggest_investigations(self, top_k=5):
        return self._analytics.suggest_investigations(top_k)

    def discover(self, top_k=5):
        return self._analytics.discover(top_k)

    def compile(self, topic, *args, **kwargs):
        return self._analytics.compile(topic, *args, **kwargs)

    def corroboration_report(self, min_sources=2):
        return self._analytics.corroboration_report(min_sources)

    def diagnose_corroboration(self, content_id):
        return self._analytics.diagnose_corroboration(content_id)
