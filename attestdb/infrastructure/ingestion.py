"""Ingestion pipeline implementing all 13 validation rules."""

from __future__ import annotations

import logging
import time

import jsonschema

from attestdb.core.confidence import (
    tier1_confidence,
    source_confidence_ceiling,
    calibrate_llm_confidence,
    LLM_SOURCE_TYPES,
)
from attestdb.core.errors import (
    CircularProvenanceError,
    ClaimRejectedByQualityGateError,
    DimensionalityError,
    DuplicateClaimError,
    PredicateConstraintError,
    ProvenanceError,
    SchemaValidationError,
    VocabularyError,
)
from attestdb.core.hashing import compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import (
    BatchResult,
    Claim,
    ClaimInput,
    ClaimStatus,
    EntityRef,
    Payload,
    PredicateRef,
    Provenance,
)
from attestdb.core.vocabulary import (
    BUILT_IN_ENTITY_TYPES,
    BUILT_IN_PREDICATE_TYPES,
    BUILT_IN_SOURCE_TYPES,
)
from attestdb.infrastructure.embedding_index import EmbeddingIndex
from attestdb.infrastructure.tracing import span as _span, set_attribute as _set_attr, record_exception as _record_exc

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Validates and ingests claims, enforcing all 13 rules from the API spec."""

    def __init__(
        self,
        store,
        embedding_index: EmbeddingIndex | None = None,
        embedding_dim: int | None = None,
        strict: bool = False,
    ):
        self._store = store
        self._embedding_index = embedding_index
        self._embedding_dim = embedding_dim
        self._strict = strict
        # Optional EntityResolver, set via AttestDB.enable_entity_resolution()
        self._resolver = None
        # Optional auto-embedding callback: (text) -> list[float]
        self._embed_fn: callable | None = None
        # Optional DomainSpec for display name quality warnings
        self._domain_spec = None
        # Runtime entity aliases (registered via AttestDB.add_entity_alias)
        self._entity_aliases: dict[str, str] = {}
        # Vocabulary caches (invalidated via invalidate_vocab_caches)
        self._valid_entity_types: set[str] | None = None
        self._valid_pred_types: set[str] | None = None
        self._valid_source_types: set[str] | None = None
        # Quality gate callbacks: run after structural validation, before persist
        self._quality_gates: list[tuple[callable, str]] = []  # (gate_fn, name)

    def invalidate_vocab_caches(self) -> None:
        """Clear cached vocabulary type sets. Called by AttestDB.register_vocabulary()."""
        self._valid_entity_types = None
        self._valid_pred_types = None
        self._valid_source_types = None

    def register_quality_gate(self, gate: callable, name: str = "") -> None:
        """Register a quality gate callback.

        Gates run in registration order after structural validation (all 13
        rules) but before persistence. A claim must pass ALL gates to be
        ingested.

        gate: callable that takes a ClaimInput and returns:
          ("accept", None) — claim passes this gate
          ("reject", "reason") — claim is rejected, not ingested
          ("flag", "reason") — claim is ingested but flagged in metadata
        name: optional name for logging/debugging
        """
        self._quality_gates.append((gate, name or f"gate_{len(self._quality_gates)}"))
        logger.info("Registered quality gate: %s", name or f"gate_{len(self._quality_gates) - 1}")

    def _get_valid_entity_types(self) -> set[str]:
        if self._valid_entity_types is None:
            types = set(BUILT_IN_ENTITY_TYPES)
            for vocab in self._store.get_registered_vocabularies().values():
                types.update(vocab.get("entity_types", []))
            self._valid_entity_types = types
        return self._valid_entity_types

    def _get_valid_predicate_types(self) -> set[str]:
        if self._valid_pred_types is None:
            types = set(BUILT_IN_PREDICATE_TYPES)
            for vocab in self._store.get_registered_vocabularies().values():
                types.update(vocab.get("predicate_types", []))
            self._valid_pred_types = types
        return self._valid_pred_types

    def _get_valid_source_types(self) -> set[str]:
        if self._valid_source_types is None:
            types = set(BUILT_IN_SOURCE_TYPES)
            for vocab in self._store.get_registered_vocabularies().values():
                types.update(vocab.get("source_types", []))
            self._valid_source_types = types
        return self._valid_source_types

    def _validate_and_build(self, claim_input: ClaimInput) -> tuple[Claim, list[float] | None]:
        """Apply all validation rules and build a Claim object.

        Returns (claim, embedding). Raises on validation failure.
        Does NOT write to the store or track corroboration.
        """
        with _span("attestdb.ingest.validate", {"subject": claim_input.subject[0], "predicate": claim_input.predicate[0]}):
            return self._validate_and_build_inner(claim_input)

    def _validate_and_build_inner(self, claim_input: ClaimInput) -> tuple[Claim, list[float] | None]:
        # Rule 1: Normalize entity IDs + resolve aliases
        from attestdb.core.vocabulary import normalize_entity_name
        subj_canonical = normalize_entity_name(
            claim_input.subject[0], self._entity_aliases
        )
        obj_canonical = normalize_entity_name(
            claim_input.object[0], self._entity_aliases
        )
        if not subj_canonical:
            raise ValueError("Subject entity ID is empty after normalization")
        if not obj_canonical:
            raise ValueError("Object entity ID is empty after normalization")
        subj_type = claim_input.subject[1]
        obj_type = claim_input.object[1]
        pred_id = claim_input.predicate[0]
        pred_type = claim_input.predicate[1]

        # Normalize predicate to controlled vocabulary
        from attestdb.core.vocabulary import normalize_predicate
        pred_id = normalize_predicate(pred_id)

        # Display name quality warning (informational, non-blocking)
        if self._domain_spec is not None:
            if self._domain_spec.looks_opaque(subj_canonical, subj_type):
                logger.warning(
                    "Opaque display name: %s (type=%s) — consider display name resolution",
                    subj_canonical, subj_type,
                )
            if self._domain_spec.looks_opaque(obj_canonical, obj_type):
                logger.warning(
                    "Opaque display name: %s (type=%s) — consider display name resolution",
                    obj_canonical, obj_type,
                )

        # Entity resolution (optional): resolve names to existing entities
        if self._resolver is not None:
            subj_ext = (claim_input.external_ids or {}).get("subject", {})
            obj_ext = (claim_input.external_ids or {}).get("object", {})
            resolved_subj, subj_conf = self._resolver.resolve(
                subj_canonical, subj_type, subj_ext,
            )
            if resolved_subj is not None and resolved_subj != subj_canonical:
                logger.info(
                    "Entity resolved: %r -> %r (conf=%.2f)",
                    subj_canonical, resolved_subj, subj_conf,
                )
                subj_canonical = resolved_subj
            resolved_obj, obj_conf = self._resolver.resolve(
                obj_canonical, obj_type, obj_ext,
            )
            if resolved_obj is not None and resolved_obj != obj_canonical:
                logger.info(
                    "Entity resolved: %r -> %r (conf=%.2f)",
                    obj_canonical, resolved_obj, obj_conf,
                )
                obj_canonical = resolved_obj

        # Parse provenance
        prov_dict = claim_input.provenance
        source_type = str(prov_dict.get("source_type", ""))
        source_id = str(prov_dict.get("source_id", ""))
        if not source_type or not source_id:
            raise ProvenanceError("source_type and source_id are required")

        method = prov_dict.get("method")
        chain = list(prov_dict.get("chain", []))
        model_version = prov_dict.get("model_version")
        organization = prov_dict.get("organization")
        project = prov_dict.get("project")
        agent_id = prov_dict.get("agent_id")
        source_version = prov_dict.get("source_version")
        labels = dict(prov_dict.get("labels", {}))

        timestamp = claim_input.timestamp or int(time.time() * 1_000_000_000)

        # Rule 2: Compute claim_id
        claim_id = compute_claim_id(
            subj_canonical, pred_id, obj_canonical, source_id, source_type, timestamp
        )

        # Rule 3: Compute content_id
        content_id = compute_content_id(subj_canonical, pred_id, obj_canonical)

        # Rule 4: Duplicate check (exact claim_id match)
        if self._store.claim_exists(claim_id):
            raise DuplicateClaimError(f"Claim {claim_id} already exists")

        # Rule 6: Validate source_type
        if source_type not in self._get_valid_source_types():
            if self._strict:
                raise VocabularyError(f"Unknown source_type: {source_type}")
            logger.debug("Custom source_type: %s (allowed in non-strict mode)", source_type)

        # Rule 7: Validate entity types
        valid_entity_types = self._get_valid_entity_types()
        if subj_type not in valid_entity_types:
            if self._strict:
                raise VocabularyError(f"Unknown entity_type for subject: {subj_type}")
            logger.debug("Custom entity_type: %s", subj_type)
        if obj_type not in valid_entity_types:
            if self._strict:
                raise VocabularyError(f"Unknown entity_type for object: {obj_type}")
            logger.debug("Custom entity_type: %s", obj_type)

        # Rule 8: Validate predicate type
        if pred_type not in self._get_valid_predicate_types():
            if self._strict:
                raise VocabularyError(f"Unknown predicate_type: {pred_type}")
            logger.debug("Custom predicate_type: %s", pred_type)

        # Rule 9: Predicate constraints
        constraints = self._store.get_predicate_constraints().get(pred_id)
        if constraints:
            allowed_subj = constraints.get("subject_types", [])
            if allowed_subj and subj_type not in allowed_subj:
                raise PredicateConstraintError(
                    f"Predicate '{pred_id}' does not accept subject type '{subj_type}'. "
                    f"Allowed: {allowed_subj}"
                )
            allowed_obj = constraints.get("object_types", [])
            if allowed_obj and obj_type not in allowed_obj:
                raise PredicateConstraintError(
                    f"Predicate '{pred_id}' does not accept object type '{obj_type}'. "
                    f"Allowed: {allowed_obj}"
                )

        # Rule 10: Payload schema validation
        payload = None
        if claim_input.payload:
            schema_ref = claim_input.payload.get("schema_ref") or claim_input.payload.get("schema", "")
            data = claim_input.payload.get("data", {})
            if schema_ref:
                registered = self._store.get_payload_schemas().get(schema_ref)
                if registered:
                    try:
                        jsonschema.validate(instance=data, schema=registered)
                    except jsonschema.ValidationError as e:
                        raise SchemaValidationError(
                            f"Payload validation failed for schema '{schema_ref}': {e.message}"
                        ) from e
            payload = Payload(schema_ref=schema_ref, data=data)

        # Rule 11: Provenance chain existence
        for ref_id in chain:
            if not self._store.claim_exists(ref_id):
                raise ProvenanceError(
                    f"Provenance chain references non-existent claim: {ref_id}"
                )

        # Rule 12: Provenance DAG acyclicity
        if chain:
            self._check_provenance_acyclicity(claim_id, chain)

        # Rule 13: Embedding dimensionality
        embedding = claim_input.embedding
        if not embedding and self._embed_fn and self._embedding_dim:
            # Auto-embed: build text from subject + predicate + object
            text = f"{claim_input.subject[0]} {claim_input.predicate[0]} {claim_input.object[0]}"
            try:
                embedding = self._embed_fn(text)
            except Exception as e:
                logger.warning("Auto-embedding failed: %s", e)
                embedding = None
        if embedding and self._embedding_dim:
            if len(embedding) != self._embedding_dim:
                raise DimensionalityError(
                    f"Expected {self._embedding_dim}-dim embedding, got {len(embedding)}-dim"
                )

        # Compute confidence — calibrate LLM sources, cap by source ceiling
        confidence = claim_input.confidence
        if confidence is None:
            confidence = tier1_confidence(source_type)
        else:
            # Calibrate LLM source types (compresses overconfident values)
            if source_type in LLM_SOURCE_TYPES:
                confidence = calibrate_llm_confidence(confidence)
            # Cap by source type ceiling
            ceiling = source_confidence_ceiling(source_type)
            if confidence > ceiling:
                confidence = ceiling

        subj_display = claim_input.subject[0]
        obj_display = claim_input.object[0]
        subj_ext = (claim_input.external_ids or {}).get("subject", {})
        obj_ext = (claim_input.external_ids or {}).get("object", {})

        # Rule 4b: Content-level dedup — same fact from same source is a duplicate,
        # not corroboration. Without this, re-submitting the same claim (e.g. via API
        # retry or migration re-run) inflates counts and fakes corroboration.
        # Placed after all other validation so that genuinely invalid claims still
        # fail with the correct error (e.g. SchemaValidationError, not DuplicateClaimError).
        existing_by_content = self._store.claims_by_content_id(content_id)
        for existing_claim in existing_by_content:
            prov = existing_claim.get("provenance", {})
            ex_source_id = prov.get("source_id", "") if isinstance(prov, dict) else ""
            if ex_source_id == source_id:
                raise DuplicateClaimError(
                    f"Claim with content_id {content_id} from source {source_id} already exists"
                )

        # Compute expires_at from ttl_seconds
        ttl_seconds = getattr(claim_input, "ttl_seconds", 0) or 0
        expires_at = 0
        if ttl_seconds > 0:
            expires_at = timestamp + ttl_seconds * 1_000_000_000

        namespace = getattr(claim_input, "namespace", "") or ""
        # Auto-populate namespace from project when namespace is empty
        if not namespace and project:
            namespace = str(project)

        claim = Claim(
            claim_id=claim_id,
            content_id=content_id,
            subject=EntityRef(
                id=subj_canonical, entity_type=subj_type,
                display_name=subj_display, external_ids=subj_ext,
            ),
            predicate=PredicateRef(id=pred_id, predicate_type=pred_type),
            object=EntityRef(
                id=obj_canonical, entity_type=obj_type,
                display_name=obj_display, external_ids=obj_ext,
            ),
            confidence=confidence,
            provenance=Provenance(
                source_type=source_type,
                source_id=source_id,
                method=str(method) if method else None,
                chain=chain,
                model_version=str(model_version) if model_version else None,
                organization=str(organization) if organization else None,
                project=str(project) if project else None,
                agent_id=str(agent_id) if agent_id else None,
                source_version=str(source_version) if source_version else None,
                labels=labels,
            ),
            payload=payload,
            timestamp=timestamp,
            status=ClaimStatus.ACTIVE,
            namespace=namespace,
            expires_at=expires_at,
        )
        return claim, embedding

    def _persist(self, claim: Claim, embedding: list[float] | None) -> None:
        """Write validated claim to store and embedding index."""
        with _span("attestdb.ingest.persist", {"claim_id": claim.claim_id}):
            self._persist_inner(claim, embedding)

    def _persist_inner(self, claim: Claim, embedding: list[float] | None) -> None:
        self._store.upsert_entity(
            claim.subject.id, claim.subject.entity_type,
            claim.subject.display_name, claim.subject.external_ids, claim.timestamp,
        )
        self._store.upsert_entity(
            claim.object.id, claim.object.entity_type,
            claim.object.display_name, claim.object.external_ids, claim.timestamp,
        )
        self._store.insert_claim(claim)

        # Keep resolver index current with new external_ids
        if self._resolver is not None:
            for ns, eid in (claim.subject.external_ids or {}).items():
                self._resolver.register_external_id(claim.subject.id, ns, eid)
            for ns, eid in (claim.object.external_ids or {}).items():
                self._resolver.register_external_id(claim.object.id, ns, eid)

        if embedding is not None and self._embedding_index is not None:
            self._embedding_index.add(claim.claim_id, embedding)

    def ingest(self, claim_input: ClaimInput) -> str:
        """Ingest a single claim, applying all 13 validation rules. Returns claim_id."""
        claim, embedding = self._validate_and_build(claim_input)

        # Quality gates — run after structural validation, before persistence
        for gate_fn, gate_name in self._quality_gates:
            decision, reason = gate_fn(claim_input)
            if decision == "reject":
                raise ClaimRejectedByQualityGateError(
                    f"Claim rejected by quality gate '{gate_name}': {reason}"
                )
            elif decision == "flag":
                if claim.payload is None:
                    claim.payload = Payload(schema_ref="", data={})
                flags = claim.payload.data.get("quality_flags", [])
                flags.append(reason)
                claim.payload.data["quality_flags"] = flags

        # Rule 5: Corroboration tracking
        existing = self._store.claims_by_content_id(claim.content_id)
        if existing:
            logger.info(
                "Claim %s corroborates existing content %s (now %d sources)",
                claim.claim_id, claim.content_id, len(existing) + 1,
            )

        self._persist(claim, embedding)
        return claim.claim_id

    def _check_provenance_acyclicity(self, new_claim_id: str, chain: list[str]) -> None:
        """Ensure adding this claim doesn't create a cycle in the provenance DAG."""
        visited: set[str] = set()
        stack = list(chain)
        while stack:
            current = stack.pop()
            if current == new_claim_id:
                raise CircularProvenanceError(
                    f"Provenance chain creates a cycle involving {new_claim_id}"
                )
            if current in visited:
                continue
            visited.add(current)
            upstream = self._store.get_claim_provenance_chain(current)
            stack.extend(upstream)

    def ingest_batch(
        self,
        claims: list[ClaimInput],
        on_ingested: "Callable[[str], None] | None" = None,  # noqa: F821
    ) -> BatchResult:
        """Batch ingestion — validates and persists without per-claim corroboration tracking.

        Batches entity upserts into a single Rust/LMDB transaction when
        ``upsert_entities_batch`` is available, reducing transaction count
        from 3N to N+1 for N claims.

        Args:
            on_ingested: Optional callback invoked with each claim_id on successful persist.
        """
        with _span("attestdb.ingest_batch", {"batch_size": len(claims)}):
            return self._ingest_batch_inner(claims, on_ingested)

    def _ingest_batch_inner(
        self,
        claims: list[ClaimInput],
        on_ingested: "Callable[[str], None] | None" = None,  # noqa: F821
    ) -> BatchResult:
        self._store.warm_caches()
        result = BatchResult()

        # Phase 1: Validate all claims
        validated: list[tuple[Claim, list[float] | None]] = []
        with _span("attestdb.ingest_batch.validate", {"batch_size": len(claims)}):
            for ci in claims:
                try:
                    claim, embedding = self._validate_and_build_inner(ci)
                    validated.append((claim, embedding))
                except DuplicateClaimError:
                    result.duplicates += 1
                except Exception as e:
                    result.errors.append(str(e))

        if not validated:
            _set_attr("validated_count", 0)
            return result

        _set_attr("validated_count", len(validated))

        # Phase 2: Batch persist — entities + claims in 2 LMDB transactions
        use_batch = hasattr(self._store, "insert_claims_batch")
        if use_batch:
            import json as _json
            with _span("attestdb.ingest_batch.persist", {"claim_count": len(validated)}):
                # 2a. Batch entity upserts (1 transaction)
                entities: dict[str, tuple[str, str, str]] = {}
                max_ts = 0
                for claim, _ in validated:
                    subj, obj = claim.subject, claim.object
                    if subj.id not in entities:
                        ext = _json.dumps(subj.external_ids) if subj.external_ids else "{}"
                        entities[subj.id] = (subj.entity_type, subj.display_name, ext)
                    if obj.id not in entities:
                        ext = _json.dumps(obj.external_ids) if obj.external_ids else "{}"
                        entities[obj.id] = (obj.entity_type, obj.display_name, ext)
                    if claim.timestamp > max_ts:
                        max_ts = claim.timestamp
                self._store.upsert_entities_batch(entities, max_ts)

                # 2b. Batch claim inserts (1 transaction per 100K chunk)
                claim_objects = [claim for claim, _ in validated]
                result.ingested = self._store.insert_claims_batch(claim_objects)

            # 2c. Embeddings, callbacks, resolver (Python-side)
            for claim, embedding in validated:
                if embedding is not None and self._embedding_index is not None:
                    self._embedding_index.add(claim.claim_id, embedding)
                if on_ingested is not None:
                    on_ingested(claim.claim_id)
                if self._resolver is not None:
                    for ns, eid in (claim.subject.external_ids or {}).items():
                        self._resolver.register_external_id(claim.subject.id, ns, eid)
                    for ns, eid in (claim.object.external_ids or {}).items():
                        self._resolver.register_external_id(claim.object.id, ns, eid)
        else:
            # Fallback: per-claim persist (old wheel without insert_claims_batch)
            for claim, embedding in validated:
                self._persist(claim, embedding)
                result.ingested += 1
                if on_ingested is not None:
                    on_ingested(claim.claim_id)

        return result


# ---------------------------------------------------------------------------
# Built-in quality gates
# ---------------------------------------------------------------------------


def min_entity_length_gate(min_chars: int = 2) -> callable:
    """Reject claims where subject or object entity IDs are too short."""
    def gate(claim_input: ClaimInput) -> tuple[str, str | None]:
        subj_id = claim_input.subject[0]
        obj_id = claim_input.object[0]
        if len(subj_id) < min_chars or len(obj_id) < min_chars:
            return ("reject", f"Entity ID too short (min {min_chars} chars)")
        return ("accept", None)
    return gate


def no_self_reference_gate() -> callable:
    """Reject claims where subject and object are the same entity."""
    def gate(claim_input: ClaimInput) -> tuple[str, str | None]:
        from attestdb.core.normalization import normalize_entity_id
        subj_norm = normalize_entity_id(claim_input.subject[0])
        obj_norm = normalize_entity_id(claim_input.object[0])
        if subj_norm == obj_norm:
            return ("reject", "Self-referencing claim (subject == object)")
        return ("accept", None)
    return gate
