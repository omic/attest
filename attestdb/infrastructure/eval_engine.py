"""EvalEngine — domain evaluation set generation from the knowledge graph."""

from __future__ import annotations

import hashlib
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.types import (
    Claim,
    EvalItem,
    EvalResult,
    EvalSet,
    claim_from_dict,
    entity_summary_from_dict,
)


class EvalEngine:
    """Generates domain-specific evaluation sets from the knowledge graph.

    The key insight: AttestDB already has ground truth — real, provenanced
    claims.  We sample from the graph to create benchmarks, then score
    agent answers against expected answers.
    """

    def __init__(self, db: "AttestDB"):
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_eval(
        self,
        domains: list[str] | None = None,
        n: int = 50,
        difficulty: str = "mixed",
        seed: int = 42,
    ) -> EvalSet:
        """Generate an eval set from the knowledge graph.

        Samples real claims and creates questions of varying difficulty:
        - easy: entity_recall, relationship
        - medium: multi_hop, contradiction
        - hard: provenance, holdout
        """
        rng = random.Random(seed)

        # Determine item counts per difficulty
        if difficulty == "easy":
            n_easy, n_medium, n_hard = n, 0, 0
        elif difficulty == "medium":
            n_easy, n_medium, n_hard = 0, n, 0
        elif difficulty == "hard":
            n_easy, n_medium, n_hard = 0, 0, n
        else:  # mixed
            n_easy = int(n * 0.4)
            n_medium = int(n * 0.4)
            n_hard = n - n_easy - n_medium

        # Load candidate claims from the store, optionally filtered by domain
        all_claims = self._load_candidate_claims(domains)
        if not all_claims:
            eval_id = hashlib.sha256(f"eval:{seed}:{time.time_ns()}".encode()).hexdigest()[:16]
            return EvalSet(eval_id=eval_id, created_at=time.time_ns())

        items: list[EvalItem] = []

        # Easy items
        if n_easy > 0:
            easy_items = self._generate_easy(all_claims, n_easy, domains, rng)
            items.extend(easy_items)

        # Medium items
        if n_medium > 0:
            medium_items = self._generate_medium(all_claims, n_medium, domains, rng)
            items.extend(medium_items)

        # Hard items
        if n_hard > 0:
            hard_items = self._generate_hard(all_claims, n_hard, domains, rng)
            items.extend(hard_items)

        eval_id = hashlib.sha256(
            f"eval:{seed}:{len(items)}:{time.time_ns()}".encode()
        ).hexdigest()[:16]

        return EvalSet(
            eval_id=eval_id,
            items=items,
            domains=domains or [],
            created_at=time.time_ns(),
            version=1,
            n_easy=sum(1 for i in items if i.difficulty == "easy"),
            n_medium=sum(1 for i in items if i.difficulty == "medium"),
            n_hard=sum(1 for i in items if i.difficulty == "hard"),
        )

    def score_eval(
        self,
        eval_set: EvalSet,
        agent_answers: list[str],
        agent_id: str = "unknown",
    ) -> EvalResult:
        """Score agent answers against expected answers.

        Uses fuzzy matching for text, numeric tolerance for numbers,
        exact match for yes/no.
        """
        start = time.time()
        items = eval_set.items
        total = len(items)
        correct = 0
        per_item: list[dict] = []
        domain_correct: dict[str, int] = {}
        domain_total: dict[str, int] = {}
        type_correct: dict[str, int] = {}
        type_total: dict[str, int] = {}

        for i, item in enumerate(items):
            agent_answer = agent_answers[i] if i < len(agent_answers) else ""
            is_correct = self._check_answer(item.expected_answer, agent_answer)
            if is_correct:
                correct += 1

            per_item.append({
                "question": item.question,
                "expected": item.expected_answer,
                "agent_answer": agent_answer,
                "correct": is_correct,
                "difficulty": item.difficulty,
                "eval_type": item.eval_type,
                "domain": item.domain,
            })

            # Track by domain
            dom = item.domain or "general"
            domain_total[dom] = domain_total.get(dom, 0) + 1
            if is_correct:
                domain_correct[dom] = domain_correct.get(dom, 0) + 1

            # Track by eval_type
            et = item.eval_type or "unknown"
            type_total[et] = type_total.get(et, 0) + 1
            if is_correct:
                type_correct[et] = type_correct.get(et, 0) + 1

        score = correct / total if total > 0 else 0.0
        elapsed = time.time() - start
        ts = time.time_ns()

        domain_scores = {
            dom: domain_correct.get(dom, 0) / domain_total[dom]
            for dom in domain_total
        }
        type_scores = {
            et: type_correct.get(et, 0) / type_total[et]
            for et in type_total
        }

        result = EvalResult(
            eval_id=eval_set.eval_id,
            agent_id=agent_id,
            score=score,
            total=total,
            correct=correct,
            per_item=per_item,
            domain_scores=domain_scores,
            type_scores=type_scores,
            elapsed=elapsed,
            timestamp=ts,
        )

        # Persist result as a claim in the graph
        self._persist_result(result)

        return result

    def eval_history(self, agent_id: str | None = None) -> list[dict]:
        """Get eval results from the graph (stored as claims)."""
        results = []
        raw_claims = self.db._store.claims_by_predicate_id("eval_scored")
        for d in raw_claims:
            claim = claim_from_dict(d)
            if agent_id and claim.subject.id != agent_id:
                continue
            entry = {
                "agent_id": claim.subject.id,
                "eval_id": claim.object.id,
                "timestamp": claim.timestamp,
                "confidence": claim.confidence,
            }
            if claim.payload and claim.payload.data:
                entry.update(claim.payload.data)
            results.append(entry)
        return results

    # ------------------------------------------------------------------
    # Internal — claim loading
    # ------------------------------------------------------------------

    def _load_candidate_claims(self, domains: list[str] | None) -> list[Claim]:
        """Load claims from the store, optionally filtered by entity type domain."""
        claims: list[Claim] = []
        # Page through all claims (cap at 10K to avoid memory issues on huge DBs)
        batch = self.db._store.all_claims(0, 10_000)
        for d in batch:
            claim = claim_from_dict(d)
            # Filter out eval/agent metadata claims
            if claim.provenance.source_type == "eval":
                continue
            if claim.predicate.id in ("eval_scored", "has_domain", "has_capability", "uses_model"):
                continue
            if domains:
                subj_type = claim.subject.entity_type
                obj_type = claim.object.entity_type
                if subj_type not in domains and obj_type not in domains:
                    continue
            claims.append(claim)
        return claims

    # ------------------------------------------------------------------
    # Internal — easy generators
    # ------------------------------------------------------------------

    def _generate_easy(
        self, claims: list[Claim], n: int, domains: list[str] | None, rng: random.Random
    ) -> list[EvalItem]:
        """Generate easy items: entity_recall and relationship."""
        items: list[EvalItem] = []
        half = max(1, n // 2)

        # Entity recall questions
        entity_map: dict[str, tuple[str, str, int]] = {}  # id → (name, type, claim_count)
        for c in claims:
            eid = c.subject.id
            ename = c.subject.display_name or c.subject.id
            etype = c.subject.entity_type
            if eid not in entity_map:
                entity_map[eid] = (ename, etype, 0)
            old = entity_map[eid]
            entity_map[eid] = (old[0], old[1], old[2] + 1)

        # Pick entities with at least 2 claims for entity recall
        eligible = [(eid, info) for eid, info in entity_map.items() if info[2] >= 2]
        if eligible:
            rng.shuffle(eligible)
            for eid, (ename, etype, _count) in eligible[:half]:
                display = ename if ename != eid else eid
                items.append(EvalItem(
                    question=f"What type of entity is '{display}'?",
                    expected_answer=etype,
                    supporting_claims=[],
                    difficulty="easy",
                    domain=etype if domains else "",
                    eval_type="entity_recall",
                ))
                if len(items) >= half:
                    break

        # Relationship questions
        remaining = n - len(items)
        if remaining > 0 and claims:
            shuffled = list(claims)
            rng.shuffle(shuffled)
            for c in shuffled[:remaining]:
                subj_name = c.subject.display_name or c.subject.id
                obj_name = c.object.display_name or c.object.id
                items.append(EvalItem(
                    question=f"What is the relationship between '{subj_name}' and '{obj_name}'?",
                    expected_answer=c.predicate.id,
                    supporting_claims=[c.claim_id],
                    difficulty="easy",
                    domain=c.subject.entity_type if domains else "",
                    eval_type="relationship",
                ))

        return items[:n]

    # ------------------------------------------------------------------
    # Internal — medium generators
    # ------------------------------------------------------------------

    def _generate_medium(
        self, claims: list[Claim], n: int, domains: list[str] | None, rng: random.Random
    ) -> list[EvalItem]:
        """Generate medium items: multi_hop and contradiction."""
        items: list[EvalItem] = []
        half = max(1, n // 2)

        # Multi-hop: find A→B→C paths
        # Build adjacency: entity → list of (claim, target_entity)
        adjacency: dict[str, list[tuple[Claim, str]]] = {}
        for c in claims:
            sid = c.subject.id
            oid = c.object.id
            if sid not in adjacency:
                adjacency[sid] = []
            adjacency[sid].append((c, oid))

        # Find 2-hop paths
        entity_ids = list(adjacency.keys())
        rng.shuffle(entity_ids)
        for start_id in entity_ids:
            if len(items) >= half:
                break
            neighbors = adjacency.get(start_id, [])
            if not neighbors:
                continue
            rng.shuffle(neighbors)
            for claim_a, mid_id in neighbors:
                if mid_id == start_id:
                    continue
                second_hop = adjacency.get(mid_id, [])
                if not second_hop:
                    continue
                rng.shuffle(second_hop)
                for claim_b, end_id in second_hop:
                    if end_id == start_id or end_id == mid_id:
                        continue
                    subj_name = claim_a.subject.display_name or claim_a.subject.id
                    mid_name = claim_a.object.display_name or claim_a.object.id
                    end_name = claim_b.object.display_name or claim_b.object.id
                    pred_a = claim_a.predicate.id
                    pred_b = claim_b.predicate.id
                    items.append(EvalItem(
                        question=(
                            f"If '{subj_name}' {pred_a} '{mid_name}' and "
                            f"'{mid_name}' {pred_b} '{end_name}', what can we "
                            f"infer about the relationship chain from "
                            f"'{subj_name}' to '{end_name}'?"
                        ),
                        expected_answer=f"{pred_a}, {pred_b}",
                        supporting_claims=[claim_a.claim_id, claim_b.claim_id],
                        difficulty="medium",
                        domain=claim_a.subject.entity_type if domains else "",
                        eval_type="multi_hop",
                    ))
                    break  # one per start→mid
                if len(items) >= half:
                    break

        # Contradiction questions
        remaining = n - len(items)
        if remaining > 0:
            contradiction_items = self._generate_contradiction_items(claims, remaining, domains, rng)
            items.extend(contradiction_items)

        return items[:n]

    def _generate_contradiction_items(
        self, claims: list[Claim], n: int, domains: list[str] | None, rng: random.Random
    ) -> list[EvalItem]:
        """Find entity pairs with opposing predicates."""
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        items: list[EvalItem] = []
        # Group claims by (subject_id, object_id)
        pair_claims: dict[tuple[str, str], list[Claim]] = {}
        for c in claims:
            key = (c.subject.id, c.object.id)
            if key not in pair_claims:
                pair_claims[key] = []
            pair_claims[key].append(c)

        pairs = list(pair_claims.keys())
        rng.shuffle(pairs)

        for subj_id, obj_id in pairs:
            if len(items) >= n:
                break
            group = pair_claims[(subj_id, obj_id)]
            predicates = {c.predicate.id for c in group}
            for pred in predicates:
                opposite = OPPOSITE_PREDICATES.get(pred)
                if opposite and opposite in predicates:
                    subj_name = group[0].subject.display_name or subj_id
                    obj_name = group[0].object.display_name or obj_id
                    items.append(EvalItem(
                        question=(
                            f"Sources disagree about '{subj_name}' and '{obj_name}'. "
                            f"What are the two opposing predicates?"
                        ),
                        expected_answer=f"{pred}, {opposite}",
                        supporting_claims=[c.claim_id for c in group],
                        difficulty="medium",
                        domain=group[0].subject.entity_type if domains else "",
                        eval_type="contradiction",
                    ))
                    break  # one per pair

        return items[:n]

    # ------------------------------------------------------------------
    # Internal — hard generators
    # ------------------------------------------------------------------

    def _generate_hard(
        self, claims: list[Claim], n: int, domains: list[str] | None, rng: random.Random
    ) -> list[EvalItem]:
        """Generate hard items: provenance and holdout."""
        items: list[EvalItem] = []
        half = max(1, n // 2)

        # Provenance questions: count distinct source types per entity
        entity_claims: dict[str, list[Claim]] = {}
        for c in claims:
            eid = c.subject.id
            if eid not in entity_claims:
                entity_claims[eid] = []
            entity_claims[eid].append(c)

        # Pick entities with multiple source types
        multi_source = [
            (eid, cs)
            for eid, cs in entity_claims.items()
            if len({c.provenance.source_type for c in cs}) >= 2
        ]
        rng.shuffle(multi_source)
        for eid, cs in multi_source[:half]:
            source_count = len({c.provenance.source_type for c in cs})
            ename = cs[0].subject.display_name or eid
            items.append(EvalItem(
                question=f"How many independent source types support claims about '{ename}'?",
                expected_answer=str(source_count),
                supporting_claims=[c.claim_id for c in cs],
                difficulty="hard",
                domain=cs[0].subject.entity_type if domains else "",
                eval_type="provenance",
            ))
            if len(items) >= half:
                break

        # Holdout questions: pick high-confidence claims and ask if the relationship exists
        remaining = n - len(items)
        if remaining > 0:
            high_conf = [c for c in claims if c.confidence >= 0.7]
            rng.shuffle(high_conf)
            for c in high_conf[:remaining]:
                subj_name = c.subject.display_name or c.subject.id
                obj_name = c.object.display_name or c.object.id
                pred = c.predicate.id
                items.append(EvalItem(
                    question=(
                        f"Based on the knowledge graph, is it likely that "
                        f"'{subj_name}' {pred} '{obj_name}'?"
                    ),
                    expected_answer="yes",
                    supporting_claims=[c.claim_id],
                    difficulty="hard",
                    domain=c.subject.entity_type if domains else "",
                    eval_type="holdout",
                ))

        return items[:n]

    # ------------------------------------------------------------------
    # Internal — answer checking
    # ------------------------------------------------------------------

    def _check_answer(self, expected: str, agent_answer: str) -> bool:
        """Fuzzy-check an agent answer against expected.

        - Text: normalized containment check
        - Numeric: 10% tolerance
        - Yes/no: exact match
        """
        expected_norm = expected.strip().lower()
        agent_norm = agent_answer.strip().lower()

        if not expected_norm or not agent_norm:
            return False

        # Yes/no exact match
        if expected_norm in ("yes", "no"):
            return agent_norm.startswith(expected_norm)

        # Numeric tolerance
        try:
            expected_num = float(expected_norm)
            agent_num = float(agent_norm)
            if expected_num == 0:
                return abs(agent_num) < 0.1
            return abs(agent_num - expected_num) / abs(expected_num) <= 0.1
        except ValueError:
            pass

        # Comma-separated sets (for contradiction questions)
        if "," in expected_norm:
            expected_parts = {p.strip() for p in expected_norm.split(",")}
            agent_parts = {p.strip() for p in agent_norm.split(",")}
            if expected_parts and expected_parts <= agent_parts:
                return True

        # Text containment (fuzzy)
        return expected_norm in agent_norm

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _persist_result(self, result: EvalResult) -> None:
        """Store eval result as a claim in the graph."""
        import json

        try:
            self.db.ingest(
                subject=(result.agent_id, "agent"),
                predicate=("eval_scored", "eval"),
                object=(result.eval_id, "eval"),
                provenance={"source_type": "eval", "source_id": f"eval:{result.eval_id}"},
                confidence=result.score,
                payload={
                    "schema_ref": "eval_result",
                    "data": {
                        "score": result.score,
                        "total": result.total,
                        "correct": result.correct,
                        "domain_scores": result.domain_scores,
                        "type_scores": result.type_scores,
                        "elapsed": result.elapsed,
                    },
                },
            )
        except Exception:
            pass  # Non-critical — eval still returns result
