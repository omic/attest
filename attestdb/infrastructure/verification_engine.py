"""Verification engine — claim verification, freshness checks, and staleness sweeps."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.types import Claim, claim_from_dict

logger = logging.getLogger(__name__)


class VerificationEngine:
    """Manages claim verification pipeline, freshness checks, and staleness sweeps.

    Owns verification budget state (per-session cost tracking).
    """

    def __init__(self, db: "AttestDB") -> None:
        self.db = db

    def configure_verification_budget(self, max_usd: float) -> None:
        """Set a per-session verification budget. Once spent, verify_claim() will
        skip checks that exceed the remaining budget."""
        self.db._verification_budget = max_usd
        self.db._verification_spent = 0.0

    def verification_budget_status(self) -> dict:
        """Return current verification budget and spending."""
        return {
            "budget_usd": self.db._verification_budget,
            "spent_usd": round(self.db._verification_spent, 6),
            "remaining_usd": round(self.db._verification_budget - self.db._verification_spent, 6)
            if self.db._verification_budget is not None else None,
        }

    def verify_claim(
        self,
        claim: "Claim | str",
        tier: "int | None" = None,
        checks: "list[str] | None" = None,
    ) -> "VerificationVerdict":
        """Run the verification pipeline on a claim.

        Args:
            claim: A Claim object or claim_id string.
            tier: Override tier (1, 2, or 3). None = auto-assign.
            checks: Override specific checks to run.

        Returns:
            VerificationVerdict with per-check results and overall verdict.
        """
        from attestdb.intelligence.verification import verify_claim as _verify, VerificationTier

        if isinstance(claim, str):
            claim = self.db.get_claim(claim)
            if claim is None:
                from attestdb.core.types import VerificationVerdict
                return VerificationVerdict(
                    verdict="FAILED",
                    overall_confidence=0.0,
                    recommended_status="verification_failed",
                    check_results=[],
                    checks_skipped=["all"],
                    verification_completeness=0.0,
                )

        tier_enum = VerificationTier(tier) if isinstance(tier, int) else tier
        verdict = _verify(claim, self.db, tier=tier_enum, checks=checks)
        self.db._verification_spent += verdict.total_cost_usd
        return verdict

    # --- Enterprise Staleness ---

    def check_freshness(
        self,
        source_type: str | None = None,
        entity_filter: str | None = None,
    ) -> dict:
        """Check freshness of claims, optionally filtered by source type or entity.

        Returns stale claims, approaching-stale claims, and summary stats.
        """
        import time as _time
        from attestdb.core.vocabulary import SOURCE_TYPE_REGISTRY

        now_ns = int(_time.time() * 1e9)
        stale = []
        approaching = []
        fresh = 0

        # Get claims to check
        if entity_filter:
            claims = self.db.claims_for(entity_filter)
        else:
            claims = self.db._all_claims()

        for c in claims[:5000]:  # Cap scan
            if source_type and c.provenance.source_type != source_type:
                continue

            src_info = SOURCE_TYPE_REGISTRY.get(c.provenance.source_type, {})
            staleness_days = src_info.get("staleness_days")
            if staleness_days is None:
                fresh += 1
                continue

            age_days = (now_ns - c.timestamp) / (86400 * 1e9)
            if age_days > staleness_days:
                stale.append({
                    "claim_id": c.claim_id,
                    "subject": c.subject.id,
                    "predicate": c.predicate.id,
                    "object": c.object.id,
                    "source_type": c.provenance.source_type,
                    "age_days": round(age_days, 1),
                    "threshold_days": staleness_days,
                })
            elif age_days > staleness_days * 0.8:
                approaching.append({
                    "claim_id": c.claim_id[:16],
                    "source_type": c.provenance.source_type,
                    "age_days": round(age_days, 1),
                    "threshold_days": staleness_days,
                })
            else:
                fresh += 1

        return {
            "stale_count": len(stale),
            "approaching_stale_count": len(approaching),
            "fresh_count": fresh,
            "stale_claims": stale[:50],
            "approaching_stale": approaching[:20],
        }

    def sweep_stale(
        self,
        dry_run: bool = True,
        source_type: str | None = None,
        decay_factor: float = 0.9,
        confidence_floor: float = 0.2,
    ) -> dict:
        """Sweep stale claims and optionally apply confidence decay.

        dry_run=True: report only, no changes.
        dry_run=False: transitions stale agent-extracted claims to
        PROVENANCE_DEGRADED status. Append-only invariant preserved.
        """
        freshness = self.check_freshness(source_type=source_type)
        decayed = 0
        flagged = 0
        degraded_ids: list[str] = []

        if not dry_run:
            for item in freshness["stale_claims"]:
                full_cid = item.get("claim_id", "")
                # claim_id in freshness is truncated to 16 chars; find full ID
                claim = self.db.get_claim(full_cid)
                if claim is None:
                    # Try finding via source lookup
                    flagged += 1
                    continue
                # Only decay agent-extracted claims
                if claim.provenance.source_type in ("claude_chat", "claude_code", "agent_session", "agent", "llm_inference"):
                    age_days = item["age_days"]
                    periods = age_days / 30
                    new_conf = max(confidence_floor, claim.confidence * (decay_factor ** periods))
                    if new_conf < claim.confidence:
                        # Transition to PROVENANCE_DEGRADED (append-only safe)
                        try:
                            self.db._store.update_claim_status(claim.claim_id, "provenance_degraded")
                            decayed += 1
                            degraded_ids.append(claim.claim_id)
                        except Exception as e:
                            logger.warning("Failed to degrade claim %s: %s", claim.claim_id[:16], e)
                            flagged += 1
                else:
                    flagged += 1

        return {
            "stale_claims_found": freshness["stale_count"],
            "approaching_stale": freshness["approaching_stale_count"],
            "claims_degraded": decayed,
            "flagged": flagged,
            "degraded_claim_ids": degraded_ids,
            "dry_run": dry_run,
        }
