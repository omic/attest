"""Claim normalizer — convert raw extracted claims into ClaimInput-compatible dicts.

Handles predicate alias mapping (40+ entries), sentiment word normalization
to 0-1 floats, date normalization to ISO 8601, and currency string parsing.

Confidence is capped at 0.6 for normalized unstructured claims.

Source authority is adjusted based on content type and author role:
- exec email about account health > random Slack message
- QBR document > meeting notes > email > Slack

Each extraction decision is recorded in PredictionLog (if provided)
so the calibration engine can track extraction accuracy over time.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

_MAX_NORMALIZED_CONFIDENCE = 0.6

# ── Source authority weights ───────────────────────────────────────────

_CONTENT_TYPE_AUTHORITY: dict[str, float] = {
    "qbr": 0.60,
    "document": 0.55,
    "meeting_notes": 0.50,
    "email": 0.45,
    "slack_message": 0.35,
    "generic": 0.40,
}

_AUTHOR_ROLE_BOOST: dict[str, float] = {
    "executive": 0.10,
    "vp": 0.08,
    "director": 0.06,
    "csm": 0.05,
    "ae": 0.04,
    "manager": 0.03,
}

# ── Predicate alias mapping ───────────────────────────────────────────

_PREDICATE_ALIASES: dict[str, str] = {
    # Risk
    "at_risk": "risk.escalation", "atrisk": "risk.escalation",
    "at risk": "risk.escalation", "risk": "risk.escalation",
    "escalation": "risk.escalation", "escalated": "risk.escalation",
    "high_risk": "risk.escalation", "critical_risk": "risk.escalation",
    # Churn
    "churn": "churn.status", "churned": "churn.status",
    "churning": "churn.status", "churn_risk": "churn.status",
    "likely_to_churn": "churn.status",
    # Renewal
    "renewal": "renewal.status", "renewed": "renewal.status",
    "renewing": "renewal.status", "auto_renewal": "renewal.status",
    "renewal_date": "renewal.date",
    # Satisfaction
    "nps": "satisfaction.nps", "nps_score": "satisfaction.nps",
    "csat": "satisfaction.csat", "csat_score": "satisfaction.csat",
    "sentiment": "satisfaction.sentiment",
    "customer_sentiment": "satisfaction.sentiment",
    "health": "satisfaction.health", "health_score": "satisfaction.health",
    "happiness": "satisfaction.sentiment",
    # Revenue
    "revenue": "revenue.amount", "arr": "revenue.arr", "mrr": "revenue.mrr",
    "contract_value": "revenue.amount", "deal_size": "revenue.amount",
    "expansion": "revenue.expansion", "contraction": "revenue.contraction",
    # Actions
    "blocker": "action.blocker", "blocked": "action.blocker",
    "blocking": "action.blocker", "action_item": "action.item",
    "todo": "action.item", "task": "action.item",
    "decision": "action.decision",
    # Relationship
    "status": "relationship.status", "relationship": "relationship.status",
    "champion": "relationship.champion",
    "champion_change": "relationship.champion_change",
    "champion_departure": "relationship.champion_change",
    "stakeholder": "relationship.stakeholder",
    "contact": "relationship.contact",
    # Timeline
    "delay": "timeline.delay", "delayed": "timeline.delay",
    "pushed_back": "timeline.delay", "moved_up": "timeline.acceleration",
    "deadline": "commitment.deadline",
    # Commitment
    "commitment": "commitment.promise", "promised": "commitment.promise",
}

# ── Sentiment word mapping ─────────────────────────────────────────────

_SENTIMENT_WORDS: dict[str, float] = {
    "furious": 0.0, "angry": 0.05, "outraged": 0.05, "terrible": 0.05,
    "frustrated": 0.15, "dissatisfied": 0.15, "unhappy": 0.15,
    "disappointed": 0.2, "annoyed": 0.2, "upset": 0.2, "concerned": 0.25,
    "mixed": 0.35, "lukewarm": 0.35, "so-so": 0.4, "meh": 0.4,
    "neutral": 0.5, "ok": 0.5, "okay": 0.5, "fine": 0.5, "indifferent": 0.5,
    "positive": 0.65, "good": 0.65, "satisfied": 0.7,
    "happy": 0.75, "pleased": 0.75, "glad": 0.75, "content": 0.7,
    "delighted": 0.85, "thrilled": 0.9, "ecstatic": 0.95, "love": 0.9,
    "excellent": 0.9, "amazing": 0.9, "fantastic": 0.9, "great": 0.8,
    "wonderful": 0.85,
}

# ── Date patterns (input → ISO 8601) ──────────────────────────────────

_DATE_FORMATS: list[str] = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ",
]

# ── Currency parsing ──────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"^\$?\s*([\d,]+(?:\.\d+)?)\s*([MmKkBb]?)$")
_CURRENCY_MULTIPLIERS: dict[str, float] = {
    "": 1.0, "k": 1_000.0, "K": 1_000.0, "m": 1_000_000.0,
    "M": 1_000_000.0, "b": 1_000_000_000.0, "B": 1_000_000_000.0,
}


class ClaimNormalizer:
    """Normalize raw extracted claims into ClaimInput-compatible dicts.

    Parameters
    ----------
    prediction_log:
        Optional ``PredictionLog`` instance. When provided, each normalized
        claim is recorded as decision_type ``"unstructured_claim"`` so the
        calibration engine can track extraction accuracy over time.
    tenant_id:
        Tenant scope for prediction log records.
    """

    def __init__(
        self,
        prediction_log: object | None = None,
        tenant_id: str = "default",
    ) -> None:
        self._prediction_log = prediction_log
        self._tenant_id = tenant_id

    def normalize(
        self,
        raw_claims: list[dict],
        content_type: str = "generic",
        author_role: str | None = None,
        source_id: str = "unstructured",
    ) -> list[dict]:
        """Normalize a list of raw claim dicts into ClaimInput-compatible dicts.

        Parameters
        ----------
        raw_claims:
            Each dict must have ``subject``, ``predicate``, ``object``.
            Optional: ``confidence``, ``source_snippet``, ``content_type``.
        content_type:
            Content type for source authority calculation.
        author_role:
            Author's role for authority boost (e.g. "executive", "csm").
        source_id:
            Source identifier for provenance.
        """
        authority = self._compute_authority(content_type, author_role)
        results: list[dict] = []
        for raw in raw_claims:
            normalized = self._normalize_one(raw, content_type, source_id, authority)
            if normalized is not None:
                results.append(normalized)
                self._record_prediction(normalized, source_id, content_type)
        return results

    def _compute_authority(self, content_type: str, author_role: str | None) -> float:
        """Compute source authority from content type + author role."""
        base = _CONTENT_TYPE_AUTHORITY.get(content_type, 0.40)
        boost = _AUTHOR_ROLE_BOOST.get(author_role or "", 0.0)
        return min(base + boost, _MAX_NORMALIZED_CONFIDENCE)

    def _normalize_one(
        self,
        raw: dict,
        content_type: str,
        source_id: str,
        authority: float,
    ) -> dict | None:
        subject = str(raw.get("subject", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        obj = str(raw.get("object", "")).strip()
        if not subject or not predicate or not obj:
            return None

        predicate = self._normalize_predicate(predicate)
        obj = self._normalize_object(predicate, obj)

        # Confidence = min(raw_confidence, source_authority, cap)
        confidence = _MAX_NORMALIZED_CONFIDENCE
        try:
            raw_conf = float(raw.get("confidence", _MAX_NORMALIZED_CONFIDENCE))
            confidence = min(raw_conf, authority, _MAX_NORMALIZED_CONFIDENCE)
        except (TypeError, ValueError):
            confidence = min(authority, _MAX_NORMALIZED_CONFIDENCE)

        snippet = str(raw.get("source_snippet", ""))[:200]

        return {
            "subject": (subject, "entity"),
            "predicate": (predicate, "unstructured_extraction"),
            "object": (obj, "value"),
            "confidence": confidence,
            "provenance": {
                "source_type": "unstructured",
                "source_id": source_id,
                "method": "normalized_extraction",
                "labels": {
                    "content_type": content_type,
                    **({"source_snippet": snippet} if snippet else {}),
                },
            },
        }

    def _record_prediction(self, claim: dict, source_id: str, content_type: str) -> None:
        """Record the extraction decision in PredictionLog."""
        if self._prediction_log is None:
            return
        if not hasattr(self._prediction_log, "record"):
            return
        try:
            from attestdb.calibration.prediction_log import PredictionRecord
            pred = PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                tenant_id=self._tenant_id,
                decision_type="unstructured_claim",
                source_id=source_id,
                field_semantic_type=content_type,
                predicted_confidence=claim["confidence"],
                predicted_value=f"{claim['subject'][0]}|{claim['predicate'][0]}|{claim['object'][0]}",
                review_outcome="pending",
                corrected_value=None,
                reviewed_at=None,
                reviewed_by=None,
                created_at=time.time(),
            )
            self._prediction_log.record(pred)
        except Exception as exc:
            logger.debug("Failed to record prediction: %s", exc)

    def _normalize_predicate(self, predicate: str) -> str:
        lower = predicate.lower().strip()
        canonical = _PREDICATE_ALIASES.get(lower)
        if canonical:
            return canonical
        spaced = lower.replace("_", " ").replace("-", " ")
        canonical = _PREDICATE_ALIASES.get(spaced)
        if canonical:
            return canonical
        return predicate

    def _normalize_object(self, predicate: str, obj: str) -> str:
        if predicate in ("satisfaction.sentiment", "satisfaction.health"):
            normalized = _normalize_sentiment(obj)
            if normalized is not None:
                return str(normalized)
        if predicate.endswith(".date") or predicate.endswith(".deadline"):
            normalized_date = _normalize_date(obj)
            if normalized_date is not None:
                return normalized_date
        if any(k in predicate for k in ("revenue", "amount", "arr", "mrr")):
            normalized_amount = _normalize_currency(obj)
            if normalized_amount is not None:
                return str(normalized_amount)
        return obj


# ── Module-level helpers ──────────────────────────────────────────────


def _normalize_sentiment(value: str) -> float | None:
    lower = value.lower().strip()
    score = _SENTIMENT_WORDS.get(lower)
    if score is not None:
        return score
    try:
        f = float(lower)
        if 0.0 <= f <= 1.0:
            return f
    except ValueError:
        pass
    return None


def _normalize_date(value: str) -> str | None:
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _normalize_currency(value: str) -> float | None:
    match = _CURRENCY_RE.match(value.strip())
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    return number * _CURRENCY_MULTIPLIERS.get(match.group(2), 1.0)
