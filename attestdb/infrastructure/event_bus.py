"""Event bus — lifecycle hooks and topic subscriptions."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Callable

from attestdb.core.normalization import normalize_entity_id

if TYPE_CHECKING:
    from attestdb.core.types import ClaimInput
    from attestdb.infrastructure.webhooks import WebhookManager

logger = logging.getLogger(__name__)


class EventBus:
    """Manages event hooks and topic subscriptions.

    Args:
        webhooks: WebhookManager for dispatching events to HTTP endpoints.
        get_claim_fn: Callable that takes a claim_id and returns a Claim object
            (used by subscriptions to provide the full claim to callbacks).
    """

    def __init__(
        self,
        webhooks: "WebhookManager",
        get_claim_fn: Callable | None = None,
    ) -> None:
        self._hooks: dict[str, list] = {}
        self._subscriptions: dict[str, dict] = {}
        self._webhooks = webhooks
        self._get_claim = get_claim_fn

    def on(self, event: str, callback) -> None:
        """Register a callback for a lifecycle event."""
        self._hooks.setdefault(event, []).append(callback)

    def off(self, event: str, callback) -> None:
        """Remove a registered callback."""
        hooks = self._hooks.get(event, [])
        if callback in hooks:
            hooks.remove(callback)

    def fire(self, event: str, **kwargs) -> None:
        """Fire all callbacks for an event and POST to registered webhooks."""
        for cb in self._hooks.get(event, []):
            try:
                cb(**kwargs)
            except Exception as exc:
                logger.warning("Event hook %s failed: %s", event, exc)
        self._webhooks.fire(event, kwargs)

    def subscribe(
        self,
        subscriber_id: str,
        topics: list[str],
        callback,
        predicates: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> str:
        """Register interest in topics. Callback fires on matching ingested claims.

        Returns subscription ID (use to unsubscribe).
        """
        sub_id = str(uuid.uuid4())
        self._subscriptions[sub_id] = {
            "subscriber_id": subscriber_id,
            "topics": {normalize_entity_id(t) for t in topics},
            "callback": callback,
            "predicates": set(predicates) if predicates else None,
            "min_confidence": min_confidence,
        }
        return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """Remove a subscription. Returns True if it existed."""
        return self._subscriptions.pop(sub_id, None) is not None

    def check_subscriptions(self, claim_id: str, claim_input: "ClaimInput") -> None:
        """Check all subscriptions against a newly ingested claim."""
        if not self._subscriptions:
            return
        subj_id = normalize_entity_id(claim_input.subject[0])
        obj_id = normalize_entity_id(claim_input.object[0])
        pred_id = claim_input.predicate[0]
        conf = claim_input.confidence if claim_input.confidence is not None else 0.5
        claim = self._get_claim(claim_id) if self._get_claim and claim_id else None
        for sub_id, sub in list(self._subscriptions.items()):
            if conf < sub["min_confidence"]:
                continue
            if sub["predicates"] is not None and pred_id not in sub["predicates"]:
                continue
            if subj_id in sub["topics"] or obj_id in sub["topics"]:
                try:
                    sub["callback"](claim, sub_id, sub["subscriber_id"])
                except Exception as exc:
                    logger.warning("Subscription %s callback failed: %s", sub_id, exc)
