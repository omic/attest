"""Webhook notification manager — push events to registered HTTP endpoints."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class WebhookManager:
    """Manages webhook registrations, persistence, and fire-and-forget delivery.

    Args:
        webhooks_path: Path to the JSON sidecar file, or None for in-memory only.
    """

    def __init__(self, webhooks_path: str | None = None) -> None:
        self._webhooks: list[dict] = []
        self._path: str | None = webhooks_path
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="attest-webhook")
        self._load()

    def _load(self) -> None:
        if self._path and os.path.exists(self._path):
            try:
                with builtins.open(self._path) as f:
                    self._webhooks = json.load(f)
            except Exception:
                self._webhooks = []

    def _save(self) -> None:
        if self._path:
            with builtins.open(self._path, "w") as f:
                json.dump(self._webhooks, f)

    def register(
        self, url: str, events: list[str] | None = None, secret: str | None = None,
    ) -> None:
        """Register an HTTP endpoint to receive event notifications.

        Args:
            url: The URL to POST event payloads to.
            events: List of event names to subscribe to. None = all events.
            secret: Optional HMAC-SHA256 secret for payload signing.
        """
        self._webhooks = [w for w in self._webhooks if w["url"] != url]
        self._webhooks.append({
            "url": url,
            "events": events,
            "secret": secret,
        })
        self._save()

    def remove(self, url: str) -> bool:
        """Remove a registered webhook by URL. Returns True if found."""
        before = len(self._webhooks)
        self._webhooks = [w for w in self._webhooks if w["url"] != url]
        if len(self._webhooks) < before:
            self._save()
            return True
        return False

    def list(self) -> list[dict]:
        """Return registered webhooks (secrets are masked)."""
        return [
            {"url": w["url"], "events": w["events"], "has_secret": bool(w.get("secret"))}
            for w in self._webhooks
        ]

    def fire(self, event: str, data: dict) -> None:
        """POST event payload to matching webhook URLs (fire-and-forget)."""
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
            self._executor.submit(self._post, _requests, wh["url"], payload, headers)

    @staticmethod
    def _post(requests_mod, url: str, payload: str, headers: dict) -> None:
        """Execute a single webhook POST (runs in thread pool)."""
        try:
            requests_mod.post(url, data=payload, headers=headers, timeout=5)
        except Exception as exc:
            logger.warning("Webhook POST to %s failed: %s", url, exc)

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self._executor.shutdown(wait=False)
