"""Webhook notification manager — push events to registered HTTP endpoints."""

from __future__ import annotations

import builtins
import hashlib
import hmac
import json
import logging
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_RETRY_DELAY_S = 2
_RETRYABLE_STATUS_MIN = 500


class WebhookManager:
    """Manages webhook registrations, persistence, and fire-and-forget delivery.

    Args:
        webhooks_path: Path to the JSON sidecar file, or None for in-memory only.
        failure_log_path: Path for JSONL failure log, or None to skip.
    """

    def __init__(
        self,
        webhooks_path: str | None = None,
        failure_log_path: str | None = None,
    ) -> None:
        self._webhooks: list[dict] = []
        self._path: str | None = webhooks_path
        self._failure_log_path: str | None = failure_log_path
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
            self._executor.submit(
                self._post, _requests, wh["url"], payload, headers,
                event, self._failure_log_path,
            )

    @staticmethod
    def _post(
        requests_mod, url: str, payload: str, headers: dict,
        event: str, failure_log_path: str | None,
    ) -> None:
        """Execute a single webhook POST with one retry on transient errors."""
        last_error: str | None = None
        for attempt in range(2):
            try:
                resp = requests_mod.post(url, data=payload, headers=headers, timeout=5)
                if resp.status_code < _RETRYABLE_STATUS_MIN:
                    return  # success or 4xx (no retry)
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)
            if attempt == 0:
                _time.sleep(_RETRY_DELAY_S)

        logger.warning("Webhook POST to %s failed after retry: %s", url, last_error)
        if failure_log_path:
            try:
                entry = json.dumps({
                    "timestamp": _time.time(),
                    "url": url,
                    "event": event,
                    "error": last_error,
                })
                with builtins.open(failure_log_path, "a") as f:
                    f.write(entry + "\n")
            except Exception:
                pass

    def shutdown(self) -> None:
        """Shut down the thread pool executor, waiting for in-flight deliveries."""
        self._executor.shutdown(wait=True)
