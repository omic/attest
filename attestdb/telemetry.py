"""Opt-in anonymous usage telemetry.

Only tracks: event type, timestamp, Python version, OS, claim count (bucketed).
NEVER tracks: query content, claim content, API keys, user-identifiable data.
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Any


def _bucket_claims(n: int | None) -> str:
    """Bucket claim count to avoid leaking exact database size."""
    if n is None:
        return "unknown"
    if n < 100:
        return "<100"
    if n < 1_000:
        return "<1K"
    if n < 10_000:
        return "<10K"
    if n < 100_000:
        return "<100K"
    return "100K+"


def _send(payload: dict[str, Any]) -> None:
    """POST payload to telemetry endpoint. Swallows all errors."""
    try:
        import urllib.request
        import json

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.attestdb.com/telemetry",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def track(event: str, properties: dict[str, Any] | None = None) -> None:
    """Fire-and-forget telemetry event.

    Only sends if telemetry is enabled via ``attestdb.config``.
    Runs in a background daemon thread so it never blocks the CLI.
    """
    try:
        from attestdb import config

        if not config.is_telemetry_enabled():
            return
    except Exception:
        return

    props = dict(properties or {})

    # Auto-include standard fields
    props["python_version"] = platform.python_version()
    props["os"] = platform.system()
    try:
        from attestdb import __version__
        props["attestdb_version"] = __version__
    except Exception:
        props["attestdb_version"] = "unknown"
    props["timestamp"] = time.time()

    # Bucket claim counts if present
    if "claims" in props:
        props["claims"] = _bucket_claims(props["claims"])

    payload = {"event": event, "properties": props}

    t = threading.Thread(target=_send, args=(payload,), daemon=True)
    t.start()
