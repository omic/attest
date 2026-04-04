"""Append-only JSONL audit log for mutation tracking and operational event logging."""

from __future__ import annotations

import builtins
import json
import logging
import os
import time
from dataclasses import dataclass

from attestdb.core.types import AuditEvent

logger = logging.getLogger(__name__)


class AuditLog:
    """Manages actor identity and append-only audit log persistence.

    All mutations (ingest, retract, grant, etc.) are recorded with
    actor attribution and nanosecond timestamps.
    """

    def __init__(self, db_path: str, is_memory: bool) -> None:
        self._actor: str = ""
        self._audit_path: str | None = None
        if not is_memory:
            self._audit_path = db_path + ".audit.jsonl"

    @property
    def actor(self) -> str:
        return self._actor

    def set_actor(self, actor: str) -> None:
        """Set the current actor identity for audit logging.

        All subsequent mutations (ingest, retract, etc.) will be
        attributed to this actor in the audit log.
        """
        self._actor = actor

    def write(self, event: str, **kwargs) -> None:
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

    def query(
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


@dataclass
class OpsEvent:
    """A single operational event."""
    event: str
    timestamp: float
    details: dict


class OpsLog:
    """Append-only JSONL log for operational (read-path) events.

    Separate from AuditLog (which tracks mutations). Records:
    - curator_triage: claim identifier, decision, provider, tokens, cost
    - ask_query: question, entity count, tokens, elapsed time
    """

    def __init__(self, db_path: str, is_memory: bool) -> None:
        self._path: str | None = None
        if not is_memory:
            self._path = db_path + ".ops.jsonl"

    def write(self, event: str, **kwargs) -> None:
        """Append an operational event."""
        if not self._path:
            return
        entry = {"event": event, "timestamp": time.time(), **kwargs}
        try:
            with builtins.open(self._path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write ops event: %s", exc)

    def query(
        self,
        since: float = 0.0,
        event_type: str | None = None,
        limit: int = 1000,
    ) -> list[OpsEvent]:
        """Query the ops log.

        Args:
            since: Unix timestamp. Returns events after this time.
            event_type: Filter to a specific event type.
            limit: Max events to return.
        """
        if not self._path or not os.path.exists(self._path):
            return []
        events: list[OpsEvent] = []
        with builtins.open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = d.get("timestamp", 0.0)
                if ts <= since:
                    continue
                if event_type and d.get("event") != event_type:
                    continue
                evt = d.pop("event", "")
                ts = d.pop("timestamp", 0.0)
                events.append(OpsEvent(event=evt, timestamp=ts, details=d))
                if len(events) >= limit:
                    break
        return events
