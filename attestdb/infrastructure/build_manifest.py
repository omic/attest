"""Build manifest — append-only JSONL log of reference DB build events.

Sidecar file at ``{db_path}.build.jsonl``. Records build lifecycle events
(start, source start/complete, build complete) so builds can be inspected,
resumed, and diffed after the fact.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class SourceResult:
    """Outcome of loading a single data source."""

    source: str
    status: str  # "ok" | "failed" | "timeout" | "skipped"
    ingested: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    total_claims_after: int = 0
    entity_count_after: int = 0
    error_detail: str = ""  # full traceback on failure


@dataclass
class BuildReport:
    """Assembled view of a complete build."""

    build_id: str
    started_at: int
    completed_at: int
    builder: str
    sources: dict[str, SourceResult]
    total_claims: int
    total_entities: int


class BuildManifest:
    """Append-only JSONL at ``{db_path}.build.jsonl``."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path.rstrip("/") + ".build.jsonl"
        self._build_id: str | None = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def build_id(self) -> str | None:
        return self._build_id

    # ── Write API ─────────────────────────────────────────────────────

    def start_build(
        self,
        builder: str,
        sources: list[str],
        args: dict | None = None,
    ) -> str:
        """Record a build_started event and return the build_id."""
        bid = f"build_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self._build_id = bid
        self._append({
            "event": "build_started",
            "build_id": bid,
            "timestamp": _now_ns(),
            "builder": builder,
            "sources_planned": sources,
            "args": args or {},
        })
        return bid

    def record_source_start(self, source: str) -> None:
        self._append({
            "event": "source_started",
            "build_id": self._build_id,
            "timestamp": _now_ns(),
            "source": source,
        })

    def record_source_result(self, result: SourceResult) -> None:
        self._append({
            "event": "source_completed",
            "build_id": self._build_id,
            "timestamp": _now_ns(),
            **asdict(result),
        })

    def end_build(self, summary: dict | None = None) -> None:
        self._append({
            "event": "build_completed",
            "build_id": self._build_id,
            "timestamp": _now_ns(),
            **(summary or {}),
        })

    # ── Read API ──────────────────────────────────────────────────────

    def load_events(
        self,
        build_id: str | None = None,
        source: str | None = None,
        event_type: str | None = None,
    ) -> list[dict]:
        """Load events with optional filters."""
        events = self._read_all()
        if build_id:
            events = [e for e in events if e.get("build_id") == build_id]
        if source:
            events = [e for e in events if e.get("source") == source]
        if event_type:
            events = [e for e in events if e.get("event") == event_type]
        return events

    def latest_build(self) -> BuildReport | None:
        """Assemble a BuildReport from the most recent build."""
        events = self._read_all()
        if not events:
            return None

        # Find the last build_started
        starts = [e for e in events if e["event"] == "build_started"]
        if not starts:
            return None
        start_event = starts[-1]
        bid = start_event["build_id"]

        # Collect events for this build
        build_events = [e for e in events if e.get("build_id") == bid]

        sources: dict[str, SourceResult] = {}
        for e in build_events:
            if e["event"] == "source_completed":
                sources[e["source"]] = SourceResult(
                    source=e["source"],
                    status=e.get("status", "unknown"),
                    ingested=e.get("ingested", 0),
                    duplicates=e.get("duplicates", 0),
                    errors=e.get("errors", []),
                    warnings=e.get("warnings", []),
                    elapsed_sec=e.get("elapsed_sec", 0.0),
                    total_claims_after=e.get("total_claims_after", 0),
                    entity_count_after=e.get("entity_count_after", 0),
                    error_detail=e.get("error_detail", ""),
                )

        completed = [e for e in build_events if e["event"] == "build_completed"]
        completed_at = completed[-1]["timestamp"] if completed else 0
        total_claims = completed[-1].get("total_claims", 0) if completed else 0
        total_entities = completed[-1].get("total_entities", 0) if completed else 0

        return BuildReport(
            build_id=bid,
            started_at=start_event["timestamp"],
            completed_at=completed_at,
            builder=start_event.get("builder", ""),
            sources=sources,
            total_claims=total_claims,
            total_entities=total_entities,
        )

    def completed_sources(self, build_id: str | None = None) -> set[str]:
        """Return source names that completed successfully in a build.

        If *build_id* is ``None``, uses the latest build.
        """
        if build_id is None:
            report = self.latest_build()
            if report is None:
                return set()
            build_id = report.build_id

        events = self.load_events(build_id=build_id, event_type="source_completed")
        return {e["source"] for e in events if e.get("status") == "ok"}

    def source_history(self, source_id: str) -> list[dict]:
        """All events (across all builds) for a given source."""
        return self.load_events(source=source_id)

    def diff_builds(self, build_id_a: str, build_id_b: str) -> dict:
        """Compare two builds: new/removed/changed sources."""
        events_a = self.load_events(build_id=build_id_a, event_type="source_completed")
        events_b = self.load_events(build_id=build_id_b, event_type="source_completed")

        sources_a = {e["source"]: e for e in events_a}
        sources_b = {e["source"]: e for e in events_b}

        keys_a = set(sources_a)
        keys_b = set(sources_b)

        changed = []
        for s in keys_a & keys_b:
            a, b = sources_a[s], sources_b[s]
            if a.get("ingested") != b.get("ingested") or a.get("status") != b.get("status"):
                changed.append({
                    "source": s,
                    "a_status": a.get("status"),
                    "b_status": b.get("status"),
                    "a_ingested": a.get("ingested", 0),
                    "b_ingested": b.get("ingested", 0),
                })

        return {
            "added": sorted(keys_b - keys_a),
            "removed": sorted(keys_a - keys_b),
            "changed": changed,
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _append(self, event: dict) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        events = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events


def _now_ns() -> int:
    return int(time.time() * 1_000_000_000)
