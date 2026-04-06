"""Synthetic data generators for connector validation.

Four deterministic connectors — one per pattern — that exercise the
real sealed ``run()`` orchestration with known data.  Used by the
connector validation matrices in ``tests/integration/test_connector_matrices.py``.
"""

from __future__ import annotations

from typing import Iterator

from attestdb.connectors.base import (
    HybridConnector,
    QueryConnector,
    StructuredConnector,
    TextConnector,
)

T0 = 1_700_000_000_000_000_000  # Base timestamp (~Nov 2023)
DAY_NS = 86_400_000_000_000


def _ns(i: int) -> int:
    """Return T0 + i days in nanoseconds."""
    return T0 + i * DAY_NS


# ══════════════════════════════════════════════════════════════════════
# Pattern 1: Structured
# ══════════════════════════════════════════════════════════════════════

_STATUSES = ["open", "in_progress", "review", "done"]
_PRIORITIES = ["high", "low"]


class SyntheticStructuredConnector(StructuredConnector):
    """Deterministic structured connector for validation.

    Generates a project management graph: tickets, people, statuses,
    priorities, org membership.  Interleaves malformed dicts to test
    error handling in ``_flush()``.
    """

    name = "synthetic_structured"

    def __init__(
        self,
        n_claims: int = 12,
        n_malformed: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._n_claims = n_claims
        self._n_malformed = n_malformed

    def fetch(self) -> Iterator[dict]:
        # Build all good claims first, then interleave malformed
        good: list[dict] = []

        # Claims 0-3: ticket assigned_to person (directional)
        for i in range(4):
            good.append(self._make_claim(
                f"ticket-{i}", "ticket",
                "assigned_to",
                f"person-{i % 3}", "person",
                f"synth_struct_{len(good):03d}",
                confidence=round(0.7 + 0.02 * len(good), 2),
                subj_ext={"jira": f"PROJ-{i}"},
            ))

        # Claims 4-7: ticket has_status status (attribute)
        for i in range(4):
            good.append(self._make_claim(
                f"ticket-{i}", "ticket",
                "has_status",
                f"status-{_STATUSES[i]}", "status",
                f"synth_struct_{len(good):03d}",
                confidence=round(0.7 + 0.02 * len(good), 2),
            ))

        # Claims 8-9: ticket has_priority (attribute)
        for i in range(2):
            good.append(self._make_claim(
                f"ticket-{i}", "ticket",
                "has_priority",
                f"priority-{_PRIORITIES[i]}", "priority",
                f"synth_struct_{len(good):03d}",
                confidence=round(0.7 + 0.02 * len(good), 2),
            ))

        # Claims 10-11: person works_at org (directional)
        for i in range(2):
            good.append(self._make_claim(
                f"person-{i}", "person",
                "works_at",
                "org-alpha", "organization",
                f"synth_struct_{len(good):03d}",
                confidence=round(0.7 + 0.02 * len(good), 2),
                obj_ext={"crm": "ORG-001"},
            ))

        # Malformed dicts to interleave at positions 5 and 9
        malformed: list[tuple[int, dict]] = []
        if self._n_malformed > 0:
            # Missing "subject" key entirely
            malformed.append((5, {
                "predicate": ("assigned_to", "directional"),
                "object": ("orphan", "entity"),
                "provenance": {
                    "source_type": self.name,
                    "source_id": "synth_struct_bad_0",
                },
            }))
        if self._n_malformed > 1:
            # Missing "object" key
            malformed.append((9, {
                "subject": ("orphan-2", "entity"),
                "predicate": ("has_status", "relates_to"),
                "provenance": {
                    "source_type": self.name,
                    "source_id": "synth_struct_bad_1",
                },
            }))

        # Yield interleaved stream
        mal_idx = 0
        for pos, claim in enumerate(good):
            while mal_idx < len(malformed) and malformed[mal_idx][0] == pos:
                yield malformed[mal_idx][1]
                mal_idx += 1
            yield claim

        self._checkpoint = {"last_offset": self._n_claims}


# ══════════════════════════════════════════════════════════════════════
# Pattern 2: Text
# ══════════════════════════════════════════════════════════════════════

_TEXT_TEMPLATES = [
    "Person-{i} authored report-{i} on 2023-11-14.",
    "Report-{i} covers risk assessment for project-alpha.",
    "Department-{i} approved budget-{i} for Q4 2023.",
    "Person-{i} presented findings at conference-{i}.",
    "Report-{i} cites regulation-{i} as primary authority.",
    "Department-{i} escalated issue-{i} to management.",
]


class SyntheticTextConnector(TextConnector):
    """Deterministic text connector for validation.

    Yields ``(source_id, text)`` pairs with known content.  Empty texts
    interleaved to test filtering.  Tests mock ``db.ingest_text()``.
    """

    name = "synthetic_text"

    def __init__(
        self,
        n_texts: int = 6,
        n_empty: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._n_texts = n_texts
        self._n_empty = n_empty

    def fetch_texts(self) -> Iterator[tuple[str, str]]:
        text_idx = 0
        empty_idx = 0
        total = self._n_texts + self._n_empty
        last_source_id = ""

        for pos in range(total):
            # Interleave empties at positions 2 and 4
            if empty_idx < self._n_empty and pos in (2, 4):
                if empty_idx == 0:
                    yield (f"text_src_empty_{empty_idx}", "")
                else:
                    yield (f"text_src_empty_{empty_idx}", "   \n  ")
                empty_idx += 1
                continue

            if text_idx < self._n_texts:
                source_id = f"text_src_{text_idx:03d}"
                template = _TEXT_TEMPLATES[
                    text_idx % len(_TEXT_TEMPLATES)
                ]
                text = template.format(i=text_idx)
                last_source_id = source_id
                yield (source_id, text)
                text_idx += 1

        if last_source_id:
            self._checkpoint = {"last_source_id": last_source_id}


# ══════════════════════════════════════════════════════════════════════
# Pattern 3: Hybrid
# ══════════════════════════════════════════════════════════════════════

_LABELS = ["bug", "feature"]


class SyntheticHybridConnector(HybridConnector):
    """Deterministic hybrid connector for validation.

    ``fetch()`` yields structural claims (issues → users, labels, repo).
    ``fetch_bodies()`` yields text bodies for extraction.
    """

    name = "synthetic_hybrid"

    def __init__(
        self,
        n_structural: int = 8,
        n_bodies: int = 4,
        n_malformed: int = 1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._n_structural = n_structural
        self._n_bodies = n_bodies
        self._n_malformed = n_malformed
        self._bodies: list[tuple[str, str]] = []

    def fetch(self) -> Iterator[dict]:
        self._bodies = []
        idx = 0

        # Claims 0-3: issue opened_by user (directional)
        for i in range(4):
            if self._n_malformed > 0 and idx == 3:
                yield {
                    "subject": (f"issue-bad", "issue"),
                    "predicate": ("opened_by", "directional"),
                    # missing "object" key
                    "provenance": {
                        "source_type": self.name,
                        "source_id": "synth_hybrid_bad",
                    },
                }
            ext: dict[str, str] = {"github": f"ISSUE-{i}"}
            if i < 2:
                ext["employee_id"] = f"person-{i}"
            yield self._make_claim(
                f"issue-{i}", "issue",
                "opened_by",
                f"user-{i % 2}", "user",
                f"synth_hybrid_{idx:03d}",
                subj_ext=ext,
            )
            idx += 1

        # Claims 4-5: issue labeled (attribute)
        for i in range(2):
            yield self._make_claim(
                f"issue-{i}", "issue",
                "labeled",
                _LABELS[i], "label",
                f"synth_hybrid_{idx:03d}",
            )
            idx += 1

        # Claims 6-7: issue belongs_to repo (directional)
        for i in range(2):
            yield self._make_claim(
                f"issue-{i}", "issue",
                "belongs_to",
                "repo-main", "repository",
                f"synth_hybrid_{idx:03d}",
            )
            idx += 1

        # Collect bodies for text extraction pass
        for i in range(self._n_bodies):
            source_id = f"hybrid_body_{i:03d}"
            if i == 1:
                self._bodies.append((source_id, ""))
            else:
                self._bodies.append((
                    source_id,
                    f"Issue {i} involves a null pointer in module-{i}.",
                ))

        self._checkpoint = {"structural_count": self._n_structural}

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        return iter(self._bodies)


# ══════════════════════════════════════════════════════════════════════
# Pattern 4: Query
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_QUERY_MAPPING = {
    "subject": "employee_name",
    "subject_type": "person",
    "predicate": "relationship",
    "predicate_type": "directional",
    "object": "department_name",
    "object_type": "department",
    "confidence": "score",
}


def _default_rows() -> list[dict]:
    """Generate 11 deterministic rows: 10 good + 1 bad at position 5."""
    rows: list[dict] = []
    for i in range(11):
        if i == 5:
            # Bad row: missing employee_name
            rows.append({
                "relationship": "works_at",
                "department_name": f"dept-{i % 3}",
                "score": 0.85,
            })
        else:
            rows.append({
                "employee_name": f"emp-{i}",
                "relationship": "works_at",
                "department_name": f"dept-{i % 3}",
                "score": round(0.8 + 0.01 * i, 2),
            })
    return rows


class SyntheticQueryConnector(QueryConnector):
    """Deterministic query connector for validation.

    ``_open_cursor()`` returns an in-memory row iterator — no real
    database needed.
    """

    name = "synthetic_query"

    def __init__(
        self,
        rows: list[dict] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(
            query="SELECT * FROM synthetic_hr",
            mapping=_DEFAULT_QUERY_MAPPING,
            source_type="database_import",
            **kwargs,
        )
        self._rows = rows if rows is not None else _default_rows()

    def _open_cursor(self) -> Iterator[dict]:
        return iter(self._rows)
