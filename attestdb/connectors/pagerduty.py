"""PagerDuty connector — imports incidents as knowledge claims.

Uses the PagerDuty REST API to fetch incidents and converts them to
claims representing incident management relationships (status, urgency,
service, assignee).

Usage::

    from attestdb.connectors.pagerduty import PagerDutyConnector

    conn = PagerDutyConnector(
        token=os.environ["PAGERDUTY_API_KEY"],
        since="2025-01-01T00:00:00Z",
        statuses=["triggered", "acknowledged"],
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator

from attestdb.connectors.base import ConnectorResult, StructuredConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)

PAGERDUTY_API = "https://api.pagerduty.com"


class PagerDutyConnector(StructuredConnector):
    """Sync connector that imports PagerDuty incidents as structured claims."""

    name = "pagerduty"

    def __init__(
        self,
        token: str,
        since: str | None = None,
        statuses: list[str] | None = None,
        max_items: int = 500,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the PagerDuty connector")
        self._token = token
        self._since = since
        self._statuses = statuses
        self._max_items = max_items
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Token token={token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts from PagerDuty incidents."""
        offset = 0
        count = 0
        pages = 0

        while count < self._max_items and pages < self._MAX_PAGES:
            limit = min(100, self._max_items - count)
            params: dict = {
                "offset": offset,
                "limit": limit,
                "sort_by": "created_at:desc",
            }
            if self._since:
                params["since"] = self._since
            if self._statuses:
                params["statuses[]"] = self._statuses

            resp = self._request_with_retry(
                "GET",
                f"{PAGERDUTY_API}/incidents",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            incidents = data.get("incidents", [])
            if not incidents:
                break

            for incident in incidents:
                yield from self._incident_to_claims(incident)
                count += 1

            if not data.get("more", False):
                break

            offset += len(incidents)
            pages += 1
            time.sleep(0.1)

    def _incident_to_claims(self, incident: dict) -> Iterator[dict]:
        inc_id = incident.get("incident_number", incident.get("id", ""))
        display_id = (
            f"PD-{inc_id}" if isinstance(inc_id, int) else str(inc_id)
        )
        source_id = f"pagerduty:{display_id}"

        # Status
        status = incident.get("status", "")
        if status:
            yield self._make_claim(
                display_id, "incident", "has_status",
                status, "status", source_id,
            )

        # Urgency
        urgency = incident.get("urgency", "")
        if urgency:
            yield self._make_claim(
                display_id, "incident", "has_urgency",
                urgency, "urgency", source_id,
            )

        # Service
        service = (incident.get("service") or {}).get("summary", "")
        if service:
            yield self._make_claim(
                display_id, "incident", "affects_service",
                service, "service", source_id,
            )

        # Assignments (with external_ids)
        for assignment in incident.get("assignments", []):
            assignee_data = assignment.get("assignee") or {}
            assignee = assignee_data.get("summary", "")
            if assignee:
                yield self._make_claim(
                    display_id, "incident", "assigned_to",
                    assignee, "person", source_id,
                    obj_ext={"person_name": assignee},
                )

        # Escalation policy
        policy = (
            (incident.get("escalation_policy") or {})
            .get("summary", "")
        )
        if policy:
            yield self._make_claim(
                display_id, "incident", "escalation_policy",
                policy, "policy", source_id,
            )
