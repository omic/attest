"""Linear connector — imports issues as knowledge claims.

Uses the Linear GraphQL API to fetch issues and converts them to claims
representing project management relationships (state, assignee, labels,
team, priority, creator, title).  Optionally extracts claims from issue
descriptions via text extraction.

Usage::

    from attestdb.connectors.linear import LinearConnector

    conn = LinearConnector(
        token=os.environ["LINEAR_API_KEY"],
        team="ENG",              # optional — filter by team key
        max_items=1000,
        extraction="heuristic",  # extract claims from descriptions
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

from attestdb.connectors.base import HybridConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LINEAR_API = "https://api.linear.app/graphql"

_ISSUES_QUERY = """
query($first: Int!, $after: String, $teamId: String) {
  issues(
    first: $first,
    after: $after,
    filter: { team: { id: { eq: $teamId } } }
    orderBy: updatedAt
  ) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      id
      identifier
      title
      description
      state { name type }
      assignee { name email }
      creator { name email }
      labels { nodes { name } }
      team { name key }
      priority
      project { name }
    }
  }
}
"""

# Simpler query when no team filter is applied
_ISSUES_QUERY_NO_TEAM = """
query($first: Int!, $after: String) {
  issues(
    first: $first,
    after: $after,
    orderBy: updatedAt
  ) {
    pageInfo {
      hasNextPage
      endCursor
    }
    nodes {
      id
      identifier
      title
      description
      state { name type }
      assignee { name email }
      creator { name email }
      labels { nodes { name } }
      team { name key }
      priority
      project { name }
    }
  }
}
"""

# Query to resolve a team key ("ENG") to its internal ID
_TEAM_BY_KEY_QUERY = """
query($key: String!) {
  teams(filter: { key: { eq: $key } }) {
    nodes { id name key }
  }
}
"""

_PRIORITY_LABELS = {
    0: "no_priority",
    1: "urgent",
    2: "high",
    3: "medium",
    4: "low",
}


_SEARCH_QUERY = """
query($term: String!, $first: Int!) {
  issueSearch(query: $term, first: $first) {
    nodes {
      identifier
      title
      description
      state { name }
      assignee { name }
    }
  }
}
"""


class LinearConnector(HybridConnector):
    """Sync connector that imports Linear issues as structured claims.

    Produces structural claims (assignee, creator, state, labels, team,
    priority, project) with external_ids for cross-connector entity
    resolution, plus optional text extraction from issue descriptions.
    """

    name = "linear"
    _MAX_PAGES = 10000

    def __init__(
        self,
        token: str,
        team: str | None = None,
        team_id: str | None = None,
        project: str | None = None,
        state: str | None = None,
        max_items: int = 1000,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError(
                "pip install requests for the Linear connector"
            )
        self._token = token
        self._team = team
        self._team_id = team_id
        self._project = project
        self._state = state
        self._max_items = max_items
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": token,
            "Content-Type": "application/json",
        })
        self._bodies: list[tuple[str, str]] = []

    @property
    def supports_search(self) -> bool:
        return True

    def search(self, query: str) -> str:
        """Search Linear issues via GraphQL issueSearch."""
        try:
            resp = self._request_with_retry(
                "POST",
                LINEAR_API,
                json={
                    "query": _SEARCH_QUERY,
                    "variables": {"term": query, "first": 10},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            nodes = (
                data.get("data", {})
                .get("issueSearch", {})
                .get("nodes", [])
            )
            parts = []
            for node in nodes[:10]:
                ident = node.get("identifier", "")
                title = node.get("title", "")
                state = (node.get("state") or {}).get("name", "")
                assignee = (
                    (node.get("assignee") or {}).get("name", "")
                )
                desc = (node.get("description") or "")[:500]
                entry = f"[{ident}] {title}"
                if state:
                    entry += f" ({state})"
                if assignee:
                    entry += f" — {assignee}"
                if desc:
                    entry += f"\n{desc}"
                parts.append(entry)
            return "\n\n".join(parts)[:4000]
        except Exception as exc:
            logger.warning("Linear search failed: %s", exc)
            return ""

    def _resolve_team_key(self, key: str) -> str | None:
        """Resolve a team key (e.g. ``ENG``) to a Linear team ID."""
        resp = self._request_with_retry(
            "POST",
            LINEAR_API,
            json={"query": _TEAM_BY_KEY_QUERY, "variables": {"key": key}},
        )
        resp.raise_for_status()
        data = resp.json()
        nodes = data.get("data", {}).get("teams", {}).get("nodes", [])
        if nodes:
            return nodes[0].get("id")
        logger.warning("linear: team key %r not found", key)
        return None

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts from Linear issues."""
        # Resolve team key → team ID if needed
        effective_team_id = self._team_id
        if not effective_team_id and self._team:
            effective_team_id = self._resolve_team_key(self._team)

        cursor: str | None = None
        count = 0
        pages = 0
        self._bodies = []

        while count < self._max_items and pages < self._MAX_PAGES:
            page_size = min(50, self._max_items - count)
            variables: dict = {"first": page_size}
            if cursor:
                variables["after"] = cursor

            if effective_team_id:
                variables["teamId"] = effective_team_id
                query = _ISSUES_QUERY
            else:
                query = _ISSUES_QUERY_NO_TEAM

            resp = self._request_with_retry(
                "POST",
                LINEAR_API,
                json={"query": query, "variables": variables},
            )
            resp.raise_for_status()
            data = resp.json()

            if "errors" in data:
                logger.error("Linear GraphQL errors: %s", data["errors"])
                break

            issues_data = data.get("data", {}).get("issues", {})
            nodes = issues_data.get("nodes", [])
            if not nodes:
                break

            for node in nodes:
                # Client-side filters
                if self._project:
                    proj = (node.get("project") or {}).get("name", "")
                    if proj != self._project:
                        continue
                if self._state:
                    st = (node.get("state") or {}).get("type", "")
                    if st != self._state:
                        continue

                yield from self._issue_to_claims(node)

                # Collect description for text extraction
                desc = node.get("description") or ""
                if desc.strip():
                    identifier = node.get("identifier", node.get("id", ""))
                    self._bodies.append((f"linear:{identifier}:body", desc))

                count += 1

            page_info = issues_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            pages += 1
            time.sleep(0.1)

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield collected issue description bodies."""
        return iter(self._bodies)

    @staticmethod
    def _person_ext(person_data: dict) -> dict[str, str] | None:
        """Build external_ids for a person (assignee or creator)."""
        name = (person_data or {}).get("name", "")
        if not name:
            return None
        ext: dict[str, str] = {"person_name": name}
        email = (person_data or {}).get("email", "")
        if email:
            ext["email"] = email
        return ext

    def _issue_to_claims(self, node: dict) -> Iterator[dict]:
        identifier = node.get("identifier", node.get("id", ""))
        source_id = f"linear:{identifier}"

        # Title
        title = node.get("title", "")
        if title:
            yield self._make_claim(
                identifier, "issue", "titled",
                title, "title", source_id,
            )

        # Creator (with external_ids)
        creator_data = node.get("creator") or {}
        creator = creator_data.get("name", "")
        if creator:
            yield self._make_claim(
                identifier, "issue", "authored_by",
                creator, "person", source_id,
                obj_ext=self._person_ext(creator_data),
            )

        # State
        state = (node.get("state") or {}).get("name", "")
        if state:
            yield self._make_claim(
                identifier, "issue", "has_state",
                state, "status", source_id,
            )

        # Assignee (with external_ids)
        assignee_data = node.get("assignee") or {}
        assignee = assignee_data.get("name", "")
        if assignee:
            yield self._make_claim(
                identifier, "issue", "assigned_to",
                assignee, "person", source_id,
                obj_ext=self._person_ext(assignee_data),
            )

        # Labels
        labels_data = node.get("labels") or {}
        for label in labels_data.get("nodes", []):
            label_name = label.get("name", "")
            if label_name:
                yield self._make_claim(
                    identifier, "issue", "labeled",
                    label_name, "label", source_id,
                )

        # Team
        team = (node.get("team") or {}).get("name", "")
        if team:
            yield self._make_claim(
                identifier, "issue", "belongs_to",
                team, "team", source_id,
            )

        # Priority
        priority = node.get("priority")
        if priority is not None:
            priority_label = _PRIORITY_LABELS.get(
                priority, f"priority_{priority}",
            )
            yield self._make_claim(
                identifier, "issue", "has_priority",
                priority_label, "priority", source_id,
            )

        # Project
        project = (node.get("project") or {}).get("name", "")
        if project:
            yield self._make_claim(
                identifier, "issue", "belongs_to",
                project, "project", source_id,
            )
