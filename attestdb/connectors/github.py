"""GitHub connector — imports issues and PRs as knowledge claims.

Uses the GitHub REST API to fetch issues and pull requests from a
repository and converts them to claims representing the project's
issue/PR activity.

Usage::

    from attestdb.connectors.github import GitHubConnector

    conn = GitHubConnector(
        token=os.environ["GITHUB_TOKEN"],
        repo="owner/repo",
        include_prs=True,
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

GITHUB_API = "https://api.github.com"


class GitHubConnector(HybridConnector):
    """Sync connector that imports GitHub issues/PRs as claims."""

    name = "github"

    def __init__(
        self,
        token: str,
        repo: str,
        include_prs: bool = True,
        state: str = "all",
        max_items: int = 500,
        labels: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the GitHub connector")
        self._token = token
        self._repo = repo
        self._include_prs = include_prs
        self._state = state
        self._max_items = max_items
        self._labels = labels
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })

    @property
    def supports_search(self) -> bool:
        return True

    def search(self, query: str) -> str:
        """Search GitHub issues/PRs matching *query*."""
        try:
            resp = self._request_with_retry(
                "GET",
                f"{GITHUB_API}/search/issues",
                params={"q": f"{query} repo:{self._repo}", "per_page": 10},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            parts = []
            for item in items[:10]:
                title = item.get("title", "")
                body = (item.get("body") or "")[:500]
                labels = ", ".join(l["name"] for l in item.get("labels", []))
                entry = f"[#{item.get('number', '')}] {title}"
                if labels:
                    entry += f" ({labels})"
                if body:
                    entry += f"\n{body}"
                parts.append(entry)
            return "\n\n".join(parts)[:4000]
        except Exception as exc:
            logger.warning("GitHub search failed: %s", exc)
            return ""

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts from GitHub issues."""
        page = 1
        count = 0
        while count < self._max_items:
            params: dict = {
                "state": self._state,
                "per_page": min(100, self._max_items - count),
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            if self._labels:
                params["labels"] = ",".join(self._labels)

            resp = self._request_with_retry(
                "GET",
                f"{GITHUB_API}/repos/{self._repo}/issues",
                params=params,
            )
            resp.raise_for_status()
            items = resp.json()

            if not items:
                break

            for item in items:
                is_pr = "pull_request" in item
                if is_pr and not self._include_prs:
                    continue

                item_type = "pull_request" if is_pr else "issue"
                number = item["number"]
                subj = f"{self._repo}#{number}"
                sid = f"github:{subj}"
                state = item.get("state", "unknown")
                author = (
                    item.get("user", {}).get("login", "unknown")
                )

                yield self._make_claim(
                    subj, item_type, "authored_by",
                    author, "person", sid,
                    obj_ext={"github_login": author},
                )

                yield self._make_claim(
                    subj, item_type, "has_state",
                    state, "status", sid,
                )

                for label in item.get("labels", []):
                    label_name = label.get("name", "")
                    if label_name:
                        yield self._make_claim(
                            subj, item_type, "labeled",
                            label_name, "label", sid,
                        )

                for assignee in item.get("assignees", []):
                    login = assignee.get("login", "")
                    if login:
                        yield self._make_claim(
                            subj, item_type, "assigned_to",
                            login, "person", sid,
                            obj_ext={"github_login": login},
                        )

                count += 1

            page += 1
            time.sleep(0.1)

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, body_text)`` from issue/PR bodies."""
        page = 1
        count = 0
        while count < self._max_items:
            params: dict = {
                "state": self._state,
                "per_page": min(100, self._max_items - count),
                "page": page,
                "sort": "updated",
                "direction": "desc",
            }
            if self._labels:
                params["labels"] = ",".join(self._labels)
            resp = self._request_with_retry(
                "GET",
                f"{GITHUB_API}/repos/{self._repo}/issues",
                params=params,
            )
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break

            for item in items:
                is_pr = "pull_request" in item
                if is_pr and not self._include_prs:
                    count += 1
                    continue
                body = item.get("body") or ""
                if body.strip():
                    source_id = (
                        f"github:{self._repo}#{item['number']}:body"
                    )
                    yield (source_id, body)
                count += 1

            page += 1
            time.sleep(0.1)
