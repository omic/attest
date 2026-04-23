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
        include_reviews: bool = False,
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
        self._include_reviews = include_reviews
        self._state = state
        self._max_items = max_items
        self._labels = labels
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self._user_cache: dict[str, dict[str, str]] = {}
        self._pr_numbers: list[int] = []

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

    def _fetch_user_info(self, login: str) -> dict[str, str]:
        """Fetch user profile and cache login, email, and display name.

        Returns cached result on repeat calls. Falls back gracefully
        if the API call fails (returns login-only external_ids).
        """
        if login in self._user_cache:
            return self._user_cache[login]

        ext: dict[str, str] = {"github_login": login}
        try:
            resp = self._request_with_retry(
                "GET", f"{GITHUB_API}/users/{login}",
            )
            resp.raise_for_status()
            data = resp.json()
            email = data.get("email") or ""
            name = data.get("name") or ""
            if email:
                ext["email"] = email
            if name:
                ext["person_name"] = name
        except Exception as exc:
            logger.debug("github: failed to fetch user %s: %s", login, exc)

        self._user_cache[login] = ext
        return ext

    def _person_ext(self, login: str) -> dict[str, str]:
        """Build external_ids for a person from their GitHub login."""
        return self._fetch_user_info(login)

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
                author = (
                    item.get("user", {}).get("login", "unknown")
                )

                # Detect merged PRs — GitHub Issues API includes
                # pull_request.merged_at for PRs. Compute state BEFORE the
                # payload so the payload reflects the true final state.
                state = item.get("state", "unknown")
                if is_pr:
                    merged_at = (
                        item.get("pull_request", {}).get("merged_at")
                    )
                    if merged_at:
                        state = "merged"
                        yield self._make_claim(
                            subj, item_type, "merged_on",
                            merged_at[:10], "date", sid,
                        )
                    self._pr_numbers.append(number)

                pl = {"schema_ref": f"github/{item_type}", "data": {
                    "record_id": str(number),
                    "url": item.get("html_url", f"https://github.com/{self._repo}/issues/{number}"),
                    "title": (item.get("title") or "")[:200],
                    "state": state,
                    "author": author,
                }}

                yield self._make_claim(
                    subj, item_type, "authored_by",
                    author, "person", sid, payload=pl,
                    obj_ext=self._person_ext(author),
                )

                yield self._make_claim(
                    subj, item_type, "has_state",
                    state, "status", sid, payload=pl,
                )

                # Timestamps
                created_at = item.get("created_at", "")
                if created_at:
                    yield self._make_claim(
                        subj, item_type, "created_on",
                        created_at[:10], "date", sid,
                    )
                closed_at = item.get("closed_at", "")
                if closed_at:
                    yield self._make_claim(
                        subj, item_type, "closed_on",
                        closed_at[:10], "date", sid,
                    )
                updated_at = item.get("updated_at", "")
                if updated_at:
                    yield self._make_claim(
                        subj, item_type, "updated_on",
                        updated_at[:10], "date", sid,
                    )

                # Milestone
                milestone = (
                    item.get("milestone") or {}
                ).get("title", "")
                if milestone:
                    yield self._make_claim(
                        subj, item_type, "belongs_to",
                        milestone, "milestone", sid,
                    )

                for label in item.get("labels", []):
                    label_name = label.get("name", "")
                    if label_name:
                        yield self._make_claim(
                            subj, item_type, "labeled",
                            label_name, "label", sid, payload=pl,
                        )

                for assignee in item.get("assignees", []):
                    login = assignee.get("login", "")
                    if login:
                        yield self._make_claim(
                            subj, item_type, "assigned_to",
                            login, "person", sid, payload=pl,
                            obj_ext=self._person_ext(login),
                        )

                count += 1

            page += 1
            time.sleep(0.1)

    def run(self, db, *, batch_size: int = 500):
        """Structural pass, then review pass, then text pass."""
        # Let the parent HybridConnector handle fetch() + fetch_bodies()
        result = super().run(db, batch_size=batch_size)

        # Review pass — after structural pass so _pr_numbers is populated
        if self._include_reviews and self._pr_numbers:
            batch: list[dict] = []
            for claim_dict in self.fetch_reviews():
                batch.append(claim_dict)
                if len(batch) >= batch_size:
                    self._flush(db, batch, result)
                    batch = []
            if batch:
                self._flush(db, batch, result)
            logger.info(
                "github: fetched reviews for %d PRs",
                len(self._pr_numbers),
            )

        return result

    def fetch_reviews(self) -> Iterator[dict]:
        """Yield review claim dicts for PRs collected during fetch().

        Calls the Pull Request Reviews API for each PR number seen
        during the structural pass. Emits approved/rejected claims.

        Only runs when ``include_reviews=True``.
        """
        if not self._include_reviews or not self._pr_numbers:
            return

        logger.info(
            "github: fetching reviews for %d PRs",
            len(self._pr_numbers),
        )
        for pr_num in self._pr_numbers:
            subj = f"{self._repo}#{pr_num}"
            sid = f"github:{subj}:review"
            try:
                resp = self._request_with_retry(
                    "GET",
                    f"{GITHUB_API}/repos/{self._repo}"
                    f"/pulls/{pr_num}/reviews",
                    params={"per_page": 100},
                )
                resp.raise_for_status()
                reviews = resp.json()
            except Exception as exc:
                logger.debug(
                    "github: reviews for PR #%d failed: %s",
                    pr_num, exc,
                )
                continue

            for review in reviews:
                review_state = review.get("state", "").lower()
                reviewer = (
                    review.get("user", {}).get("login", "")
                )
                if not reviewer or review_state not in (
                    "approved", "changes_requested",
                ):
                    continue

                predicate = (
                    "approved" if review_state == "approved"
                    else "rejected"
                )
                yield self._make_claim(
                    subj, "pull_request", predicate,
                    reviewer, "person", sid,
                    obj_ext=self._person_ext(reviewer),
                )

            time.sleep(0.05)

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
