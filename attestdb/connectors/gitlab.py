"""GitLab connector — imports issues, merge requests, and commits as knowledge claims.

Uses the GitLab REST API v4 to fetch issues, MRs, and commit activity
from projects and converts them to claims representing the project's
development activity.

Usage::

    from attestdb.connectors.gitlab import GitLabConnector

    conn = GitLabConnector(
        token=os.environ["GITLAB_TOKEN"],
        project_id="omic/next/research/substratedb",
        include_mrs=True,
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator
from urllib.parse import quote_plus

from attestdb.connectors.base import HybridConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GITLAB_API = "https://gitlab.com/api/v4"


class GitLabConnector(HybridConnector):
    """Sync connector that imports GitLab issues/MRs as claims."""

    name = "gitlab"

    def __init__(
        self,
        token: str,
        project_id: str | int | None = None,
        group_id: str | int | None = None,
        base_url: str = GITLAB_API,
        include_mrs: bool = True,
        include_commits: bool = True,
        state: str = "all",
        max_items: int = 500,
        labels: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the GitLab connector")
        self._token = token
        self._project_id = project_id
        self._group_id = group_id
        self._base_url = base_url.rstrip("/")
        self._include_mrs = include_mrs
        self._include_commits = include_commits
        self._state = state
        self._max_items = max_items
        self._labels = labels
        self._session = requests.Session()
        self._session.headers.update({
            "PRIVATE-TOKEN": token,
            "Accept": "application/json",
        })
        self._user_cache: dict[str, dict[str, str]] = {}  # username -> {"email": ...}

    @property
    def supports_search(self) -> bool:
        return True

    def search(self, query: str) -> str:
        """Search GitLab issues/MRs matching *query*."""
        if not self._project_id:
            return ""
        try:
            pid = self._encode_project_id()
            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/projects/{pid}/search",
                params={"scope": "issues", "search": query, "per_page": 10},
            )
            resp.raise_for_status()
            items = resp.json()
            parts = []
            for item in items[:10]:
                title = item.get("title", "")
                body = (item.get("description") or "")[:500]
                labels = ", ".join(item.get("labels", []))
                entry = f"[#{item.get('iid', '')}] {title}"
                if labels:
                    entry += f" ({labels})"
                if body:
                    entry += f"\n{body}"
                parts.append(entry)
            return "\n\n".join(parts)[:4000]
        except Exception as exc:
            logger.warning("GitLab search failed: %s", exc)
            return ""

    def _encode_project_id(self) -> str:
        """URL-encode project path for API calls."""
        pid = str(self._project_id)
        if "/" in pid:
            return quote_plus(pid)
        return pid

    def _list_projects(self) -> list[dict]:
        """List projects from group or return single project."""
        if self._project_id:
            pid = self._encode_project_id()
            resp = self._request_with_retry(
                "GET", f"{self._base_url}/projects/{pid}",
            )
            resp.raise_for_status()
            return [resp.json()]

        if self._group_id:
            projects = []
            page = 1
            while page <= self._MAX_PAGES:
                resp = self._request_with_retry(
                    "GET",
                    f"{self._base_url}/groups/{self._group_id}/projects",
                    params={
                        "per_page": 100,
                        "page": page,
                        "include_subgroups": "true",
                        "archived": "false",
                    },
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                projects.extend(batch)
                page += 1
                time.sleep(0.1)
            return projects

        return []

    def _person_name(self, author_data: dict) -> str:
        """Return display name for a GitLab user, falling back to username."""
        return author_data.get("name") or author_data.get("username") or "unknown"

    def _person_ext(self, author_data: dict) -> dict[str, str] | None:
        """Build external_ids for a person from GitLab user data.

        Fetches and caches user profiles to extract email for cross-connector
        entity resolution. Includes both gitlab_login and email.
        """
        username = author_data.get("username", "")
        if not username:
            return None

        if username not in self._user_cache:
            ext: dict[str, str] = {"gitlab_login": username}
            user_id = author_data.get("id")
            if user_id:
                try:
                    resp = self._request_with_retry(
                        "GET", f"{self._base_url}/users/{user_id}",
                    )
                    if resp.status_code == 200:
                        profile = resp.json()
                        email = profile.get("public_email") or profile.get("email") or ""
                        name = profile.get("name") or ""
                        if email:
                            ext["email"] = email
                        if name:
                            ext["person_name"] = name
                    time.sleep(0.05)
                except Exception:
                    pass
            self._user_cache[username] = ext

        return self._user_cache[username] or None

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts from GitLab issues and MRs."""
        self._bodies: list[tuple[str, str]] = []
        projects = self._list_projects()
        logger.info("GitLab: %d projects to scan", len(projects))

        for project in projects:
            proj_path = project.get("path_with_namespace", str(project["id"]))
            proj_id = project["id"]
            logger.info("GitLab: scanning %s", proj_path)

            # Issues
            yield from self._fetch_issues(proj_id, proj_path)

            # Merge requests
            if self._include_mrs:
                yield from self._fetch_merge_requests(proj_id, proj_path)

            # Commits (recent)
            if self._include_commits:
                yield from self._fetch_commits(proj_id, proj_path)

    def _fetch_issues(self, proj_id: int, proj_path: str) -> Iterator[dict]:
        """Fetch issues for a project."""
        page = 1
        count = 0
        while count < self._max_items:
            params: dict = {
                "state": self._state,
                "per_page": min(100, self._max_items - count),
                "page": page,
                "order_by": "updated_at",
                "sort": "desc",
            }
            if self._labels:
                params["labels"] = ",".join(self._labels)

            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/projects/{proj_id}/issues",
                params=params,
            )
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break

            for item in items:
                iid = item["iid"]
                subj = f"{proj_path}#{iid}"
                sid = f"gitlab:{subj}"
                author_data = item.get("author", {})
                author = self._person_name(author_data)
                state = item.get("state", "unknown")

                yield self._make_claim(
                    subj, "issue", "authored_by",
                    author, "person", sid,
                    obj_ext=self._person_ext(author_data),
                )

                yield self._make_claim(
                    subj, "issue", "has_state",
                    state, "status", sid,
                )

                for label in item.get("labels", []):
                    yield self._make_claim(
                        subj, "issue", "labeled",
                        label, "label", sid,
                    )

                for assignee in item.get("assignees", []):
                    assignee_name = self._person_name(assignee)
                    if assignee_name and assignee_name != "unknown":
                        yield self._make_claim(
                            subj, "issue", "assigned_to",
                            assignee_name, "person", sid,
                            obj_ext=self._person_ext(assignee),
                        )

                if item.get("milestone"):
                    ms_title = item["milestone"].get("title", "")
                    if ms_title:
                        yield self._make_claim(
                            subj, "issue", "in_milestone",
                            ms_title, "milestone", sid,
                        )

                yield self._make_claim(
                    subj, "issue", "belongs_to",
                    proj_path, "project", sid,
                )

                desc = item.get("description") or ""
                if desc.strip():
                    self._bodies.append((f"{sid}:description", desc))

                count += 1

            page += 1
            time.sleep(0.1)

    def _fetch_merge_requests(self, proj_id: int, proj_path: str) -> Iterator[dict]:
        """Fetch merge requests for a project."""
        page = 1
        count = 0
        while count < self._max_items:
            params: dict = {
                "state": self._state,
                "per_page": min(100, self._max_items - count),
                "page": page,
                "order_by": "updated_at",
                "sort": "desc",
            }

            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/projects/{proj_id}/merge_requests",
                params=params,
            )
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break

            for item in items:
                iid = item["iid"]
                subj = f"{proj_path}!{iid}"
                sid = f"gitlab:{subj}"
                author_data = item.get("author", {})
                author = self._person_name(author_data)
                state = item.get("state", "unknown")

                yield self._make_claim(
                    subj, "merge_request", "authored_by",
                    author, "person", sid,
                    obj_ext=self._person_ext(author_data),
                )

                yield self._make_claim(
                    subj, "merge_request", "has_state",
                    state, "status", sid,
                )

                # Timestamps (velocity metrics — already in the API response)
                created_at = item.get("created_at", "")
                if created_at:
                    yield self._make_claim(
                        subj, "merge_request", "created_on",
                        created_at[:10], "date", sid,
                    )
                closed_at = item.get("closed_at", "")
                if closed_at:
                    yield self._make_claim(
                        subj, "merge_request", "closed_on",
                        closed_at[:10], "date", sid,
                    )
                merged_at = item.get("merged_at", "")
                if merged_at:
                    yield self._make_claim(
                        subj, "merge_request", "merged_on",
                        merged_at[:10], "date", sid,
                    )
                updated_at = item.get("updated_at", "")
                if updated_at:
                    yield self._make_claim(
                        subj, "merge_request", "updated_on",
                        updated_at[:10], "date", sid,
                    )

                if item.get("merged_by"):
                    merged_data = item["merged_by"]
                    merger = self._person_name(merged_data)
                    if merger and merger != "unknown":
                        yield self._make_claim(
                            subj, "merge_request", "merged_by",
                            merger, "person", sid,
                            obj_ext=self._person_ext(merged_data),
                        )

                for label in item.get("labels", []):
                    if label:
                        yield self._make_claim(
                            subj, "merge_request", "labeled",
                            label, "label", sid,
                        )

                for reviewer in item.get("reviewers", []):
                    reviewer_name = self._person_name(reviewer)
                    if reviewer_name and reviewer_name != "unknown":
                        yield self._make_claim(
                            subj, "merge_request", "reviewed_by",
                            reviewer_name, "person", sid,
                            obj_ext=self._person_ext(reviewer),
                        )

                # Source and target branches
                src_branch = item.get("source_branch", "")
                if src_branch:
                    yield self._make_claim(
                        subj, "merge_request", "from_branch",
                        src_branch, "branch", sid,
                    )

                target_branch = item.get("target_branch", "")
                if target_branch:
                    yield self._make_claim(
                        subj, "merge_request", "targets_branch",
                        target_branch, "branch", sid,
                    )

                # Project membership (links item to its project)
                yield self._make_claim(
                    subj, "merge_request", "belongs_to",
                    proj_path, "project", sid,
                )

                desc = item.get("description") or ""
                if desc.strip():
                    self._bodies.append((f"{sid}:description", desc))

                count += 1

            page += 1
            time.sleep(0.1)

    def _fetch_commits(self, proj_id: int, proj_path: str) -> Iterator[dict]:
        """Fetch recent commits for a project."""
        page = 1
        count = 0
        max_commits = min(self._max_items, 200)  # Cap commits
        while count < max_commits:
            resp = self._request_with_retry(
                "GET",
                f"{self._base_url}/projects/{proj_id}/repository/commits",
                params={
                    "per_page": min(100, max_commits - count),
                    "page": page,
                },
            )
            if resp.status_code in (403, 404):
                logger.debug("Commits unavailable for %s (HTTP %d)", proj_path, resp.status_code)
                return
            resp.raise_for_status()
            items = resp.json()
            if not items:
                break

            for item in items:
                sha = item["short_id"]
                subj = f"{proj_path}@{sha}"
                sid = f"gitlab:{subj}"
                author = item.get("author_name", "unknown")
                message = item.get("title", "")

                author_email = item.get("author_email", "")
                commit_ext: dict[str, str] = {}
                if author_email:
                    commit_ext["email"] = author_email
                yield self._make_claim(
                    subj, "commit", "authored_by",
                    author, "person", sid,
                    obj_ext=commit_ext or None,
                )

                # Extract meaningful commit message as a claim
                if message and len(message) > 10:
                    # Parse conventional commit prefix
                    for prefix in ("feat:", "fix:", "refactor:", "test:", "docs:", "chore:"):
                        if message.lower().startswith(prefix):
                            action = prefix.rstrip(":")
                            target = message[len(prefix):].strip()[:100]
                            if target:
                                yield self._make_claim(
                                    author, "person", action,
                                    target, "change", sid,
                                )
                            break

                count += 1

            page += 1
            time.sleep(0.1)

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield collected issue/MR description bodies."""
        return iter(getattr(self, "_bodies", []))
