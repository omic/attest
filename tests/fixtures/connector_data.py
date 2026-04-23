"""Deterministic API response generators for connector integration tests.

Each ``generate_{name}_responses()`` function returns a dict of endpoint
patterns → response data matching the real API shapes.  Use with
``make_mock_router()`` to create a ``_mock_request`` function for
``patch.object(conn._session, "request", side_effect=router)``.

Usage::

    from tests.fixtures.connector_data import (
        FakeResponse, make_mock_router, generate_github_responses,
    )

    data = generate_github_responses(n_issues=20)
    router = make_mock_router(data["routes"])
    with patch.object(conn._session, "request", side_effect=router):
        result = conn.run(db)
"""

from __future__ import annotations

import base64
import json


# ── Shared infrastructure ──────────────────────────────────────────

class FakeResponse:
    """Minimal ``requests.Response`` stand-in for connector tests."""

    def __init__(
        self,
        json_data: dict | list | None = None,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict | None = None,
    ):
        self._json = json_data
        self.status_code = status_code
        self.text = text
        self.content = content or (
            text.encode() if isinstance(text, str) else b""
        )
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def make_mock_router(
    routes: dict[str, object],
) -> callable:
    """Build a ``_mock_request(method, url, **kwargs)`` from URL routes.

    *routes* maps URL substrings to either:
    - A ``FakeResponse`` instance (returned directly)
    - A callable ``(method, url, **kwargs) -> FakeResponse``
    - A dict/list (wrapped in ``FakeResponse``)

    Falls back to ``FakeResponse({})`` for unmatched URLs.
    """
    def _mock_request(method, url, **kwargs):
        for pattern, handler in routes.items():
            if pattern in url:
                if isinstance(handler, FakeResponse):
                    return handler
                if callable(handler):
                    return handler(method, url, **kwargs)
                return FakeResponse(handler)
        return FakeResponse({})
    return _mock_request


# ── Name helpers ───────────────────────────────────────────────────

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve",
    "Frank", "Grace", "Hank", "Iris", "Jack",
]
_LAST_NAMES = [
    "Smith", "Jones", "Brown", "Davis", "Wilson",
    "Taylor", "Clark", "Hall", "Lee", "Young",
]
_STATUSES = ["open", "in_progress", "review", "done", "closed"]
_PRIORITIES = ["critical", "high", "medium", "low"]
_LABELS = [
    "bug", "feature", "docs", "refactor", "test",
    "security", "performance", "ux", "backend", "frontend",
]
_DEPARTMENTS = ["engineering", "sales", "marketing", "support", "ops"]
_INDUSTRIES = ["technology", "finance", "healthcare", "retail", "energy"]


def _person(i: int) -> tuple[str, str, str]:
    """Return ``(display_name, email, login)`` for person *i*."""
    first = _FIRST_NAMES[i % len(_FIRST_NAMES)]
    last = _LAST_NAMES[i % len(_LAST_NAMES)]
    name = f"{first} {last}"
    login = f"{first.lower()}.{last.lower()}"
    email = f"{login}@testcorp.com"
    return name, email, login


# ══════════════════════════════════════════════════════════════════════
# GitHub
# ══════════════════════════════════════════════════════════════════════

def generate_github_responses(
    n_issues: int = 20,
    n_prs: int = 5,
    repo: str = "testorg/testrepo",
    page_size: int = 100,
) -> dict:
    """Generate GitHub issues API responses.

    Returns ``{"routes": {...}, "expected": {...}}`` where expected
    contains deterministic counts for assertions.
    """
    items = []
    for i in range(n_issues + n_prs):
        is_pr = i >= n_issues
        name, email, login = _person(i)
        assignee_name, _, assignee_login = _person((i + 3) % 10)
        item: dict = {
            "number": i + 1,
            "title": f"Issue #{i + 1}: {'Add feature' if is_pr else 'Fix bug'} in module-{i % 5}",
            "state": _STATUSES[i % len(_STATUSES)],
            "user": {"login": login},
            "labels": [
                {"name": _LABELS[i % len(_LABELS)]},
                {"name": _LABELS[(i + 1) % len(_LABELS)]},
            ],
            "assignees": [{"login": assignee_login}] if i % 3 != 0 else [],
            "body": f"Description for item {i + 1}." if i % 4 != 0 else "",
        }
        if is_pr:
            item["pull_request"] = {"url": f"https://api.github.com/repos/{repo}/pulls/{i + 1}"}
        items.append(item)

    def _issues_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        page = int(params.get("page", 1))
        per_page = int(params.get("per_page", page_size))
        start = (page - 1) * per_page
        page_items = items[start:start + per_page]
        return FakeResponse(page_items)

    # User profile cache for /users/{login} requests
    _user_profiles: dict[str, dict] = {}
    for i in range(len(_FIRST_NAMES)):
        uname, uemail, ulogin = _person(i)
        _user_profiles[ulogin] = {
            "login": ulogin, "name": uname, "email": uemail,
        }

    def _users_handler(method, url, **kwargs):
        # Extract login from /users/{login}
        login = url.rstrip("/").rsplit("/", 1)[-1]
        profile = _user_profiles.get(login, {"login": login})
        return FakeResponse(profile)

    n_with_assignees = sum(1 for i in range(n_issues + n_prs) if i % 3 != 0)
    n_with_bodies = sum(1 for i in range(n_issues + n_prs) if i % 4 != 0)

    return {
        "routes": {
            f"/repos/{repo}/issues": _issues_handler,
            "/search/issues": FakeResponse({"items": []}),
            "/users/": _users_handler,
        },
        "expected": {
            "total_items": n_issues + n_prs,
            "n_issues": n_issues,
            "n_prs": n_prs,
            "n_with_assignees": n_with_assignees,
            "n_with_bodies": n_with_bodies,
            # Each item produces: authored_by + has_state + 2 labels
            # + assigned_to (if applicable)
            "structural_claims": (
                (n_issues + n_prs) * 4  # authored_by + state + 2 labels
                + n_with_assignees      # assigned_to
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# GitLab
# ══════════════════════════════════════════════════════════════════════

def generate_gitlab_responses(
    n_issues: int = 20,
    n_mrs: int = 5,
    project: str = "testorg/testrepo",
) -> dict:
    """Generate GitLab issues + merge requests API responses.

    Returns ``{"routes": {...}, "expected": {...}}`` where expected
    contains deterministic counts for assertions.
    """
    issues = []
    for i in range(n_issues):
        name, email, login = _person(i)
        assignee_name, _, assignee_login = _person((i + 3) % 10)
        issue: dict = {
            "iid": i + 1,
            "title": f"Issue #{i + 1}: Fix bug in module-{i % 5}",
            "state": _STATUSES[i % len(_STATUSES)],
            "author": {"username": login, "name": name},
            "labels": [
                _LABELS[i % len(_LABELS)],
                _LABELS[(i + 1) % len(_LABELS)],
            ],
            "assignees": (
                [{"username": assignee_login, "name": assignee_name}]
                if i % 3 != 0 else []
            ),
            "milestone": (
                {"title": f"v{(i % 3) + 1}.0"}
                if i % 4 == 0 else None
            ),
            "description": (
                f"Description for issue {i + 1}."
                if i % 4 != 0 else ""
            ),
        }
        issues.append(issue)

    mrs = []
    for i in range(n_mrs):
        name, email, login = _person(i + n_issues)
        mrs.append({
            "iid": n_issues + i + 1,
            "title": f"MR !{n_issues + i + 1}: Add feature-{i % 3}",
            "state": "merged" if i % 2 == 0 else "opened",
            "author": {"username": login, "name": name},
            "labels": [_LABELS[(i + 2) % len(_LABELS)]],
            "assignees": [],
            "milestone": None,
            "description": f"Merge request description {i + 1}.",
        })

    # User profile cache for /users?username= requests
    _user_profiles: dict[str, list[dict]] = {}
    for i in range(len(_FIRST_NAMES)):
        uname, uemail, ulogin = _person(i)
        _user_profiles[ulogin] = [{
            "username": ulogin, "name": uname, "public_email": uemail,
        }]

    def _issues_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        page = int(params.get("page", 1))
        per_page = int(params.get("per_page", 100))
        start = (page - 1) * per_page
        return FakeResponse(issues[start:start + per_page])

    def _mrs_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        page = int(params.get("page", 1))
        per_page = int(params.get("per_page", 100))
        start = (page - 1) * per_page
        return FakeResponse(mrs[start:start + per_page])

    def _users_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        username = params.get("username", "")
        return FakeResponse(_user_profiles.get(username, []))

    n_with_assignees = sum(1 for i in range(n_issues) if i % 3 != 0)
    n_with_milestones = sum(1 for i in range(n_issues) if i % 4 == 0)

    return {
        "routes": {
            "/issues": _issues_handler,
            "/merge_requests": _mrs_handler,
            "/users": _users_handler,
            "/search": FakeResponse([]),
        },
        "expected": {
            "total_issues": n_issues,
            "total_mrs": n_mrs,
            "structural_claims": (
                # Issues: authored_by + state + 2 labels + belongs_to(project)
                #   + assigned_to + milestone
                n_issues * 5  # authored_by + state + 2 labels + belongs_to
                + n_with_assignees
                + n_with_milestones
                # MRs: authored_by + state + 1 label + belongs_to(project)
                + n_mrs * 4
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Jira
# ══════════════════════════════════════════════════════════════════════

def generate_jira_responses(
    n_issues: int = 20,
    page_size: int = 50,
) -> dict:
    """Generate Jira search API responses."""
    issues = []
    for i in range(n_issues):
        name, email, _ = _person(i)
        reporter_name, reporter_email, _ = _person((i + 2) % 10)
        issue: dict = {
            "key": f"TEST-{i + 1}",
            "fields": {
                "issuetype": {"name": "Bug" if i % 2 == 0 else "Task"},
                "status": {"name": _STATUSES[i % len(_STATUSES)]},
                "assignee": {
                    "displayName": name,
                    "emailAddress": email,
                } if i % 4 != 0 else None,
                "reporter": {
                    "displayName": reporter_name,
                    "emailAddress": reporter_email,
                },
                "labels": [_LABELS[i % len(_LABELS)]],
                "components": [
                    {"name": f"component-{i % 3}"},
                ] if i % 3 == 0 else [],
                "issuelinks": [
                    {
                        "type": {"outward": "blocks"},
                        "outwardIssue": {
                            "key": f"TEST-{i + 2}",
                            "fields": {"issuetype": {"name": "Task"}},
                        },
                    },
                ] if i % 5 == 0 and i + 2 <= n_issues else [],
            },
        }
        issues.append(issue)

    def _search_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        start_at = int(params.get("startAt", 0))
        max_results = int(params.get("maxResults", page_size))
        page_issues = issues[start_at:start_at + max_results]
        return FakeResponse({
            "total": len(issues),
            "issues": page_issues,
        })

    n_with_assignee = sum(1 for i in range(n_issues) if i % 4 != 0)
    n_with_components = sum(1 for i in range(n_issues) if i % 3 == 0)
    n_with_links = sum(
        1 for i in range(n_issues)
        if i % 5 == 0 and i + 2 <= n_issues
    )

    return {
        "routes": {
            "/rest/api/3/search": _search_handler,
        },
        "expected": {
            "total_issues": n_issues,
            "structural_claims": (
                n_issues               # has_status (all)
                + n_with_assignee      # assigned_to
                + n_issues             # reported_by (all)
                + n_issues             # labeled (1 each)
                + n_with_components    # belongs_to (component)
                + n_with_links         # issue links
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Linear
# ══════════════════════════════════════════════════════════════════════

def generate_linear_responses(
    n_issues: int = 20,
    page_size: int = 50,
) -> dict:
    """Generate Linear GraphQL API responses."""
    nodes = []
    for i in range(n_issues):
        name, email, _ = _person(i)
        creator_name, creator_email, _ = _person((i + 1) % 10)
        node: dict = {
            "id": f"linear-id-{i}",
            "identifier": f"ENG-{100 + i}",
            "title": f"Issue {i}: work on module-{i % 5}",
            "description": (
                f"Detailed description for ENG-{100 + i}."
                if i % 3 != 0 else ""
            ),
            "state": {
                "name": _STATUSES[i % len(_STATUSES)],
                "type": "started" if i % 2 == 0 else "backlog",
            },
            "assignee": {"name": name, "email": email} if i % 4 != 0 else None,
            "creator": {"name": creator_name, "email": creator_email},
            "labels": {
                "nodes": [{"name": _LABELS[i % len(_LABELS)]}],
            },
            "team": {"name": "Engineering", "key": "ENG"},
            "priority": i % 5,
            "project": (
                {"name": f"Project-{i % 3}"}
                if i % 3 != 0 else None
            ),
        }
        nodes.append(node)

    def _graphql_handler(method, url, **kwargs):
        body = kwargs.get("json", {})
        variables = body.get("variables", {})
        query_text = body.get("query", "")

        if "teams" in query_text:
            return FakeResponse({
                "data": {"teams": {"nodes": [
                    {"id": "team-eng", "name": "Engineering", "key": "ENG"},
                ]}},
            })

        if "issueSearch" in query_text:
            return FakeResponse({
                "data": {"issueSearch": {"nodes": nodes[:5]}},
            })

        first = variables.get("first", page_size)
        after = variables.get("after")
        start = 0
        if after:
            start = int(after.split("-")[-1]) + 1
        page_nodes = nodes[start:start + first]
        has_next = start + first < len(nodes)
        end_cursor = (
            f"cursor-{start + len(page_nodes) - 1}"
            if has_next else None
        )
        return FakeResponse({
            "data": {
                "issues": {
                    "pageInfo": {
                        "hasNextPage": has_next,
                        "endCursor": end_cursor,
                    },
                    "nodes": page_nodes,
                },
            },
        })

    n_with_assignee = sum(1 for i in range(n_issues) if i % 4 != 0)
    n_with_project = sum(1 for i in range(n_issues) if i % 3 != 0)
    n_with_description = sum(1 for i in range(n_issues) if i % 3 != 0)

    return {
        "routes": {
            "api.linear.app/graphql": _graphql_handler,
        },
        "expected": {
            "total_issues": n_issues,
            "n_with_description": n_with_description,
            "structural_claims": (
                n_issues               # titled
                + n_issues             # authored_by (creator, all)
                + n_issues             # has_state
                + n_with_assignee      # assigned_to
                + n_issues             # labeled (1 each)
                + n_issues             # belongs_to (team, all)
                + n_issues             # has_priority
                + n_with_project       # belongs_to (project)
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# ServiceNow
# ══════════════════════════════════════════════════════════════════════

def generate_servicenow_responses(
    n_incidents: int = 10,
    n_changes: int = 5,
) -> dict:
    """Generate ServiceNow Table API responses."""
    def _make_records(table, n):
        records = []
        prefix = "INC" if table == "incident" else "CHG"
        for i in range(n):
            name, _, _ = _person(i)
            opener_name, _, _ = _person((i + 2) % 10)
            records.append({
                "number": f"{prefix}{i + 1:07d}",
                "assigned_to": name,
                "state": _STATUSES[i % len(_STATUSES)],
                "priority": _PRIORITIES[i % len(_PRIORITIES)],
                "category": _DEPARTMENTS[i % len(_DEPARTMENTS)],
                "opened_by": opener_name,
                "description": (
                    f"Description for {prefix}{i + 1:07d}."
                    if i % 3 != 0 else ""
                ),
                "close_notes": "",
            })
        return records

    incidents = _make_records("incident", n_incidents)
    changes = _make_records("change_request", n_changes)

    def _table_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        offset = int(params.get("sysparm_offset", 0))
        limit = int(params.get("sysparm_limit", 100))
        if "/incident" in url:
            page = incidents[offset:offset + limit]
        elif "/change_request" in url:
            page = changes[offset:offset + limit]
        else:
            page = []
        return FakeResponse({"result": page})

    total = n_incidents + n_changes
    return {
        "routes": {
            "/api/now/table/": _table_handler,
        },
        "expected": {
            "total_records": total,
            # Each record: assigned_to + has_state + has_priority
            # + categorized_as + opened_by = 5
            "structural_claims": total * 5,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# PagerDuty
# ══════════════════════════════════════════════════════════════════════

def generate_pagerduty_responses(n_incidents: int = 10) -> dict:
    """Generate PagerDuty incidents API responses."""
    incidents = []
    for i in range(n_incidents):
        name, _, _ = _person(i)
        incidents.append({
            "incident_number": i + 1,
            "title": f"Alert: service-{i % 4} degraded",
            "status": ["triggered", "acknowledged", "resolved"][i % 3],
            "urgency": ["high", "low"][i % 2],
            "service": {"summary": f"service-{i % 4}"},
            "assignments": [
                {"assignee": {"summary": name}},
            ] if i % 3 != 0 else [],
            "escalation_policy": {
                "summary": f"policy-{i % 2}",
            } if i % 2 == 0 else None,
        })

    def _incidents_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 100))
        page = incidents[offset:offset + limit]
        return FakeResponse({
            "incidents": page,
            "more": offset + limit < len(incidents),
        })

    n_with_assignee = sum(1 for i in range(n_incidents) if i % 3 != 0)
    n_with_policy = sum(1 for i in range(n_incidents) if i % 2 == 0)

    return {
        "routes": {
            "/incidents": _incidents_handler,
        },
        "expected": {
            "total_incidents": n_incidents,
            "structural_claims": (
                n_incidents           # has_status
                + n_incidents         # has_urgency
                + n_incidents         # affects_service
                + n_with_assignee     # assigned_to
                + n_with_policy       # escalation_policy
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Zendesk
# ══════════════════════════════════════════════════════════════════════

def generate_zendesk_responses(
    n_tickets: int = 20,
    page_size: int = 100,
) -> dict:
    """Generate Zendesk tickets API responses."""
    tickets = []
    for i in range(n_tickets):
        tickets.append({
            "id": i + 1,
            "status": _STATUSES[i % len(_STATUSES)],
            "priority": _PRIORITIES[i % len(_PRIORITIES)] if i % 3 != 0 else None,
            "tags": [_LABELS[i % len(_LABELS)], _LABELS[(i + 1) % len(_LABELS)]],
            "subject": f"Ticket #{i + 1}: issue with feature-{i % 5}",
            "description": (
                f"Customer reports problem with feature-{i % 5}."
                if i % 4 != 0 else ""
            ),
        })

    def _tickets_handler(method, url, **kwargs):
        return FakeResponse({
            "tickets": tickets[:page_size],
            "meta": {"has_more": len(tickets) > page_size},
            "links": {},
        })

    n_with_priority = sum(1 for i in range(n_tickets) if i % 3 != 0)

    return {
        "routes": {
            "/tickets.json": _tickets_handler,
        },
        "expected": {
            "total_tickets": n_tickets,
            "structural_claims": (
                n_tickets              # has_status
                + n_with_priority      # has_priority
                + n_tickets * 2        # tagged (2 per ticket)
                + n_tickets            # titled
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Salesforce
# ══════════════════════════════════════════════════════════════════════

def generate_salesforce_responses(
    n_opportunities: int = 5,
    n_contacts: int = 5,
    n_cases: int = 5,
) -> dict:
    """Generate Salesforce SOQL API responses."""
    opps = []
    for i in range(n_opportunities):
        name, _, _ = _person(i)
        opps.append({
            "Id": f"opp-{i:03d}",
            "Name": f"Deal-{i}: Enterprise License",
            "StageName": ["Prospecting", "Qualification", "Proposal", "Closed Won", "Closed Lost"][i % 5],
            "Amount": 10000 + i * 5000,
            "CloseDate": f"2024-0{(i % 9) + 1}-15",
            "Account": {"Name": f"Acme-{i % 3}"},
            "Owner": {"Name": name},
        })

    contacts = []
    for i in range(n_contacts):
        name, email, _ = _person(i)
        first, last = name.split()
        contacts.append({
            "Id": f"con-{i:03d}",
            "FirstName": first,
            "LastName": last,
            "Email": email,
            "Title": f"{'VP' if i % 2 == 0 else 'Director'} of {_DEPARTMENTS[i % len(_DEPARTMENTS)]}",
            "Account": {"Name": f"Acme-{i % 3}"},
        })

    cases = []
    for i in range(n_cases):
        name, _, _ = _person(i)
        cases.append({
            "Id": f"case-{i:03d}",
            "CaseNumber": f"{10000 + i}",
            "Subject": f"Case {i}: support request",
            "Status": _STATUSES[i % len(_STATUSES)],
            "Priority": _PRIORITIES[i % len(_PRIORITIES)],
            "Contact": {"Name": name},
            "Account": {"Name": f"Acme-{i % 3}"},
            "Description": (
                f"Customer needs help with module-{i % 5}."
                if i % 3 != 0 else ""
            ),
        })

    def _query_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        if "query" in url:
            q = params.get("q", "")
            # Check "FROM Xxx" to avoid substring conflicts
            # (e.g. Case query contains "Contact.Name")
            if "FROM Opportunity" in q:
                return FakeResponse({"records": opps, "done": True})
            if "FROM Case" in q:
                return FakeResponse({"records": cases, "done": True})
            if "FROM Contact" in q:
                return FakeResponse({"records": contacts, "done": True})
        return FakeResponse({"records": [], "done": True})

    def _auth_handler(method, url, **kwargs):
        return FakeResponse({
            "access_token": "test-token",
            "instance_url": "https://test.salesforce.com",
        })

    return {
        "routes": {
            "login.salesforce.com": _auth_handler,
            "/services/data/": _query_handler,
        },
        "expected": {
            "n_opportunities": n_opportunities,
            "n_contacts": n_contacts,
            "n_cases": n_cases,
            # Opps: stage + amount + belongs_to + owned_by + closes_on = 5 each
            # Contacts: has_role + works_at = 2 each
            # Cases: has_status + has_priority + submitted_by + belongs_to = 4 each
            "structural_claims": (
                n_opportunities * 5
                + n_contacts * 2
                + n_cases * 4
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Slack
# ══════════════════════════════════════════════════════════════════════

def generate_slack_responses(
    n_channels: int = 2,
    n_messages: int = 10,
    n_users: int = 5,
) -> dict:
    """Generate Slack Web API responses."""
    channels = []
    for i in range(n_channels + 1):  # +1 for archived
        channels.append({
            "id": f"C{i:03d}",
            "name": f"channel-{i}" if i < n_channels else "archived",
            "is_archived": i >= n_channels,
        })

    users = []
    for i in range(n_users):
        name, email, login = _person(i)
        users.append({
            "id": f"U{i:03d}",
            "name": login,
            "profile": {
                "display_name": name,
                "email": email,
            },
        })

    messages = []
    for i in range(n_messages):
        uid = f"U{i % n_users:03d}"
        text = f"Message {i} about topic-{i % 4}"
        # Add mentions every 3rd message
        if i % 3 == 0 and n_users > 1:
            mentioned_uid = f"U{(i + 1) % n_users:03d}"
            text += f" cc <@{mentioned_uid}>"
        messages.append({
            "text": text,
            "user": uid,
            "ts": f"{1000 + i}",
        })
    # Add system message (should be filtered)
    messages.append({
        "text": "joined the channel",
        "subtype": "channel_join",
        "ts": f"{1000 + n_messages}",
    })

    def _slack_handler(method, url, **kwargs):
        if "conversations.list" in url:
            return FakeResponse({"ok": True, "channels": channels})
        if "users.list" in url:
            return FakeResponse({"ok": True, "members": users})
        if "conversations.history" in url:
            return FakeResponse({"ok": True, "messages": messages})
        if "search.messages" in url:
            return FakeResponse({
                "ok": True,
                "messages": {"matches": []},
            })
        return FakeResponse({"ok": True})

    n_non_trivial = n_messages  # all generated messages are > 8 chars
    n_with_mentions = sum(
        1 for i in range(n_messages)
        if i % 3 == 0 and n_users > 1
    )

    return {
        "routes": {
            "slack.com/api": _slack_handler,
        },
        "expected": {
            "n_channels": n_channels,
            "n_active_channels": n_channels,
            "n_messages": n_messages,
            "n_users": n_users,
            "n_non_trivial": n_non_trivial,
            "n_with_mentions": n_with_mentions,
            # posted_in: 1 per non-trivial message per channel
            # mentioned: 1 per mention per channel
            "structural_claims_per_channel": (
                n_non_trivial + n_with_mentions
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# HubSpot
# ══════════════════════════════════════════════════════════════════════

def generate_hubspot_responses(
    n_contacts: int = 5,
    n_companies: int = 3,
    n_deals: int = 3,
    n_notes: int = 2,
) -> dict:
    """Generate HubSpot CRM API responses."""
    contacts = []
    for i in range(n_contacts):
        name, email, _ = _person(i)
        first, last = name.split()
        contacts.append({
            "id": f"c-{i:03d}",
            "properties": {
                "firstname": first,
                "lastname": last,
                "email": email,
                "company": f"Corp-{i % 3}",
                "jobtitle": f"Engineer {i}",
                "lifecyclestage": ["lead", "subscriber", "opportunity"][i % 3],
            },
        })

    companies = []
    for i in range(n_companies):
        companies.append({
            "id": f"co-{i:03d}",
            "properties": {
                "name": f"Corp-{i}",
                "industry": _INDUSTRIES[i % len(_INDUSTRIES)],
                "domain": f"corp{i}.com",
                "numberofemployees": str(100 * (i + 1)),
            },
        })

    deals = []
    for i in range(n_deals):
        deals.append({
            "id": f"d-{i:03d}",
            "properties": {
                "dealname": f"Deal-{i}: Enterprise",
                "dealstage": f"stage-{i % 3}",
                "pipeline": "default",
                "amount": str(10000 + i * 5000),
                "closedate": f"2024-0{i + 1}-01",
            },
        })

    notes = []
    for i in range(n_notes):
        notes.append({
            "id": f"n-{i:03d}",
            "properties": {
                "hs_note_body": f"Follow up on deal {i}. Customer interested.",
                "hs_timestamp": "2024-01-15T10:00:00Z",
            },
            "associations": {
                "contacts": {"results": [{"id": f"c-{i:03d}"}]},
                "companies": {"results": [{"id": f"co-{i % n_companies:03d}"}]},
                "deals": {"results": []},
            },
        })

    stages = [
        {"id": f"stage-{i}", "label": f"Stage {i} Label"}
        for i in range(5)
    ]

    def _hubspot_handler(method, url, **kwargs):
        if "/pipelines/" in url:
            return FakeResponse({
                "results": [{"stages": stages}],
            })
        if "/contacts" in url:
            return FakeResponse({"results": contacts})
        if "/companies" in url:
            return FakeResponse({"results": companies})
        if "/deals" in url:
            return FakeResponse({"results": deals})
        if "/notes" in url:
            return FakeResponse({"results": notes})
        return FakeResponse({"results": []})

    return {
        "routes": {
            "api.hubapi.com": _hubspot_handler,
        },
        "expected": {
            "n_contacts": n_contacts,
            "n_companies": n_companies,
            "n_deals": n_deals,
            "n_notes": n_notes,
            # Contacts: works_at + has_role + has_stage = 3 each
            # Companies: in_industry + has_domain + has_size = 3 each
            # Deals: has_stage + has_amount + in_pipeline + closes_on = 4 each
            # Notes: 2 associations each (contact + company)
            "structural_claims": (
                n_contacts * 3
                + n_companies * 3
                + n_deals * 4
                + n_notes * 2
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Box
# ══════════════════════════════════════════════════════════════════════

def generate_box_responses(
    n_files: int = 10,
    folder_id: str = "0",
) -> dict:
    """Generate Box API responses for folder listing and file downloads."""
    files = []
    file_contents = {}
    for i in range(n_files):
        name, email, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        file_id = f"f-{i:04d}"
        filename = f"report_{dept}_{i}.txt"
        files.append({
            "type": "file",
            "id": file_id,
            "name": filename,
        })
        file_contents[file_id] = (
            f"Quarterly Report for {dept.title()} Department\n"
            f"Author: {name} ({email})\n"
            f"Status: {'complete' if i % 3 != 0 else 'draft'}\n"
            f"Revenue target achieved: {70 + i * 3}%\n"
            f"Headcount: {10 + i * 5}\n"
        )

    def _box_handler(method, url, **kwargs):
        if f"/folders/{folder_id}/items" in url:
            return FakeResponse({
                "entries": files,
                "total_count": len(files),
            })
        for fid, content in file_contents.items():
            if f"/files/{fid}/content" in url:
                return FakeResponse(text=content)
        return FakeResponse({})

    return {
        "routes": {"api.box.com": _box_handler},
        "expected": {
            "n_files": n_files,
            # Box is a text connector — claim count depends on extraction mode
            # With extraction="none", each file produces 1 raw text claim
            "structural_claims": n_files,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Confluence
# ══════════════════════════════════════════════════════════════════════

def generate_confluence_responses(
    n_pages: int = 10,
    space_key: str = "ENG",
) -> dict:
    """Generate Confluence REST API responses for page listing and content."""
    pages = []
    for i in range(n_pages):
        name, email, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        title = f"{dept.title()} {['Runbook', 'Architecture', 'Onboarding', 'Postmortem', 'RFC'][i % 5]} #{i}"
        body = (
            f"<h1>{title}</h1>"
            f"<p>Author: {name}. Department: {dept}.</p>"
            f"<p>This document describes the {dept} team's "
            f"{'operational procedures' if i % 2 == 0 else 'technical design'}.</p>"
            f"<p>Status: {'approved' if i % 3 != 0 else 'draft'}. "
            f"Last reviewed: 2024-0{(i % 9) + 1}-15.</p>"
        )
        pages.append({
            "id": f"page-{i:04d}",
            "title": title,
            "body": {"storage": {"value": body}},
            "space": {"key": space_key},
            "status": "current",
        })

    def _confluence_handler(method, url, **kwargs):
        params = kwargs.get("params", {})
        if "/rest/api/content" in url:
            start = int(params.get("start", 0))
            limit = int(params.get("limit", 25))
            chunk = pages[start:start + limit]
            return FakeResponse({
                "results": chunk,
                "start": start,
                "limit": limit,
                "size": len(chunk),
                "_links": {},
            })
        return FakeResponse({"results": []})

    return {
        "routes": {"atlassian.net": _confluence_handler},
        "expected": {
            "n_pages": n_pages,
            # TextConnector — with extraction="none", no claims
            "structural_claims": 0,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Google Sheets
# ══════════════════════════════════════════════════════════════════════

def generate_google_sheets_data(
    n_rows: int = 20,
) -> dict:
    """Generate Google Sheets API-compatible row data.

    Returns row data suitable for mocking the Sheets API values.get response,
    plus a mapping dict for the GoogleSheetsConnector.
    """
    headers = ["employee_name", "department", "title", "status", "manager"]
    rows = [headers]
    for i in range(n_rows):
        name, _, _ = _person(i % len(_FIRST_NAMES))
        mgr_name, _, _ = _person((i + 3) % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        title = ["Engineer", "Manager", "Director", "VP", "Analyst"][i % 5]
        status = "active" if i % 5 != 0 else "on_leave"
        rows.append([name, dept, title, status, mgr_name])

    mapping = {
        "subject": "employee_name",
        "subject_type": "person",
        "predicate": "department",
        "predicate_type": "part_of",
        "object": "department",
        "object_type": "entity",
    }

    return {
        "values": rows,
        "mapping": mapping,
        "expected": {
            "n_rows": n_rows,
            "structural_claims": n_rows,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Teams
# ══════════════════════════════════════════════════════════════════════

def generate_teams_responses(
    n_channels: int = 2,
    n_messages: int = 5,
) -> dict:
    """Generate Microsoft Teams Graph API responses."""
    channels = [
        {"id": f"ch-{i}", "displayName": f"channel-{i}"}
        for i in range(n_channels)
    ]

    messages = []
    for i in range(n_messages):
        name, email, _ = _person(i % len(_FIRST_NAMES))
        messages.append({
            "messageType": "message",
            "from": {"user": {"displayName": name, "mail": email}},
            "body": {
                "contentType": "text",
                "content": f"Update from {name}: {_DEPARTMENTS[i % len(_DEPARTMENTS)]} "
                           f"team completed sprint {i} review.",
            },
        })
    # Add a system message (should be skipped)
    messages.append({
        "messageType": "systemEventMessage",
        "body": {"contentType": "text", "content": "User joined the team"},
    })

    def _teams_handler(method, url, **kwargs):
        if "/channels" in url and "/messages" not in url:
            return FakeResponse({"value": channels})
        if "/messages" in url:
            return FakeResponse({"value": messages})
        return FakeResponse({})

    return {
        "routes": {"graph.microsoft.com": _teams_handler},
        "expected": {
            "n_channels": n_channels,
            "n_messages": n_messages,
            # Text connector — structural claims depend on extraction mode
            "structural_claims": 0,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Gmail
# ══════════════════════════════════════════════════════════════════════

def generate_gmail_responses(n_threads: int = 5) -> dict:
    """Generate Gmail REST API responses."""
    import base64 as _b64

    threads = [{"id": f"thread-{i:03d}"} for i in range(n_threads)]

    def _gmail_handler(method, url, **kwargs):
        if "/users/me/threads" in url and "format" not in str(kwargs.get("params", {})):
            return FakeResponse({"threads": threads})
        for i in range(n_threads):
            if f"thread-{i:03d}" in url:
                name, email, _ = _person(i % len(_FIRST_NAMES))
                dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
                body = (
                    f"Hi team,\n\n{name} from {dept} reporting: "
                    f"Q{(i % 4) + 1} results are in. "
                    f"Revenue target {'met' if i % 3 != 0 else 'missed'}.\n\n"
                    f"Best,\n{name}\n{email}"
                )
                encoded = _b64.urlsafe_b64encode(body.encode()).decode()
                return FakeResponse({
                    "messages": [{
                        "payload": {
                            "mimeType": "text/plain",
                            "body": {"data": encoded},
                        },
                    }],
                })
        return FakeResponse({})

    return {
        "routes": {"googleapis.com": _gmail_handler},
        "expected": {
            "n_threads": n_threads,
            "structural_claims": 0,  # Text connector
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Google Docs
# ══════════════════════════════════════════════════════════════════════

def generate_gdocs_responses(n_docs: int = 5) -> dict:
    """Generate Google Drive + Docs API responses."""
    files = [
        {
            "id": f"doc-{i:03d}",
            "name": f"{_DEPARTMENTS[i % len(_DEPARTMENTS)].title()} Report #{i}",
            "mimeType": "application/vnd.google-apps.document",
        }
        for i in range(n_docs)
    ]

    def _gdocs_handler(method, url, **kwargs):
        if "drive/v3/files" in url:
            return FakeResponse({"files": files})
        for i in range(n_docs):
            if f"doc-{i:03d}" in url:
                name, _, _ = _person(i % len(_FIRST_NAMES))
                dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
                return FakeResponse({
                    "body": {
                        "content": [{
                            "paragraph": {
                                "elements": [{
                                    "textRun": {
                                        "content": (
                                            f"Author: {name}, Department: {dept}. "
                                            f"This document covers the {dept} team's "
                                            f"Q{(i % 4) + 1} objectives and results.\n"
                                        ),
                                    },
                                }],
                            },
                        }],
                    },
                })
        return FakeResponse({})

    return {
        "routes": {"googleapis.com": _gdocs_handler},
        "expected": {
            "n_docs": n_docs,
            "structural_claims": n_docs,  # GDocs has custom run() that ingests raw text
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Notion
# ══════════════════════════════════════════════════════════════════════

def generate_notion_responses(n_pages: int = 5) -> dict:
    """Generate Notion API responses (POST search + GET blocks)."""
    pages = []
    for i in range(n_pages):
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        pages.append({
            "id": f"page-{i:03d}",
            "properties": {
                "Name": {
                    "type": "title",
                    "title": [{"plain_text": f"{dept.title()} Wiki #{i}"}],
                },
            },
        })

    def _notion_handler(method, url, **kwargs):
        if method == "POST":
            return FakeResponse({"results": pages, "has_more": False})
        for i in range(n_pages):
            if f"page-{i:03d}" in url:
                name, _, _ = _person(i % len(_FIRST_NAMES))
                dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
                return FakeResponse({
                    "results": [{
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [{
                                "plain_text": (
                                    f"Maintained by {name}. "
                                    f"The {dept} team's internal knowledge base "
                                    f"covering processes and runbooks."
                                ),
                            }],
                        },
                    }],
                    "has_more": False,
                })
        return FakeResponse({"results": [], "has_more": False})

    return {
        "routes": {"api.notion.com": _notion_handler},
        "expected": {
            "n_pages": n_pages,
            "structural_claims": 0,  # Text connector
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Zoho Mail
# ══════════════════════════════════════════════════════════════════════

def generate_zoho_responses(n_emails: int = 5) -> dict:
    """Generate Zoho Mail API responses."""
    emails = []
    for i in range(n_emails):
        name, email_addr, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        emails.append({
            "messageId": f"msg-{i:03d}",
            "folderId": "inbox-001",
            "subject": f"{dept.title()} Update from {name}",
            "fromAddress": email_addr,
            "receivedTime": 1700000000 + i * 3600000,
            "content": (
                f"<html><body><p>Hi team,</p>"
                f"<p>{name} from {dept}: quarterly review complete. "
                f"All targets {'achieved' if i % 3 != 0 else 'under review'}.</p>"
                f"</body></html>"
            ),
        })

    def _zoho_handler(method, url, **kwargs):
        if "/messages/view" in url:
            return FakeResponse({"data": emails})
        for i in range(n_emails):
            if f"msg-{i:03d}" in url and "/content" in url:
                return FakeResponse({
                    "data": {
                        "subject": emails[i]["subject"],
                        "content": emails[i]["content"],
                    },
                })
        return FakeResponse({"data": {}})

    return {
        "routes": {"mail.zoho.com": _zoho_handler},
        "expected": {
            "n_emails": n_emails,
            "structural_claims": 0,  # Text connector
        },
    }


# ══════════════════════════════════════════════════════════════════════
# HTTP (generic)
# ══════════════════════════════════════════════════════════════════════

def generate_http_responses(n_items: int = 10) -> dict:
    """Generate generic HTTP/JSON API responses for HTTPConnector."""
    items = []
    for i in range(n_items):
        name, _, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        items.append({
            "id": f"item-{i:03d}",
            "name": name,
            "relationship": "part_of",
            "department": dept,
            "status": _STATUSES[i % len(_STATUSES)],
        })

    return {
        "routes": {
            "example.com": FakeResponse(items),
        },
        "mapping": {
            "subject": "name", "subject_type": "person",
            "predicate": "relationship", "predicate_type": "part_of",
            "object": "department", "object_type": "entity",
        },
        "expected": {
            "n_items": n_items,
            "structural_claims": n_items,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# Airtable
# ══════════════════════════════════════════════════════════════════════

def generate_airtable_responses(n_records: int = 10) -> dict:
    """Generate Airtable API responses."""
    records = []
    for i in range(n_records):
        name, _, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        records.append({
            "id": f"rec{i:04d}",
            "fields": {
                "Name": name,
                "relationship": "part_of",
                "Department": dept,
                "Status": _STATUSES[i % len(_STATUSES)],
            },
        })

    return {
        "routes": {
            "api.airtable.com": FakeResponse({"records": records}),
        },
        "mapping": {
            "subject": "Name", "subject_type": "person",
            "predicate": "relationship", "predicate_type": "part_of",
            "object": "Department", "object_type": "entity",
        },
        "expected": {
            "n_records": n_records,
            "structural_claims": n_records,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# GDrive
# ══════════════════════════════════════════════════════════════════════

def generate_gdrive_responses(n_files: int = 5) -> dict:
    """Generate Google Drive API responses for file listing + download."""
    files = []
    file_contents = {}
    for i in range(n_files):
        name, email, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        file_id = f"gdrive-{i:03d}"
        files.append({
            "id": file_id,
            "name": f"{dept}_notes_{i}.txt",
            "mimeType": "text/plain",
        })
        file_contents[file_id] = (
            f"Meeting notes - {dept.title()} department\n"
            f"Attendee: {name} ({email})\n"
            f"Topic: Q{(i % 4) + 1} planning and resource allocation.\n"
        )

    def _gdrive_handler(method, url, **kwargs):
        # List endpoint: drive/v3/files (no specific file ID after)
        if "drive/v3/files" in url:
            # Check if this is a file-specific request (has ID in path)
            after = url.split("drive/v3/files")[-1]
            if not after or after == "":
                return FakeResponse({"files": files})
            # File content download
            for fid, content in file_contents.items():
                if fid in url:
                    return FakeResponse(text=content)
        return FakeResponse({})

    return {
        "routes": {"googleapis.com": _gdrive_handler},
        "expected": {
            "n_files": n_files,
            "structural_claims": n_files,  # 1 raw text claim per file
        },
    }


# ══════════════════════════════════════════════════════════════════════
# SharePoint
# ══════════════════════════════════════════════════════════════════════

def generate_sharepoint_responses(n_files: int = 5) -> dict:
    """Generate SharePoint Graph API responses."""
    files = []
    file_contents = {}
    for i in range(n_files):
        name, email, _ = _person(i % len(_FIRST_NAMES))
        dept = _DEPARTMENTS[i % len(_DEPARTMENTS)]
        item_id = f"sp-{i:03d}"
        dl_url = f"https://download.sharepoint.com/{item_id}"
        files.append({
            "id": item_id,
            "name": f"{dept}_policy_{i}.txt",
            "file": {"mimeType": "text/plain"},
            "size": 500 + i * 100,
            "@microsoft.graph.downloadUrl": dl_url,
        })
        file_contents[dl_url] = (
            f"Policy Document: {dept.title()} Department\n"
            f"Owner: {name}\n"
            f"Last review: 2024-0{(i % 9) + 1}-01\n"
            f"Status: {'approved' if i % 3 != 0 else 'pending review'}\n"
        )

    def _sp_handler(method, url, **kwargs):
        if "/children" in url or "/items" in url:
            return FakeResponse({"value": files})
        for dl_url, content in file_contents.items():
            if dl_url in url or url == dl_url:
                return FakeResponse(text=content)
        return FakeResponse({})

    return {
        "routes": {
            "graph.microsoft.com": _sp_handler,
            "download.sharepoint.com": _sp_handler,
        },
        "expected": {
            "n_files": n_files,
            "structural_claims": n_files,
        },
    }
