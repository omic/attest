"""PolicyStore — JSON-sidecar persistence for groups, memberships, and policy rules.

Follows the same sidecar pattern as RBACManager ({db_path}.policy.json).
Supports in-memory mode (path=None) for tests.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from attestdb.core.types import (
    Entitlement,
    Group,
    GroupMembership,
    PolicyRule,
    SensitivityLevel,
)

logger = logging.getLogger(__name__)

_ROLE_HIERARCHY = {"admin": 3, "writer": 2, "reader": 1}
_VALID_GROUP_ROLES = {"member", "manager", "owner"}
_VALID_EFFECTS = {"allow", "deny"}


class PolicyStore:
    """Manages groups, memberships, and policy rules with JSON persistence."""

    def __init__(self, path: Optional[str] = None, audit_log=None) -> None:
        self._path = path
        self._audit = audit_log
        self._groups: dict[str, Group] = {}
        self._memberships: list[GroupMembership] = []
        self._rules: dict[str, PolicyRule] = {}
        if path:
            self._load()

    def _emit(self, event: str, **kwargs) -> None:
        """Write an audit event if audit log is configured."""
        if self._audit:
            self._audit.write(event, **kwargs)

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for gd in data.get("groups", []):
                g = Group(**gd)
                self._groups[g.group_id] = g
            for md in data.get("memberships", []):
                self._memberships.append(GroupMembership(**md))
            for rd in data.get("rules", []):
                raw_sens = rd.get("sensitivity_max", 3)
                try:
                    rd["sensitivity_max"] = SensitivityLevel[raw_sens] if isinstance(raw_sens, str) else SensitivityLevel(raw_sens)
                except (KeyError, ValueError):
                    rd["sensitivity_max"] = SensitivityLevel.RESTRICTED
                r = PolicyRule(**rd)
                self._rules[r.rule_id] = r
            logger.info("PolicyStore: loaded %d groups, %d memberships, %d rules",
                        len(self._groups), len(self._memberships), len(self._rules))
        except Exception as exc:
            logger.warning("PolicyStore: failed to load %s: %s", self._path, exc)

    def _save(self) -> None:
        if not self._path:
            return
        data = {
            "groups": [
                {k: v for k, v in g.__dict__.items()}
                for g in self._groups.values()
            ],
            "memberships": [
                {k: v for k, v in m.__dict__.items()}
                for m in self._memberships
            ],
            "rules": [
                {k: (v.value if isinstance(v, SensitivityLevel) else v)
                 for k, v in r.__dict__.items()}
                for r in self._rules.values()
            ],
        }
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self._path)

    # ── Group CRUD ──────────────────────────────────────────────────

    def create_group(self, group: Group, actor: str = "") -> None:
        # Validate parent hierarchy depth
        if group.parent_group_id:
            depth = 0
            pid = group.parent_group_id
            while pid and depth < 20:
                parent = self._groups.get(pid)
                if not parent:
                    break
                pid = parent.parent_group_id
                depth += 1
            if depth >= 20:
                raise ValueError(f"Group hierarchy too deep (max 20). Cannot add {group.group_id} under {group.parent_group_id}")
        self._groups[group.group_id] = group
        logger.info("Group created: %s by %s", group.group_id, actor)
        self._emit("group_created", group_id=group.group_id, org_id=group.org_id)
        self._save()

    def get_group(self, group_id: str) -> Optional[Group]:
        return self._groups.get(group_id)

    def list_groups(self, org_id: str = "") -> list[Group]:
        if not org_id:
            return list(self._groups.values())
        return [g for g in self._groups.values() if g.org_id == org_id]

    def delete_group(self, group_id: str, actor: str = "") -> None:
        self._groups.pop(group_id, None)
        self._memberships = [m for m in self._memberships if m.group_id != group_id]
        self._rules = {k: v for k, v in self._rules.items()
                       if group_id not in v.group_ids}
        logger.info("Group deleted: %s by %s", group_id, actor)
        self._emit("group_deleted", group_id=group_id)
        self._save()

    # ── Membership CRUD ─────────────────────────────────────────────

    def add_member(self, membership: GroupMembership, actor: str = "") -> None:
        if membership.role_in_group not in _VALID_GROUP_ROLES:
            raise ValueError(f"Invalid role_in_group: {membership.role_in_group}. Must be one of {_VALID_GROUP_ROLES}")
        # Remove existing membership for this principal+group if any
        self._memberships = [
            m for m in self._memberships
            if not (m.principal_id == membership.principal_id and m.group_id == membership.group_id)
        ]
        membership.added_by = actor
        membership.added_at = int(time.time() * 1_000_000_000)
        self._memberships.append(membership)
        logger.info("Member added: %s → %s by %s",
                     membership.principal_id, membership.group_id, actor)
        self._emit("member_added", principal_id=membership.principal_id,
                    group_id=membership.group_id, role_in_group=membership.role_in_group)
        self._save()

    def remove_member(self, principal_id: str, group_id: str, actor: str = "") -> None:
        self._memberships = [
            m for m in self._memberships
            if not (m.principal_id == principal_id and m.group_id == group_id)
        ]
        logger.info("Member removed: %s from %s by %s", principal_id, group_id, actor)
        self._emit("member_removed", principal_id=principal_id, group_id=group_id)
        self._save()

    def get_memberships(self, principal_id: str) -> list[GroupMembership]:
        return [m for m in self._memberships if m.principal_id == principal_id]

    def get_membership(self, principal_id: str, group_id: str) -> Optional[GroupMembership]:
        for m in self._memberships:
            if m.principal_id == principal_id and m.group_id == group_id:
                return m
        return None

    def get_group_members(self, group_id: str) -> list[GroupMembership]:
        return [m for m in self._memberships if m.group_id == group_id]

    # ── Policy Rule CRUD ────────────────────────────────────────────

    def add_rule(self, rule: PolicyRule, actor: str = "") -> None:
        if not (0 <= rule.priority <= 10000):
            raise ValueError(f"Priority must be 0-10000, got {rule.priority}")
        if rule.effect not in _VALID_EFFECTS:
            raise ValueError(f"Invalid effect: {rule.effect}. Must be one of {_VALID_EFFECTS}")
        rule.created_by = actor
        rule.created_at = int(time.time() * 1_000_000_000)
        self._rules[rule.rule_id] = rule
        logger.info("Policy rule added: %s by %s", rule.rule_id, actor)
        self._emit("policy_created", rule_id=rule.rule_id, description=rule.description)
        self._save()

    def remove_rule(self, rule_id: str, actor: str = "") -> None:
        self._rules.pop(rule_id, None)
        logger.info("Policy rule removed: %s by %s", rule_id, actor)
        self._emit("policy_deleted", rule_id=rule_id)
        self._save()

    def list_rules(self, org_id: str = "") -> list[PolicyRule]:
        if not org_id:
            return list(self._rules.values())
        return [r for r in self._rules.values() if r.org_id == org_id]

    def get_rule(self, rule_id: str) -> Optional[PolicyRule]:
        return self._rules.get(rule_id)

    # ── Group hierarchy ─────────────────────────────────────────────

    def import_legacy_rbac(self, grants: dict[str, dict[str, str]]) -> int:
        """Import legacy RBAC grants as PolicyRules.

        Args:
            grants: {principal_id: {namespace: role}} from RBACManager.

        Returns:
            Number of rules imported.
        """
        count = 0
        for principal_id, ns_roles in grants.items():
            for namespace, role in ns_roles.items():
                rule_id = f"legacy:{principal_id}:{namespace}"
                if rule_id in self._rules:
                    continue  # already imported
                actions = {"admin": ["read", "write", "admin"],
                           "writer": ["read", "write"],
                           "reader": ["read"]}.get(role, ["read"])
                if namespace == "*":
                    logger.warning("Legacy RBAC: converting wildcard namespace for %s to org-wide %s access",
                                   principal_id, role)
                self._rules[rule_id] = PolicyRule(
                    rule_id=rule_id, org_id="", priority=50,
                    effect="allow", principal_ids=[principal_id],
                    namespaces=[namespace] if namespace != "*" else [],
                    actions=actions,
                    description=f"Imported from legacy RBAC: {principal_id} → {namespace} ({role})",
                )
                count += 1
        if count:
            logger.info("Imported %d legacy RBAC grants as policy rules", count)
            self._emit("legacy_rbac_imported", count=count)
            self._save()
        return count

    def expand_groups(self, principal_id: str, max_depth: int = 20) -> list[str]:
        """Return all group_ids for a principal, including parent groups.

        Handles cycles and caps hierarchy depth at max_depth.
        """
        direct = {m.group_id for m in self.get_memberships(principal_id)}
        all_groups = set(direct)
        frontier = list(direct)
        visited: set[str] = set()
        depth = 0
        while frontier and depth < max_depth:
            gid = frontier.pop()
            if gid in visited:
                continue
            visited.add(gid)
            all_groups.add(gid)
            group = self._groups.get(gid)
            if group and group.parent_group_id and group.parent_group_id not in visited:
                all_groups.add(group.parent_group_id)
                frontier.append(group.parent_group_id)
            depth += 1
        return sorted(all_groups)
