"""AccessResolver — resolves a principal's effective access from groups and policies.

Pipeline: principal_id → group expansion → policy collection → evaluation → entitlement.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from attestdb.core.types import (
    Entitlement,
    PolicyRule,
    Principal,
    SensitivityLevel,
)

logger = logging.getLogger(__name__)

_ROLE_HIERARCHY = {"admin": 3, "writer": 2, "reader": 1}


class AccessResolver:
    """Resolves a principal's effective access rights from groups and policies."""

    def __init__(self, policy_store) -> None:
        self._store = policy_store

    def resolve_entitlement(self, principal_id: str, org_id: str = "") -> Entitlement:
        """Full resolution pipeline. Returns computed Entitlement."""
        # Step 1: Expand groups (including parent hierarchy)
        group_ids = self._store.expand_groups(principal_id)

        # Step 2: Get all roles from memberships
        memberships = self._store.get_memberships(principal_id)
        membership_roles = {m.role_in_group for m in memberships}

        # Step 3: Collect matching policy rules
        all_rules = self._store.list_rules(org_id)
        matching = self._collect_matching_rules(
            principal_id, group_ids, list(membership_roles), all_rules,
        )

        # Step 4: Evaluate allow/deny
        return self._evaluate_rules(principal_id, org_id, matching)

    def build_principal(self, principal_id: str, org_id: str = "") -> Principal:
        """Construct a fully-resolved Principal with entitlement."""
        entitlement = self.resolve_entitlement(principal_id, org_id)
        groups = self._store.expand_groups(principal_id)
        return Principal(
            principal_id=principal_id,
            principal_type="user",
            org_id=org_id,
            roles=[entitlement.effective_role],
            clearance=entitlement.clearance,
            groups=groups,
        )

    def check_claim_access(
        self,
        claim_dict: dict,
        entitlement: Entitlement,
    ) -> bool:
        """Check if an entitlement grants access to a specific claim.

        Checks namespace, sensitivity, entity type, source type, predicates.
        """
        if entitlement.is_org_admin:
            return True

        # Namespace check
        # Empty namespace ("") = unscoped claim, visible to all authenticated users
        # Non-empty namespace = must be in allowed list
        claim_ns = claim_dict.get("namespace", "")
        if claim_ns in entitlement.denied_namespaces:
            return False
        if entitlement.allowed_namespaces and claim_ns:
            if claim_ns not in entitlement.allowed_namespaces:
                return False

        # Sensitivity check (claim sensitivity vs principal clearance)
        claim_sensitivity = claim_dict.get("sensitivity", 0)
        if isinstance(claim_sensitivity, int) and claim_sensitivity > entitlement.clearance.value:
            return False

        # Entity type check
        if entitlement.allowed_entity_types:
            subj_type = claim_dict.get("subject", {}).get("entity_type", "")
            obj_type = claim_dict.get("object", {}).get("entity_type", "")
            if subj_type and subj_type not in entitlement.allowed_entity_types:
                return False
            if obj_type and obj_type not in entitlement.allowed_entity_types:
                return False

        # Source type check
        if entitlement.allowed_source_types:
            src_type = claim_dict.get("provenance", {}).get("source_type", "")
            if src_type and src_type not in entitlement.allowed_source_types:
                return False

        # Predicate check
        if hasattr(entitlement, 'allowed_predicates') and entitlement.allowed_predicates:
            pred_id = claim_dict.get("predicate", {}).get("id", "")
            if pred_id and pred_id not in entitlement.allowed_predicates:
                return False

        return True

    # ── Internal ────────────────────────────────────────────────────

    def _collect_matching_rules(
        self,
        principal_id: str,
        group_ids: list[str],
        roles: list[str],
        all_rules: list[PolicyRule],
    ) -> list[PolicyRule]:
        """Collect all rules that apply to this principal."""
        matching = []
        group_set = set(group_ids)
        role_set = set(roles)

        for rule in all_rules:
            # Match by principal_id
            if rule.principal_ids and principal_id in rule.principal_ids:
                matching.append(rule)
                continue
            # Match by group_id
            if rule.group_ids and group_set & set(rule.group_ids):
                matching.append(rule)
                continue
            # Match by role
            if rule.roles and role_set & set(rule.roles):
                matching.append(rule)
                continue
            # Rule with no subject selectors = applies to everyone
            if not rule.principal_ids and not rule.group_ids and not rule.roles:
                matching.append(rule)

        return matching

    def _evaluate_rules(
        self,
        principal_id: str,
        org_id: str,
        rules: list[PolicyRule],
    ) -> Entitlement:
        """Evaluate matched rules to compute effective entitlement.

        Higher priority evaluated first. Explicit deny wins at same priority.
        """
        # Sort by priority descending
        rules.sort(key=lambda r: -r.priority)

        allowed_ns: set[str] = set()
        denied_ns: set[str] = set()
        max_clearance = SensitivityLevel.PUBLIC
        max_role = "reader"
        is_admin = False
        allowed_entity_types: set[str] = set()
        allowed_source_types: set[str] = set()
        allowed_predicates: set[str] = set()

        # Process DENY rules first (explicit deny always wins)
        for rule in rules:
            if rule.effect == "deny":
                denied_ns.update(rule.namespaces)

        # Then process ALLOW rules
        for rule in rules:
            if rule.effect != "allow":
                continue

            # Allow rule
            if rule.namespaces:
                allowed_ns.update(rule.namespaces)
            else:
                # Empty namespaces = wildcard (all)
                is_admin = True

            if rule.sensitivity_max.value > max_clearance.value:
                max_clearance = rule.sensitivity_max

            # Resolve role from actions
            if not rule.actions or "admin" in rule.actions:
                rule_role = "admin"
            elif "write" in rule.actions:
                rule_role = "writer"
            else:
                rule_role = "reader"
            if _ROLE_HIERARCHY.get(rule_role, 0) > _ROLE_HIERARCHY.get(max_role, 0):
                max_role = rule_role

            if rule.entity_types:
                allowed_entity_types.update(rule.entity_types)
            if rule.source_types:
                allowed_source_types.update(rule.source_types)
            if rule.predicates:
                allowed_predicates.update(rule.predicates)

        # If any rule grants wildcard namespace, mark as admin
        if is_admin and max_role == "admin":
            is_admin = True
        elif not is_admin:
            is_admin = False

        return Entitlement(
            principal_id=principal_id,
            org_id=org_id,
            effective_role=max_role,
            allowed_namespaces=sorted(allowed_ns - denied_ns) if allowed_ns else [],
            denied_namespaces=sorted(denied_ns),
            clearance=max_clearance,
            allowed_entity_types=sorted(allowed_entity_types),
            allowed_source_types=sorted(allowed_source_types),
            allowed_predicates=sorted(allowed_predicates),
            is_org_admin=is_admin,
            computed_at=int(time.time() * 1_000_000_000),
        )
