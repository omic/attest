"""Per-namespace role-based access control (RBAC)."""

from __future__ import annotations

import builtins
import json
import logging
import os
from typing import Callable

from attestdb.core.types import Role
from attestdb.infrastructure.audit_log import AuditLog

logger = logging.getLogger(__name__)

_HIERARCHY = {"admin": 3, "writer": 2, "reader": 1}


class RBACManager:
    """Manages RBAC grants, persistence, and permission checks.

    Args:
        rbac_path: Path to the JSON sidecar file, or None for in-memory only.
        audit: AuditLog instance for actor identity and audit writes.
        get_namespace_fn: Callable returning the current namespace filter list.
    """

    def __init__(
        self,
        rbac_path: str | None,
        audit: AuditLog,
        get_namespace_fn: Callable[[], list[str]],
    ) -> None:
        self._grants: dict[str, dict[str, str]] = {}
        self._enabled: bool = False
        self._path: str | None = rbac_path
        self._audit = audit
        self._get_namespace = get_namespace_fn
        self._load()

    def _load(self) -> None:
        """Load RBAC config from sidecar file."""
        if self._path and os.path.exists(self._path):
            try:
                with builtins.open(self._path) as f:
                    data = json.load(f)
                self._grants = data.get("grants", {})
                self._enabled = data.get("enabled", False)
            except Exception as exc:
                logger.warning("Failed to load RBAC config: %s", exc)

    def _save(self) -> None:
        """Persist RBAC config to sidecar file."""
        if not self._path:
            return
        try:
            with builtins.open(self._path, "w") as f:
                json.dump({"enabled": self._enabled, "grants": self._grants}, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save RBAC config: %s", exc)

    def enable(self) -> None:
        """Enable RBAC enforcement."""
        self._enabled = True
        self._save()

    def disable(self) -> None:
        """Disable RBAC enforcement (all operations permitted)."""
        self._enabled = False
        self._save()

    def grant(self, principal: str, namespace: str, role: str | Role) -> None:
        """Grant a role to a principal on a namespace."""
        if isinstance(role, Role):
            role = role.value
        if role not in ("admin", "writer", "reader"):
            raise ValueError(f"Invalid role: {role}. Must be admin, writer, or reader.")
        self.check_permission("admin")
        grants = self._grants.setdefault(principal, {})
        grants[namespace] = role
        self._save()
        self._audit.write("rbac_grant", principal=principal, namespace=namespace, role=role)

    def revoke(self, principal: str, namespace: str) -> None:
        """Revoke a principal's access to a namespace."""
        self.check_permission("admin")
        grants = self._grants.get(principal, {})
        grants.pop(namespace, None)
        if not grants:
            self._grants.pop(principal, None)
        self._save()
        self._audit.write("rbac_revoke", principal=principal, namespace=namespace)

    def list_grants(self, principal: str | None = None) -> dict:
        """List all RBAC grants, optionally filtered by principal."""
        if principal:
            grants = self._grants.get(principal, {})
            return {principal: grants} if grants else {}
        return dict(self._grants)

    def get_actor_role(self, namespace: str = "") -> str | None:
        """Get the current actor's effective role for a namespace."""
        actor = self._audit.actor
        if not actor:
            return None
        grants = self._grants.get(actor, {})
        return grants.get(namespace) or grants.get("*")

    def check_permission(self, required_role: str) -> None:
        """Raise PermissionError if RBAC is enabled and actor lacks permission.

        Role hierarchy: admin > writer > reader.
        """
        if not self._enabled:
            return
        actor = self._audit.actor
        if not actor:
            raise PermissionError("RBAC is enabled but no actor set. Call db.set_actor() first.")

        ns_filter = self._get_namespace()
        namespace = ns_filter[0] if len(ns_filter) == 1 else ""

        role = self.get_actor_role(namespace)
        if role is None:
            raise PermissionError(
                f"Actor '{actor}' has no access to namespace '{namespace or '*'}'."
            )

        required_level = _HIERARCHY.get(required_role, 0)
        actual_level = _HIERARCHY.get(role, 0)
        if actual_level < required_level:
            raise PermissionError(
                f"Actor '{actor}' has role '{role}' but '{required_role}' is required."
            )
