"""SecureStoreProxy — wraps the Rust store, filtering ALL claim-returning methods.

Single enforcement choke point: if a claim passes through this proxy,
it has been authorized. When enterprise_rbac is disabled, this is a
zero-overhead passthrough.
"""

from __future__ import annotations

import logging
from typing import Optional

from attestdb.core.types import Principal

logger = logging.getLogger(__name__)


class SecureStoreProxy:
    """Wraps the Rust store with claim-level access filtering.

    When no principal is set or enterprise_rbac is off, all methods
    pass through directly to the underlying store (zero overhead).
    """

    def __init__(self, store, access_resolver=None) -> None:
        self._store = store
        self._resolver = access_resolver
        self._principal: Optional[Principal] = None
        self._entitlement = None
        self._enabled = False

    def enable(self) -> None:
        """Activate security filtering."""
        self._enabled = True

    def disable(self) -> None:
        """Deactivate — passthrough mode."""
        self._enabled = False
        self._principal = None
        self._entitlement = None

    def set_principal(self, principal: Principal) -> None:
        """Set the active principal and apply namespace filter."""
        if self._resolver and principal:
            # Resolve entitlement FIRST, then update atomically
            entitlement = self._resolver.resolve_entitlement(
                principal.principal_id, principal.org_id or "",
            )
            self._entitlement = entitlement
            self._principal = principal
            # Apply namespace filter at Rust level (first-pass)
            ns = self._entitlement.allowed_namespaces
            if ns and not self._entitlement.is_org_admin:
                try:
                    self._store.set_namespace_filter(ns)
                except (AttributeError, TypeError) as exc:
                    logger.warning("Failed to set namespace filter: %s", exc)
        else:
            self._entitlement = None

    @property
    def active(self) -> bool:
        return self._enabled and self._principal is not None and self._entitlement is not None

    # ── Filtered claim-returning methods ─────────────────────────────

    def claims_for(self, entity_id, pred=None, src=None, min_conf=0.0, limit=0):
        try:
            raw = self._store.claims_for(entity_id, pred, src, min_conf, limit)
        except TypeError:
            raw = self._store.claims_for(entity_id, pred, src, min_conf)
        return self._filter(raw)

    def get_claim(self, claim_id):
        raw = self._store.get_claim(claim_id)
        if raw is None:
            return None
        if not self.active:
            return raw
        filtered = self._filter([raw])
        return filtered[0] if filtered else None

    def all_claims(self, offset=0, limit=0):
        try:
            raw = self._store.all_claims(offset, limit)
        except TypeError:
            raw = self._store.all_claims()
        return self._filter(raw)

    def claims_in_range(self, since, until):
        raw = self._store.claims_in_range(since, until)
        return self._filter(raw)

    def claims_by_content_id(self, content_id):
        raw = self._store.claims_by_content_id(content_id)
        return self._filter(raw)

    def claims_by_source_id(self, source_id):
        raw = self._store.claims_by_source_id(source_id)
        return self._filter(raw)

    def claims_by_predicate_id(self, predicate_id):
        raw = self._store.claims_by_predicate_id(predicate_id)
        return self._filter(raw)

    def bfs_claims(self, entity_id, max_depth):
        raw = self._store.bfs_claims(entity_id, max_depth)
        if not self.active:
            return raw
        # raw is list of (claim_dict, depth) tuples
        filtered_claims = self._filter([c for c, _ in raw])
        filtered_ids = set()
        for c in filtered_claims:
            if isinstance(c, dict):
                filtered_ids.add(c.get("claim_id", ""))
        return [(c, d) for c, d in raw
                if isinstance(c, dict) and c.get("claim_id", "") in filtered_ids]

    def changes(self, since=0, limit=1000):
        raw_claims, cursor = self._store.changes(since, limit)
        return self._filter(raw_claims), cursor

    # ── Core filter ──────────────────────────────────────────────────

    def _filter(self, claim_dicts: list) -> list:
        """Apply entitlement-based filtering to claim dicts."""
        if not self.active:
            return claim_dicts
        result = []
        denied = 0
        for d in claim_dicts:
            if not isinstance(d, dict):
                result.append(d)
                continue
            if self._resolver.check_claim_access(d, self._entitlement):
                result.append(d)
            else:
                denied += 1
        if denied > 0:
            logger.debug("RBAC filtered %d/%d claims for %s",
                         denied, len(claim_dicts),
                         self._principal.principal_id if self._principal else "?")
        return result

    # ── Passthrough for everything else ──────────────────────────────

    def __getattr__(self, name):
        """Delegate all non-overridden methods to the underlying store."""
        return getattr(self._store, name)
