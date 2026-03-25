"""Security layer — sensitivity classification, access control, and redaction."""

from __future__ import annotations

import builtins
import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.core.types import Claim, Principal, SensitivityLevel
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)


class SecurityLayer:
    """Manages claim sensitivity classification, access control filtering, and redaction.

    Owns security config, metadata sidecar, and audit logger state via self.db.
    """

    def __init__(self, db: "AttestDB") -> None:
        self.db = db

    def configure_security(self, **kwargs) -> None:
        """Configure security settings.

        Example::

            db.configure_security(enterprise_mode=True, audit_logging=True)
            db.configure_security(
                source_sensitivity_defaults={"internal_chat": "CONFIDENTIAL"},
                predicate_sensitivity_defaults={"salary": "RESTRICTED"},
            )
        """
        from attestdb.core.security import SecurityConfig, SecurityAuditLogger, SensitivityLevel
        for key, val in kwargs.items():
            if key == "default_sensitivity" and isinstance(val, str):
                val = SensitivityLevel[val.upper()]
            if key == "default_principal" and isinstance(val, dict):
                from attestdb.core.types import Principal
                val = Principal(**val)
            if hasattr(self.db._security, key):
                setattr(self.db._security, key, val)
        self.db._security_audit = SecurityAuditLogger(
            self.db._security.audit_log_path,
            enabled=self.db._security.audit_logging,
        )

    def _load_security_meta(self) -> None:
        """Load security metadata sidecar."""
        if not self.db._security_path or not os.path.exists(self.db._security_path):
            return
        try:
            with builtins.open(self.db._security_path) as f:
                self.db._security_meta = json.load(f)
        except Exception:
            self.db._security_meta = {}

    def _save_security_meta(self) -> None:
        """Persist security metadata sidecar."""
        if not self.db._security_path:
            return
        if not self.db._security_meta:
            return
        try:
            with builtins.open(self.db._security_path, "w") as f:
                json.dump(self.db._security_meta, f)
        except Exception as e:
            logger.warning("Failed to save security metadata: %s", e)

    def _apply_security_overlay(self, claim: "Claim") -> "Claim":
        """Apply security metadata overlay from sidecar to a Claim object."""
        meta = self.db._security_meta.get(claim.claim_id)
        if meta:
            from attestdb.core.types import SensitivityLevel
            try:
                claim.sensitivity = SensitivityLevel(meta.get("sensitivity", 0))
            except (ValueError, KeyError):
                pass
            claim.owner = meta.get("owner")
            claim.acl = meta.get("acl", [])
        return claim

    def _set_claim_security(self, claim_id: str, sensitivity: "SensitivityLevel",
                            owner: str | None, acl: list[str]) -> None:
        """Store security metadata for a claim."""
        self.db._security_meta[claim_id] = {
            "sensitivity": sensitivity.value,
            "owner": owner,
            "acl": acl or [],
        }

    def _security_filter(self, claims: list["Claim"],
                         principal: "Principal | None" = None) -> list["Claim"]:
        """Apply access control filtering + redaction to a list of claims.

        When enterprise_mode is on but no principal is provided, falls back to
        the default_principal from config. If that's also None, skips filtering
        (internal call).
        """
        if not self.db._security.enterprise_mode:
            # Apply overlays only (for metadata population), skip filtering
            for c in claims:
                self._apply_security_overlay(c)
            return claims

        from attestdb.core.security import filter_claims, redact_claim, _should_redact
        # Apply security overlays
        for c in claims:
            self._apply_security_overlay(c)
        # Resolve principal
        effective_principal = principal or self.db._security.default_principal
        if effective_principal is None:
            # Internal call with no principal context -- skip filtering
            return claims
        # Filter by access
        visible = filter_claims(
            claims, effective_principal, True,
            self.db._security.admin_sees_restricted,
        )
        # Apply redaction
        result = []
        for c in visible:
            if _should_redact(c, effective_principal, True):
                result.append(redact_claim(c))
            else:
                result.append(c)
        return result
