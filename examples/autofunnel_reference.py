"""Synthetic, offline account brief and approval-gated local outbox example."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from attestdb import AttestDB
from attestdb.core.types import ClaimStatus

ACCOUNT = "demo:account-cedar"
PRODUCT = "demo:product-lantern"
REQUIRED = {"account_priority": ACCOUNT, "product_capability": PRODUCT}
SCHEMA = "demo/autofunnel/fact/v1"
RECIPIENT = "contact@cedar.invalid"


class EvidenceGap(ValueError):
    """The synthetic brief lacks an unambiguous required fact."""


@dataclass(frozen=True)
class Draft:
    recipient: str
    subject: str
    body: str
    claim_ids: tuple[str, ...]

    def digest(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Approval:
    draft_digest: str
    reviewer: str


def add_fact(db: AttestDB, key: str, text: str, source: str) -> str:
    if key not in REQUIRED or not text.strip() or not source.strip():
        raise ValueError("A required fact key, nonempty text and source are required")
    return db.ingest(
        subject=(REQUIRED[key], "demo_entity"),
        predicate=(key, "relates_to"),
        object=(text, "demo_statement"),
        confidence=0.5,
        provenance={"source_id": source, "source_type": "manual"},
        payload={"schema_ref": SCHEMA, "data": {"text": text, "synthetic": True}},
    )


def account_brief(db: AttestDB) -> dict:
    facts, gaps = {}, []
    for key, entity in REQUIRED.items():
        rows = db.claims_for(entity, limit=101)
        if len(rows) > 100:
            gaps.append({"fact": key, "reason": "evidence_limit"})
            continue
        evidence = [
            c
            for c in rows
            if c.predicate.id == key
            and c.status == ClaimStatus.ACTIVE
            and c.payload is not None
            and c.payload.schema_ref == SCHEMA
        ]
        texts = {c.payload.data["text"] for c in evidence}
        if len(texts) != 1:
            gaps.append({"fact": key, "reason": "missing" if not texts else "conflicting"})
        else:
            facts[key] = {
                "text": next(iter(texts)),
                "claim_ids": sorted(c.claim_id for c in evidence),
                "sources": sorted({c.provenance.source_id for c in evidence}),
            }
    return {"facts": facts, "gaps": gaps}


def make_draft(db: AttestDB, recipient: str = RECIPIENT) -> Draft:
    local, separator, domain = recipient.partition("@")
    if (
        not local
        or not separator
        or "@" in domain
        or not domain.endswith(".invalid")
        or any(character.isspace() for character in recipient)
    ):
        raise ValueError("This demo accepts synthetic .invalid recipients only")
    brief = account_brief(db)
    if brief["gaps"]:
        raise EvidenceGap(json.dumps(brief["gaps"]))
    facts = brief["facts"]
    return Draft(
        recipient=recipient,
        subject="Synthetic account brief for review",
        body=(
            f"Account priority: {facts['account_priority']['text']}\n"
            f"Product capability: {facts['product_capability']['text']}\n"
            "Synthetic draft; review before any real use."
        ),
        claim_ids=tuple(sorted(claim for fact in facts.values() for claim in fact["claim_ids"])),
    )


def approve(draft: Draft, reviewer: str) -> Approval:
    """Simulate an approval receipt; reviewer text is not authenticated identity."""
    if not reviewer.strip():
        raise ValueError("A named synthetic reviewer is required")
    return Approval(draft.digest(), reviewer.strip())


def outbox_blocks(
    db: AttestDB,
    draft: Draft,
    approval: Approval | None,
    *,
    consent: bool = False,
    suppressed: frozenset[str] = frozenset(),
) -> list[str]:
    blocks = []
    if consent is not True:
        blocks.append("consent_required")
    if draft.recipient.casefold() in {address.casefold() for address in suppressed}:
        blocks.append("recipient_suppressed")
    if approval is None or not approval.reviewer.strip() or approval.draft_digest != draft.digest():
        blocks.append("approval_required")
    try:
        current = make_draft(db, draft.recipient)
    except EvidenceGap:
        blocks.append("evidence_gap")
    else:
        if current.digest() != draft.digest():
            blocks.append("draft_or_evidence_changed")
    return blocks


def stage_outbox(
    db: AttestDB,
    draft: Draft,
    approval: Approval | None,
    *,
    consent: bool = False,
    suppressed: frozenset[str] = frozenset(),
) -> dict:
    blocks = outbox_blocks(db, draft, approval, consent=consent, suppressed=suppressed)
    if blocks:
        raise ValueError("Outbox blocked: " + ", ".join(blocks))
    record = {
        "draft": asdict(draft),
        "approval": asdict(approval),
        "delivery_enabled": False,
        "observed_outcome": None,
        "synthetic": True,
    }
    claim_id = db.ingest(
        subject=(f"draft:{draft.digest()}", "demo_draft"),
        predicate=("staged_in", "relates_to"),
        object=("local-outbox", "demo_outbox"),
        confidence=0.5,
        provenance={"source_id": "synthetic-outbox", "source_type": "manual"},
        payload={"schema_ref": "demo/autofunnel/outbox/v1", "data": record},
    )
    return {"claim_id": claim_id, **record}


def run_demo(db_path: str) -> dict:
    db = AttestDB(db_path, embedding_dim=None)
    try:
        add_fact(db, "account_priority", "Keep an evidence register", "synthetic-account-note")
        before = account_brief(db)
        add_fact(
            db,
            "product_capability",
            "Store claims with source references",
            "synthetic-product-sheet",
        )
        after = account_brief(db)
        draft = make_draft(db)
        unapproved = outbox_blocks(db, draft, None, consent=True)
        approval = approve(draft, "synthetic-reviewer")
        staged = stage_outbox(db, draft, approval, consent=True)
        db.retract("synthetic-product-sheet", reason="Synthetic product evidence withdrawn")
        return {
            "before": before,
            "after_evidence": after,
            "before_approval": unapproved,
            "local_outbox": staged,
            "after_withdrawal": account_brief(db),
            "withdrawn_draft_blocks": outbox_blocks(db, draft, approval, consent=True),
        }
    finally:
        db.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="attest-autofunnel-demo-") as directory:
        print(json.dumps(run_demo(str(Path(directory) / "demo.attest")), indent=2))
