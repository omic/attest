"""Synthetic reference uses real claims, source withdrawal and persisted outbox records."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from attestdb import AttestDB

PATH = Path(__file__).resolve().parents[2] / "examples/autofunnel_reference.py"
SPEC = importlib.util.spec_from_file_location("autofunnel_reference", PATH)
funnel = importlib.util.module_from_spec(SPEC)
# Dataclasses inspect the defining module while resolving annotations.

sys.modules[SPEC.name] = funnel
SPEC.loader.exec_module(funnel)


@pytest.fixture
def db(tmp_path):
    store = AttestDB(str(tmp_path / "funnel.attest"), embedding_dim=None)
    try:
        funnel.add_fact(store, "account_priority", "Keep evidence", "synthetic-account")
        funnel.add_fact(store, "product_capability", "Store sourced claims", "synthetic-product")
        yield store
    finally:
        store.close()


def test_evidence_gap_closes_and_withdrawal_blocks_old_approval(tmp_path):
    result = funnel.run_demo(str(tmp_path / "funnel.attest"))
    assert result["before"]["gaps"] == [{"fact": "product_capability", "reason": "missing"}]
    assert result["after_evidence"]["gaps"] == []
    assert result["before_approval"] == ["approval_required"]
    assert result["withdrawn_draft_blocks"] == ["evidence_gap"]
    assert result["after_withdrawal"]["gaps"] == result["before"]["gaps"]
    staged = result["local_outbox"]
    assert staged["delivery_enabled"] is False
    assert staged["observed_outcome"] is None
    db = AttestDB(str(tmp_path / "funnel.attest"), embedding_dim=None)
    try:
        claim = db.get_claim(staged["claim_id"])
        assert claim.payload.data["draft"]["body"] == staged["draft"]["body"]
        assert len(claim.payload.data["draft"]["claim_ids"]) == 2
    finally:
        db.close()


@pytest.mark.parametrize("change", ["body", "subject", "recipient", "claim_ids"])
def test_approval_is_bound_to_exact_draft(db, change):
    draft = funnel.make_draft(db)
    values = {
        "body": draft.body + " Unapproved claim",
        "subject": "Changed",
        "recipient": "another@cedar.invalid",
        "claim_ids": (),
    }
    altered = replace(draft, **{change: values[change]})
    with pytest.raises(ValueError, match="approval_required"):
        funnel.stage_outbox(db, altered, funnel.approve(draft, "reviewer"), consent=True)


@pytest.mark.parametrize(
    "consent,suppressed,expected",
    [
        (False, frozenset(), "consent_required"),
        (True, frozenset({funnel.RECIPIENT.upper()}), "recipient_suppressed"),
    ],
)
def test_permissions_and_suppression_override_approval(db, consent, suppressed, expected):
    draft = funnel.make_draft(db)
    with pytest.raises(ValueError, match=expected):
        funnel.stage_outbox(
            db, draft, funnel.approve(draft, "reviewer"), consent=consent, suppressed=suppressed
        )
    assert db.claims_for(f"draft:{draft.digest()}", limit=10) == []


def test_missing_approval_blocks_outbox(db):
    with pytest.raises(ValueError, match="approval_required"):
        funnel.stage_outbox(db, funnel.make_draft(db), None, consent=True)


def test_forged_content_with_new_receipt_still_fails_source_check(db):
    draft = replace(funnel.make_draft(db), body="Unsupported performance promise")
    with pytest.raises(ValueError, match="draft_or_evidence_changed"):
        funnel.stage_outbox(db, draft, funnel.approve(draft, "reviewer"), consent=True)


def test_conflicting_evidence_requires_investigation(db):
    funnel.add_fact(db, "product_capability", "A conflicting capability", "synthetic-other")
    assert funnel.account_brief(db)["gaps"] == [
        {"fact": "product_capability", "reason": "conflicting"},
    ]
    with pytest.raises(funnel.EvidenceGap):
        funnel.make_draft(db)


def test_new_evidence_invalidates_previous_review(db):
    draft = funnel.make_draft(db)
    approval = funnel.approve(draft, "reviewer")
    funnel.add_fact(db, "product_capability", "Store sourced claims", "synthetic-corroboration")
    assert funnel.outbox_blocks(db, draft, approval, consent=True) == ["draft_or_evidence_changed"]


@pytest.mark.parametrize(
    "recipient", ["person@example.com", "not-an-email.invalid", "x @a.invalid"]
)
def test_non_demo_recipient_is_rejected(db, recipient):
    with pytest.raises(ValueError, match="synthetic .invalid"):
        funnel.make_draft(db, recipient)


def test_draft_citations_resolve_to_real_source_claims(db):
    draft = funnel.make_draft(db)
    claims = [db.get_claim(claim_id) for claim_id in draft.claim_ids]
    assert {claim.provenance.source_id for claim in claims} == {
        "synthetic-account",
        "synthetic-product",
    }
    assert all(claim.payload.data["text"] in draft.body for claim in claims)


def test_excess_evidence_blocks_draft_instead_of_using_truncated_view(db):
    for index in range(100):
        funnel.add_fact(
            db, "product_capability", "Store sourced claims", f"synthetic-source-{index}"
        )
    assert funnel.account_brief(db)["gaps"] == [
        {"fact": "product_capability", "reason": "evidence_limit"},
    ]
    with pytest.raises(funnel.EvidenceGap):
        funnel.make_draft(db)
