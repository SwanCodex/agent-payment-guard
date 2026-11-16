
import pytest

from sentinel_contracts import (
    PaymentProposal,
    PaymentDecision,
    ApprovalRecord,
    ApprovalStatus,
    Verdict,
)

from orchestration.authorization_service import AuthorizationService


def make_proposal(**overrides):
    data = {
        "proposal_id": "p1",
        "request_id": "r1",
        "agent_id": "agent1",
        "user_id": "user1",
        "amount": "150.75",
        "currency": "INR",
        "recipient_id": "merchant1",
        "recipient_name": "Demo Merchant",
        "recipient_account_ref": "demo-account-001",
    }
    data.update(overrides)
    return PaymentProposal(**data)


def make_decision(proposal, verdict):
    return PaymentDecision(
        decision_id="d1",
        proposal_id=proposal.proposal_id,
        fingerprint=proposal.fingerprint,
        verdict=verdict,
        risk_score=0.2,
    )


def make_approval(proposal):
    return ApprovalRecord(
        approval_id="a1",
        proposal_id=proposal.proposal_id,
        fingerprint=proposal.fingerprint,
        status=ApprovalStatus.APPROVED,
        reviewed_by="reviewer1",
    )


def test_allow_decision_can_issue_authorization():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    authorization = service.issue_authorization(proposal, decision)

    assert authorization.proposal_id == proposal.proposal_id
    assert authorization.fingerprint == proposal.fingerprint
    assert service.verify_authorization(authorization, proposal)


def test_block_decision_cannot_issue_authorization():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.BLOCK)

    with pytest.raises(ValueError, match="BLOCK"):
        service.issue_authorization(proposal, decision)


def test_step_up_requires_approval():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)

    with pytest.raises(ValueError, match="requires an approval"):
        service.issue_authorization(proposal, decision)


def test_step_up_with_approval_can_issue_authorization():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)
    approval = make_approval(proposal)

    authorization = service.issue_authorization(
        proposal,
        decision,
        approval_record=approval,
    )

    assert authorization.approved_by == "reviewer1"
    assert service.verify_authorization(authorization, proposal)


def test_changed_payment_invalidates_authorization():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    authorization = service.issue_authorization(proposal, decision)
    changed = make_proposal(amount="999.00")

    assert not service.verify_authorization(authorization, changed)


def test_authorization_cannot_be_consumed_twice():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    authorization = service.issue_authorization(proposal, decision)

    assert service.consume_authorization(authorization, proposal)
    assert not service.consume_authorization(authorization, proposal)
    assert not service.verify_authorization(authorization, proposal)


def test_unregistered_authorization_is_rejected():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    authorization = service.issue_authorization(proposal, decision)

    another_service = AuthorizationService()

    assert not another_service.verify_authorization(
        authorization,
        proposal,
    )


def test_invalid_ttl_is_rejected():
    service = AuthorizationService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    with pytest.raises(ValueError, match="ttl_seconds"):
        service.issue_authorization(
            proposal,
            decision,
            ttl_seconds=0,
        )
