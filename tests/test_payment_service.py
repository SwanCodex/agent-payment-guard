
import pytest

from sentinel_contracts import (
    PaymentProposal,
    PaymentDecision,
    PaymentStatus,
    Verdict,
    ExecutionStatus,
)

from orchestration.payment_service import PaymentService


def make_proposal(proposal_id="p1", **overrides):
    data = {
        "proposal_id": proposal_id,
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
        decision_id=f"d-{proposal.proposal_id}",
        proposal_id=proposal.proposal_id,
        fingerprint=proposal.fingerprint,
        verdict=verdict,
        risk_score=0.2,
    )


def test_allow_payment_executes_successfully():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    result = service.process_payment(proposal, decision)

    assert result.status == PaymentStatus.EXECUTED
    assert result.receipt.status == ExecutionStatus.SUCCEEDED
    assert service.get_lifecycle("p1").status == PaymentStatus.EXECUTED


def test_block_payment_never_executes():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.BLOCK)

    result = service.process_payment(proposal, decision)

    assert result.status == PaymentStatus.BLOCKED
    assert result.receipt is None
    assert service.executor.list_receipts() == []


def test_step_up_waits_for_human_approval():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)

    result = service.process_payment(proposal, decision)

    assert result.status == PaymentStatus.AWAITING_APPROVAL
    assert result.approval_id is not None
    assert result.receipt is None
    assert service.executor.list_receipts() == []


def test_approved_step_up_payment_executes():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)

    initial = service.process_payment(proposal, decision)

    result = service.approve_payment(
        proposal=proposal,
        decision=decision,
        approval_id=initial.approval_id,
        reviewer_id="reviewer1",
    )

    assert result.status == PaymentStatus.EXECUTED
    assert result.receipt.status == ExecutionStatus.SUCCEEDED


def test_rejected_step_up_payment_never_executes():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)

    initial = service.process_payment(proposal, decision)

    result = service.reject_payment(
        proposal=proposal,
        approval_id=initial.approval_id,
        reviewer_id="reviewer1",
    )

    assert result.status == PaymentStatus.REJECTED
    assert result.receipt is None
    assert service.executor.list_receipts() == []


def test_duplicate_proposal_processing_is_rejected():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    service.process_payment(proposal, decision)

    with pytest.raises(ValueError, match="already been processed"):
        service.process_payment(proposal, decision)


def test_mismatched_decision_is_rejected():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.ALLOW)

    changed = make_proposal(amount="999.00")

    with pytest.raises(ValueError, match="does not match"):
        service.process_payment(changed, decision)


def test_approval_cannot_be_used_for_changed_payment():
    service = PaymentService()
    proposal = make_proposal()
    decision = make_decision(proposal, Verdict.STEP_UP)

    initial = service.process_payment(proposal, decision)
    changed = make_proposal(amount="999.00")

    with pytest.raises(ValueError):
        service.approve_payment(
            proposal=changed,
            decision=decision,
            approval_id=initial.approval_id,
            reviewer_id="reviewer1",
        )

    assert service.executor.list_receipts() == []
