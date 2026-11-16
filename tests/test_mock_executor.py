
import pytest

from sentinel_contracts import (
    PaymentProposal,
    PaymentDecision,
    Verdict,
    ExecutionStatus,
)

from orchestration.authorization_service import AuthorizationService
from orchestration.mock_executor import MockPaymentExecutor


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


def make_authorized_executor(proposal):
    auth_service = AuthorizationService()

    decision = PaymentDecision(
        decision_id="d1",
        proposal_id=proposal.proposal_id,
        fingerprint=proposal.fingerprint,
        verdict=Verdict.ALLOW,
        risk_score=0.1,
    )

    authorization = auth_service.issue_authorization(
        proposal,
        decision,
    )

    executor = MockPaymentExecutor(auth_service)

    return executor, authorization


def test_valid_authorization_executes_payment():
    proposal = make_proposal()
    executor, authorization = make_authorized_executor(proposal)

    receipt = executor.execute(proposal, authorization)

    assert receipt.status == ExecutionStatus.SUCCEEDED
    assert receipt.amount == "150.75"
    assert receipt.currency == "INR"


def test_direct_execution_without_authorization_is_rejected():
    proposal = make_proposal()
    auth_service = AuthorizationService()
    executor = MockPaymentExecutor(auth_service)

    with pytest.raises(TypeError):
        executor.execute(proposal, None)


def test_forged_authorization_is_rejected():
    proposal = make_proposal()
    auth_service = AuthorizationService()
    executor = MockPaymentExecutor(auth_service)

    other_service = AuthorizationService()
    decision = PaymentDecision(
        decision_id="d1",
        proposal_id=proposal.proposal_id,
        fingerprint=proposal.fingerprint,
        verdict=Verdict.ALLOW,
        risk_score=0.1,
    )
    forged = other_service.issue_authorization(proposal, decision)

    with pytest.raises(PermissionError):
        executor.execute(proposal, forged)


def test_changed_payment_is_rejected():
    original = make_proposal()
    executor, authorization = make_authorized_executor(original)

    changed = make_proposal(amount="999.00")

    with pytest.raises(PermissionError):
        executor.execute(changed, authorization)


def test_duplicate_execution_returns_same_receipt():
    proposal = make_proposal()
    executor, authorization = make_authorized_executor(proposal)

    first = executor.execute(proposal, authorization)
    second = executor.execute(proposal, authorization)

    assert first.execution_id == second.execution_id
    assert len(executor.list_receipts()) == 1


def test_consumed_authorization_cannot_execute_again():
    proposal = make_proposal()
    executor, authorization = make_authorized_executor(proposal)

    executor.execute(proposal, authorization)

    # A new executor cannot bypass the consumed authorization.
    second_executor = MockPaymentExecutor(
        executor._authorization_service
    )

    with pytest.raises(PermissionError):
        second_executor.execute(proposal, authorization)


def test_receipt_can_be_retrieved():
    proposal = make_proposal()
    executor, authorization = make_authorized_executor(proposal)

    receipt = executor.execute(proposal, authorization)

    retrieved = executor.get_receipt(receipt.execution_id)

    assert retrieved == receipt


def test_idempotency_key_cannot_be_reused_for_different_payment():
    proposal = make_proposal()
    executor, authorization = make_authorized_executor(proposal)

    executor.execute(proposal, authorization)

    changed = make_proposal(
        proposal_id="p2",
        amount="999.00",
    )

    # Simulate a key collision/reuse attempt.
    from dataclasses import replace

    reused_authorization = replace(
        authorization,
        proposal_id=changed.proposal_id,
        fingerprint=changed.fingerprint,
    )

    with pytest.raises(ValueError, match="Idempotency key"):
        executor.execute(changed, reused_authorization)
