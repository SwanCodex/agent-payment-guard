
import pytest
from datetime import datetime, timezone, timedelta

from sentinel_contracts import (
    PaymentProposal,
    PaymentStatus,
    ApprovalStatus,
    ApprovalRequest,
    ExecutionAuthorization,
)


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


def test_payment_proposal_normalizes_amount_and_currency():
    proposal = make_proposal(amount="00150.7500", currency="inr")

    assert proposal.amount == "150.75"
    assert proposal.currency == "INR"


@pytest.mark.parametrize("amount", ["0", "-10", "NaN", "Infinity", "abc"])
def test_payment_proposal_rejects_invalid_amount(amount):
    with pytest.raises((ValueError, TypeError)):
        make_proposal(amount=amount)


def test_payment_proposal_rejects_empty_recipient():
    with pytest.raises(ValueError):
        make_proposal(recipient_id="")


def test_payment_proposal_fingerprint_is_deterministic():
    first = make_proposal()
    second = make_proposal()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_payment_proposal_fingerprint_changes_when_amount_changes():
    first = make_proposal(amount="150.75")
    second = make_proposal(amount="200.00")

    assert first.fingerprint != second.fingerprint


def test_payment_proposal_is_immutable():
    proposal = make_proposal()

    with pytest.raises(AttributeError):
        proposal.amount = "500.00"


def test_approval_request_requires_timezone_aware_expiry():
    with pytest.raises(ValueError):
        ApprovalRequest(
            approval_id="a1",
            proposal_id="p1",
            fingerprint="abc",
            expires_at=datetime(2026, 11, 16),
        )


def test_execution_authorization_expiry_must_follow_issue_time():
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError):
        ExecutionAuthorization(
            authorization_id="auth1",
            proposal_id="p1",
            fingerprint="abc",
            decision_id="d1",
            idempotency_key="idem1",
            issued_at=now,
            expires_at=now - timedelta(seconds=1),
        )
