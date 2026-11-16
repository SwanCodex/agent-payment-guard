
import pytest
from datetime import datetime, timezone, timedelta

from sentinel_contracts import (
    PaymentProposal,
    ApprovalStatus,
)
from orchestration.approval_service import ApprovalService


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


def test_create_approval_request():
    service = ApprovalService()
    proposal = make_proposal()

    request = service.create_request(proposal)

    assert request.proposal_id == proposal.proposal_id
    assert request.fingerprint == proposal.fingerprint
    assert request.status == ApprovalStatus.PENDING


def test_approve_request():
    service = ApprovalService()
    proposal = make_proposal()
    request = service.create_request(proposal)

    record = service.review_request(
        request.approval_id,
        proposal,
        "reviewer1",
        approve=True,
    )

    assert record.status == ApprovalStatus.APPROVED
    assert service.is_approved(request.approval_id, proposal)


def test_reject_request():
    service = ApprovalService()
    proposal = make_proposal()
    request = service.create_request(proposal)

    record = service.review_request(
        request.approval_id,
        proposal,
        "reviewer1",
        approve=False,
    )

    assert record.status == ApprovalStatus.REJECTED
    assert not service.is_approved(request.approval_id, proposal)


def test_changed_payment_cannot_be_approved():
    service = ApprovalService()
    original = make_proposal()
    changed = make_proposal(amount="999.00")

    request = service.create_request(original)

    with pytest.raises(ValueError, match="Payment details changed"):
        service.review_request(
            request.approval_id,
            changed,
            "reviewer1",
            approve=True,
        )


def test_request_cannot_be_reviewed_twice():
    service = ApprovalService()
    proposal = make_proposal()
    request = service.create_request(proposal)

    service.review_request(
        request.approval_id,
        proposal,
        "reviewer1",
        approve=True,
    )

    with pytest.raises(ValueError, match="already been reviewed"):
        service.review_request(
            request.approval_id,
            proposal,
            "reviewer2",
            approve=False,
        )


def test_expired_request_cannot_be_approved():
    service = ApprovalService()
    proposal = make_proposal()
    request = service.create_request(proposal, ttl_seconds=1)

    # Simulate expiry without waiting.
    expired_request = type(request)(
        approval_id=request.approval_id,
        proposal_id=request.proposal_id,
        fingerprint=request.fingerprint,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    service._requests[request.approval_id] = expired_request

    with pytest.raises(ValueError, match="expired"):
        service.review_request(
            request.approval_id,
            proposal,
            "reviewer1",
            approve=True,
        )

    record = service.get_record(request.approval_id)
    assert record.status == ApprovalStatus.EXPIRED


def test_unknown_approval_request_is_rejected():
    service = ApprovalService()

    with pytest.raises(ValueError, match="not found"):
        service.review_request(
            "unknown-id",
            make_proposal(),
            "reviewer1",
            approve=True,
        )


def test_invalid_ttl_is_rejected():
    service = ApprovalService()

    with pytest.raises(ValueError):
        service.create_request(make_proposal(), ttl_seconds=0)
