
import pytest

from sentinel_contracts import PaymentStatus
from orchestration.payment_lifecycle import PaymentLifecycle


def test_payment_starts_as_received():
    lifecycle = PaymentLifecycle(proposal_id="p1")

    assert lifecycle.status == PaymentStatus.RECEIVED
    assert len(lifecycle.history) == 1


def test_normal_payment_lifecycle():
    lifecycle = PaymentLifecycle(proposal_id="p1")

    lifecycle.transition(PaymentStatus.VALIDATING)
    lifecycle.transition(PaymentStatus.EVALUATING)
    lifecycle.transition(PaymentStatus.ALLOWED)
    lifecycle.transition(PaymentStatus.EXECUTING)
    lifecycle.transition(PaymentStatus.EXECUTED)

    assert lifecycle.status == PaymentStatus.EXECUTED
    assert len(lifecycle.history) == 6


def test_approval_payment_lifecycle():
    lifecycle = PaymentLifecycle(proposal_id="p2")

    lifecycle.transition(PaymentStatus.VALIDATING)
    lifecycle.transition(PaymentStatus.EVALUATING)
    lifecycle.transition(PaymentStatus.AWAITING_APPROVAL)
    lifecycle.transition(PaymentStatus.APPROVED)
    lifecycle.transition(PaymentStatus.ALLOWED)

    assert lifecycle.status == PaymentStatus.ALLOWED


def test_blocked_payment_cannot_execute():
    lifecycle = PaymentLifecycle(proposal_id="p3")

    lifecycle.transition(PaymentStatus.VALIDATING)
    lifecycle.transition(PaymentStatus.EVALUATING)
    lifecycle.transition(PaymentStatus.BLOCKED)

    with pytest.raises(ValueError):
        lifecycle.transition(PaymentStatus.EXECUTING)


def test_executed_payment_is_terminal():
    lifecycle = PaymentLifecycle(proposal_id="p4")

    lifecycle.transition(PaymentStatus.VALIDATING)
    lifecycle.transition(PaymentStatus.EVALUATING)
    lifecycle.transition(PaymentStatus.ALLOWED)
    lifecycle.transition(PaymentStatus.EXECUTING)
    lifecycle.transition(PaymentStatus.EXECUTED)

    with pytest.raises(ValueError):
        lifecycle.transition(PaymentStatus.EXECUTING)


def test_invalid_transition_is_rejected():
    lifecycle = PaymentLifecycle(proposal_id="p5")

    with pytest.raises(ValueError):
        lifecycle.transition(PaymentStatus.EXECUTED)


def test_history_records_every_transition():
    lifecycle = PaymentLifecycle(proposal_id="p6")

    lifecycle.transition(PaymentStatus.VALIDATING)

    assert lifecycle.history[-1]["status"] == "VALIDATING"
    assert len(lifecycle.history) == 2
