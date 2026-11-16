import pytest

from sentinel_contracts import PaymentProposal

from mock_agent.agent import MockPaymentAgent
from mock_agent.payment_parser import PaymentParseError
from mock_agent.beneficiary_directory import BeneficiaryNotFoundError


def test_agent_creates_payment_proposal():
    agent = MockPaymentAgent(agent_id="MOCK-AGENT-001")

    proposal = agent.propose_payment(
        instruction="Pay ₹5,000 to ABC Traders for invoice INV-102",
        user_id="USER-001",
    )

    assert isinstance(proposal, PaymentProposal)
    assert proposal.agent_id == "MOCK-AGENT-001"
    assert proposal.user_id == "USER-001"
    assert proposal.amount == "5000"
    assert proposal.currency == "INR"
    assert proposal.recipient_id == "BEN-001"
    assert proposal.recipient_name == "ABC Traders"
    assert proposal.recipient_account_ref == "MOCK-ACCOUNT-001"
    assert proposal.reference == "INV-102"


def test_agent_generates_unique_proposal_ids():
    agent = MockPaymentAgent(agent_id="MOCK-AGENT-001")

    first = agent.propose_payment(
        "Pay ₹500 to ABC Traders",
        "USER-001",
    )

    second = agent.propose_payment(
        "Pay ₹500 to ABC Traders",
        "USER-001",
    )

    assert first.proposal_id != second.proposal_id
    assert first.request_id != second.request_id


def test_agent_rejects_unknown_beneficiary():
    agent = MockPaymentAgent(agent_id="MOCK-AGENT-001")

    with pytest.raises(BeneficiaryNotFoundError):
        agent.propose_payment(
            "Pay ₹500 to Unknown Company",
            "USER-001",
        )


def test_agent_rejects_incomplete_instruction():
    agent = MockPaymentAgent(agent_id="MOCK-AGENT-001")

    with pytest.raises(PaymentParseError):
        agent.propose_payment(
            "Pay ₹500",
            "USER-001",
        )


def test_agent_rejects_missing_user_id():
    agent = MockPaymentAgent(agent_id="MOCK-AGENT-001")

    with pytest.raises(ValueError):
        agent.propose_payment(
            "Pay ₹500 to ABC Traders",
            "",
        )
