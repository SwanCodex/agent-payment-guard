from uuid import uuid4

from sentinel_contracts import PaymentProposal

from mock_agent.payment_parser import parse_payment_instruction
from mock_agent.beneficiary_directory import BeneficiaryDirectory


class MockPaymentAgent:
    """
    Simulated autonomous payment agent.

    Converts natural-language payment instructions into
    structured payment proposals.

    Does not authorize or execute payments.
    """

    def __init__(
        self,
        agent_id: str,
        beneficiary_directory=None,
    ):
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must not be empty")

        self.agent_id = agent_id
        self.beneficiary_directory = (
            beneficiary_directory or BeneficiaryDirectory()
        )

    def propose_payment(
        self,
        instruction: str,
        user_id: str,
    ) -> PaymentProposal:

        if not user_id or not user_id.strip():
            raise ValueError("user_id must not be empty")

        # 1. Parse the natural-language instruction.
        parsed = parse_payment_instruction(instruction)

        # 2. Resolve the recipient using trusted directory data.
        beneficiary = self.beneficiary_directory.resolve(
            parsed.recipient_name
        )

        # 3. Construct a validated payment proposal.
        proposal = PaymentProposal(
            proposal_id=f"PROP-{uuid4().hex[:12].upper()}",
            request_id=f"REQ-{uuid4().hex[:12].upper()}",
            agent_id=self.agent_id,
            user_id=user_id,
            amount=parsed.amount,
            currency=parsed.currency,
            recipient_id=beneficiary.recipient_id,
            recipient_name=beneficiary.recipient_name,
            recipient_account_ref=beneficiary.recipient_account_ref,
            reference=parsed.reference,
        )

        return proposal
