from dataclasses import dataclass
from uuid import uuid4

from sentinel_contracts import (
    PaymentDecision,
    PaymentProposal,
    SentinelRequest,
    SentinelResponse,
)

from mock_agent.transaction_profile import build_transaction_profile
from orchestration.api.guard_service import process_request
from orchestration.payment_service import PaymentResult, PaymentService


@dataclass
class PaymentWorkflowResult:
    evaluation: SentinelResponse
    payment: PaymentResult


class PaymentWorkflowService:
    """Trusted backend workflow connecting SENTINEL to payment execution."""

    def __init__(self, payment_service=None):
        self.payment_service = payment_service or PaymentService()

    def submit_payment(
        self,
        proposal: PaymentProposal,
        prompt: str,
        url: str = None,
    ) -> PaymentWorkflowResult:

        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        transaction = build_transaction_profile(
            user_id=proposal.user_id,
            amount=proposal.amount,
            currency=proposal.currency,
            recipient_id=proposal.recipient_id,
            reference=proposal.reference,
        )

        request = SentinelRequest(
            request_id=proposal.request_id,
            agent_id=proposal.agent_id,
            user_id=proposal.user_id,
            transaction_id=proposal.proposal_id,
            prompt=prompt,
            url=url,
            transaction=transaction,
            metadata={
                "proposal_id": proposal.proposal_id,
                "payment_fingerprint": proposal.fingerprint,
                "simulation": True,
            },
        )

        evaluation = process_request(request)

        decision = PaymentDecision(
            decision_id=f"DEC-{uuid4().hex[:12].upper()}",
            proposal_id=proposal.proposal_id,
            fingerprint=proposal.fingerprint,
            verdict=evaluation.verdict,
            risk_score=evaluation.risk_score,
            reason_codes=tuple(evaluation.reason_codes),
        )

        payment_result = self.payment_service.process_payment(
            proposal=proposal,
            decision=decision,
        )

        return PaymentWorkflowResult(
            evaluation=evaluation,
            payment=payment_result,
        )
