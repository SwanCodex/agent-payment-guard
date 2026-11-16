
from dataclasses import dataclass
from typing import Optional

from sentinel_contracts import (
    PaymentProposal,
    PaymentDecision,
    PaymentStatus,
    ApprovalRecord,
    ApprovalStatus,
    ExecutionReceipt,
    Verdict,
)

from orchestration.payment_lifecycle import PaymentLifecycle
from orchestration.approval_service import ApprovalService
from orchestration.authorization_service import AuthorizationService
from orchestration.mock_executor import MockPaymentExecutor


@dataclass
class PaymentResult:
    proposal_id: str
    status: PaymentStatus
    decision: PaymentDecision
    approval_id: Optional[str] = None
    receipt: Optional[ExecutionReceipt] = None
    message: str = ""


class PaymentService:
    """Coordinates SENTINEL payment evaluation, approval and execution."""

    def __init__(
        self,
        approval_service=None,
        authorization_service=None,
        executor=None,
    ):
        self.approval_service = approval_service or ApprovalService()
        self.authorization_service = (
            authorization_service or AuthorizationService()
        )
        self.executor = executor or MockPaymentExecutor(
            self.authorization_service
        )

        if self.executor._authorization_service is not self.authorization_service:
            raise ValueError(
                "Executor and PaymentService must share the same "
                "AuthorizationService instance"
            )

        self._lifecycles = {}
        self._results = {}

    def get_lifecycle(self, proposal_id: str):
        return self._lifecycles.get(proposal_id)

    def process_payment(
        self,
        proposal: PaymentProposal,
        decision: PaymentDecision,
    ) -> PaymentResult:

        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(decision, PaymentDecision):
            raise TypeError("decision must be a PaymentDecision")

        if decision.proposal_id != proposal.proposal_id:
            raise ValueError("Decision belongs to another proposal")

        if decision.fingerprint != proposal.fingerprint:
            raise ValueError("Decision does not match payment details")

        if proposal.proposal_id in self._lifecycles:
            raise ValueError("This proposal has already been processed")

        lifecycle = PaymentLifecycle(proposal.proposal_id)
        self._lifecycles[proposal.proposal_id] = lifecycle

        lifecycle.transition(PaymentStatus.VALIDATING)
        lifecycle.transition(PaymentStatus.EVALUATING)

        if decision.verdict == Verdict.BLOCK:
            lifecycle.transition(PaymentStatus.BLOCKED)

            result = PaymentResult(
                proposal_id=proposal.proposal_id,
                status=lifecycle.status,
                decision=decision,
                message="Payment blocked by SENTINEL",
            )

        elif decision.verdict == Verdict.STEP_UP:
            request = self.approval_service.create_request(proposal)

            lifecycle.transition(PaymentStatus.AWAITING_APPROVAL)

            result = PaymentResult(
                proposal_id=proposal.proposal_id,
                status=lifecycle.status,
                decision=decision,
                approval_id=request.approval_id,
                message="Human approval required",
            )

        elif decision.verdict == Verdict.ALLOW:
            result = self._execute_allowed_payment(
                proposal,
                decision,
                lifecycle,
            )

        else:
            lifecycle.transition(PaymentStatus.FAILED)
            raise ValueError("Unsupported decision verdict")

        self._results[proposal.proposal_id] = result
        return result

    def approve_payment(
        self,
        proposal: PaymentProposal,
        decision: PaymentDecision,
        approval_id: str,
        reviewer_id: str,
        comment: Optional[str] = None,
    ) -> PaymentResult:

        lifecycle = self._lifecycles.get(proposal.proposal_id)

        if lifecycle is None:
            raise ValueError("Payment has not been processed")

        if lifecycle.status != PaymentStatus.AWAITING_APPROVAL:
            raise ValueError("Payment is not awaiting approval")

        if decision.proposal_id != proposal.proposal_id:
            raise ValueError("Decision belongs to another proposal")

        if decision.fingerprint != proposal.fingerprint:
            raise ValueError("Decision does not match payment details")

        if decision.verdict != Verdict.STEP_UP:
            raise ValueError("Approval is only valid for STEP_UP decisions")

        record = self.approval_service.review_request(
            approval_id=approval_id,
            proposal=proposal,
            reviewer_id=reviewer_id,
            approve=True,
            comment=comment,
        )

        if record.status != ApprovalStatus.APPROVED:
            raise ValueError("Payment approval was not granted")

        lifecycle.transition(PaymentStatus.APPROVED)
        lifecycle.transition(PaymentStatus.ALLOWED)

        result = self._execute_allowed_payment(
            proposal,
            decision,
            lifecycle,
            approval_record=record,
            approval_id=approval_id,
        )

        self._results[proposal.proposal_id] = result
        return result

    def reject_payment(
        self,
        proposal: PaymentProposal,
        approval_id: str,
        reviewer_id: str,
        comment: Optional[str] = None,
    ) -> PaymentResult:

        lifecycle = self._lifecycles.get(proposal.proposal_id)

        if lifecycle is None:
            raise ValueError("Payment has not been processed")

        if lifecycle.status != PaymentStatus.AWAITING_APPROVAL:
            raise ValueError("Payment is not awaiting approval")

        record = self.approval_service.review_request(
            approval_id=approval_id,
            proposal=proposal,
            reviewer_id=reviewer_id,
            approve=False,
            comment=comment,
        )

        if record.status != ApprovalStatus.REJECTED:
            raise ValueError("Payment rejection was not recorded")

        lifecycle.transition(PaymentStatus.REJECTED)

        previous = self._results[proposal.proposal_id]

        result = PaymentResult(
            proposal_id=proposal.proposal_id,
            status=lifecycle.status,
            decision=previous.decision,
            approval_id=approval_id,
            message="Payment rejected by human reviewer",
        )

        self._results[proposal.proposal_id] = result
        return result

    def _execute_allowed_payment(
        self,
        proposal,
        decision,
        lifecycle,
        approval_record=None,
        approval_id=None,
    ):

        authorization = self.authorization_service.issue_authorization(
            proposal=proposal,
            decision=decision,
            approval_record=approval_record,
        )

        # Ensure the payment reaches ALLOWED before execution.
        if lifecycle.status == PaymentStatus.EVALUATING:
            lifecycle.transition(PaymentStatus.ALLOWED)

        lifecycle.transition(PaymentStatus.EXECUTING)

        try:
            receipt = self.executor.execute(proposal, authorization)
            lifecycle.transition(PaymentStatus.EXECUTED)

            return PaymentResult(
                proposal_id=proposal.proposal_id,
                status=lifecycle.status,
                decision=decision,
                approval_id=approval_id,
                receipt=receipt,
                message="Mock payment executed successfully",
            )

        except Exception:
            lifecycle.transition(PaymentStatus.FAILED)
            raise

    def get_result(self, proposal_id: str):
        return self._results.get(proposal_id)
