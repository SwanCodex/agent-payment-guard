
from threading import RLock
from uuid import uuid4
from datetime import datetime, timezone

from sentinel_contracts import (
    PaymentProposal,
    ExecutionAuthorization,
    ExecutionReceipt,
    ExecutionStatus,
)

from orchestration.authorization_service import AuthorizationService


class MockPaymentExecutor:
    """Simulates payments. Never connects to a real payment network."""

    def __init__(self, authorization_service: AuthorizationService):
        if not isinstance(authorization_service, AuthorizationService):
            raise TypeError("A valid AuthorizationService is required")

        self._authorization_service = authorization_service
        self._receipts = {}
        self._idempotency_records = {}
        self._lock = RLock()

    def execute(
        self,
        proposal: PaymentProposal,
        authorization: ExecutionAuthorization,
    ) -> ExecutionReceipt:

        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(authorization, ExecutionAuthorization):
            raise TypeError("authorization must be an ExecutionAuthorization")

        key = authorization.idempotency_key

        with self._lock:
            # Check idempotency before consuming authorization.
            existing = self._idempotency_records.get(key)

            if existing is not None:
                existing_fingerprint, receipt = existing

                if existing_fingerprint != proposal.fingerprint:
                    raise ValueError(
                        "Idempotency key was already used for another payment"
                    )

                if receipt.authorization_id != authorization.authorization_id:
                    raise ValueError(
                        "Idempotency key belongs to another authorization"
                    )

                return receipt

            # Only a registered, valid, unused authorization can proceed.
            if not self._authorization_service.consume_authorization(
                authorization,
                proposal,
            ):
                raise PermissionError(
                    "Execution authorization is invalid, expired, "
                    "consumed, or does not match the payment"
                )

            # Simulate a successful payment.
            receipt = ExecutionReceipt(
                execution_id=str(uuid4()),
                authorization_id=authorization.authorization_id,
                proposal_id=proposal.proposal_id,
                fingerprint=proposal.fingerprint,
                status=ExecutionStatus.SUCCEEDED,
                amount=proposal.amount,
                currency=proposal.currency,
                executed_at=datetime.now(timezone.utc),
            )

            self._receipts[receipt.execution_id] = receipt
            self._idempotency_records[key] = (
                proposal.fingerprint,
                receipt,
            )

            return receipt

    def get_receipt(self, execution_id: str):
        with self._lock:
            return self._receipts.get(execution_id)

    def list_receipts(self):
        with self._lock:
            return list(self._receipts.values())
