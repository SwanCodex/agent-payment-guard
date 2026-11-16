
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import uuid4

from sentinel_contracts import (
    PaymentProposal,
    PaymentDecision,
    ApprovalRecord,
    ApprovalStatus,
    ExecutionAuthorization,
    Verdict,
)


class AuthorizationService:
    """Issues and verifies backend-controlled execution authorizations."""

    def __init__(self):
        self._authorizations = {}
        self._consumed = set()
        self._lock = RLock()

    def issue_authorization(
        self,
        proposal: PaymentProposal,
        decision: PaymentDecision,
        approval_record: ApprovalRecord | None = None,
        ttl_seconds: int = 60,
    ) -> ExecutionAuthorization:

        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(decision, PaymentDecision):
            raise TypeError("decision must be a PaymentDecision")

        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")

        if decision.proposal_id != proposal.proposal_id:
            raise ValueError("Decision belongs to another proposal")

        if decision.fingerprint != proposal.fingerprint:
            raise ValueError("Payment details do not match the decision")

        if decision.verdict == Verdict.BLOCK:
            raise ValueError("BLOCK decisions cannot be authorized")

        approved_by = None

        if decision.verdict == Verdict.STEP_UP:
            if not isinstance(approval_record, ApprovalRecord):
                raise ValueError("STEP_UP requires an approval record")

            if approval_record.status != ApprovalStatus.APPROVED:
                raise ValueError("Human approval has not been granted")

            if approval_record.proposal_id != proposal.proposal_id:
                raise ValueError("Approval belongs to another proposal")

            if approval_record.fingerprint != proposal.fingerprint:
                raise ValueError("Approved payment details do not match")

            approved_by = approval_record.reviewed_by

        elif decision.verdict != Verdict.ALLOW:
            raise ValueError("Unsupported decision verdict")

        now = datetime.now(timezone.utc)

        authorization = ExecutionAuthorization(
            authorization_id=str(uuid4()),
            proposal_id=proposal.proposal_id,
            fingerprint=proposal.fingerprint,
            decision_id=decision.decision_id,
            idempotency_key=str(uuid4()),
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            approved_by=approved_by,
        )

        with self._lock:
            self._authorizations[authorization.authorization_id] = authorization

        return authorization

    def verify_authorization(
        self,
        authorization: ExecutionAuthorization,
        proposal: PaymentProposal,
    ) -> bool:

        if not isinstance(authorization, ExecutionAuthorization):
            return False

        if not isinstance(proposal, PaymentProposal):
            return False

        now = datetime.now(timezone.utc)

        with self._lock:
            registered = self._authorizations.get(
                authorization.authorization_id
            )

            if registered != authorization:
                return False

            if authorization.authorization_id in self._consumed:
                return False

            if now >= authorization.expires_at:
                return False

            if authorization.proposal_id != proposal.proposal_id:
                return False

            if authorization.fingerprint != proposal.fingerprint:
                return False

            return True

    def consume_authorization(
        self,
        authorization: ExecutionAuthorization,
        proposal: PaymentProposal,
    ) -> bool:
        """Atomically verifies and marks an authorization as consumed."""

        with self._lock:
            if not self.verify_authorization(authorization, proposal):
                return False

            self._consumed.add(authorization.authorization_id)
            return True
