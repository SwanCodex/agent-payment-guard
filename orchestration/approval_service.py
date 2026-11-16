
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import uuid4

from sentinel_contracts import (
    PaymentProposal,
    ApprovalRequest,
    ApprovalRecord,
    ApprovalStatus,
)


class ApprovalService:
    """In-memory approval manager for the SENTINEL demo."""

    def __init__(self):
        self._requests = {}
        self._records = {}
        self._lock = RLock()

    def create_request(
        self,
        proposal: PaymentProposal,
        ttl_seconds: int = 300,
    ) -> ApprovalRequest:
        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")

        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            approval_id=str(uuid4()),
            proposal_id=proposal.proposal_id,
            fingerprint=proposal.fingerprint,
            expires_at=now + timedelta(seconds=ttl_seconds),
        )

        with self._lock:
            self._requests[request.approval_id] = request

        return request

    def review_request(
        self,
        approval_id: str,
        proposal: PaymentProposal,
        reviewer_id: str,
        approve: bool,
        comment: str | None = None,
    ) -> ApprovalRecord:
        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise ValueError("reviewer_id must be a non-empty string")

        if not isinstance(approve, bool):
            raise TypeError("approve must be a boolean")

        now = datetime.now(timezone.utc)

        with self._lock:
            request = self._requests.get(approval_id)

            if request is None:
                raise ValueError("Approval request not found")

            if approval_id in self._records:
                raise ValueError("Approval request has already been reviewed")

            if request.proposal_id != proposal.proposal_id:
                raise ValueError("Approval request belongs to another proposal")

            if request.fingerprint != proposal.fingerprint:
                raise ValueError("Payment details changed after approval was requested")

            if now >= request.expires_at:
                record = ApprovalRecord(
                    approval_id=approval_id,
                    proposal_id=proposal.proposal_id,
                    fingerprint=proposal.fingerprint,
                    status=ApprovalStatus.EXPIRED,
                    reviewed_by=reviewer_id,
                    reviewed_at=now,
                    comment="Approval request expired",
                )
                self._records[approval_id] = record
                raise ValueError("Approval request has expired")

            status = (
                ApprovalStatus.APPROVED
                if approve
                else ApprovalStatus.REJECTED
            )

            record = ApprovalRecord(
                approval_id=approval_id,
                proposal_id=proposal.proposal_id,
                fingerprint=proposal.fingerprint,
                status=status,
                reviewed_by=reviewer_id,
                reviewed_at=now,
                comment=comment,
            )

            self._records[approval_id] = record
            return record

    def get_record(self, approval_id: str):
        with self._lock:
            return self._records.get(approval_id)

    def is_approved(
        self,
        approval_id: str,
        proposal: PaymentProposal,
    ) -> bool:
        with self._lock:
            record = self._records.get(approval_id)

            return bool(
                record
                and record.status == ApprovalStatus.APPROVED
                and record.proposal_id == proposal.proposal_id
                and record.fingerprint == proposal.fingerprint
            )
