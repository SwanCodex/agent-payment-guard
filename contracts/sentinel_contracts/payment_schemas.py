
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Dict, Optional

from .schemas import Verdict


# ---------------------------------------------------------
# Payment lifecycle states
# ---------------------------------------------------------

class PaymentStatus(str, Enum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    EVALUATING = "EVALUATING"
    ALLOWED = "ALLOWED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


class ApprovalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ExecutionStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------

def _validate_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _normalized_amount(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("amount must be supplied as a decimal string")

    try:
        amount = Decimal(value)
    except (InvalidOperation, ValueError):
        raise ValueError("amount must be a valid decimal string")

    if not amount.is_finite() or amount <= 0:
        raise ValueError("amount must be finite and greater than zero")

    return format(amount.normalize(), "f")


# ---------------------------------------------------------
# Payment proposal
# ---------------------------------------------------------

@dataclass(frozen=True)
class PaymentProposal:
    proposal_id: str
    request_id: str
    agent_id: str
    user_id: str

    amount: str
    currency: str

    recipient_id: str
    recipient_name: str
    recipient_account_ref: str

    reference: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        for name in (
            "proposal_id",
            "request_id",
            "agent_id",
            "user_id",
            "recipient_id",
            "recipient_name",
            "recipient_account_ref",
        ):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        object.__setattr__(
            self,
            "amount",
            _normalized_amount(self.amount),
        )

        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isalpha()
        ):
            raise ValueError("currency must be a three-letter code")

        object.__setattr__(self, "currency", self.currency.upper())

        if self.reference is not None and not isinstance(self.reference, str):
            raise TypeError("reference must be a string or None")

        _validate_aware_datetime(self.created_at, "created_at")

    @property
    def fingerprint(self) -> str:
        """
        Stable SHA-256 fingerprint of security-relevant payment details.

        Formatting differences such as 150.0 and 150.00 normalize
        to the same amount.
        """
        payload = {
            "proposal_id": self.proposal_id,
            "amount": self.amount,
            "currency": self.currency,
            "recipient_id": self.recipient_id,
            "recipient_name": self.recipient_name,
            "recipient_account_ref": self.recipient_account_ref,
            "reference": self.reference,
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        return sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# Central payment decision
# ---------------------------------------------------------

@dataclass(frozen=True)
class PaymentDecision:
    decision_id: str
    proposal_id: str
    fingerprint: str
    verdict: Verdict

    risk_score: float
    reason_codes: tuple[str, ...] = ()

    decided_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self):
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be between 0 and 1")

        if not isinstance(self.verdict, Verdict):
            raise TypeError("verdict must be a Verdict")

        _validate_aware_datetime(self.decided_at, "decided_at")


# ---------------------------------------------------------
# Human approval
# ---------------------------------------------------------

@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    proposal_id: str
    fingerprint: str

    expires_at: datetime

    status: ApprovalStatus = ApprovalStatus.PENDING

    def __post_init__(self):
        _validate_aware_datetime(self.expires_at, "expires_at")


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    proposal_id: str
    fingerprint: str

    status: ApprovalStatus
    reviewed_by: str

    reviewed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    comment: Optional[str] = None

    def __post_init__(self):
        if self.status not in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
        ):
            raise ValueError("Invalid final approval status")

        if not self.reviewed_by.strip():
            raise ValueError("reviewed_by must not be empty")

        _validate_aware_datetime(self.reviewed_at, "reviewed_at")


# ---------------------------------------------------------
# Execution authorization
# ---------------------------------------------------------

@dataclass(frozen=True)
class ExecutionAuthorization:
    authorization_id: str
    proposal_id: str
    fingerprint: str
    decision_id: str

    idempotency_key: str

    issued_at: datetime
    expires_at: datetime

    approved_by: Optional[str] = None

    def __post_init__(self):
        _validate_aware_datetime(self.issued_at, "issued_at")
        _validate_aware_datetime(self.expires_at, "expires_at")

        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")

        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")


# ---------------------------------------------------------
# Mock execution receipt
# ---------------------------------------------------------

@dataclass(frozen=True)
class ExecutionReceipt:
    execution_id: str
    authorization_id: str
    proposal_id: str
    fingerprint: str

    status: ExecutionStatus

    amount: str
    currency: str

    executed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    failure_reason: Optional[str] = None

    def __post_init__(self):
        _normalized_amount(self.amount)
        _validate_aware_datetime(self.executed_at, "executed_at")


# ---------------------------------------------------------
# Audit event
# ---------------------------------------------------------

@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    request_id: str
    event_type: str

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    proposal_id: Optional[str] = None
    actor_id: Optional[str] = None

    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.event_type.strip():
            raise ValueError("event_type must not be empty")

        _validate_aware_datetime(self.timestamp, "timestamp")
