
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from sentinel_contracts import PaymentStatus


ALLOWED_TRANSITIONS = {
    PaymentStatus.RECEIVED: {
        PaymentStatus.VALIDATING,
        PaymentStatus.FAILED,
    },
    PaymentStatus.VALIDATING: {
        PaymentStatus.EVALUATING,
        PaymentStatus.FAILED,
        PaymentStatus.REJECTED,
    },
    PaymentStatus.EVALUATING: {
        PaymentStatus.ALLOWED,
        PaymentStatus.AWAITING_APPROVAL,
        PaymentStatus.BLOCKED,
        PaymentStatus.FAILED,
    },
    PaymentStatus.AWAITING_APPROVAL: {
        PaymentStatus.APPROVED,
        PaymentStatus.REJECTED,
        PaymentStatus.EXPIRED,
    },
    PaymentStatus.APPROVED: {
        PaymentStatus.ALLOWED,
        PaymentStatus.EXPIRED,
    },
    PaymentStatus.ALLOWED: {
        PaymentStatus.EXECUTING,
        PaymentStatus.EXPIRED,
    },
    PaymentStatus.EXECUTING: {
        PaymentStatus.EXECUTED,
        PaymentStatus.FAILED,
    },
}


@dataclass
class PaymentLifecycle:
    proposal_id: str
    status: PaymentStatus = PaymentStatus.RECEIVED
    history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.proposal_id, str) or not self.proposal_id.strip():
            raise ValueError("proposal_id must be a non-empty string")

        self.history.append({
            "status": self.status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def transition(self, new_status: PaymentStatus) -> None:
        if not isinstance(new_status, PaymentStatus):
            raise TypeError("new_status must be a PaymentStatus")

        permitted = ALLOWED_TRANSITIONS.get(self.status, set())

        if new_status not in permitted:
            raise ValueError(
                f"Invalid payment transition: "
                f"{self.status.value} -> {new_status.value}"
            )

        self.status = new_status

        self.history.append({
            "status": new_status.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
