
from dataclasses import dataclass


class BeneficiaryNotFoundError(LookupError):
    pass


class AmbiguousBeneficiaryError(LookupError):
    pass


@dataclass(frozen=True)
class Beneficiary:
    recipient_id: str
    recipient_name: str
    recipient_account_ref: str


class BeneficiaryDirectory:
    """
    Trusted directory of mock payment beneficiaries.

    These records are for simulation only.
    """

    def __init__(self):
        self._beneficiaries = (
            Beneficiary(
                recipient_id="BEN-001",
                recipient_name="ABC Traders",
                recipient_account_ref="MOCK-ACCOUNT-001",
            ),
            Beneficiary(
                recipient_id="BEN-002",
                recipient_name="XYZ Enterprises",
                recipient_account_ref="MOCK-ACCOUNT-002",
            ),
            Beneficiary(
                recipient_id="BEN-003",
                recipient_name="ABC Technologies",
                recipient_account_ref="MOCK-ACCOUNT-003",
            ),
        )

    def resolve(self, recipient_name: str) -> Beneficiary:
        if not isinstance(recipient_name, str) or not recipient_name.strip():
            raise BeneficiaryNotFoundError("Recipient name is required.")

        normalized = recipient_name.strip().casefold()

        matches = [
            beneficiary
            for beneficiary in self._beneficiaries
            if beneficiary.recipient_name.casefold() == normalized
        ]

        if not matches:
            raise BeneficiaryNotFoundError(
                f"No beneficiary found for '{recipient_name}'."
            )

        if len(matches) > 1:
            raise AmbiguousBeneficiaryError(
                f"Multiple beneficiaries match '{recipient_name}'."
            )

        return matches[0]
