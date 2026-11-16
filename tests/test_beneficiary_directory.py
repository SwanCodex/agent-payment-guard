import pytest

from mock_agent.beneficiary_directory import (
    BeneficiaryDirectory,
    BeneficiaryNotFoundError,
)


def test_resolve_existing_beneficiary():
    directory = BeneficiaryDirectory()

    beneficiary = directory.resolve("ABC Traders")

    assert beneficiary.recipient_id == "BEN-001"
    assert beneficiary.recipient_name == "ABC Traders"
    assert beneficiary.recipient_account_ref == "MOCK-ACCOUNT-001"


def test_case_insensitive_lookup():
    directory = BeneficiaryDirectory()

    beneficiary = directory.resolve("abc traders")

    assert beneficiary.recipient_id == "BEN-001"


def test_unknown_beneficiary_is_rejected():
    directory = BeneficiaryDirectory()

    with pytest.raises(BeneficiaryNotFoundError):
        directory.resolve("Unknown Company")
