
import pytest

from mock_agent.payment_parser import (
    parse_payment_instruction,
    PaymentParseError,
)


def test_parse_valid_inr_payment():
    payment = parse_payment_instruction(
        "Pay ₹5,000 to ABC Traders for invoice INV-102"
    )

    assert payment.amount == "5000"
    assert payment.currency == "INR"
    assert payment.recipient_name == "ABC Traders"
    assert payment.reference == "INV-102"


def test_parse_payment_without_reference():
    payment = parse_payment_instruction(
        "Pay ₹2500 to ABC Traders"
    )

    assert payment.amount == "2500"
    assert payment.currency == "INR"
    assert payment.recipient_name == "ABC Traders"
    assert payment.reference is None


def test_reject_missing_recipient():
    with pytest.raises(PaymentParseError):
        parse_payment_instruction("Pay ₹5,000")


def test_reject_missing_currency():
    with pytest.raises(PaymentParseError):
        parse_payment_instruction("Pay 5000 to ABC Traders")


def test_reject_multiple_amounts():
    with pytest.raises(PaymentParseError):
        parse_payment_instruction(
            "Pay ₹5,000 to ABC Traders and ₹2,000 to XYZ"
        )


def test_reject_zero_amount():
    with pytest.raises(PaymentParseError):
        parse_payment_instruction("Pay ₹0 to ABC Traders")
