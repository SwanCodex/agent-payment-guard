
import re
from dataclasses import dataclass
from typing import Optional


class PaymentParseError(ValueError):
    """Raised when a payment instruction is incomplete or ambiguous."""


@dataclass(frozen=True)
class ParsedPayment:
    amount: str
    currency: str
    recipient_name: str
    reference: Optional[str] = None


# Supported currency markers.
CURRENCY_PATTERNS = [
    (r"₹", "INR"),
    (r"\bINR\b|\brupees?\b|\bRs\.?\b", "INR"),
    (r"\$", "USD"),
    (r"\bUSD\b", "USD"),
    (r"€", "EUR"),
    (r"\bEUR\b", "EUR"),
    (r"£", "GBP"),
    (r"\bGBP\b", "GBP"),
]

AMOUNT_PATTERN = (
    r"(?<![A-Za-z0-9_.-])"
    r"((?:\d{1,3}(?:,\d{2,3})+)|\d+)"
    r"(?:\.(\d{1,2}))?"
    r"(?![A-Za-z0-9_.-])"
)

def parse_payment_instruction(instruction: str) -> ParsedPayment:
    if not isinstance(instruction, str) or not instruction.strip():
        raise PaymentParseError("Payment instruction cannot be empty.")

    text = instruction.strip()

    # Find currency markers.
    currencies = {
        currency
        for pattern, currency in CURRENCY_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    }

    if not currencies:
        raise PaymentParseError(
            "Currency is missing. Specify a currency such as ₹ or INR."
        )

    if len(currencies) > 1:
        raise PaymentParseError("Multiple currencies found in the instruction.")

    currency = currencies.pop()

    # Find monetary amounts.
    amount_matches = list(re.finditer(AMOUNT_PATTERN, text))

    if len(amount_matches) != 1:
        raise PaymentParseError(
            "Specify exactly one unambiguous payment amount."
        )

    amount_match = amount_matches[0]
    whole = amount_match.group(1).replace(",", "")
    decimal = amount_match.group(2)

    amount = whole if decimal is None else f"{whole}.{decimal}"

    if float(amount) <= 0:
        raise PaymentParseError("Payment amount must be greater than zero.")

    # Extract recipient name after the word 'to'.
    recipient_match = re.search(
        r"\bto\s+(.+?)(?=\s+\b(?:for|invoice|reference|ref)\b|$)",
        text,
        re.IGNORECASE,
    )

    if not recipient_match:
        raise PaymentParseError("Recipient name is missing.")

    recipient_name = recipient_match.group(1).strip(" .,:;")

    if not recipient_name:
        raise PaymentParseError("Recipient name is missing.")

    # Extract an optional invoice/reference identifier.
    reference_match = re.search(
        r"\b(?:invoice|reference|ref)\s*(?:number|no\.?)?\s*[:#-]?\s*"
        r"([A-Za-z0-9][A-Za-z0-9/_-]*)",
        text,
        re.IGNORECASE,
    )

    reference = reference_match.group(1) if reference_match else None

    return ParsedPayment(
        amount=amount,
        currency=currency,
        recipient_name=recipient_name,
        reference=reference,
    )
