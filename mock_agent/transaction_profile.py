import hashlib
import time


def _stable_mock_key(namespace: str, user_id: str) -> str:
    value = f"SENTINEL-MOCK:{namespace}:{user_id}"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"MOCK-{namespace.upper()}-{digest}"


def build_transaction_profile(
    user_id: str,
    amount: str,
    currency: str,
    recipient_id: str,
    reference=None,
) -> dict:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must not be empty")

    return {
        "amount": float(amount),
        "event_time": time.time(),
        "card_key": _stable_mock_key("card", user_id),
        "addr_key": _stable_mock_key("address", user_id),
        "device_key": _stable_mock_key("device", user_id),
        "currency": currency,
        "recipient_id": recipient_id,
        "reference": reference,
        "simulation": True,
    }
