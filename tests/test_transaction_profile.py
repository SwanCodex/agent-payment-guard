from mock_agent.transaction_profile import build_transaction_profile


def test_transaction_profile_contains_l3a_fields():
    profile = build_transaction_profile(
        user_id="USER-001",
        amount="5000",
        currency="INR",
        recipient_id="BEN-001",
    )

    required = {
        "amount",
        "event_time",
        "card_key",
        "addr_key",
        "device_key",
    }

    assert required.issubset(profile.keys())


def test_profile_identifiers_are_stable():
    first = build_transaction_profile(
        "USER-001", "5000", "INR", "BEN-001"
    )

    second = build_transaction_profile(
        "USER-001", "2500", "INR", "BEN-002"
    )

    assert first["card_key"] == second["card_key"]
    assert first["addr_key"] == second["addr_key"]
    assert first["device_key"] == second["device_key"]


def test_different_users_have_different_profiles():
    first = build_transaction_profile(
        "USER-001", "5000", "INR", "BEN-001"
    )

    second = build_transaction_profile(
        "USER-002", "5000", "INR", "BEN-001"
    )

    assert first["card_key"] != second["card_key"]
    assert first["addr_key"] != second["addr_key"]
    assert first["device_key"] != second["device_key"]


def test_profile_uses_numeric_amount_and_timestamp():
    profile = build_transaction_profile(
        "USER-001", "5000", "INR", "BEN-001"
    )

    assert profile["amount"] == 5000.0
    assert isinstance(profile["event_time"], float)
    assert profile["simulation"] is True
