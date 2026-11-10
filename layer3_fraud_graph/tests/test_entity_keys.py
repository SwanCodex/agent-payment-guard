import pandas as pd
import pytest

from layer3_fraud_graph.sentinel_l3b.graph.entity_keys import (
    add_entity_keys,
    entity_key_coverage,
    make_entity_key,
)


def test_same_values_produce_same_key():
    values = [123, 456, 1, 2, 100, 200]
    assert make_entity_key("card", values) == make_entity_key("card", values)


def test_different_entity_types_do_not_collide():
    assert make_entity_key("card", [1, 2, 3, 4, 5, 6]) != (
        make_entity_key("address", [1, 2])
    )


def test_all_missing_values_return_none():
    assert make_entity_key("card", [None, None, None, None, None, None]) is None
    assert make_entity_key("address", [float("nan"), None]) is None
    assert make_entity_key("device", [None, None, None]) is None


def test_partial_identity_is_supported():
    key = make_entity_key("card", [123, None, None, None, None, None])
    assert key is not None
    assert key.startswith("card:v1:")


def test_email_is_case_insensitive():
    first = make_entity_key("email", ["EXAMPLE.COM"])
    second = make_entity_key("email", ["example.com"])
    assert first == second


def test_invalid_entity_type_raises():
    with pytest.raises(ValueError):
        make_entity_key("merchant", ["123"])


def test_wrong_number_of_values_raises():
    with pytest.raises(ValueError):
        make_entity_key("card", [123, 456])


def test_add_entity_keys():
    df = pd.DataFrame({
        "card1": [100, 100, None],
        "card2": [200, 200, None],
        "card3": [1, 1, None],
        "card5": [2, 2, None],
        "addr1": [10, 10, None],
        "addr2": [20, 20, None],
        "P_emaildomain": ["EXAMPLE.COM", "example.com", None],
        "DeviceInfo": ["Device-A", "Device-A", None],
        "id_30": ["Windows", "Windows", None],
        "id_31": ["Chrome", "Chrome", None],
    })

    result = add_entity_keys(df)

    assert result.loc[0, "card_key"] == result.loc[1, "card_key"]
    assert result.loc[0, "addr_key"] == result.loc[1, "addr_key"]
    assert result.loc[0, "device_key"] == result.loc[1, "device_key"]
    assert result.loc[0, "email_key"] == result.loc[1, "email_key"]

    assert pd.isna(result.loc[2, "card_key"])
    assert pd.isna(result.loc[2, "addr_key"])
    assert pd.isna(result.loc[2, "device_key"])
    assert pd.isna(result.loc[2, "email_key"])

    # Original DataFrame remains unchanged by default.
    assert "card_key" not in df.columns


def test_device_columns_can_be_absent():
    df = pd.DataFrame({
        "card1": [1],
        "card2": [2],
        "card3": [3],
        "card5": [5],
        "addr1": [10],
        "addr2": [20],
        "P_emaildomain": ["example.com"],
    })

    result = add_entity_keys(df)
    assert pd.isna(result.loc[0, "device_key"])


def test_coverage_report():
    df = pd.DataFrame({
        "card_key": ["a", None, "b"],
        "addr_key": ["x", "y", None],
        "device_key": [None, None, "d"],
        "email_key": ["e", "e", "e"],
    })

    coverage = entity_key_coverage(df)

    assert coverage["card_key"]["present"] == 2
    assert coverage["device_key"]["coverage"] == pytest.approx(1 / 3)
    assert coverage["email_key"]["coverage"] == 1.0
