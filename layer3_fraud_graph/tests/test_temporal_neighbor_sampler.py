
import pandas as pd
import pytest

from sentinel_l3b.graph.temporal_neighbor_sampler import (
    TemporalNeighborSampler,
)


@pytest.fixture
def sampler():
    transactions = pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5, 6, 7],
        "timestamp":     [100, 200, 300, 300, 400, 500, 600],
    })

    # Transactions 1-6 share a card.
    # Transaction 7 has a different card.
    card_edges = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7],
        "entity_key": [
            "card_A",
            "card_A",
            "card_A",
            "card_A",
            "card_A",
            "card_A",
            "card_B",
        ],
        "timestamp": [100, 200, 300, 300, 400, 500, 600],
    })

    # Transactions 1, 3, 5, 6 share a device.
    device_edges = pd.DataFrame({
        "TransactionID": [1, 3, 5, 6],
        "entity_key": [
            "device_A",
            "device_A",
            "device_A",
            "device_A",
        ],
        "timestamp": [100, 300, 400, 500],
    })

    return TemporalNeighborSampler(
        transactions,
        card_edges,
        device_edges,
    )


def test_only_strictly_earlier_card_neighbors(sampler):
    # Target timestamp = 400.
    # Transactions at timestamp 400 and later must be excluded.
    assert sampler.sample_neighbors(
        5,
        relation="card",
        fanout=10,
    ) == [1, 2, 3, 4]


def test_same_timestamp_transactions_are_excluded(sampler):
    # Transactions 3 and 4 both have timestamp 300.
    # Neither can be a historical neighbor of the other.
    assert sampler.sample_neighbors(
        3,
        relation="card",
        fanout=10,
    ) == [1, 2]

    assert sampler.sample_neighbors(
        4,
        relation="card",
        fanout=10,
    ) == [1, 2]


def test_future_transactions_are_excluded(sampler):
    # Target timestamp = 200.
    assert sampler.sample_neighbors(
        2,
        relation="card",
        fanout=10,
    ) == [1]


def test_device_neighbors(sampler):
    # Target timestamp = 500.
    assert sampler.sample_neighbors(
        6,
        relation="device",
        fanout=10,
    ) == [1, 3, 5]


def test_fanout_limit(sampler):
    assert sampler.sample_neighbors(
        6,
        relation="card",
        fanout=2,
    ) == [4, 5]


def test_cold_start_transaction(sampler):
    # Transaction 7 has a card with no historical interactions.
    assert sampler.sample_all_relations(7) == {
        "card": [],
        "device": [],
    }

    assert sampler.get_warm_status(7) is False


def test_unknown_transaction_raises_error(sampler):
    with pytest.raises(KeyError):
        sampler.sample_neighbors(999)


def test_zero_fanout(sampler):
    assert sampler.sample_neighbors(
        6,
        relation="card",
        fanout=0,
    ) == []


def test_negative_fanout_raises_error(sampler):
    with pytest.raises(ValueError):
        sampler.sample_neighbors(
            6,
            relation="card",
            fanout=-1,
        )


def test_timestamp_mismatch_is_rejected():
    transactions = pd.DataFrame({
        "transaction_id": [1, 2],
        "timestamp": [100, 200],
    })

    card_edges = pd.DataFrame({
        "TransactionID": [1, 2],
        "entity_key": ["card_A", "card_A"],
        "timestamp": [100, 250],  # Incorrect timestamp
    })

    device_edges = pd.DataFrame(
        columns=["TransactionID", "entity_key", "timestamp"]
    )

    with pytest.raises(ValueError):
        TemporalNeighborSampler(
            transactions,
            card_edges,
            device_edges,
        )
