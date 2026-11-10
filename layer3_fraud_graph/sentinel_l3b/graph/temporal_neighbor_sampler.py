
from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd


Relation = Literal["card", "device"]


class TemporalNeighborSampler:
    """
    Retrieves historical transaction neighbors through shared entities.

    A neighbor is eligible only when:

        neighbor_timestamp < target_timestamp

    Same-timestamp and future transactions are always excluded.

    The sampler does not use transaction labels.
    """

    def __init__(
        self,
        transaction_nodes: pd.DataFrame,
        card_edges: pd.DataFrame,
        device_edges: pd.DataFrame,
    ) -> None:

        required_nodes = {
            "transaction_id",
            "timestamp",
        }

        if not required_nodes.issubset(transaction_nodes.columns):
            raise ValueError(
                f"Transaction nodes must contain {required_nodes}"
            )

        if transaction_nodes["transaction_id"].isna().any():
            raise ValueError("Null transaction IDs found")

        if transaction_nodes["transaction_id"].duplicated().any():
            raise ValueError("Duplicate transaction IDs found")

        if transaction_nodes["timestamp"].isna().any():
            raise ValueError("Null transaction timestamps found")

        self.transaction_timestamps = (
            transaction_nodes
            .set_index("transaction_id")["timestamp"]
            .to_dict()
        )

        self.entity_by_transaction: dict[
            Relation, dict[int, str]
        ] = {
            "card": {},
            "device": {},
        }

        self.entity_history: dict[
            Relation, dict[str, tuple[np.ndarray, np.ndarray]]
        ] = {
            "card": {},
            "device": {},
        }

        self._build_relation_index("card", card_edges)
        self._build_relation_index("device", device_edges)

    def _build_relation_index(
        self,
        relation: Relation,
        edges: pd.DataFrame,
    ) -> None:
        required = {
            "TransactionID",
            "entity_key",
            "timestamp",
        }

        if not required.issubset(edges.columns):
            raise ValueError(
                f"{relation} edges must contain {required}"
            )

        if edges["TransactionID"].isna().any():
            raise ValueError(
                f"Null transaction ID in {relation} edges"
            )

        if edges["entity_key"].isna().any():
            raise ValueError(
                f"Null entity key in {relation} edges"
            )

        if edges["timestamp"].isna().any():
            raise ValueError(
                f"Null timestamp in {relation} edges"
            )

        if edges["TransactionID"].duplicated().any():
            raise ValueError(
                f"Duplicate transaction in {relation} edges"
            )

        unknown_ids = (
            set(edges["TransactionID"])
            - set(self.transaction_timestamps)
        )

        if unknown_ids:
            raise ValueError(
                f"{relation} edges contain unknown transactions"
            )

        expected_timestamps = edges["TransactionID"].map(
            self.transaction_timestamps
        )

        if not np.array_equal(
            expected_timestamps.to_numpy(),
            edges["timestamp"].to_numpy(),
        ):
            raise ValueError(
                f"{relation} edge timestamps do not match "
                "transaction node timestamps"
            )

        self.entity_by_transaction[relation] = dict(
            zip(
                edges["TransactionID"].to_numpy(),
                edges["entity_key"].to_numpy(),
            )
        )

        # Build an independently sorted history for each entity.
        grouped = edges.groupby("entity_key", sort=False)

        for entity_key, group in grouped:
            ordered = group.sort_values(
                ["timestamp", "TransactionID"],
                kind="mergesort",
            )

            timestamps = ordered["timestamp"].to_numpy()
            transaction_ids = ordered["TransactionID"].to_numpy()

            self.entity_history[relation][entity_key] = (
                timestamps,
                transaction_ids,
            )

    def sample_neighbors(
        self,
        transaction_id: int,
        fanout: int = 10,
        relation: Relation = "card",
    ) -> list[int]:
        """
        Return up to `fanout` most recent historical neighbors.

        The target transaction's timestamp is used automatically.
        """

        if fanout < 0:
            raise ValueError("fanout must be non-negative")

        if relation not in ("card", "device"):
            raise ValueError(f"Unsupported relation: {relation}")

        if transaction_id not in self.transaction_timestamps:
            raise KeyError(
                f"Unknown transaction ID: {transaction_id}"
            )

        if fanout == 0:
            return []

        target_timestamp = self.transaction_timestamps[
            transaction_id
        ]

        entity_key = self.entity_by_transaction[
            relation
        ].get(transaction_id)

        if entity_key is None:
            return []

        history = self.entity_history[relation].get(entity_key)

        if history is None:
            return []

        timestamps, transaction_ids = history

        # side="left" excludes every transaction whose timestamp
        # is equal to the target timestamp.
        cutoff = np.searchsorted(
            timestamps,
            target_timestamp,
            side="left",
        )

        if cutoff == 0:
            return []

        # Select the most recent eligible historical transactions.
        start = max(0, cutoff - fanout)

        neighbors = transaction_ids[start:cutoff]

        # Defensive checks.
        result = [
            int(neighbor_id)
            for neighbor_id in neighbors
            if int(neighbor_id) != transaction_id
        ]

        return result

    def sample_all_relations(
        self,
        transaction_id: int,
        card_fanout: int = 10,
        device_fanout: int = 5,
    ) -> dict[str, list[int]]:
        """
        Sample historical neighbors for both graph relations.
        """

        return {
            "card": self.sample_neighbors(
                transaction_id,
                fanout=card_fanout,
                relation="card",
            ),
            "device": self.sample_neighbors(
                transaction_id,
                fanout=device_fanout,
                relation="device",
            ),
        }

    def get_warm_status(self, transaction_id: int) -> bool:
        """
        A transaction is warm if it has at least one historical
        neighbor through either relation.
        """

        neighbors = self.sample_all_relations(
            transaction_id,
            card_fanout=1,
            device_fanout=1,
        )

        return bool(
            neighbors["card"] or neighbors["device"]
        )
