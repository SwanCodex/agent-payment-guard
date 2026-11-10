from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "interim" / "train_merged.parquet"
OUTPUT = ROOT / "data" / "processed" / "graph"
REPORTS = ROOT / "layer3_fraud_graph" / "reports"

CARD_COLUMNS = [
    "card1", "card2", "card3", "card5", "addr1", "addr2"
]

DEVICE_COLUMNS = [
    "DeviceInfo", "id_30", "id_31"
]


def generate_complete_keys(
    df: pd.DataFrame,
    entity_type: str,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate keys only for complete identities.

    Partial identities are deliberately excluded from graph edges.
    Transactions themselves are never removed.
    """
    from sentinel_l3b.graph.entity_keys import make_entity_key

    available = [column for column in columns if column in df.columns]

    if len(available) != len(columns):
        missing = sorted(set(columns) - set(available))

        # Device identity columns may be absent in another dataset version.
        if entity_type == "device":
            print(f"Optional device columns unavailable: {missing}")
            return (
                pd.DataFrame(columns=["entity_key"]),
                pd.DataFrame(
                    columns=[
                        "TransactionID",
                        "entity_key",
                        "timestamp",
                    ]
                ),
            )

        raise ValueError(f"Missing required {entity_type} columns: {missing}")

    complete_mask = df[columns].notna().all(axis=1)
    complete = df.loc[
        complete_mask,
        ["TransactionID", "TransactionDT", *columns],
    ]

    print(
        f"Generating {entity_type} keys for "
        f"{len(complete):,} complete identities..."
    )

    keys = [
        make_entity_key(entity_type, values)
        for values in complete[columns].itertuples(
            index=False,
            name=None,
        )
    ]

    edges = pd.DataFrame({
        "TransactionID": complete["TransactionID"].to_numpy(),
        "entity_key": keys,
        "timestamp": complete["TransactionDT"].to_numpy(),
    })

    if edges["entity_key"].isna().any():
        raise AssertionError(
            f"Unexpected null {entity_type} key for complete identity"
        )

    nodes = (
        edges[["entity_key"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return nodes, edges


def validate_graph(
    transactions: pd.DataFrame,
    card_nodes: pd.DataFrame,
    device_nodes: pd.DataFrame,
    card_edges: pd.DataFrame,
    device_edges: pd.DataFrame,
) -> None:
    transaction_ids = set(transactions["transaction_id"])

    for name, edges, nodes in [
        ("card", card_edges, card_nodes),
        ("device", device_edges, device_nodes),
    ]:
        if edges["TransactionID"].duplicated().any():
            raise AssertionError(
                f"Duplicate transaction-to-{name} edges found"
            )

        if not set(edges["TransactionID"]).issubset(transaction_ids):
            raise AssertionError(
                f"{name} edge references unknown transaction"
            )

        if not set(edges["entity_key"]).issubset(
            set(nodes["entity_key"])
        ):
            raise AssertionError(
                f"{name} edge references unknown entity node"
            )

        if edges["timestamp"].isna().any():
            raise AssertionError(f"Null timestamp in {name} edges")

    if transactions["TransactionID"].duplicated().any():
        raise AssertionError("Duplicate transaction nodes")

    if transactions["TransactionDT"].isna().any():
        raise AssertionError("Null transaction timestamp")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Training Parquet not found: {INPUT}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("Loading training transactions...")
    df = pd.read_parquet(INPUT)

    required = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "has_identity",
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # One node for every transaction.
    transaction_nodes = df[
        [
            "TransactionID",
            "TransactionDT",
            "isFraud",
            "has_identity",
        ]
    ].copy()

    transaction_nodes.rename(
        columns={
            "TransactionID": "transaction_id",
            "TransactionDT": "timestamp",
            "isFraud": "label",
        },
        inplace=True,
    )

    print("\nBuilding card entities and edges...")
    card_nodes, card_edges = generate_complete_keys(
        df,
        "card",
        CARD_COLUMNS,
    )

    print("\nBuilding device entities and edges...")
    device_nodes, device_edges = generate_complete_keys(
        df,
        "device",
        DEVICE_COLUMNS,
    )

    # Explicit relation labels make downstream heterogeneous graph
    # construction unambiguous.
    card_edges["relation"] = "transaction_to_card"
    device_edges["relation"] = "transaction_to_device"

    validate_graph(
        transaction_nodes,
        card_nodes,
        device_nodes,
        card_edges,
        device_edges,
    )

    # Save graph components separately.
    artifacts = {
        "transaction_nodes.parquet": transaction_nodes,
        "card_nodes.parquet": card_nodes,
        "device_nodes.parquet": device_nodes,
        "transaction_card_edges.parquet": card_edges,
        "transaction_device_edges.parquet": device_edges,
    }

    print("\nSaving graph artifacts...")

    for filename, frame in artifacts.items():
        path = OUTPUT / filename
        frame.to_parquet(
            path,
            index=False,
            compression="snappy",
        )
        print(
            f"{filename}: {len(frame):,} rows | "
            f"{path.stat().st_size / (1024**2):.2f} MB"
        )

    # Record graph statistics.
    report = {
        "dataset": "IEEE-CIS",
        "split": "train",
        "graph_type": "heterogeneous_temporal",
        "transaction_nodes": len(transaction_nodes),
        "card_nodes": len(card_nodes),
        "device_nodes": len(device_nodes),
        "transaction_to_card_edges": len(card_edges),
        "transaction_to_device_edges": len(device_edges),
        "transactions_without_card_edge": (
            len(transaction_nodes) - len(card_edges)
        ),
        "transactions_without_device_edge": (
            len(transaction_nodes) - len(device_edges)
        ),
        "unique_transaction_timestamps": int(
            transaction_nodes["timestamp"].nunique()
        ),
        "relations": [
            "transaction_to_card",
            "transaction_to_device",
        ],
        "temporal_edge_policy": (
            "An edge is eligible as historical context only when "
            "edge timestamp is strictly less than target timestamp."
        ),
    }

    report_path = REPORTS / "graph_build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== GRAPH BUILD COMPLETE ==========")
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
