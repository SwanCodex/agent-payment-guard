from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "layer3_fraud_graph" / "reports"

FILES = {
    "train_transaction": RAW / "train_transaction.csv",
    "train_identity": RAW / "train_identity.csv",
    "test_transaction": RAW / "test_transaction.csv",
    "test_identity": RAW / "test_identity.csv",
    "sample_submission": RAW / "sample_submission.csv",
}


def inspect_csv(path: Path, track_ids: bool = False) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    row_count = 0
    headers = []
    ids = set() if track_ids else None
    duplicate_ids = 0
    fraud_counts = Counter()
    missing_counts = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []

        if track_ids and "TransactionID" not in headers:
            raise ValueError(f"TransactionID missing from {path.name}")

        for row in reader:
            row_count += 1

            for column, value in row.items():
                if value is None or value.strip() == "":
                    missing_counts[column] += 1

            if track_ids:
                transaction_id = row["TransactionID"]

                if transaction_id in ids:
                    duplicate_ids += 1
                else:
                    ids.add(transaction_id)

            if "isFraud" in row:
                fraud_counts[row["isFraud"]] += 1

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "rows": row_count,
        "columns_count": len(headers),
        "columns": headers,
        "duplicate_transaction_ids": duplicate_ids,
        "transaction_ids": ids,
        "fraud_distribution": dict(fraud_counts),
        "missing_values": dict(missing_counts),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    results = {}

    print("Inspecting IEEE-CIS CSV files...\n")

    for name, path in FILES.items():
        print(f"Reading {path.name} ...")
        results[name] = inspect_csv(
            path,
            track_ids=name in {
                "train_transaction",
                "train_identity",
                "test_transaction",
                "test_identity",
            },
        )

    train_tx = results["train_transaction"]["transaction_ids"]
    train_id = results["train_identity"]["transaction_ids"]
    test_tx = results["test_transaction"]["transaction_ids"]
    test_id = results["test_identity"]["transaction_ids"]

    join_results = {
        "train_transaction_rows": results["train_transaction"]["rows"],
        "train_identity_rows": results["train_identity"]["rows"],
        "train_identity_ids_in_transactions": len(train_id & train_tx),
        "unmatched_train_identity_ids": len(train_id - train_tx),
        "test_transaction_rows": results["test_transaction"]["rows"],
        "test_identity_rows": results["test_identity"]["rows"],
        "test_identity_ids_in_transactions": len(test_id & test_tx),
        "unmatched_test_identity_ids": len(test_id - test_tx),
    }

    # Remove in-memory ID sets before serializing the report.
    for result in results.values():
        result.pop("transaction_ids", None)

    report = {
        "dataset": "IEEE-CIS Fraud Detection",
        "files": results,
        "join_validation": join_results,
    }

    output_path = REPORTS / "dataset_inspection.json"
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== DATASET SUMMARY ==========")

    for name, result in results.items():
        print(
            f"{name}: {result['rows']:,} rows | "
            f"{result['columns_count']} columns | "
            f"{result['duplicate_transaction_ids']} duplicate IDs"
        )

    print("\n========== JOIN VALIDATION ==========")
    for key, value in join_results.items():
        print(f"{key}: {value:,}")

    print("\n========== FRAUD DISTRIBUTION ==========")
    print(results["train_transaction"]["fraud_distribution"])

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
