from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
REPORTS = ROOT / "layer3_fraud_graph" / "reports"

INTERIM.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_and_merge(split: str) -> dict:
    print(f"\nProcessing {split} dataset...")

    transaction_path = RAW / f"{split}_transaction.csv"
    identity_path = RAW / f"{split}_identity.csv"
    output_path = INTERIM / f"{split}_merged.parquet"

    print("Loading transaction data...")
    transactions = pd.read_csv(transaction_path, low_memory=False).copy()

    print("Loading identity data...")
    identity = pd.read_csv(identity_path, low_memory=False).copy()

    # Validate primary keys.
    for name, frame in [
        ("transactions", transactions),
        ("identity", identity),
    ]:
        if "TransactionID" not in frame.columns:
            raise ValueError(f"TransactionID missing from {name}")

        if frame["TransactionID"].isna().any():
            raise ValueError(f"Null TransactionID found in {name}")

        if frame["TransactionID"].duplicated().any():
            raise ValueError(f"Duplicate TransactionID found in {name}")

    # Prevent accidental duplicate columns during merging.
    overlap = (
        set(transactions.columns)
        & set(identity.columns)
        - {"TransactionID"}
    )

    if overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(overlap)}")

    original_rows = len(transactions)

    print("Merging on TransactionID...")
    merged = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
        sort=False,
        indicator="_identity_merge",
    )

    merged["has_identity"] = (
        merged["_identity_merge"] == "both"
    )

    identity_matches = int(merged["has_identity"].sum())
    merged = merged.drop(columns=["_identity_merge"])
    
    # Integrity checks.
    if len(merged) != original_rows:
        raise AssertionError("Row count changed after left join")

    if merged["TransactionID"].duplicated().any():
        raise AssertionError("Duplicate IDs introduced by merge")

    if merged["has_identity"].sum() != len(identity):
        raise AssertionError("Not all identity rows matched")

    print(f"Saving {output_path.name}...")
    merged.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )

    result = {
        "split": split,
        "transaction_rows": original_rows,
        "identity_rows": len(identity),
        "merged_rows": len(merged),
        "identity_matches": identity_matches,
        "transactions_without_identity": original_rows - identity_matches,
        "has_identity_coverage": (
            identity_matches / original_rows if original_rows else 0
        ),
        "columns": len(merged.columns),
        "output": str(output_path.relative_to(ROOT)),
        "output_size_bytes": output_path.stat().st_size,
    }

    print(f"Rows: {result['merged_rows']:,}")
    print(f"Columns: {result['columns']}")
    print(f"Identity matches: {identity_matches:,}")
    print(
        "Transactions without identity: "
        f"{result['transactions_without_identity']:,}"
    )
    print(
        f"Saved size: {result['output_size_bytes'] / (1024**2):.2f} MB"
    )

    del transactions, identity, merged

    return result


def main():
    results = {
        "train": load_and_merge("train"),
        "test": load_and_merge("test"),
    }

    report_path = REPORTS / "ingestion_report.json"
    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n========== INGESTION COMPLETE ==========")
    print(json.dumps(results, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
