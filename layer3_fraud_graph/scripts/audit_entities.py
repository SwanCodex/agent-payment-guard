import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "interim" / "train_merged.parquet"
REPORTS = ROOT / "layer3_fraud_graph" / "reports"

ENTITY_GROUPS = {
    "card": ["card1", "card2", "card3", "card5", "addr1", "addr2"],
    "address": ["addr1", "addr2"],
    "device": ["DeviceInfo", "id_30", "id_31"],
    "email_domain": ["P_emaildomain"],
}


def audit_group(df, name, columns):
    subset = df[columns]

    present = subset.notna().any(axis=1)
    complete = subset.notna().all(axis=1)
    partial = present & ~complete

    complete_rows = df.loc[complete, columns]

    counts = (
        complete_rows
        .groupby(columns, sort=False)
        .size()
    )

    degree = counts

    return {
        "columns": columns,
        "transactions_with_any_component": int(present.sum()),
        "transactions_with_all_components": int(complete.sum()),
        "transactions_with_partial_identity": int(partial.sum()),
        "unique_complete_entities": int(len(degree)),
        "singleton_entities": int((degree == 1).sum()),
        "entities_with_multiple_transactions": int((degree > 1).sum()),
        "transactions_in_repeated_entities": int(
            degree[degree > 1].sum()
        ),
        "degree_quantiles": {
            str(q): float(degree.quantile(q))
            for q in [0.50, 0.90, 0.95, 0.99, 0.999]
        },
        "maximum_degree": int(degree.max()) if len(degree) else 0,
        "top_10_entity_degrees": [
            int(value) for value in degree.nlargest(10).tolist()
        ],
    }


def main():
    print("Loading training Parquet...")
    df = pd.read_parquet(DATA)

    results = {}

    print("\n========== ENTITY CONNECTIVITY ==========")

    for name, columns in ENTITY_GROUPS.items():
        print(f"\nAuditing {name}...")
        results[name] = audit_group(df, name, columns)

        result = results[name]

        print("Transactions with any component:",
            f"{result['transactions_with_any_component']:,}")

        print("Complete identities:",
            f"{result['transactions_with_all_components']:,}")

        print("Partial identities:",
            f"{result['transactions_with_partial_identity']:,}")

        print("Unique complete entities:",
            f"{result['unique_complete_entities']:,}")

        print("Singleton entities:",
            f"{result['singleton_entities']:,}")

        print("Repeated entities:",
            f"{result['entities_with_multiple_transactions']:,}")

        print("Transactions in repeated entities:",
            f"{result['transactions_in_repeated_entities']:,}")

        print("Maximum degree:", result["maximum_degree"])

        print("Degree quantiles:", result["degree_quantiles"])

        print("Top 10 degrees:", result["top_10_entity_degrees"])

    # Timestamp ties must stay together during chronological splitting.
    timestamp_counts = df.groupby("TransactionDT").size()

    results["temporal"] = {
        "unique_timestamps": int(timestamp_counts.size),
        "transactions_in_tied_timestamps": int(
            timestamp_counts[timestamp_counts > 1].sum()
        ),
        "number_of_tied_timestamp_groups": int(
            (timestamp_counts > 1).sum()
        ),
        "maximum_transactions_at_same_timestamp": int(
            timestamp_counts.max()
        ),
    }

    output = REPORTS / "entity_connectivity_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n========== TEMPORAL AUDIT ==========")
    print(json.dumps(results["temporal"], indent=2))
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
