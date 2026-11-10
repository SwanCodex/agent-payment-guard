"""Deterministic entity-key generation for SENTINEL Layer 3B."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any, Iterable

import pandas as pd


KEY_VERSION = "v1"

ENTITY_COLUMNS = {
    "card": ("card1", "card2", "card3", "card5", "addr1", "addr2"),    
    "address": ("addr1", "addr2"),
    "device": ("DeviceInfo", "id_30", "id_31"),
    "email": ("P_emaildomain",),
}

# These columns are expected in the merged IEEE-CIS transaction dataset.
# Device columns are optional because identity information may be unavailable.
REQUIRED_COLUMNS = (
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "P_emaildomain",
)


def _normalise_value(value: Any, *, entity_type: str) -> str | None:
    """Convert a scalar into a stable string representation."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None

    # Convert NumPy scalar values into their Python equivalents.
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass

    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if value.is_integer():
            value = int(value)

    if isinstance(value, str):
        value = unicodedata.normalize("NFKC", value).strip()
        if not value:
            return None
        if entity_type == "email":
            value = value.casefold()
        return value

    if isinstance(value, bool):
        return str(value).lower()

    return str(value)


def make_entity_key(
    entity_type: str,
    values: Iterable[Any],
) -> str | None:
    """
    Generate a deterministic SHA-256 entity key.

    Returns None when every component is missing.
    Partial composite identities are allowed and retain explicit
    missing-value markers. This policy must be reviewed with Layer 3A.
    """
    if entity_type not in ENTITY_COLUMNS:
        raise ValueError(
            f"Unknown entity type: {entity_type!r}. "
            f"Expected one of {tuple(ENTITY_COLUMNS)}."
        )

    columns = ENTITY_COLUMNS[entity_type]
    values = list(values)

    if len(values) != len(columns):
        raise ValueError(
            f"{entity_type} requires {len(columns)} values "
            f"for columns {columns}; received {len(values)}."
        )

    normalised = [
        _normalise_value(value, entity_type=entity_type)
        for value in values
    ]

    if all(value is None for value in normalised):
        return None

    payload = {
        "version": KEY_VERSION,
        "entity_type": entity_type,
        "fields": list(zip(columns, normalised)),
    }

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{entity_type}:{KEY_VERSION}:{digest}"


def add_entity_keys(
    transactions: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add card_key, addr_key, device_key and email_key columns.

    Device source columns are optional. Other source columns are
    required so accidental schema mismatches fail explicitly.
    """
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in transactions.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required IEEE-CIS columns: {missing}"
        )

    result = transactions.copy() if copy else transactions

    for entity_type, columns in ENTITY_COLUMNS.items():
        available_columns = [
            column for column in columns
            if column in result.columns
        ]

        output_column = {
            "card": "card_key",
            "address": "addr_key",
            "device": "device_key",
            "email": "email_key",
        }[entity_type]

        if not available_columns:
            result[output_column] = None
            continue

        def generate(row: pd.Series) -> str | None:
            values = [
                row[column] if column in available_columns else None
                for column in columns
            ]
            return make_entity_key(entity_type, values)

        result[output_column] = result.apply(generate, axis=1)

    return result


def entity_key_coverage(
    transactions: pd.DataFrame,
) -> dict[str, dict[str, int | float]]:
    """Report non-null entity-key coverage."""
    total = len(transactions)
    result = {}

    for column in ("card_key", "addr_key", "device_key", "email_key"):
        if column not in transactions.columns:
            raise ValueError(f"Missing generated column: {column}")

        present = int(transactions[column].notna().sum())

        result[column] = {
            "present": present,
            "missing": total - present,
            "coverage": present / total if total else 0.0,
        }

    return result
