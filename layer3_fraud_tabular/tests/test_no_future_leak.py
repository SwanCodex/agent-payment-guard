"""Strict causality test: scramble the FUTURE rows and check that the features of earlier rows do not change."""
import numpy as np
import pandas as pd

from sentinel_l3a.features.history import distinct_before, make_entity_features, make_history_features


def make_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 60 * 86400, n))
    dt[100:110] = dt[100]                        # rows with the same timestamp
    cards = rng.integers(0, 120, n)
    frame = pd.DataFrame({
        "TransactionID": np.arange(n),
        "TransactionDT": dt,
        "TransactionAmt": np.round(rng.lognormal(4, 1, n), 2),
        "card_key": ["c%d" % c for c in cards],
        "card_base_key": ["b%d" % (c // 2) for c in cards],
        "addr_key": np.where(rng.random(n) < 0.2, "missing", ["a%d" % x for x in rng.integers(0, 40, n)]),
        "device_key": np.where(rng.random(n) < 0.5, "missing", ["d%d" % x for x in rng.integers(0, 30, n)]),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", None, "aol.com"], n),
    })
    return frame.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def build_all(d):
    return pd.concat([
        make_history_features(d),
        make_entity_features(d, "addr_key", "addr"),
        make_entity_features(d, "device_key", "dev"),
        distinct_before(d, "card_key", "P_emaildomain").rename("card_distinct_emails"),
        distinct_before(d, "card_base_key", "addr_key").rename("card_distinct_addr"),
        distinct_before(d, "device_key", "card_key").rename("dev_distinct_cards"),
    ], axis=1)


def test_features_of_earlier_rows_do_not_change_when_the_future_is_scrambled():
    d = make_frame()
    cutoff = d["TransactionDT"].iloc[len(d) // 2]
    before = build_all(d)

    d2 = d.copy()
    future = d2["TransactionDT"] >= cutoff
    rng = np.random.default_rng(1)
    for c in ["TransactionAmt", "card_key", "card_base_key", "addr_key", "device_key", "P_emaildomain"]:
        d2.loc[future, c] = rng.permutation(d2.loc[future, c].values)
    after = build_all(d2)

    pd.testing.assert_frame_equal(before[~future], after[~future])
