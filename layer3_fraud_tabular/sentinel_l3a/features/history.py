"""Offline (training-time) history features. Every window looks only at rows strictly BEFORE the current one.

The serving code (sentinel_l3a/entity_stats.py + api.py) computes the same numbers one transaction at a time.
tests/test_serving_parity.py checks that both give the same values.
"""
import numpy as np
import pandas as pd


def make_history_features(data):
    t = data[["card_key", "TransactionDT", "TransactionAmt"]].copy()
    t["time"] = pd.to_datetime(t["TransactionDT"], unit="s")

    t_sorted = t.sort_values(["card_key", "time"], kind="stable")
    grouped = t_sorted.groupby("card_key")
    out = pd.DataFrame(index=t.index)

    def look_back(window, column, stat):
        r = grouped.rolling(window, on="time", closed="left")[column]   # closed="left": current row excluded
        values = getattr(r, stat)().to_numpy()
        assert len(values) == len(t_sorted)
        return pd.Series(values, index=t_sorted.index).reindex(t.index)

    out["card_cnt_1h"] = look_back("1h", "TransactionAmt", "count")
    out["card_cnt_24h"] = look_back("24h", "TransactionAmt", "count")
    out["card_cnt_7d"] = look_back("7D", "TransactionAmt", "count")
    out["card_cnt_all"] = look_back("400D", "TransactionAmt", "count")

    # empty window -> pandas gives NaN, but the true count is 0
    for c in ["card_cnt_1h", "card_cnt_24h", "card_cnt_7d", "card_cnt_all"]:
        out[c] = out[c].fillna(0)
    out["card_first_seen"] = (out["card_cnt_all"] == 0).astype(int)

    out["card_amt_mean"] = look_back("400D", "TransactionAmt", "mean")
    out["card_amt_std"] = look_back("400D", "TransactionAmt", "std")

    # z-score only with >= 3 earlier transactions, clipped to [-10, 10]
    z = (t["TransactionAmt"] - out["card_amt_mean"]) / out["card_amt_std"].replace(0, np.nan)
    out["card_amt_z"] = z.where(out["card_cnt_all"] >= 3).clip(-10, 10)

    last_time = look_back("400D", "TransactionDT", "max")
    out["card_secs_since_last"] = t["TransactionDT"] - last_time
    return out


def make_entity_features(data, key, prefix):
    t = data[[key, "TransactionDT", "TransactionAmt"]].copy()
    t["time"] = pd.to_datetime(t["TransactionDT"], unit="s")
    t_sorted = t.sort_values([key, "time"], kind="stable")
    grouped = t_sorted.groupby(key)
    out = pd.DataFrame(index=t.index)

    def look_back(window, column, stat):
        r = grouped.rolling(window, on="time", closed="left")[column]
        values = getattr(r, stat)().to_numpy()
        assert len(values) == len(t_sorted)
        return pd.Series(values, index=t_sorted.index).reindex(t.index)

    for name, window in [("cnt_1h", "1h"), ("cnt_24h", "24h"), ("cnt_7d", "7D")]:
        out[f"{prefix}_{name}"] = look_back(window, "TransactionAmt", "count").fillna(0)
    out[f"{prefix}_amt_mean"] = look_back("400D", "TransactionAmt", "mean")
    out[f"{prefix}_secs_since_last"] = t["TransactionDT"] - look_back("400D", "TransactionDT", "max")

    out.loc[t[key] == "missing", :] = np.nan    # unknown entity: undefined, not zero
    return out


def distinct_before(data, key, other):
    t = data[[key, other, "TransactionDT"]].copy()
    t[other] = t[other].fillna("missing").astype(str)
    t = t.sort_values([key, "TransactionDT"], kind="stable")

    is_new = (~t.duplicated(subset=[key, other])).astype(int)
    running = is_new.groupby(t[key]).cumsum() - is_new
    strict = running.groupby([t[key], t["TransactionDT"]]).transform("min")   # same-second rows don't see each other
    return strict.reindex(data.index)
