# %% [markdown]
# # SENTINEL Layer 3A (P2): serving module + parity test
# Run this in a NEW Kaggle notebook named  layer3a-serving-parity   (CPU, no GPU).
# Add these 3 inputs:
#   1. your raw IEEE-CIS dataset      (train_transaction.csv and train_identity.csv)
#   2. layer3a-data-features-eda      (core_features.parquet = the offline features)
#   3. layer3a-models-calibration     (models_serving/, lgbm_serving_bundle/, predictions_serving/)
# What it does: writes the serving code (sentinel_l3a/entity_stats.py and api.py), replays the TEST period one
# transaction at a time through it, and checks the live features and scores equal the offline ones.
# Everything it saves is in /kaggle/working/sentinel_l3a/  (these are the files that go to GitHub later).

# %%
import os
os.makedirs("/kaggle/working/sentinel_l3a", exist_ok=True)
open("/kaggle/working/sentinel_l3a/__init__.py", "w").close()

ENTITY_STATS = r'''
"""Entity history store for Layer 3A (a plain dict now; the same methods can sit on top of Redis later).

For every key (a card, an address, a device) it remembers the earlier transactions.
All look-ups are STRICTLY before the current time t: a transaction never sees itself
or another transaction with the same timestamp. That is the same rule as the training features.
"""
import bisect

import numpy as np

HOUR = 3600.0
DAY = 86400.0
LONG_WINDOW = 400 * DAY      # "all history" window used in training


class EntityStore:
    def __init__(self):
        self.times = {}      # key -> list of transaction times (kept in increasing order)
        self.amounts = {}    # key -> list of amounts, same order
        self.distinct = {}   # (kind, key) -> {value: time it was first seen}

    # ---------- write ----------
    def add_transaction(self, key, t, amount):
        self.times.setdefault(key, []).append(t)
        self.amounts.setdefault(key, []).append(amount)

    def add_value(self, kind, key, value, t):
        self.distinct.setdefault((kind, key), {}).setdefault(value, t)   # keep the first time only

    # ---------- read (strictly before t) ----------
    def stats(self, key, t):
        times = self.times.get(key, [])
        k = bisect.bisect_left(times, t)                       # number of transactions with time < t

        def count_since(seconds):
            return k - bisect.bisect_left(times, t - seconds)  # window is [t - seconds, t)

        lo = bisect.bisect_left(times, t - LONG_WINDOW)
        amounts = np.asarray(self.amounts.get(key, [])[lo:k], dtype="float64")
        n = len(amounts)
        return {
            "cnt_1h": count_since(HOUR),
            "cnt_24h": count_since(DAY),
            "cnt_7d": count_since(7 * DAY),
            "cnt_all": n,
            "amt_mean": float(amounts.mean()) if n >= 1 else np.nan,
            "amt_std": float(amounts.std(ddof=1)) if n >= 2 else np.nan,
            "secs_since_last": float(t - times[k - 1]) if n >= 1 else np.nan,
        }

    def distinct_before(self, kind, key, t):
        seen = self.distinct.get((kind, key), {})
        return sum(1 for first_time in seen.values() if first_time < t)
'''
open("/kaggle/working/sentinel_l3a/entity_stats.py", "w").write(ENTITY_STATS.lstrip("\n"))

API = r'''
"""Layer 3A serving: one transaction in, one calibrated fraud probability out.

Input = a dict with the CoreTxn field names (see the handbook contract):
  amount, product_cd, card_network, card_type, payer_email_domain, recipient_email_domain,
  billing_addr1, billing_addr2, dist1, device_type, event_time, card_key, addr_key, device_key
event_time: seconds (the IEEE-CIS TransactionDT) or a datetime (seconds since 1970-01-01 are used).
"""
import datetime as dt
import glob
import json
import math
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from sentinel_l3a.entity_stats import EntityStore


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _num(v):
    return np.nan if _missing(v) else float(v)


def _seconds(event_time):
    if isinstance(event_time, dt.datetime):
        return (event_time - dt.datetime(1970, 1, 1)).total_seconds()
    return float(event_time)


STAT_NAMES = ["cnt_1h", "cnt_24h", "cnt_7d", "amt_mean", "secs_since_last"]


class L3AScorer:
    def __init__(self, models_dir, bundle_dir, store=None):
        files = sorted(glob.glob(os.path.join(models_dir, "lgbm_seed*.txt")))
        assert files, f"no lgbm_seed*.txt in {models_dir}"
        self.boosters = [lgb.Booster(model_file=f) for f in files]
        with open(os.path.join(bundle_dir, "calibration.json")) as f:
            self.T = json.load(f)["T"]
        with open(os.path.join(bundle_dir, "frequency_maps.json")) as f:
            self.freq = json.load(f)
        with open(os.path.join(bundle_dir, "feature_order.json")) as f:
            spec = json.load(f)
        self.features = spec["features"]
        self.categorical = spec["categorical"]
        self.categories = {c: sorted(self.freq[c].keys()) for c in self.categorical}
        self.store = store if store is not None else EntityStore()

    # ---------- 1. turn one transaction into feature values (does NOT change the store) ----------
    def features_for(self, txn):
        t = _seconds(txn["event_time"])
        amt = float(txn["amount"])
        card, addr, dev = txn["card_key"], txn["addr_key"], txn["device_key"]
        payer, recipient = txn.get("payer_email_domain"), txn.get("recipient_email_domain")

        raw_cat = {"ProductCD": txn.get("product_cd"), "card4": txn.get("card_network"),
                   "card6": txn.get("card_type"), "P_emaildomain": payer,
                   "R_emaildomain": recipient, "DeviceType": txn.get("device_type")}

        row = {
            "log_amount": float(np.log1p(amt)),
            "cents": float(np.round(amt % 1, 2)),
            "is_round_amount": int(amt % 1 == 0),
            "hour": (t // 3600) % 24,
            "dayofweek": (t // 86400) % 7,
            "addr1": _num(txn.get("billing_addr1")),
            "addr2": _num(txn.get("billing_addr2")),
            "dist1": _num(txn.get("dist1")),
            "has_identity": int(not _missing(txn.get("device_type"))),     # live rule: device type is known
            "same_email_domain": int((not _missing(payer)) and payer == recipient),
        }
        for c, v in raw_cat.items():
            row[c] = None if _missing(v) else str(v)
            row[c + "_freq"] = 0.0 if _missing(v) else self.freq[c].get(str(v), 0.0)

        s = self.store.stats(card, t)
        for name in STAT_NAMES:
            row["card_" + name] = s[name]
        row["card_cnt_all"] = s["cnt_all"]
        row["card_first_seen"] = int(s["cnt_all"] == 0)
        row["card_amt_std"] = s["amt_std"]
        z = np.nan
        if s["cnt_all"] >= 3 and not np.isnan(s["amt_std"]) and s["amt_std"] != 0:
            z = float(np.clip((amt - s["amt_mean"]) / s["amt_std"], -10, 10))
        row["card_amt_z"] = z

        for prefix, key in [("addr", addr), ("dev", dev)]:
            if key == "missing":                       # unknown entity: undefined, not zero
                for name in STAT_NAMES:
                    row[f"{prefix}_{name}"] = np.nan
            else:
                s2 = self.store.stats(key, t)
                for name in STAT_NAMES:
                    row[f"{prefix}_{name}"] = s2[name]

        row["card_distinct_emails"] = self.store.distinct_before("email", card, t)
        row["dev_distinct_cards"] = np.nan if dev == "missing" else self.store.distinct_before("devcard", dev, t)
        return row

    # ---------- 2. remember the transaction (call AFTER scoring it) ----------
    def remember(self, txn):
        t = _seconds(txn["event_time"])
        amt = float(txn["amount"])
        card, addr, dev = txn["card_key"], txn["addr_key"], txn["device_key"]
        self.store.add_transaction(card, t, amt)
        if addr != "missing":
            self.store.add_transaction(addr, t, amt)
        if dev != "missing":
            self.store.add_transaction(dev, t, amt)
        email = txn.get("payer_email_domain")
        self.store.add_value("email", card, "missing" if _missing(email) else str(email), t)
        if dev != "missing":
            self.store.add_value("devcard", dev, card, t)

    # ---------- 3. model ----------
    def to_frame(self, rows):
        X = pd.DataFrame(rows)[self.features]
        for c in self.categorical:
            X[c] = pd.Categorical(X[c], categories=self.categories[c])     # unseen value -> missing
        return X

    def predict_raw(self, X):
        return np.mean([b.predict(X) for b in self.boosters], axis=0)

    def calibrate(self, p):
        return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / self.T)

    # ---------- 4. the one call the guard makes ----------
    def score(self, txn, explain=False, commit=True):
        t0 = time.time()
        row = self.features_for(txn)
        X = self.to_frame([row])
        p_raw = float(self.predict_raw(X)[0])
        out = {"layer": "L3A", "p_raw": p_raw, "p": float(self.calibrate(np.array([p_raw]))[0]),
               "warm": bool(row["card_cnt_all"] > 0), "abstain": False, "evidence": []}
        if explain:                                    # only needed for non-ALLOW cases (1 seed is enough)
            contrib = self.boosters[0].predict(X, pred_contrib=True)[0]
            order = np.argsort(-np.abs(contrib[:-1]))[:5]
            out["evidence"] = [{"feature": self.features[j],
                                "value": None if _missing(row[self.features[j]]) else row[self.features[j]],
                                "shap_logodds": round(float(contrib[j]), 4)} for j in order]
        if commit:
            self.remember(txn)
        out["latency_ms"] = (time.time() - t0) * 1000
        return out
'''
open("/kaggle/working/sentinel_l3a/api.py", "w").write(API.lstrip("\n"))
print(os.listdir("/kaggle/working/sentinel_l3a"))

# %%
import glob
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/kaggle/working")

txn_path = glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0]
id_path = glob.glob("/kaggle/input/**/train_identity.csv", recursive=True)[0]
tcols = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4",
         "card5", "card6", "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain"]
icols = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31"]
df = pd.read_csv(txn_path, usecols=tcols).merge(pd.read_csv(id_path, usecols=icols), on="TransactionID", how="left")
df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

df["card_key"] = df[["card1", "card2", "card3", "card5", "addr1", "addr2"]].astype(str).agg("_".join, axis=1)
df["addr_key"] = df["addr1"].astype(str) + "_" + df["addr2"].astype(str)
df.loc[df["addr1"].isna() | df["addr2"].isna(), "addr_key"] = "missing"
df["device_key"] = df["DeviceInfo"].astype(str) + "_" + df["id_30"].astype(str) + "_" + df["id_31"].astype(str)
df.loc[df[["DeviceInfo", "id_30", "id_31"]].isna().all(axis=1), "device_key"] = "missing"

# the split comes from the saved offline features (same rows, same order rule)
offline = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0])
df = df.merge(offline[["TransactionID", "split"]], on="TransactionID", how="left")
assert df["split"].notna().all()
print(df["split"].value_counts().to_dict())

# one dict per transaction, with the CoreTxn field names the real system will send
def to_txn(r):
    return {"amount": r.TransactionAmt, "event_time": r.TransactionDT, "product_cd": r.ProductCD,
            "card_network": r.card4, "card_type": r.card6, "payer_email_domain": r.P_emaildomain,
            "recipient_email_domain": r.R_emaildomain, "billing_addr1": r.addr1, "billing_addr2": r.addr2,
            "dist1": r.dist1, "device_type": r.DeviceType, "card_key": r.card_key,
            "addr_key": r.addr_key, "device_key": r.device_key}

# %%
from sentinel_l3a.api import L3AScorer

models_dir = glob.glob("/kaggle/input/**/models_serving", recursive=True)[0]
bundle_dir = glob.glob("/kaggle/input/**/lgbm_serving_bundle", recursive=True)[0]
scorer = L3AScorer(models_dir, bundle_dir)
print(len(scorer.boosters), "models | T =", round(scorer.T, 4), "|", len(scorer.features), "features")

cat_order = [c for c in scorer.features if c in scorer.categorical]
for c, saved in zip(cat_order, scorer.boosters[0].pandas_categorical):
    assert scorer.categories[c] == list(saved), f"category list of {c} differs from the model's"
print("category lists match the ones stored inside the model: OK")

# %%
before_test = df[df["split"] != "test"]
t0 = time.time()
for r in before_test.itertuples():
    scorer.remember(to_txn(r))
print(len(before_test), "past transactions loaded in", round(time.time() - t0, 1), "s")

# %%
test_rows = df[df["split"] == "test"]
live_rows = []
t0 = time.time()
for r in test_rows.itertuples():
    txn = to_txn(r)
    live_rows.append(scorer.features_for(txn))     # what the guard would compute at request time
    scorer.remember(txn)                           # then the transaction becomes history
print(len(live_rows), "transactions replayed in", round(time.time() - t0, 1), "s")
live = pd.DataFrame(live_rows)

# %%
off = offline[offline["split"] == "test"].set_index("TransactionID").loc[test_rows["TransactionID"]]
off = off.assign(has_identity=off["DeviceType"].notna().astype(int))        # live rule for has_identity

report = []
for c in scorer.features:
    if c in scorer.categorical:
        a, b = off[c].astype(object).to_numpy(), live[c].astype(object).to_numpy()
        same = np.array([(pd.isna(x) and pd.isna(y)) or x == y for x, y in zip(a, b)])
    else:
        a, b = off[c].to_numpy(dtype="float64"), live[c].to_numpy(dtype="float64")
        same = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
    report.append({"feature": c, "match %": 100 * same.mean(), "rows differing": int((~same).sum())})
report = pd.DataFrame(report).sort_values("match %")
print(report.head(12).round(4).to_string(index=False))
print("features with 100% match:", int((report["match %"] == 100).sum()), "of", len(report))
report.round(4).to_csv("/kaggle/working/serving_parity_report.csv", index=False)

# %%
X_live = scorer.to_frame(live_rows)
p_live = scorer.predict_raw(X_live)
saved = pd.read_parquet(glob.glob("/kaggle/input/**/predictions_serving/preds_lgbm_core_test.parquet", recursive=True)[0])
saved = saved.set_index("TransactionID").loc[test_rows["TransactionID"]]
diff = np.abs(p_live - saved["p_raw"].to_numpy(dtype="float64"))
print("largest difference in probability:", float(diff.max()))
print("rows with a difference above 0.001:", int((diff > 1e-3).sum()), "of", len(diff))

# %%
sample = [to_txn(r) for r in test_rows.sample(300, random_state=0).itertuples()]
t_plain = [scorer.score(t, commit=False)["latency_ms"] for t in sample]
t_explain = [scorer.score(t, explain=True, commit=False)["latency_ms"] for t in sample[:100]]
print("score only:    median", round(np.median(t_plain), 1), "ms | p95", round(np.percentile(t_plain, 95), 1), "ms")
print("score + SHAP:  median", round(np.median(t_explain), 1), "ms | p95", round(np.percentile(t_explain, 95), 1), "ms")

# %%
import json
print(json.dumps(scorer.score(sample[0], explain=True, commit=False), indent=2, default=str))
for name in sorted(os.listdir("/kaggle/working/sentinel_l3a")):
    print(name)
