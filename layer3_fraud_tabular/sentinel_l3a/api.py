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
