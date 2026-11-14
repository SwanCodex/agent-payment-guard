"""Serving parity: the live code (EntityStore + L3AScorer.features_for) must give the same history features
as the offline training code. This test uses synthetic transactions so it runs anywhere. The same check on the real
IEEE-CIS test period is in reports/serving/serving_parity_report.csv (notebook layer3a_serving_parity)."""
import numpy as np
import pandas as pd

from sentinel_l3a.api import L3AScorer
from sentinel_l3a.entity_stats import EntityStore
from sentinel_l3a.features.history import distinct_before, make_entity_features, make_history_features
from test_no_future_leak import make_frame


def make_scorer():
    scorer = L3AScorer.__new__(L3AScorer)        # no model files are needed for the history features
    scorer.store = EntityStore()
    scorer.freq = {c: {} for c in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]}
    return scorer


def test_live_history_features_equal_offline_features():
    d = make_frame(n=3000)
    offline = pd.concat([
        make_history_features(d),
        make_entity_features(d, "addr_key", "addr"),
        make_entity_features(d, "device_key", "dev"),
        distinct_before(d, "card_key", "P_emaildomain").rename("card_distinct_emails"),
        distinct_before(d, "device_key", "card_key").rename("dev_distinct_cards"),
    ], axis=1)
    offline.loc[d["device_key"] == "missing", "dev_distinct_cards"] = np.nan

    scorer = make_scorer()
    rows = []
    for r in d.itertuples():
        txn = {"amount": r.TransactionAmt, "event_time": r.TransactionDT, "card_key": r.card_key,
               "addr_key": r.addr_key, "device_key": r.device_key, "payer_email_domain": r.P_emaildomain}
        rows.append(scorer.features_for(txn))     # what the guard computes at request time
        scorer.remember(txn)                      # afterwards the transaction becomes history
    live = pd.DataFrame(rows)

    for column in offline.columns:
        a, b = offline[column].to_numpy(dtype="float64"), live[column].to_numpy(dtype="float64")
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True), f"{column} differs between offline and live"
