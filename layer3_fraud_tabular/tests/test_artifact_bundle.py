"""Checks the model bundle artifacts/layer3_tabular/lgbm_core/1.0.0 the way the handbook's validate_artifacts.py does:
schema of each file, tau_allow < tau_block, calibration monotonic, probabilities in [0, 1], and a 100-row smoke predict.
Put this file in layer3_fraud_tabular/tests/ and run:  python -m pytest -q
(The official contracts/validate_artifacts.py is owned by P4 and is run separately once it exists.)"""
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml

from sentinel_l3a.api import L3AScorer

BUNDLE = pathlib.Path(__file__).resolve().parents[2] / "artifacts" / "layer3_tabular" / "lgbm_core" / "1.0.0"
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle folder not found next to layer3_fraud_tabular/")

REQUIRED = ["calibration.json", "thresholds.json", "feature_spec.yaml", "model_card.md", "eval_report.md",
            "manifest.json", "policy_config.json", "frequency_maps.json", "feature_order.json",
            "preds_lgbm_core_val1.parquet", "preds_lgbm_core_val2.parquet", "preds_lgbm_core_test.parquet"]


def test_all_required_files_exist():
    missing = [name for name in REQUIRED if not (BUNDLE / name).exists()]
    assert not missing, f"missing files: {missing}"
    assert len(list(BUNDLE.glob("lgbm_seed*.txt"))) == 5


def test_thresholds_schema_and_order():
    th = json.loads((BUNDLE / "thresholds.json").read_text())
    for key in ["tau_allow", "tau_block", "qhat", "fpr_at_block", "fnr_at_allow", "val_split"]:
        assert key in th, f"thresholds.json lacks {key}"
    assert 0 < th["tau_allow"] < th["tau_block"] < 1


def test_calibration_is_a_monotone_temperature():
    cal = json.loads((BUNDLE / "calibration.json").read_text())
    assert cal["method"] == "temperature" and cal["T"] > 0      # dividing logits by T > 0 keeps the order


@pytest.mark.parametrize("split", ["val1", "val2", "test"])
def test_prediction_files_follow_the_exchange_schema(split):
    p = pd.read_parquet(BUNDLE / f"preds_lgbm_core_{split}.parquet")
    assert list(p.columns) == ["TransactionID", "p_raw", "p_cal", "warm", "fold_seed"]
    assert str(p["TransactionID"].dtype) == "int64" and str(p["p_cal"].dtype) == "float32"
    assert str(p["warm"].dtype) == "bool" and str(p["fold_seed"].dtype) == "int16"
    assert p["TransactionID"].is_unique
    assert p["p_raw"].between(0, 1).all() and p["p_cal"].between(0, 1).all()


def test_feature_spec_matches_the_model_features():
    spec = yaml.safe_load((BUNDLE / "feature_spec.yaml").read_text())
    order = json.loads((BUNDLE / "feature_order.json").read_text())
    for name in order["features"]:
        assert name in spec and spec[name]["serving_available"] is True


def test_smoke_predict_on_100_transactions():
    scorer = L3AScorer(str(BUNDLE), str(BUNDLE))
    freq = scorer.freq
    first = {c: next(iter(freq[c])) for c in freq}            # a known value of each categorical column
    rng = np.random.default_rng(0)
    for i in range(100):
        txn = {"amount": float(np.round(rng.lognormal(4, 1), 2)), "event_time": 1000 + 60 * i,
               "product_cd": first["ProductCD"], "card_network": first["card4"], "card_type": first["card6"],
               "payer_email_domain": first["P_emaildomain"], "recipient_email_domain": first["R_emaildomain"],
               "billing_addr1": 300.0, "billing_addr2": 87.0, "dist1": None,
               "device_type": first["DeviceType"] if i % 2 else None,
               "card_key": f"card{i % 10}", "addr_key": "300.0_87.0", "device_key": "missing" if i % 2 == 0 else f"dev{i % 5}"}
        out = scorer.score(txn)
        assert 0.0 <= out["p"] <= 1.0 and 0.0 <= out["p_raw"] <= 1.0
