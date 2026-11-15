# Model card: Layer 3A tabular fraud model (lgbm_core 1.0.0)

**Task.** Estimate P(fraud) for one card transaction from the CoreTxn fields plus the entity history store.
Output is a calibrated probability; the decision engine turns it into ALLOW / REVIEW / BLOCK.

**Data.** IEEE-CIS Fraud Detection (Vesta, via Kaggle): train_transaction + train_identity, 590,540 rows, 3.5% fraud.
Real data only; no resampling or synthetic data.

**Splits.** Time order (TransactionDT, ties by TransactionID): train 70%, val1 10%, val2 10%, test 10%;
rows with the same timestamp never straddle a boundary. Checksums (md5 of the sorted TransactionID list):
- train: md5 `2053703c02def8162b92a8f768984d6d`
- val1: md5 `c8853e342298fbc55764e24d195d9d36`
- val2: md5 `7cc09ee22961e113e97d4a2e32dac560`
- test: md5 `d23b45bde0a6a9ef337f7733f1ac5c5f`

**Features.** 43 features available at authorization time: amount features, categorical fields, frequency encodings
(fitted on train), and strictly-past history aggregates per card, address and device key (counts in 1 h / 24 h / 7 d,
amount mean / std / z-score, time since last transaction, distinct emails and addresses). See `feature_spec.yaml`.
Every aggregate uses only earlier transactions (tested by shuffling the future).

**Model.** LightGBM, 5 seeds averaged (11, 22, 33, 44, 55), hyper-parameters tuned with Optuna on expanding-window time CV,
early stopping on val1. Calibration: temperature scaling, T = 1.0680, fitted on val1.

**Decision rule.** Cost based: ALLOW if p < 0.05; BLOCK if p >= 0.833; otherwise REVIEW
(c = 0.05, b = 0.3; these costs are assumptions, change them in `policy_config.json`).

**Test results.** PR-AUC 0.294, ROC-AUC 0.836, recall at 1% FPR 0.231,
ECE 0.0029. Details in `eval_report.md`.

**Limits (read before using).**
- IEEE-CIS has no real card id: `card_key` is a heuristic (card1, card2, card3, card5, addr1, addr2). Entity quality is bounded by it.
- One dataset from one period; fraud patterns drift. PR-AUC is lower on later splits than on earlier ones.
- The live system must keep an entity history store. A new entity gets default features and `warm = false`.
- `hour` and `dayofweek` are relative: the reference start of TransactionDT is unknown.
- Absolute PR-AUC is moderate, so a large share of fraud is only sent to REVIEW, not blocked. The tabular model alone blocks very few transactions.
- The conformal qhat on val1 is 0.0724; it is below 0.5, so a prediction set never holds both classes and the layer never abstains.
- The SHAP and conformal reports in `reports/` were produced for the development model (48 features) before the five live-unavailable features were removed. The shipping model's explanations come from `api.py` (top-5 SHAP of the first seed).
- Not covered: the L3 blender with the graph model (separate component), ULB sanity check.
