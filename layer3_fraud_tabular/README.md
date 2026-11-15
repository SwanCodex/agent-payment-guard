# Layer 3A: tabular transaction fraud (P2)

LightGBM fraud model on IEEE-CIS (Core features that exist at authorization time plus entity history),
temperature-calibrated, with a cost-based ALLOW / REVIEW / BLOCK rule and serving code.

## Layout
```
sentinel_l3a/
  entity_stats.py      incremental per-card / address / device history (strictly-past look-ups)
  api.py               L3AScorer: one transaction in -> calibrated probability (+ top-5 SHAP evidence)
  features/history.py  the offline version of the same history features (training time)
configs/l3a.yaml      seeds, split, tuned parameters, policy
notebooks/            the Kaggle notebooks as .py files (percent format)
reports/              EDA, model comparison, ablations, SHAP, serving parity, final test report, data manifest
model_cards/          model_card.md
tests/                no-future-leak test, serving-parity test
scripts/              download_data.sh, export_notebooks.py
```
The trained model bundle is in `artifacts/layer3_tabular/lgbm_core/1.0.0/` (Git LFS):
5 LightGBM seeds, calibration.json, thresholds.json, policy_config.json, feature_spec.yaml, frequency_maps.json,
feature_order.json, manifest.json, model_card.md, eval_report.md and the three prediction files for the blender.

## Use
```python
from sentinel_l3a.api import L3AScorer
scorer = L3AScorer("<bundle folder>", "<bundle folder>")   # model files and json files sit in the same folder
out = scorer.score(txn_dict, explain=False)                # out["p"] is the calibrated fraud probability
```
Keep ONE scorer (it holds the entity store). Load past transactions first with `scorer.remember(txn)` (warm start).
Decide with `policy_config.json`: ALLOW if p < allow_if_p_below, BLOCK if p >= block_if_p_at_least, otherwise REVIEW.
`event_time` is seconds (IEEE-CIS TransactionDT) or a datetime counted from 1970-01-01.

## Tests
```
pip install -e ".[dev]"
make test
```

## Not done / known gaps
- `validate_artifacts.py` (contracts/) is owned by P4 and is pending. `tests/test_artifact_bundle.py` checks the same things in the meantime.
- The L3 blender with the graph model is not part of this folder (orchestration/decision).
- The ULB sanity check is done: see `reports/ulb_sanity/`.


