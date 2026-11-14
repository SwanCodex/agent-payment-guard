# %%
# %% [markdown]
# # SENTINEL Layer 3A (P2): two checks before deployment
# Inputs to attach: layer3a-data-features-eda and layer3a-models-calibration. No GPU needed.
#
# Check A: if some features are NOT available when the model runs live, how much does it lose?
# Check B: what do different cost settings do to ALLOW / REVIEW / BLOCK on a later period (val2)?

# %%
# %% 1. Setup: load features, the 5 final models and the temperature
import glob
import json
 
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_curve
 
core = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0])
 
cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "DeviceType", "browser", "os"]
for c in cat_cols:                       # same train-only categories as in training
    cats = core.loc[core["split"] == "train", c].astype("category").cat.categories
    core[c] = pd.Categorical(core[c], categories=cats)
 
all_features = [c for c in core.columns if c not in ["TransactionID", "split", "isFraud"]]
val2 = core[core["split"] == "val2"]
y_val2 = val2["isFraud"].to_numpy()
 
files = sorted(glob.glob("/kaggle/input/**/models_final/lgbm_seed*.txt", recursive=True))
assert len(files) == 5, "attach the layer3a-models-calibration notebook as an input"
boosters = [lgb.Booster(model_file=f) for f in files]
print("val2 rows:", len(val2), "| features:", len(all_features))

# %%
# %% 2. Check A helper: predict while pretending some features are unavailable (set to missing)
def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))
 
def predict_variant(frame, missing_numeric=(), missing_categorical=(), identity_from_device=False):
    X = frame[all_features].copy()
    for c in missing_numeric:
        X[c] = np.nan
    for c in missing_categorical:
        X[c] = pd.Categorical([None] * len(X), categories=frame[c].cat.categories)
    if identity_from_device:             # live replacement: "has identity" = "device type is known"
        X["has_identity"] = frame["DeviceType"].notna().astype(int).to_numpy()
    return np.mean([b.predict(X) for b in boosters], axis=0)

# %%
# %% 3. Check A: what each missing feature costs (no retraining, val2 only)
variants = {
    "as trained (all 48 features)": {},
    "browser and os not available": dict(missing_numeric=["browser_freq", "os_freq"],
                                         missing_categorical=["browser", "os"]),
    "card_distinct_addr not available": dict(missing_numeric=["card_distinct_addr"]),
    "has_identity taken from device type": dict(identity_from_device=True),
    "all three changes together": dict(missing_numeric=["browser_freq", "os_freq", "card_distinct_addr"],
                                       missing_categorical=["browser", "os"],
                                       identity_from_device=True),
}
 
rows = []
for name, kwargs in variants.items():
    p = predict_variant(val2, **kwargs)
    rows.append({"version": name,
                 "PR-AUC val2": average_precision_score(y_val2, p),
                 "recall@1%FPR val2": recall_at_fpr(y_val2, p, 0.01)})
    print(name, "done", flush=True)
 
check_a = pd.DataFrame(rows).set_index("version")
check_a["PR-AUC change"] = check_a["PR-AUC val2"] - check_a["PR-AUC val2"].iloc[0]
print(check_a.round(4))
check_a.round(4).to_csv("/kaggle/working/serving_feature_check.csv")

# %%
# %% 4. Check B setup: saved calibrated predictions for val2 (temperature-scaled)
labels = core[["TransactionID", "isFraud"]]
pred_path = glob.glob("/kaggle/input/**/preds_lgbm_core_val2.parquet", recursive=True)[0]
saved = pd.read_parquet(pred_path).merge(labels, on="TransactionID", how="left")
p = saved["p_cal"].to_numpy(dtype="float64")
y = saved["isFraud"].to_numpy()
print("rows:", len(saved), "| fraud rate:", round(y.mean(), 4), "| mean predicted:", round(p.mean(), 4))

# %%
# %% 5. Check B: the decision rule driven by costs
# Costs, measured in units of "one fraud that gets through" (= 1):
#   c = cost of sending a transaction to review (the LLM arbiter or a human)
#   b = cost of blocking a good transaction
# For one transaction with fraud probability p, the expected cost is:
#   ALLOW: p        REVIEW: c        BLOCK: (1 - p) * b
# We simply choose the cheapest action. That gives two cutoffs, with no fitting to the data.
def simulate(p, y, b, c):
    t_allow = c                                                  # allow while p < c
    t_block = max(1 - c / b, b / (1 + b)) if c < b else 2.0     # 2.0 means "never block"
    block = p >= t_block
    allow = (p < t_allow) & ~block
    review = ~allow & ~block
    fraud, legit = y == 1, y == 0
    cost = ((allow & fraud).sum() * 1.0 + review.sum() * c + (block & legit).sum() * b) / len(p)
    return {"cost of review c": c, "cost of wrong block b": b,
            "allow cutoff": round(t_allow, 3), "block cutoff": round(t_block, 3),
            "ALLOW %": 100 * allow.mean(), "REVIEW %": 100 * review.mean(), "BLOCK %": 100 * block.mean(),
            "fraud allowed %": 100 * (allow & fraud).sum() / fraud.sum(),
            "fraud blocked %": 100 * (block & fraud).sum() / fraud.sum(),
            "fraud in review %": 100 * (review & fraud).sum() / fraud.sum(),
            "good blocked %": 100 * (block & legit).sum() / legit.sum(),
            "cost per transaction": cost}
 
rows = [simulate(p, y, b, c) for b in [0.1, 0.3, 1.0] for c in [0.01, 0.02, 0.05, 0.10]]
check_b = pd.DataFrame(rows).round(3)
print(check_b.to_string(index=False))
check_b.to_csv("/kaggle/working/cost_policy_simulation.csv", index=False)
 
print("\nfor comparison, cost per transaction if we ALLOW everything:", round(y.mean(), 4))
print("                                    if we REVIEW everything: equals c")

# %%
same = (val2["DeviceType"].notna().astype(int).to_numpy() == val2["has_identity"].to_numpy()).mean()
print(round(same * 100, 2), "% of val2 rows where has_identity equals 'device type is known'")
