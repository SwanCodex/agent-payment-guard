# %%
import glob
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score

OUT = "/kaggle/working/shap"
os.makedirs(OUT, exist_ok=True)

# %%
core = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0])

cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "DeviceType", "browser", "os"]
for c in cat_cols:                       # category lists come from train only (same as before)
    cats = core.loc[core["split"] == "train", c].astype("category").cat.categories
    core[c] = pd.Categorical(core[c], categories=cats)

all_features = [c for c in core.columns if c not in ["TransactionID", "split", "isFraud"]]
val1 = core[core["split"] == "val1"]
print(len(all_features), "features | val1 rows:", len(val1))

# %%
files = sorted(glob.glob("/kaggle/input/**/models_final/lgbm_seed*.txt", recursive=True))
print(files)
assert len(files) == 5, "attach the layer3a-models-calibration notebook as an input"

seeds = [int(f.split("seed")[-1].split(".")[0]) for f in files]
boosters = [lgb.Booster(model_file=f) for f in files]

cal_path = glob.glob("/kaggle/input/**/lgbm_core_bundle/calibration.json", recursive=True)[0]
with open(cal_path) as f:
    cal = json.load(f)
assert cal["method"] == "temperature", "expected the temperature calibration file"
T = cal["T"]
print("temperature T =", T)

def predict_raw(frame):
    return np.mean([b.predict(frame[all_features]) for b in boosters], axis=0)

def calibrate(p):
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / T)

# check: this must reproduce the PR-AUC from the models notebook (about 0.411 on val1)
p_val1 = predict_raw(val1)
print("PR-AUC val1:", round(average_precision_score(val1["isFraud"], p_val1), 4))

# %%
sample = val1.sample(20000, random_state=0)
X = sample[all_features]

# one array per model: one column per feature, plus a last column (the model's base value)
contribs = [b.predict(X, pred_contrib=True) for b in boosters]
print("shape of one model's SHAP array:", contribs[0].shape)

# check: the contributions of one model must add up to that model's own log-odds output
margin = contribs[0].sum(axis=1)
direct = logit(np.clip(boosters[0].predict(X), 1e-9, 1 - 1e-9))
print("largest difference between sum of SHAP and the model's log-odds:", float(np.abs(margin - direct).max()))

# explanation of the 5-seed model = the average of the 5 models' SHAP values
shap_avg = np.mean(contribs, axis=0)
shap_values = pd.DataFrame(shap_avg[:, :-1], index=X.index, columns=all_features)

# the average is in log-odds, while our final score averages probabilities, so check how close they are
p_from_shap = expit(shap_avg.sum(axis=1))
p_final = predict_raw(sample)
print("largest gap between sigmoid(sum of averaged SHAP) and the final score:",
      round(float(np.abs(p_from_shap - p_final).max()), 4))

# %%
importance = shap_values.abs().mean().sort_values(ascending=False)
importance.to_csv(f"{OUT}/shap_global_importance.csv", header=["mean_abs_shap"])
print(importance.head(15).round(4))

top15 = importance.head(15)[::-1]
plt.figure(figsize=(7, 5))
plt.barh(top15.index, top15.values)
plt.xlabel("mean |SHAP value| (log-odds)")
plt.title("Top 15 features, final LightGBM (val1 sample)")
plt.tight_layout()
plt.savefig(f"{OUT}/shap_global_importance.png", dpi=120)
plt.show()

# %%
numeric = [c for c in all_features if c not in cat_cols]
direction = pd.Series({c: X[c].corr(shap_values[c], method="spearman") for c in numeric})

table = pd.DataFrame({"mean_abs_shap": importance,
                      "value_vs_shap_correlation": direction}).loc[importance.index]
table.head(15).round(3).to_csv(f"{OUT}/shap_direction_top15.csv")
print(table.head(15).round(3))
# value_vs_shap_correlation > 0: larger values push the fraud score up. < 0: larger values push it down.
# It is only a rough guide. Blank means the feature is a category (no order).

# %%
rows = []
top10_ensemble = set(importance.index[:10])
for s, c in zip(seeds, contribs):
    imp_s = pd.Series(np.abs(c[:, :-1]).mean(axis=0), index=all_features)
    rows.append({"seed": s,
                 "rank correlation with ensemble": imp_s.corr(importance, method="spearman"),
                 "top-10 overlap": len(set(imp_s.sort_values(ascending=False).index[:10]) & top10_ensemble)})
stability = pd.DataFrame(rows).round(3)
stability.to_csv(f"{OUT}/shap_seed_stability.csv", index=False)
print(stability)

# %%
def clean(v):
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)

def top5_evidence(row_values, row_shap, k=5):
    order = np.argsort(-np.abs(row_shap))[:k]          # biggest absolute contributions first
    return [{"feature": all_features[j],
             "value": clean(row_values.iloc[j]),
             "shap_logodds": round(float(row_shap[j]), 4)} for j in order]

p_sample = calibrate(p_final)
order = np.argsort(p_sample)
picks = {"highest risk score": order[-1], "middle": order[len(order) // 2], "lowest risk score": order[0]}

examples = {}
for name, i in picks.items():
    ev = top5_evidence(X.iloc[i], shap_values.iloc[i].to_numpy())
    examples[name] = {"risk_probability": round(float(p_sample[i]), 4),
                      "actual_isFraud": int(sample["isFraud"].iloc[i]),
                      "top5": ev}
    print(name, "| p =", examples[name]["risk_probability"], "| actual:", examples[name]["actual_isFraud"])
    for e in ev:
        print("    ", e)

with open(f"{OUT}/shap_top5_examples.json", "w") as f:
    json.dump(examples, f, indent=2)

# %%
one = X.iloc[[0]]
t0 = time.time()
for _ in range(20):
    np.mean([b.predict(one, pred_contrib=True) for b in boosters], axis=0)
ms = (time.time() - t0) / 20 * 1000
print("milliseconds to explain one transaction with 5 models:", round(ms, 1))

# %%
for name in sorted(os.listdir(OUT)):
    print(f"{name:35s} {os.path.getsize(f'{OUT}/{name}') / 1e3:8.1f} KB")

# %%
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve

core = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0],
                       columns=["TransactionID", "isFraud"])
exp = pd.read_parquet(glob.glob("/kaggle/input/**/val1_predictions_experiments.parquet", recursive=True)[0])

def load_final(name):
    p = pd.read_parquet(glob.glob(f"/kaggle/input/**/preds_lgbm_core_{name}.parquet", recursive=True)[0])
    return p.merge(core, on="TransactionID", how="left")

final_val1, final_val2 = load_final("val1"), load_final("val2")

def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])

def scores(name, y, p, t):
    pred = p >= t
    return {"model": name, "threshold": round(t, 3),
            "accuracy %": round(100 * accuracy_score(y, pred), 2),
            "precision %": round(100 * precision_score(y, pred, zero_division=0), 2),
            "recall %": round(100 * recall_score(y, pred), 2),
            "F1 %": round(100 * f1_score(y, pred), 2)}

# --- val1: every saved model, at 0.5 and at the best-F1 threshold (chosen on val1 itself, so a bit optimistic)
y1 = exp["isFraud"].to_numpy()
models_val1 = {"Logistic regression": exp["p_logreg"].to_numpy(),
               "LightGBM (single seed)": exp["p_lgbm"].to_numpy(),
               "XGBoost": exp["p_xgb"].to_numpy(),
               "CatBoost": exp["p_catboost"].to_numpy(),
               "LightGBM final (5 seeds, calibrated)": final_val1["p_cal"].to_numpy()}
y1_final = final_val1["isFraud"].to_numpy()

rows = []
for name, p in models_val1.items():
    y = y1_final if "final" in name else y1
    rows.append(scores(name + " @0.5", y, p, 0.5))
    rows.append(scores(name + " @best-F1", y, p, best_f1_threshold(y, p)))
print("val1 (majority-class accuracy:", round(100 * (1 - y1.mean()), 2), "%)")
print(pd.DataFrame(rows).to_string(index=False))

# --- val2: the final model only, using thresholds fixed on val1
y2 = final_val2["isFraud"].to_numpy()
t_f1 = best_f1_threshold(y1_final, final_val1["p_cal"].to_numpy())
rows = [scores("final @0.5", y2, final_val2["p_cal"].to_numpy(), 0.5),
        scores("final @best-F1 from val1", y2, final_val2["p_cal"].to_numpy(), t_f1),
        scores("final @tau_block 0.328", y2, final_val2["p_cal"].to_numpy(), 0.32816)]
print("\nval2 (majority-class accuracy:", round(100 * (1 - y2.mean()), 2), "%)")
print(pd.DataFrame(rows).to_string(index=False))

# %%
# 1. Are history features really important as a group?
history = [c for c in all_features if c.startswith(("card_", "addr_", "dev_"))]
row_level = [c for c in all_features if c not in history]

imp = shap_values.abs().mean()
share = pd.Series({f"row-level features ({len(row_level)})": imp[row_level].sum(),
                   f"history features ({len(history)})": imp[history].sum()})
print((share / share.sum() * 100).round(1))

# %%
# 2. Where does the 170 ms go?
one = X.iloc[[0]]
for label, fn in [("5 models, predict only", lambda: np.mean([b.predict(one) for b in boosters], axis=0)),
                  ("1 model, predict only", lambda: boosters[0].predict(one)),
                  ("1 model, with SHAP", lambda: boosters[0].predict(one, pred_contrib=True))]:
    t0 = time.time()
    for _ in range(20):
        fn()
    print(label, round((time.time() - t0) / 20 * 1000, 1), "ms")

# %%
# Idea: split conformal prediction.
#   1. On a calibration set (val1) give every transaction a "surprise score":
#        score = 1 - (probability the model gave to the TRUE class)
#   2. qhat = the score that 90% of val1 transactions stay below (alpha = 0.1).
#   3. For a new transaction a class goes into its prediction set if 1 - P(class) <= qhat.
#   4. If the set holds BOTH classes, the model is unsure and the layer abstains.

# %%
# %% 1. Load the saved calibrated predictions and the true labels
import glob
import json
import os
 
import numpy as np
import pandas as pd
 
OUTC = "/kaggle/working/conformal"
os.makedirs(OUTC, exist_ok=True)
 
labels = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0],
                         columns=["TransactionID", "isFraud"])
 
def load(name):
    path = glob.glob(f"/kaggle/input/**/preds_lgbm_core_{name}.parquet", recursive=True)[0]
    out = pd.read_parquet(path).merge(labels, on="TransactionID", how="left")
    assert out["isFraud"].notna().all(), "some TransactionIDs have no label"
    return out
 
val1, val2 = load("val1"), load("val2")
p1, y1 = val1["p_cal"].to_numpy(dtype="float64"), val1["isFraud"].to_numpy()
p2, y2 = val2["p_cal"].to_numpy(dtype="float64"), val2["isFraud"].to_numpy()
print("rows:", len(val1), len(val2), "| fraud rate val1:", round(y1.mean(), 4), "val2:", round(y2.mean(), 4))

# %%
# %% 2. Helper functions
alpha = 0.1
 
def conformal_quantile(scores, alpha):
    s = np.sort(scores)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))      # the "finite sample" correction
    return float(s[min(k, n) - 1])
 
def report(name, y, p, q_legit, q_fraud):
    in_legit = p <= q_legit              # legit stays in the set when 1 - P(legit) <= q_legit
    in_fraud = (1 - p) <= q_fraud        # fraud stays in the set when 1 - P(fraud) <= q_fraud
    both = in_legit & in_fraud
    only_fraud = in_fraud & ~in_legit
    only_legit = in_legit & ~in_fraud
    empty = ~in_legit & ~in_fraud
    covered = np.where(y == 1, in_fraud, in_legit)    # is the true class inside the set?
    return {"version": name,
            "abstain: both classes %": 100 * both.mean(),
            "set = {fraud} only %": 100 * only_fraud.mean(),
            "set = {legit} only %": 100 * only_legit.mean(),
            "empty set %": 100 * empty.mean(),
            "true class in set (all) %": 100 * covered.mean(),
            "true class in set (fraud rows) %": 100 * covered[y == 1].mean(),
            "true class in set (legit rows) %": 100 * covered[y == 0].mean()}

# %%
# %% 3. Standard split conformal (the version in the handbook): one qhat for everything
score_true_class = np.where(y1 == 1, 1 - p1, p1)
qhat = conformal_quantile(score_true_class, alpha)
print("qhat (standard, alpha = 0.1):", round(qhat, 5))
if qhat < 0.5:
    print("qhat < 0.5, so a set can never hold both classes: this version never abstains")
 
rows = [report("standard, val1 (calibration set)", y1, p1, qhat, qhat),
        report("standard, val2 (out of sample)", y2, p2, qhat, qhat)]
print(pd.DataFrame(rows).set_index("version").round(2).T)

# %%
# %% 4. Class-conditional conformal: a separate qhat for each class (each gets its own 90%)
q_legit = conformal_quantile(p1[y1 == 0], alpha)           # legit rows: score = P(fraud)
q_fraud = conformal_quantile(1 - p1[y1 == 1], alpha)       # fraud rows: score = 1 - P(fraud)
print("q_legit:", round(q_legit, 5), "| q_fraud:", round(q_fraud, 5))
print("band where both classes are in the set: from p =", round(1 - q_fraud, 5), "to p =", round(q_legit, 5))
if 1 - q_fraud > q_legit:
    print("the band is empty: no transaction has both classes in its set")
 
rows = [report("class-conditional, val1 (calibration set)", y1, p1, q_legit, q_fraud),
        report("class-conditional, val2 (out of sample)", y2, p2, q_legit, q_fraud)]
print(pd.DataFrame(rows).set_index("version").round(2).T)

# %%
# %% 5. Save a thresholds.json candidate (contract keys only) and a separate conformal report
draft_path = glob.glob("/kaggle/input/**/lgbm_core_bundle/thresholds_draft.json", recursive=True)[0]
with open(draft_path) as f:
    draft = json.load(f)
 
thresholds = {"tau_allow": draft["tau_allow"], "tau_block": draft["tau_block"], "qhat": qhat,
              "fpr_at_block": 0.005, "fnr_at_allow": 0.01, "val_split": "val1"}
assert thresholds["tau_allow"] < thresholds["tau_block"], "tau_allow must be below tau_block"
 
with open(f"{OUTC}/thresholds_candidate.json", "w") as f:
    json.dump(thresholds, f, indent=2)
 
with open(f"{OUTC}/conformal_report.json", "w") as f:
    json.dump({"alpha": alpha, "calibration_split": "val1",
               "qhat_standard": qhat, "q_legit": q_legit, "q_fraud": q_fraud}, f, indent=2)
 
print(json.dumps(thresholds, indent=2))
for name in sorted(os.listdir(OUTC)):
    print(f"{name:30s} {os.path.getsize(f'{OUTC}/{name}') / 1e3:6.1f} KB")
