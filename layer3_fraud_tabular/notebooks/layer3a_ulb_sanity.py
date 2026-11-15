# %% [markdown]
# # SENTINEL Layer 3A (P2): sanity check on the ULB credit card fraud data
# Run this in a NEW Kaggle notebook named  layer3a-ulb-sanity  (CPU only, a few minutes).
# Add ONE input: the public Kaggle dataset "Credit Card Fraud Detection" (ULB / Machine Learning Group, file creditcard.csv).
#
# Why (handbook 8.2, secondary check): run the SAME pipeline steps on a different dataset that has NO entities
# (no card, address or device history), to show that the split, training, calibration and metric code is not
# IEEE-CIS-specific in a misleading way. This data is never merged with IEEE-CIS and is reported separately.
# Everything is saved to /kaggle/working/ulb_sanity/

# %% 1. Load the data
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

OUT = "/kaggle/working/ulb_sanity"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(glob.glob("/kaggle/input/**/creditcard.csv", recursive=True)[0])
print(df.shape, "| fraud:", int(df["Class"].sum()), "| fraud rate:", round(df["Class"].mean(), 5))
assert df.shape[0] == 284807 and int(df["Class"].sum()) == 492, "this does not look like the ULB creditcard.csv"

# %% 2. Time split 70 / 10 / 10 / 10 (same rule as IEEE-CIS: sort by time, never cut through rows with the same time)
df["row_id"] = np.arange(len(df))                       # ULB has no transaction id, so the row number breaks ties
df = df.sort_values(["Time", "row_id"]).reset_index(drop=True)
n = len(df)

def boundary(fraction):
    i = int(n * fraction)
    while df.loc[i, "Time"] == df.loc[i - 1, "Time"]:
        i -= 1
    return i

b1, b2, b3 = boundary(0.70), boundary(0.80), boundary(0.90)
df["split"] = "test"
df.loc[:b3 - 1, "split"] = "val2"
df.loc[:b2 - 1, "split"] = "val1"
df.loc[:b1 - 1, "split"] = "train"

split_summary = df.groupby("split").agg(rows=("Class", "count"), fraud=("Class", "sum"), fraud_rate=("Class", "mean"),
                                        first_time=("Time", "min"), last_time=("Time", "max")).loc[["train", "val1", "val2", "test"]]
print(split_summary)
split_summary.to_csv(f"{OUT}/ulb_split_summary.csv")

checksums = {s: hashlib.md5(",".join(str(i) for i in sorted(df.loc[df["split"] == s, "row_id"])).encode()).hexdigest()
             for s in ["train", "val1", "val2", "test"]}
with open(f"{OUT}/ulb_split_checksums.json", "w") as f:
    json.dump(checksums, f, indent=2)

# %% 3. Features: amount features, hour, and the 28 anonymized PCA columns (there are no entities, so no history features)
v_cols = [f"V{i}" for i in range(1, 29)]
X = pd.concat([pd.DataFrame({"log_amount": np.log1p(df["Amount"]),
                             "cents": (df["Amount"] % 1).round(2),
                             "is_round_amount": (df["Amount"] % 1 == 0).astype(int),
                             "hour": (df["Time"] // 3600) % 24}),
               df[v_cols]], axis=1)
y = df["Class"].to_numpy()
m = {s: (df["split"] == s).to_numpy() for s in ["train", "val1", "val2", "test"]}
print(X.shape[1], "features")

# %% 4. Metric helpers (same definitions as the IEEE-CIS report)
def recall_at_fpr(y_true, p, fpr_target):
    fpr, tpr, _ = roc_curve(y_true, p)
    return float(np.interp(fpr_target, fpr, tpr))

def ece_equal_mass(y_true, p, bins=15):
    order = np.argsort(p)
    y_s, p_s = np.asarray(y_true)[order], np.asarray(p)[order]
    return sum(len(i) / len(p_s) * abs(y_s[i].mean() - p_s[i].mean()) for i in np.array_split(np.arange(len(p_s)), bins))

def metrics(y_true, p):
    return {"PR-AUC": average_precision_score(y_true, p), "ROC-AUC": roc_auc_score(y_true, p),
            "recall@0.1%FPR": recall_at_fpr(y_true, p, 0.001), "recall@1%FPR": recall_at_fpr(y_true, p, 0.01),
            "ECE": ece_equal_mass(y_true, p), "mean p": float(np.mean(p)), "fraud rows": int(np.sum(y_true))}

# %% 5. Baseline: logistic regression
scaler = StandardScaler().fit(X[m["train"]])
lr = LogisticRegression(max_iter=2000).fit(scaler.transform(X[m["train"]]), y[m["train"]])
p_lr = {s: lr.predict_proba(scaler.transform(X[m[s]]))[:, 1] for s in ["val1", "val2", "test"]}
print("logistic regression, val1 PR-AUC:", round(average_precision_score(y[m["val1"]], p_lr["val1"]), 4))

# %% 6. LightGBM, 5 seeds averaged, early stopping on val1 (same recipe as the main model)
seeds = [11, 22, 33, 44, 55]
models = []
for s in seeds:
    model = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                               colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=s, verbose=-1)
    model.fit(X[m["train"]], y[m["train"]], eval_set=[(X[m["val1"]], y[m["val1"]])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    models.append(model)
    print("seed", s, "| trees used:", model.best_iteration_)

def predict_raw(split):
    return np.mean([mdl.predict_proba(X[m[split]])[:, 1] for mdl in models], axis=0)

p_raw = {s: predict_raw(s) for s in ["val1", "val2", "test"]}

# %% 7. Temperature scaling fitted on val1 (same method as the main model)
eps = 1e-6
z1 = logit(np.clip(p_raw["val1"], eps, 1 - eps))
res = minimize_scalar(lambda T: log_loss(y[m["val1"]], expit(z1 / T)), bounds=(0.3, 5), method="bounded")
T_ulb = float(res.x)
print("temperature T =", round(T_ulb, 4))
p_cal = {s: expit(logit(np.clip(p_raw[s], eps, 1 - eps)) / T_ulb) for s in ["val1", "val2", "test"]}

# %% 8. Leakage sanity check: train on SHUFFLED labels; a clean pipeline must then score about the fraud rate
rng = np.random.default_rng(0)
y_shuffled = rng.permutation(y[m["train"]])
shuf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=0, verbose=-1).fit(X[m["train"]], y_shuffled)
shuf_pr = average_precision_score(y[m["val2"]], shuf.predict_proba(X[m["val2"]])[:, 1])
base_rate = float(y[m["val2"]].mean())
print("PR-AUC with shuffled training labels:", round(shuf_pr, 5), "| fraud rate on val2:", round(base_rate, 5))

# %% 9. Results on val2 and on the test split (the test split is evaluated once, here)
rows = []
for split in ["val2", "test"]:
    for name, p in [("logistic regression", p_lr[split]), ("LightGBM 5 seeds (raw)", p_raw[split]),
                    ("LightGBM 5 seeds (temperature calibrated)", p_cal[split])]:
        rows.append({"split": split, "model": name, **metrics(y[m[split]], p)})
results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))
results.round(5).to_csv(f"{OUT}/ulb_metrics.csv", index=False)

# %% 10. Write the report file
def md_table(frame, digits=4):
    d = frame.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.astype(str).to_numpy()]
    return "\n".join([head, sep] + body)

leak_ok = shuf_pr < 3 * base_rate
report = f"""# ULB credit card fraud: sanity check of the Layer 3A pipeline

Purpose: run the same split / training / calibration / metric steps on a dataset WITHOUT entities, to show that none of
that code is IEEE-CIS-specific in a misleading way. This dataset is never merged with IEEE-CIS, and the numbers below are
not comparable with the IEEE-CIS results (different data, no history features, anonymized PCA columns).

Data: ULB / Worldline credit card fraud (Sept 2013), 284,807 transactions, 492 fraud, columns Time, V1-V28, Amount, Class.

## Split (time order, same rule as IEEE-CIS)
{md_table(split_summary.reset_index())}

## Results
{md_table(results)}

Calibration: temperature scaling fitted on val1, T = {T_ulb:.4f}.

## Leakage check
Training on shuffled labels gives PR-AUC {shuf_pr:.5f} on val2, against a fraud rate of {base_rate:.5f}.
{"This is close to the fraud rate, so the pipeline does not leak labels." if leak_ok else "This is clearly above the fraud rate: look for leakage before trusting the pipeline."}

## Limits
- Only 492 fraud cases in total, so each of val1 / val2 / test holds few fraud rows (see the table): every metric here is noisy.
- The data covers two days, so there is no long-term drift to test.
- No history features: this checks the generic parts of the pipeline only.
"""
with open(f"{OUT}/ulb_report.md", "w") as f:
    f.write(report)
for name in sorted(os.listdir(OUT)):
    print(f"{name:30s} {os.path.getsize(f'{OUT}/{name}') / 1e3:8.1f} KB")
