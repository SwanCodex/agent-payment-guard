# %%
import glob, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_curve

core = pd.read_parquet(glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0])

cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "DeviceType", "browser", "os"]
for c in cat_cols:                                    # same train-only categories as before
    cats = core.loc[core["split"] == "train", c].astype("category").cat.categories
    core[c] = pd.Categorical(core[c], categories=cats)

all_features = [c for c in core.columns if c not in ["TransactionID", "split", "isFraud"]]
train = core[core["split"] == "train"]
val1 = core[core["split"] == "val1"]
val2 = core[core["split"] == "val2"]
y_val1, y_val2 = val1["isFraud"].to_numpy(), val2["isFraud"].to_numpy()

# Core model predictions saved by notebook 2 (matched by TransactionID, not by position)
def load_raw_preds(frame, name):
    path = glob.glob(f"/kaggle/input/**/preds_lgbm_core_{name}.parquet", recursive=True)[0]
    merged = frame[["TransactionID"]].merge(
        pd.read_parquet(path)[["TransactionID", "p_raw"]], on="TransactionID", how="left")
    assert merged["p_raw"].notna().all(), "some TransactionIDs have no saved prediction"
    return merged["p_raw"].to_numpy(dtype="float64")

p1_raw, p2_raw = load_raw_preds(val1, "val1"), load_raw_preds(val2, "val2")
print("PR-AUC val1:", round(average_precision_score(y_val1, p1_raw), 4),
      "| val2:", round(average_precision_score(y_val2, p2_raw), 4))     # should print about 0.411 / 0.2892

def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))

def tau_block_for(y, p, fpr_target):
    neg = np.sort(p[y == 0])
    return neg[len(neg) - int(fpr_target * len(neg))]

def tau_allow_for(y, p, fnr_target):
    pos = np.sort(p[y == 1])
    return pos[int(fnr_target * len(pos)) - 1]

# %%
#Cell H: Build the Full feature table
import glob

txn_path = glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)
id_path = glob.glob("/kaggle/input/**/train_identity.csv", recursive=True)
assert txn_path and id_path, "attach the raw IEEE-CIS dataset as a second input"

header_t = pd.read_csv(txn_path[0], nrows=0).columns
header_i = pd.read_csv(id_path[0], nrows=0).columns

def is_group_col(c, prefix):
    return c.startswith(prefix) and c[len(prefix):].isdigit()

extra_t = [c for c in header_t if any(is_group_col(c, p) for p in ["C", "D", "M", "V"])]
extra_i = [c for c in header_i if c.startswith("id_") and c not in ("id_30", "id_31")]  # id_30/31 are already in Core as browser/os

raw_t = pd.read_csv(txn_path[0], usecols=["TransactionID"] + extra_t)
raw_i = pd.read_csv(id_path[0], usecols=["TransactionID"] + extra_i)

full = core.merge(raw_t, on="TransactionID", how="left").merge(raw_i, on="TransactionID", how="left")
assert len(full) == len(core)

# text columns (M1-M9, many id_ columns) -> categories learned from train only
for c in extra_t + extra_i:
    if full[c].dtype == object:
        cats = full.loc[full["split"] == "train", c].astype("category").cat.categories
        full[c] = pd.Categorical(full[c], categories=cats)

# V columns: keep one representative per group of highly correlated columns
v_cols = [c for c in extra_t if c.startswith("V")]
sample = full.loc[full["split"] == "train", v_cols].sample(100000, random_state=0)
corr = sample.corr().abs()                 # how similar is each pair of V columns?

to_drop = set()
for i, col in enumerate(v_cols):
    if col in to_drop:
        continue
    for other in v_cols[i + 1:]:
        if corr.loc[col, other] >= 0.95:   # almost a copy of `col`
            to_drop.add(other)

keep_v = [c for c in v_cols if c not in to_drop]

other_extra = [c for c in extra_t if not c.startswith("V")] + extra_i
full_features = all_features + other_extra + keep_v
print(len(v_cols), "V columns ->", len(keep_v), "kept")
print("Core:", len(all_features), "| Full:", len(full_features))

# %%
#Cell I: Train Full with the same settings and compare
import time

best_params = {
    "num_leaves": 56, "min_child_samples": 162, "colsample_bytree": 0.5998368910791798,
    "subsample": 0.7571172192068059, "reg_alpha": 0.23423849847112907,
    "reg_lambda": 0.0015339162591163618, "scale_pos_weight": 1,
}

full_train = full[full["split"] == "train"]
full_val1 = full[full["split"] == "val1"]
full_val2 = full[full["split"] == "val2"]

full_models, per_seed = [], []
t0 = time.time()
for s in [11, 22, 33, 44, 55]:
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample_freq=1,
                           random_state=s, verbose=-1, **best_params)
    m.fit(full_train[full_features], full_train["isFraud"],
          eval_set=[(full_val1[full_features], y_val1)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    full_models.append(m)
    per_seed.append({"seed": s,
                     "PR-AUC val1": average_precision_score(y_val1, m.predict_proba(full_val1[full_features])[:, 1]),
                     "PR-AUC val2": average_precision_score(y_val2, m.predict_proba(full_val2[full_features])[:, 1])})
    print(f"seed {s} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

print(pd.DataFrame(per_seed).round(4))

f1 = np.mean([m.predict_proba(full_val1[full_features])[:, 1] for m in full_models], axis=0)
f2 = np.mean([m.predict_proba(full_val2[full_features])[:, 1] for m in full_models], axis=0)

def gray_share(p_a, p_b, y_a):
    tb, ta = tau_block_for(y_a, p_a, 0.005), tau_allow_for(y_a, p_a, 0.01)
    return ((p_b < tb) & (p_b > ta)).mean()

rows = []
for name, (a, b) in {"Core (deployable)": (p1_raw, p2_raw), "Full (offline ceiling)": (f1, f2)}.items():
    rows.append({"feature set": name,
                 "PR-AUC val1": average_precision_score(y_val1, a),
                 "PR-AUC val2": average_precision_score(y_val2, b),
                 "recall@1%FPR val1": recall_at_fpr(y_val1, a, 0.01),
                 "recall@1%FPR val2": recall_at_fpr(y_val2, b, 0.01),
                 "gray share val2": gray_share(a, b, y_val1)})
gap = pd.DataFrame(rows).set_index("feature set").round(4)
print(gap)
gap.to_csv("/kaggle/working/core_vs_full.csv")

# which Full-only features matter most (first seed)
imp = pd.Series(full_models[0].booster_.feature_importance("gain"), index=full_features)
print(imp.drop(index=all_features).sort_values(ascending=False).head(15).round(0))

# %%
groups = {
    "C columns": [c for c in other_extra if c.startswith("C") and c[1:].isdigit()],
    "D columns": [c for c in other_extra if c.startswith("D") and c[1:].isdigit()],
    "M columns": [c for c in other_extra if c.startswith("M") and c[1:].isdigit()],
    "id columns": extra_i,
    "V columns (kept)": keep_v,
}

rows = []
for name, cols in groups.items():
    features = all_features + cols
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample_freq=1,
                           random_state=11, verbose=-1, **best_params)
    m.fit(full_train[features], full_train["isFraud"],
          eval_set=[(full_val1[features], y_val1)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    rows.append({"Core plus": name, "n_added": len(cols),
                 "PR-AUC val1": average_precision_score(y_val1, m.predict_proba(full_val1[features])[:, 1]),
                 "PR-AUC val2": average_precision_score(y_val2, m.predict_proba(full_val2[features])[:, 1])})
    print(name, "done", flush=True)

group_table = pd.DataFrame(rows).set_index("Core plus").round(4)
print(group_table)
group_table.to_csv("/kaggle/working/core_plus_group.csv")

# %%
imp = pd.Series(full_models[0].booster_.feature_importance("gain"), index=full_features)
imp.sort_values(ascending=False).head(30).round(0).to_csv("/kaggle/working/full_top30_features.csv")
