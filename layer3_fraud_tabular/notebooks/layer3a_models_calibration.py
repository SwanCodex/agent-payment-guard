# %%
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

path = glob.glob("/kaggle/input/**/core_features.parquet", recursive=True)[0]
core = pd.read_parquet(path)
print(core.shape)
print(core["split"].value_counts())

# %%
cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "DeviceType", "browser", "os"]

# category lists come from train only
for c in cat_cols:
    cats = core.loc[core["split"] == "train", c].astype("category").cat.categories
    core[c] = pd.Categorical(core[c], categories=cats)

train = core[core["split"] == "train"]
val1 = core[core["split"] == "val1"]

all_features = [c for c in core.columns if c not in ["TransactionID", "split", "isFraud"]]
y_train, y_val = train["isFraud"], val1["isFraud"]
print(len(all_features), "features")

# %%
def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))

def evaluate(y, p):
    return {"PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p),
            "recall@1%FPR": recall_at_fpr(y, p, 0.01)}

# %%
numeric_features = [c for c in all_features if c not in cat_cols]
medians = train[numeric_features].median()

def prep(frame):
    return frame[numeric_features].fillna(medians)

scaler = StandardScaler().fit(prep(train))
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(scaler.transform(prep(train)), y_train)

p_lr = lr.predict_proba(scaler.transform(prep(val1)))[:, 1]
print({k: round(v, 4) for k, v in evaluate(y_val, p_lr).items()})

# %%
def train_lgbm(features, seed=42):
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=64,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1)
    model.fit(train[features], y_train,
              eval_set=[(val1[features], y_val)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    p = model.predict_proba(val1[features])[:, 1]
    return model, p

model_all, p_all = train_lgbm(all_features)
print("trees:", model_all.best_iteration_,
      {k: round(v, 4) for k, v in evaluate(y_val, p_all).items()})

# %%
history = [c for c in all_features if c.startswith(("card_", "addr_", "dev_"))]
long_history = ["card_cnt_all", "card_first_seen", "card_distinct_emails", "card_distinct_addr",
                "dev_distinct_cards", "card_amt_mean", "card_amt_std", "card_amt_z",
                "addr_amt_mean", "dev_amt_mean"]

feature_sets = {
    "A: all Core features": all_features,
    "B: without long-history features": [c for c in all_features if c not in long_history],
    "C: without any history features": [c for c in all_features if c not in history],
}

rows = []
for name, feats in feature_sets.items():
    m, p = train_lgbm(feats)
    r = evaluate(y_val, p)
    r.update({"set": name, "n_features": len(feats), "trees": m.best_iteration_})
    rows.append(r)

results = pd.DataFrame(rows).set_index("set").round(4)
print(results)
results.to_csv("/kaggle/working/ablation_history.csv")

# %%
val2 = core[core["split"] == "val2"]
y_val2 = val2["isFraud"]

rows = []
for name, feats in feature_sets.items():
    m, p1 = train_lgbm(feats)                      # early stopping uses val1
    p2 = m.predict_proba(val2[feats])[:, 1]        # val2 was never used for any choice
    rows.append({"set": name,
                 "PR-AUC val1": average_precision_score(y_val, p1),
                 "PR-AUC val2": average_precision_score(y_val2, p2),
                 "recall@1%FPR val2": recall_at_fpr(y_val2, p2, 0.01)})

print(pd.DataFrame(rows).set_index("set").round(4))

# %%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", enable_categorical=True,
    eval_metric="aucpr", early_stopping_rounds=100,
    random_state=42, n_jobs=-1)
xgb_model.fit(train[all_features], y_train,
              eval_set=[(val1[all_features], y_val)], verbose=False)

p_xgb = xgb_model.predict_proba(val1[all_features])[:, 1]
print("trees:", xgb_model.best_iteration,
      {k: round(float(v), 4) for k, v in evaluate(y_val, p_xgb).items()})

# %%
from catboost import CatBoostClassifier

def for_catboost(frame):
    out = frame[all_features].copy()
    for c in cat_cols:
        out[c] = out[c].astype(object).where(out[c].notna(), "missing").astype(str)
    return out

cb_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.05, depth=6,
    eval_metric="PRAUC", early_stopping_rounds=100,
    random_seed=42, verbose=0, thread_count=-1)
cb_model.fit(for_catboost(train), y_train, cat_features=cat_cols,
             eval_set=(for_catboost(val1), y_val))

p_cb = cb_model.predict_proba(for_catboost(val1))[:, 1]
print("trees:", cb_model.get_best_iteration(),
      {k: round(float(v), 4) for k, v in evaluate(y_val, p_cb).items()})

# %%
combos = {
    "Logistic regression (baseline)": p_lr,
    "LightGBM": p_all,
    "XGBoost": p_xgb,
    "CatBoost": p_cb,
    "LightGBM + CatBoost (average)": (p_all + p_cb) / 2,
    "LightGBM + XGBoost (average)": (p_all + p_xgb) / 2,
    "All three (average)": (p_all + p_xgb + p_cb) / 3,
}

rows = []
for name, p in combos.items():
    r = evaluate(y_val, p)
    r["model"] = name
    rows.append(r)

table = pd.DataFrame(rows).set_index("model").astype(float).round(4)
print(table)
table.to_csv("/kaggle/working/model_comparison_val1.csv")

# %%
#cell 11
import json
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

n_train = len(train)
folds = [(0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]   # (train ends, validation ends)

def objective(trial):
    params = dict(
        n_estimators=1500, learning_rate=0.05,
        num_leaves=trial.suggest_int("num_leaves", 16, 256, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        scale_pos_weight=trial.suggest_categorical("scale_pos_weight", [1, 3]),
        random_state=42, verbose=-1)

    scores, trees = [], []
    for tr_end, va_end in folds:
        a, b, c = 0, int(n_train * tr_end), int(n_train * va_end)
        X_a, y_a = train[all_features].iloc[a:b], y_train.iloc[a:b]
        X_b, y_b = train[all_features].iloc[b:c], y_train.iloc[b:c]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_a, y_a, eval_set=[(X_b, y_b)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False)])
        scores.append(average_precision_score(y_b, m.predict_proba(X_b)[:, 1]))
        trees.append(m.best_iteration_)
    trial.set_user_attr("mean_trees", int(np.mean(trees)))
    return float(np.mean(scores))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20, timeout=3600)    # stops after 20 trials or 1 hour

print("best CV PR-AUC:", round(study.best_value, 4))
print(study.best_params)
with open("/kaggle/working/lgbm_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=2)

# %%
#models save
import os
import joblib

OUTM = "/kaggle/working/models_experiments"
os.makedirs(OUTM, exist_ok=True)

model_all.booster_.save_model(f"{OUTM}/lgbm_core_untuned.txt")
xgb_model.save_model(f"{OUTM}/xgb_core.json")
cb_model.save_model(f"{OUTM}/catboost_core.cbm")
joblib.dump({"scaler": scaler, "model": lr, "medians": medians},
            f"{OUTM}/logreg_baseline.joblib")

val1_preds = pd.DataFrame({
    "TransactionID": val1["TransactionID"].to_numpy(),
    "isFraud": y_val.to_numpy(),
    "p_logreg": p_lr, "p_lgbm": p_all, "p_xgb": p_xgb, "p_catboost": p_cb,
})
val1_preds.to_parquet(f"{OUTM}/val1_predictions_experiments.parquet", index=False)

for f in sorted(os.listdir(OUTM)):
    print(f"{f:45s} {os.path.getsize(f'{OUTM}/{f}') / 1e6:8.1f} MB")

# %%
best_params = {
    "num_leaves": 56, "min_child_samples": 162, "colsample_bytree": 0.5998368910791798,
    "subsample": 0.7571172192068059, "reg_alpha": 0.23423849847112907,
    "reg_lambda": 0.0015339162591163618, "scale_pos_weight": 1,
}

val2 = core[core["split"] == "val2"]
y_val2 = val2["isFraud"]

tuned = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample_freq=1,
                           random_state=42, verbose=-1, **best_params)
tuned.fit(train[all_features], y_train,
          eval_set=[(val1[all_features], y_val)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])

p1 = tuned.predict_proba(val1[all_features])[:, 1]
p2 = tuned.predict_proba(val2[all_features])[:, 1]

compare = pd.DataFrame({
    "PR-AUC val1": [0.4013, average_precision_score(y_val, p1)],
    "PR-AUC val2": [0.2805, average_precision_score(y_val2, p2)],
    "recall@1%FPR val1": [0.3377, recall_at_fpr(y_val, p1, 0.01)],
    "recall@1%FPR val2": [0.2550, recall_at_fpr(y_val2, p2, 0.01)],
}, index=["LightGBM untuned (from earlier run)", f"LightGBM tuned ({tuned.best_iteration_} trees)"]).round(4)
print(compare)

# %%
import os
import time
import xgboost as xgb

best_params = {
    "num_leaves": 56, "min_child_samples": 162, "colsample_bytree": 0.5998368910791798,
    "subsample": 0.7571172192068059, "reg_alpha": 0.23423849847112907,
    "reg_lambda": 0.0015339162591163618, "scale_pos_weight": 1,
}
val2 = core[core["split"] == "val2"]
y_val2 = val2["isFraud"]

os.makedirs("/kaggle/working/models_final", exist_ok=True)
seeds = [11, 22, 33, 44, 55]
lgb_models, xgb_models = [], []
rows = []
t0 = time.time()

for s in seeds:
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample_freq=1,
                           random_state=s, verbose=-1, **best_params)
    m.fit(train[all_features], y_train,
          eval_set=[(val1[all_features], y_val)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    m.booster_.save_model(f"/kaggle/working/models_final/lgbm_seed{s}.txt")
    lgb_models.append(m)

    x = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", enable_categorical=True,
                          eval_metric="aucpr", early_stopping_rounds=100,
                          random_state=s, n_jobs=-1)
    x.fit(train[all_features], y_train,
          eval_set=[(val1[all_features], y_val)], verbose=False)
    x.save_model(f"/kaggle/working/models_final/xgb_seed{s}.json")
    xgb_models.append(x)

    for name, model in [("LightGBM", m), ("XGBoost", x)]:
        a = average_precision_score(y_val, model.predict_proba(val1[all_features])[:, 1])
        b = average_precision_score(y_val2, model.predict_proba(val2[all_features])[:, 1])
        rows.append({"model": name, "seed": s, "PR-AUC val1": a, "PR-AUC val2": b})
    print(f"seed {s} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)

per_seed = pd.DataFrame(rows)
print(per_seed.round(4))
print(per_seed.groupby("model")[["PR-AUC val1", "PR-AUC val2"]].agg(["mean", "std"]).round(4))

# average the probabilities over seeds, then compare
def avg_pred(models, frame):
    return np.mean([m.predict_proba(frame[all_features])[:, 1] for m in models], axis=0)

lgb_v1, lgb_v2 = avg_pred(lgb_models, val1), avg_pred(lgb_models, val2)
xgb_v1, xgb_v2 = avg_pred(xgb_models, val1), avg_pred(xgb_models, val2)

combos = {
    "LightGBM tuned, 5 seeds": (lgb_v1, lgb_v2),
    "XGBoost, 5 seeds": (xgb_v1, xgb_v2),
    "LightGBM + XGBoost average": ((lgb_v1 + xgb_v1) / 2, (lgb_v2 + xgb_v2) / 2),
}
out = []
for name, (p1, p2) in combos.items():
    out.append({"model": name,
                "PR-AUC val1": average_precision_score(y_val, p1),
                "PR-AUC val2": average_precision_score(y_val2, p2),
                "recall@1%FPR val1": recall_at_fpr(y_val, p1, 0.01),
                "recall@1%FPR val2": recall_at_fpr(y_val2, p2, 0.01)})
final_table = pd.DataFrame(out).set_index("model").round(4)
print(final_table)
final_table.to_csv("/kaggle/working/model_comparison_5seeds.csv")

# %%
# cell A
import os, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_curve

seeds = [11, 22, 33, 44, 55]
files = [f"/kaggle/working/models_final/lgbm_seed{s}.txt" for s in seeds]
print("model files found:", [os.path.exists(f) for f in files])
boosters = [lgb.Booster(model_file=f) for f in files]

def predict_raw(frame):
    return np.mean([b.predict(frame[all_features]) for b in boosters], axis=0)

val1 = core[core["split"] == "val1"]
val2 = core[core["split"] == "val2"]
y_val1, y_val2 = val1["isFraud"].to_numpy(), val2["isFraud"].to_numpy()

p1_raw, p2_raw = predict_raw(val1), predict_raw(val2)
print("PR-AUC raw  val1:", round(average_precision_score(y_val1, p1_raw), 4),
      "| val2:", round(average_precision_score(y_val2, p2_raw), 4))

# %%
# Cell B: Isotonic calibration and ECE
iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
iso.fit(p1_raw, y_val1)                 # calibration is fitted on val1 only

def calibrate(p):
    return iso.predict(p)

def ece_equal_mass(y, p, bins=15):
    order = np.argsort(p)
    y_s, p_s = np.asarray(y)[order], np.asarray(p)[order]
    total = 0.0
    for idx in np.array_split(np.arange(len(p_s)), bins):
        total += len(idx) / len(p_s) * abs(y_s[idx].mean() - p_s[idx].mean())
    return total

p1_cal, p2_cal = calibrate(p1_raw), calibrate(p2_raw)

print("ECE raw        val1:", round(ece_equal_mass(y_val1, p1_raw), 4), "| val2:", round(ece_equal_mass(y_val2, p2_raw), 4))
print("ECE calibrated val1:", round(ece_equal_mass(y_val1, p1_cal), 4), "| val2:", round(ece_equal_mass(y_val2, p2_cal), 4))
print("val2 mean predicted:", round(p2_cal.mean(), 4), "| actual fraud rate:", round(y_val2.mean(), 4))
print("PR-AUC calibrated val1:", round(average_precision_score(y_val1, p1_cal), 4),
      "| val2:", round(average_precision_score(y_val2, p2_cal), 4))

# save in the contract format
cal = {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
print("max gap between saved curve and sklearn:", float(np.max(np.abs(np.interp(p2_raw, cal["x"], cal["y"]) - p2_cal))))
os.makedirs("/kaggle/working/lgbm_core_bundle", exist_ok=True)
with open("/kaggle/working/lgbm_core_bundle/calibration.json", "w") as f:
    json.dump(cal, f)

# %%
# Cell C: Thresholds and how they behave on val2
# tau_block: score where val1 false-positive rate is at most 0.5%
fpr, tpr, thr = roc_curve(y_val1, p1_cal)
i = np.searchsorted(fpr, 0.005, side="right") - 1
tau_block = float(thr[i])
if np.isinf(tau_block):
    print("WARNING: the top calibration step already exceeds 0.5% FPR; check the calibration curve")
    tau_block = float(thr[1])

# tau_allow: score at or below which at most 1% of the fraud cases fall
fraud_scores = np.sort(p1_cal[y_val1 == 1])
k = int(0.01 * len(fraud_scores))
tau_allow = float(fraud_scores[k - 1]) if k > 0 else 0.0

print("tau_allow:", round(tau_allow, 5), "| tau_block:", round(tau_block, 5))
assert tau_allow < tau_block, "thresholds are in the wrong order"

def policy_report(y, p, name):
    block, allow = p >= tau_block, p <= tau_allow
    gray = ~block & ~allow
    print(f"{name}: BLOCK {block.mean():.1%} | ALLOW {allow.mean():.1%} | GRAY (LLM) {gray.mean():.1%}")
    print(f"   FPR at block: {block[y == 0].mean():.4f} | fraud caught by block: {block[y == 1].mean():.3f} | "
          f"FNR at allow: {allow[y == 1].mean():.4f}")

policy_report(y_val1, p1_cal, "val1 (thresholds fitted here)")
policy_report(y_val2, p2_cal, "val2 (out of sample)")

with open("/kaggle/working/lgbm_core_bundle/thresholds_draft.json", "w") as f:
    json.dump({"tau_allow": tau_allow, "tau_block": tau_block,
               "fpr_at_block_val1": float((p1_cal >= tau_block)[y_val1 == 0].mean()),
               "fnr_at_allow_val1": float((p1_cal <= tau_allow)[y_val1 == 1].mean()),
               "val_split": "val1"}, f, indent=2)

# %%
# Cell D: Prediction files for the blender
test = core[core["split"] == "test"]
os.makedirs("/kaggle/working/predictions", exist_ok=True)

for name, frame in [("val1", val1), ("val2", val2), ("test", test)]:
    raw = predict_raw(frame)
    out = pd.DataFrame({
        "TransactionID": frame["TransactionID"].astype("int64").to_numpy(),
        "p_raw": raw.astype("float32"),
        "p_cal": calibrate(raw).astype("float32"),
        "warm": (frame["card_cnt_all"] > 0).to_numpy(),       # card had earlier transactions
        "fold_seed": np.zeros(len(frame), dtype="int16"),      # 0 = average over 5 seeds
    })
    out.to_parquet(f"/kaggle/working/predictions/preds_lgbm_core_{name}.parquet", index=False)
    print(name, out.shape, "| warm share:", round(out["warm"].mean(), 3))

# %%
#cell E
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss

eps = 1e-6
z1 = logit(np.clip(p1_raw, eps, 1 - eps))
z2 = logit(np.clip(p2_raw, eps, 1 - eps))

# find the single number T that makes val1 probabilities most accurate (lowest log loss)
res = minimize_scalar(lambda T: log_loss(y_val1, expit(z1 / T)), bounds=(0.3, 5), method="bounded")
T = float(res.x)
print("T =", round(T, 3))

p1_t, p2_t = expit(z1 / T), expit(z2 / T)

versions = {"raw": (p1_raw, p2_raw), "isotonic": (p1_cal, p2_cal), "temperature": (p1_t, p2_t)}
rows = []
for name, (a, b) in versions.items():
    rows.append({"method": name,
                 "ECE val1": ece_equal_mass(y_val1, a), "ECE val2": ece_equal_mass(y_val2, b),
                 "PR-AUC val1": average_precision_score(y_val1, a),
                 "PR-AUC val2": average_precision_score(y_val2, b),
                 "log loss val2": log_loss(y_val2, np.clip(b, eps, 1 - eps)),
                 "mean pred val2": b.mean()})
print(pd.DataFrame(rows).set_index("method").round(4))

with open("/kaggle/working/lgbm_core_bundle/calibration_temperature_candidate.json", "w") as f:
    json.dump({"method": "temperature", "T": T}, f)

# %%
#Cell F: what thresholds would give a workable gray zone
def tau_block_for(y, p, fpr_target):
    neg = np.sort(p[y == 0])
    m = int(fpr_target * len(neg))              # how many negatives may be blocked
    return neg[len(neg) - m]

def tau_allow_for(y, p, fnr_target):
    pos = np.sort(p[y == 1])
    m = int(fnr_target * len(pos))              # how many frauds may be allowed
    return pos[m - 1]

rows = []
for fnr in [0.01, 0.02, 0.05, 0.10, 0.20]:
    for fpr in [0.005, 0.01, 0.02]:
        ta, tb = tau_allow_for(y_val1, p1_raw, fnr), tau_block_for(y_val1, p1_raw, fpr)
        if ta >= tb:
            continue
        def shares(y, p):
            block, allow = p >= tb, p <= ta
            return (~block & ~allow).mean(), allow[y == 1].mean(), block[y == 0].mean(), block[y == 1].mean()
        g1, _, _, _ = shares(y_val1, p1_raw)
        g2, fnr2, fpr2, caught2 = shares(y_val2, p2_raw)
        rows.append({"FNR target": fnr, "FPR target": fpr, "gray val1": g1, "gray val2": g2,
                     "FNR val2": fnr2, "FPR val2": fpr2, "fraud blocked val2": caught2})

tradeoff = pd.DataFrame(rows).round(3)
print(tradeoff.to_string(index=False))
tradeoff.to_csv("/kaggle/working/threshold_tradeoff.csv", index=False)

# %%
#cell G
#1. calibration file in the contract format
cal = {"method": "temperature", "T": T}
with open("/kaggle/working/lgbm_core_bundle/calibration.json", "w") as f:
    json.dump(cal, f)

def calibrate_t(p):
    return expit(logit(np.clip(p, eps, 1 - eps)) / T)

p1_cal, p2_cal = calibrate_t(p1_raw), calibrate_t(p2_raw)

# 2. thresholds with the handbook's targets (FPR <= 0.5%, FNR <= 1%), fitted on val1
tau_block = float(tau_block_for(y_val1, p1_cal, 0.005))
tau_allow = float(tau_allow_for(y_val1, p1_cal, 0.01))
assert tau_allow < tau_block
print("tau_allow:", round(tau_allow, 5), "| tau_block:", round(tau_block, 5))

policy_report(y_val1, p1_cal, "val1 (thresholds fitted here)")
policy_report(y_val2, p2_cal, "val2 (out of sample)")

with open("/kaggle/working/lgbm_core_bundle/thresholds_draft.json", "w") as f:
    json.dump({"tau_allow": tau_allow, "tau_block": tau_block,
               "fpr_at_block": 0.005, "fnr_at_allow": 0.01, "val_split": "val1",
               "note": "draft: the team may change the FPR/FNR targets via RFC"}, f, indent=2)

# 3. prediction files for the blender, now with temperature-calibrated p_cal
test = core[core["split"] == "test"]
for name, frame in [("val1", val1), ("val2", val2), ("test", test)]:
    raw = predict_raw(frame)
    out = pd.DataFrame({
        "TransactionID": frame["TransactionID"].astype("int64").to_numpy(),
        "p_raw": raw.astype("float32"),
        "p_cal": calibrate_t(raw).astype("float32"),
        "warm": (frame["card_cnt_all"] > 0).to_numpy(),
        "fold_seed": np.zeros(len(frame), dtype="int16"),
    })
    out.to_parquet(f"/kaggle/working/predictions/preds_lgbm_core_{name}.parquet", index=False)
    print(name, out.shape, "| mean p_cal:", round(float(out["p_cal"].mean()), 4))

# %%
#retraining

# %%
# %% [markdown]
# ## Serving retrain: add these cells at the END of layer3a-models-calibration
# They need Cell 1 and Cell 2 of that notebook to have run in this session (they create core, all_features, cat_cols).
# You do NOT need to re-run CatBoost, Optuna or the 5-seed cells first.
# Everything is saved to new folders, so the old files stay untouched:
#   models_serving/  lgbm_serving_bundle/  predictions_serving/

# %%
# %% R1. The features that will exist when the model runs live
import json
import os
import time
 
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, log_loss
 
dropped = ["browser", "os", "browser_freq", "os_freq", "card_distinct_addr"]
serving_features = [c for c in all_features if c not in dropped]
 
core_live = core.copy()
core_live["has_identity"] = core_live["DeviceType"].notna().astype(int)     # live: "device type is known"
agree = (core["has_identity"] == core_live["has_identity"]).mean()
print("has_identity agrees with 'device type is known' on", round(100 * agree, 2), "% of rows")
 
train_s = core_live[core_live["split"] == "train"]
val1_s = core_live[core_live["split"] == "val1"]
val2_s = core_live[core_live["split"] == "val2"]
test_s = core_live[core_live["split"] == "test"]
y1s, y2s = val1_s["isFraud"].to_numpy(), val2_s["isFraud"].to_numpy()
print(len(serving_features), "serving features (was", len(all_features), ")")

# %%
# %% R2. Retrain 5 seeds on the serving features (same tuned settings as before)
best_params = {
    "num_leaves": 56, "min_child_samples": 162, "colsample_bytree": 0.5998368910791798,
    "subsample": 0.7571172192068059, "reg_alpha": 0.23423849847112907,
    "reg_lambda": 0.0015339162591163618, "scale_pos_weight": 1,
}
os.makedirs("/kaggle/working/models_serving", exist_ok=True)
 
seeds = [11, 22, 33, 44, 55]
new_models, rows = [], []
t0 = time.time()
for s in seeds:
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, subsample_freq=1,
                           random_state=s, verbose=-1, **best_params)
    m.fit(train_s[serving_features], train_s["isFraud"],
          eval_set=[(val1_s[serving_features], y1s)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    m.booster_.save_model(f"/kaggle/working/models_serving/lgbm_seed{s}.txt")
    new_models.append(m)
    rows.append({"seed": s,
                 "PR-AUC val1": average_precision_score(y1s, m.predict_proba(val1_s[serving_features])[:, 1]),
                 "PR-AUC val2": average_precision_score(y2s, m.predict_proba(val2_s[serving_features])[:, 1])})
    print(f"seed {s} done, {(time.time() - t0) / 60:.1f} min elapsed", flush=True)
 
print(pd.DataFrame(rows).round(4))
 
def avg_raw(frame):
    return np.mean([m.predict_proba(frame[serving_features])[:, 1] for m in new_models], axis=0)
 
p1_new, p2_new = avg_raw(val1_s), avg_raw(val2_s)
summary = pd.DataFrame({
    "PR-AUC val1": [0.4110, np.nan, average_precision_score(y1s, p1_new)],
    "PR-AUC val2": [0.2892, 0.2532, average_precision_score(y2s, p2_new)],
}, index=["old model, 48 features (earlier run)",
          "old model with the 5 features unavailable (earlier check)",
          "NEW model, serving features only"]).round(4)
print(summary)

# %%
# %% R3. Calibrate with temperature scaling (fitted on val1, same method as before)
eps = 1e-6
z1 = logit(np.clip(p1_new, eps, 1 - eps))
z2 = logit(np.clip(p2_new, eps, 1 - eps))
res = minimize_scalar(lambda T: log_loss(y1s, expit(z1 / T)), bounds=(0.3, 5), method="bounded")
T_new = float(res.x)
print("new T =", round(T_new, 4), "(old T was 1.087)")
 
def ece_equal_mass(y, p, bins=15):
    order = np.argsort(p)
    y_s, p_s = np.asarray(y)[order], np.asarray(p)[order]
    total = 0.0
    for idx in np.array_split(np.arange(len(p_s)), bins):
        total += len(idx) / len(p_s) * abs(y_s[idx].mean() - p_s[idx].mean())
    return total
 
def calibrate_new(p):
    return expit(logit(np.clip(p, eps, 1 - eps)) / T_new)
 
p1_cal, p2_cal = calibrate_new(p1_new), calibrate_new(p2_new)
print("ECE raw        val1:", round(ece_equal_mass(y1s, p1_new), 4), "| val2:", round(ece_equal_mass(y2s, p2_new), 4))
print("ECE calibrated val1:", round(ece_equal_mass(y1s, p1_cal), 4), "| val2:", round(ece_equal_mass(y2s, p2_cal), 4))
print("val2 mean predicted:", round(p2_cal.mean(), 4), "| actual fraud rate:", round(y2s.mean(), 4))
 
os.makedirs("/kaggle/working/lgbm_serving_bundle", exist_ok=True)
with open("/kaggle/working/lgbm_serving_bundle/calibration.json", "w") as f:
    json.dump({"T": T_new}, f)

# %%
# %% R4. Cost-based ALLOW / REVIEW / BLOCK on val2 for the NEW model
# costs in units of "one fraud that gets through" = 1:  c = cost of a review, b = cost of blocking a good transaction
# expected cost: ALLOW = p, REVIEW = c, BLOCK = (1 - p) * b  -> pick the cheapest action
def simulate(p, y, b, c):
    t_allow = c
    t_block = max(1 - c / b, b / (1 + b)) if c < b else 2.0
    block = p >= t_block
    allow = (p < t_allow) & ~block
    review = ~allow & ~block
    fraud, legit = y == 1, y == 0
    cost = ((allow & fraud).sum() * 1.0 + review.sum() * c + (block & legit).sum() * b) / len(p)
    return {"review cost c": c, "block cost b": b, "allow below": round(t_allow, 3), "block from": round(t_block, 3),
            "ALLOW %": 100 * allow.mean(), "REVIEW %": 100 * review.mean(), "BLOCK %": 100 * block.mean(),
            "fraud allowed %": 100 * (allow & fraud).sum() / fraud.sum(),
            "fraud blocked %": 100 * (block & fraud).sum() / fraud.sum(),
            "fraud in review %": 100 * (review & fraud).sum() / fraud.sum(),
            "good blocked %": 100 * (block & legit).sum() / legit.sum(),
            "cost per transaction": cost}
 
settings = [(0.3, 0.02), (0.3, 0.05), (0.3, 0.10), (0.1, 0.05)]
print(pd.DataFrame([simulate(p2_cal, y2s, b, c) for b, c in settings]).round(3).to_string(index=False))
print("cost per transaction if we ALLOW everything:", round(y2s.mean(), 4))
 
c_review, b_block = 0.05, 0.3                  # <- the chosen setting; change these two numbers if you decide differently
policy = {"review_cost_c": c_review, "block_cost_b": b_block,
          "allow_if_p_below": c_review,
          "block_if_p_at_least": max(1 - c_review / b_block, b_block / (1 + b_block)),
          "calibration_T": T_new, "probabilities_used": "temperature-calibrated p_cal"}
with open("/kaggle/working/lgbm_serving_bundle/policy_config.json", "w") as f:
    json.dump(policy, f, indent=2)
print(json.dumps(policy, indent=2))

# %%
# %% R5. Prediction files for the blender (new folder, same exchange format; test is predicted but NOT scored)
os.makedirs("/kaggle/working/predictions_serving", exist_ok=True)
for name, frame in [("val1", val1_s), ("val2", val2_s), ("test", test_s)]:
    raw = avg_raw(frame)
    out = pd.DataFrame({
        "TransactionID": frame["TransactionID"].astype("int64").to_numpy(),
        "p_raw": raw.astype("float32"),
        "p_cal": calibrate_new(raw).astype("float32"),
        "warm": (frame["card_cnt_all"] > 0).to_numpy(),
        "fold_seed": np.zeros(len(frame), dtype="int16"),
    })
    out.to_parquet(f"/kaggle/working/predictions_serving/preds_lgbm_core_{name}.parquet", index=False)
    print(name, out.shape, "| mean p_cal:", round(float(out["p_cal"].mean()), 4))

# %%
# %% R6. Save what serving needs: feature order and the frequency maps (and check the maps reproduce the training features)
freq_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
freq_maps = {}
for c in freq_cols:
    shares = core.loc[core["split"] == "train", c].value_counts(normalize=True)
    freq_maps[c] = {str(k): float(v) for k, v in shares.items()}
    rebuilt = core[c].astype(object).map(freq_maps[c]).fillna(0).astype(float).to_numpy()
    same = np.allclose(rebuilt, core[c + "_freq"].to_numpy(dtype="float64"), atol=1e-9)
    print(c, "frequency map reproduces the training feature:", same)
 
with open("/kaggle/working/lgbm_serving_bundle/frequency_maps.json", "w") as f:
    json.dump(freq_maps, f, indent=2)
with open("/kaggle/working/lgbm_serving_bundle/feature_order.json", "w") as f:
    json.dump({"features": serving_features,
               "categorical": [c for c in serving_features if c in cat_cols],
               "has_identity_rule": "1 if device_type is known, else 0"}, f, indent=2)
 
for folder in ["models_serving", "lgbm_serving_bundle", "predictions_serving"]:
    for name in sorted(os.listdir(f"/kaggle/working/{folder}")):
        print(f"{folder}/{name:35s} {os.path.getsize(f'/kaggle/working/{folder}/{name}') / 1e3:9.1f} KB")
