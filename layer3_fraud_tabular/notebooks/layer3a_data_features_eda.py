# %% [markdown]
# # SENTINEL Layer 3A (P2): data, split, Core features
# Runs top to bottom on Kaggle (about 10-15 minutes, mostly the rolling features).
# Attach your private IEEE-CIS dataset via "Add Input" first.
# At the end: click Save Version -> "Save & Run All (Commit)" so the files are kept.

# %%
import glob
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

# %%
txn_path = glob.glob("/kaggle/input/**/train_transaction.csv", recursive=True)[0]
id_path = glob.glob("/kaggle/input/**/train_identity.csv", recursive=True)[0]

transactions = pd.read_csv(txn_path)
identity = pd.read_csv(id_path)
print("transactions:", transactions.shape, "| identity:", identity.shape)

# %%
df = transactions.merge(identity, on="TransactionID", how="left")
df["has_identity"] = df["TransactionID"].isin(identity["TransactionID"])
print(df.shape, "| share with identity:", round(df["has_identity"].mean(), 4))
print("fraud rate:", round(df["isFraud"].mean(), 4))

# %%
df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)

# %%
n = len(df)

def boundary(fraction):
    i = int(n * fraction)
    while df.loc[i, "TransactionDT"] == df.loc[i - 1, "TransactionDT"]:
        i -= 1          # never cut through rows with the same timestamp
    return i

b1, b2, b3 = boundary(0.70), boundary(0.80), boundary(0.90)

df["split"] = "test"
df.loc[:b3 - 1, "split"] = "val2"
df.loc[:b2 - 1, "split"] = "val1"
df.loc[:b1 - 1, "split"] = "train"

print(df.groupby("split").agg(rows=("TransactionID", "count"),
                              fraud_rate=("isFraud", "mean"),
                              first_time=("TransactionDT", "min"),
                              last_time=("TransactionDT", "max")))

# %%
checksums = {}
for name in ["train", "val1", "val2", "test"]:
    ids = sorted(df.loc[df["split"] == name, "TransactionID"])
    checksums[name] = hashlib.md5(",".join(str(i) for i in ids).encode()).hexdigest()
    print(name, (df["split"] == name).sum(), checksums[name])

with open("/kaggle/working/split_checksums.json", "w") as f:
    json.dump(checksums, f, indent=2)
df[["TransactionID", "split"]].to_parquet("/kaggle/working/splits.parquet", index=False)

# %%
df["card_key"] = df[["card1", "card2", "card3", "card5", "addr1", "addr2"]].astype(str).agg("_".join, axis=1)

# card without address: used only for "distinct addresses per card"
df["card_base_key"] = df[["card1", "card2", "card3", "card5"]].astype(str).agg("_".join, axis=1)

df["addr_key"] = df["addr1"].astype(str) + "_" + df["addr2"].astype(str)
df.loc[df["addr1"].isna() | df["addr2"].isna(), "addr_key"] = "missing"

df["device_key"] = df["DeviceInfo"].astype(str) + "_" + df["id_30"].astype(str) + "_" + df["id_31"].astype(str)
df.loc[df[["DeviceInfo", "id_30", "id_31"]].isna().all(axis=1), "device_key"] = "missing"

print("missing addr:", round(df["addr_key"].eq("missing").mean(), 3),
      "| missing device:", round(df["device_key"].eq("missing").mean(), 3))

# %%
core = pd.DataFrame({"TransactionID": df["TransactionID"],
                     "split": df["split"],
                     "isFraud": df["isFraud"]})

core["log_amount"] = np.log1p(df["TransactionAmt"])
core["cents"] = (df["TransactionAmt"] % 1).round(2)
core["is_round_amount"] = (df["TransactionAmt"] % 1 == 0).astype(int)
core["hour"] = (df["TransactionDT"] // 3600) % 24
core["dayofweek"] = (df["TransactionDT"] // 86400) % 7
core["addr1"] = df["addr1"]
core["addr2"] = df["addr2"]
core["dist1"] = df["dist1"]
core["has_identity"] = df["has_identity"].astype(int)

cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
for c in cat_cols:
    core[c] = df[c]
core["same_email_domain"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

# %%
def strip_version(series):
    s = series.str.lower()
    s = s.str.replace(r"[\d\._]+", "", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()

core["browser"] = strip_version(df["id_31"])
core["os"] = strip_version(df["id_30"])

# %%
for c in cat_cols + ["browser", "os"]:
    freq = core.loc[core["split"] == "train", c].value_counts(normalize=True)
    core[c + "_freq"] = core[c].map(freq).fillna(0)

# %%
def make_history_features(data):
    t = data[["card_key", "TransactionDT", "TransactionAmt"]].copy()
    t["time"] = pd.to_datetime(t["TransactionDT"], unit="s")

    t_sorted = t.sort_values(["card_key", "time"], kind="stable")
    grouped = t_sorted.groupby("card_key")
    out = pd.DataFrame(index=t.index)

    def look_back(window, column, stat):
        r = grouped.rolling(window, on="time", closed="left")[column]   # closed="left": current row excluded
        values = getattr(r, stat)().to_numpy()
        assert len(values) == len(t_sorted)
        return pd.Series(values, index=t_sorted.index).reindex(t.index)

    out["card_cnt_1h"] = look_back("1h", "TransactionAmt", "count")
    out["card_cnt_24h"] = look_back("24h", "TransactionAmt", "count")
    out["card_cnt_7d"] = look_back("7D", "TransactionAmt", "count")
    out["card_cnt_all"] = look_back("400D", "TransactionAmt", "count")

    # empty window -> pandas gives NaN, but the true count is 0
    for c in ["card_cnt_1h", "card_cnt_24h", "card_cnt_7d", "card_cnt_all"]:
        out[c] = out[c].fillna(0)
    out["card_first_seen"] = (out["card_cnt_all"] == 0).astype(int)

    out["card_amt_mean"] = look_back("400D", "TransactionAmt", "mean")
    out["card_amt_std"] = look_back("400D", "TransactionAmt", "std")

    # z-score only with >= 3 earlier transactions, clipped to [-10, 10]
    z = (t["TransactionAmt"] - out["card_amt_mean"]) / out["card_amt_std"].replace(0, np.nan)
    out["card_amt_z"] = z.where(out["card_cnt_all"] >= 3).clip(-10, 10)

    last_time = look_back("400D", "TransactionDT", "max")
    out["card_secs_since_last"] = t["TransactionDT"] - last_time
    return out

# %%
def make_entity_features(data, key, prefix):
    t = data[[key, "TransactionDT", "TransactionAmt"]].copy()
    t["time"] = pd.to_datetime(t["TransactionDT"], unit="s")
    t_sorted = t.sort_values([key, "time"], kind="stable")
    grouped = t_sorted.groupby(key)
    out = pd.DataFrame(index=t.index)

    def look_back(window, column, stat):
        r = grouped.rolling(window, on="time", closed="left")[column]
        values = getattr(r, stat)().to_numpy()
        assert len(values) == len(t_sorted)
        return pd.Series(values, index=t_sorted.index).reindex(t.index)

    for name, window in [("cnt_1h", "1h"), ("cnt_24h", "24h"), ("cnt_7d", "7D")]:
        out[f"{prefix}_{name}"] = look_back(window, "TransactionAmt", "count").fillna(0)
    out[f"{prefix}_amt_mean"] = look_back("400D", "TransactionAmt", "mean")
    out[f"{prefix}_secs_since_last"] = t["TransactionDT"] - look_back("400D", "TransactionDT", "max")

    out.loc[t[key] == "missing", :] = np.nan    # unknown entity: undefined, not zero
    return out

# %%
def distinct_before(data, key, other):
    t = data[[key, other, "TransactionDT"]].copy()
    t[other] = t[other].fillna("missing").astype(str)
    t = t.sort_values([key, "TransactionDT"], kind="stable")

    is_new = (~t.duplicated(subset=[key, other])).astype(int)
    running = is_new.groupby(t[key]).cumsum() - is_new
    strict = running.groupby([t[key], t["TransactionDT"]]).transform("min")   # same-second rows don't see each other
    return strict.reindex(data.index)

# %%
hist = make_history_features(df)
addr_feats = make_entity_features(df, "addr_key", "addr")
dev_feats = make_entity_features(df, "device_key", "dev")

extra = pd.DataFrame(index=df.index)
extra["card_distinct_emails"] = distinct_before(df, "card_key", "P_emaildomain")
extra["card_distinct_addr"] = distinct_before(df, "card_base_key", "addr_key")
extra["dev_distinct_cards"] = distinct_before(df, "device_key", "card_key")
extra.loc[df["device_key"] == "missing", "dev_distinct_cards"] = np.nan

core = pd.concat([core, hist, addr_feats, dev_feats, extra], axis=1)
print(core.shape)

# %%
core.to_parquet("/kaggle/working/core_features.parquet", index=False)

# %%
cols = ["card_key", "card_base_key", "addr_key", "device_key", "P_emaildomain", "TransactionDT", "TransactionAmt"]
sub = df[cols].iloc[:150000].copy()
cutoff = sub["TransactionDT"].iloc[75000]

def build_all(d):
    return pd.concat([
        make_history_features(d),
        make_entity_features(d, "addr_key", "addr"),
        make_entity_features(d, "device_key", "dev"),
        distinct_before(d, "card_key", "P_emaildomain").rename("card_distinct_emails"),
        distinct_before(d, "card_base_key", "addr_key").rename("card_distinct_addr"),
        distinct_before(d, "device_key", "card_key").rename("dev_distinct_cards"),
    ], axis=1)

before = build_all(sub)

sub2 = sub.copy()
future = sub2["TransactionDT"] >= cutoff
rng = np.random.default_rng(0)
for c in ["TransactionAmt", "card_key", "card_base_key", "addr_key", "device_key", "P_emaildomain"]:
    sub2.loc[future, c] = rng.permutation(sub2.loc[future, c].values)

after = build_all(sub2)
pd.testing.assert_frame_equal(before[~future], after[~future])
print("PASSED: features of earlier rows did not change when the future was scrambled")

# %%
spec = {}

def add(name, definition, window, sources, serving=True):
    spec[name] = {"definition": definition, "window": window,
                  "source_columns": sources, "serving_available": serving}

add("log_amount", "log(1 + TransactionAmt)", "row", ["TransactionAmt"])
add("cents", "decimal part of the amount", "row", ["TransactionAmt"])
add("is_round_amount", "1 if the amount has no cents", "row", ["TransactionAmt"])
add("hour", "relative hour of day from TransactionDT (reference start unknown)", "row", ["TransactionDT"])
add("dayofweek", "relative day of week from TransactionDT", "row", ["TransactionDT"])
add("addr1", "billing address code 1", "row", ["addr1"])
add("addr2", "billing address code 2", "row", ["addr2"])
add("dist1", "distance feature as provided", "row", ["dist1"])
add("has_identity", "1 if the transaction has identity info", "row", ["identity table"])
for c in cat_cols:
    add(c, f"raw categorical {c}", "row", [c])
add("same_email_domain", "1 if payer and recipient email domains match", "row", ["P_emaildomain", "R_emaildomain"])
add("browser", "browser family from id_31, version numbers removed", "row", ["id_31"], "pending")
add("os", "OS family from id_30, version numbers removed", "row", ["id_30"], "pending")

for c in cat_cols:
    add(f"{c}_freq", f"share of train rows with this {c} (0 if unseen)", "fit on train", [c])
add("browser_freq", "share of train rows with this browser family", "fit on train", ["id_31"], "pending")
add("os_freq", "share of train rows with this OS family", "fit on train", ["id_30"], "pending")

for ent, key, serving in [("card", "card_key", True), ("addr", "addr_key", True), ("dev", "device_key", "pending")]:
    for name, win in [("cnt_1h", "last 1h"), ("cnt_24h", "last 24h"), ("cnt_7d", "last 7d")]:
        add(f"{ent}_{name}", f"number of earlier transactions of this {ent}", win, [key, "TransactionDT"], serving)
    add(f"{ent}_amt_mean", f"mean amount of earlier transactions of this {ent}", "all history", [key, "TransactionAmt"], serving)
    add(f"{ent}_secs_since_last", f"seconds since this {ent}'s previous transaction", "all history", [key, "TransactionDT"], serving)

add("card_cnt_all", "number of all earlier transactions of this card", "all history", ["card_key"])
add("card_first_seen", "1 if the card has no earlier transaction", "all history", ["card_key"])
add("card_amt_std", "std of amounts of earlier transactions of this card", "all history", ["card_key", "TransactionAmt"])
add("card_amt_z", "(amount - card mean) / card std; needs >= 3 earlier rows; clipped to [-10, 10]", "all history", ["card_key", "TransactionAmt"])
add("card_distinct_emails", "distinct payer email domains seen earlier with this card", "all history", ["card_key", "P_emaildomain"])
add("card_distinct_addr", "distinct addresses seen earlier with this card (card key without address)", "all history", ["card_base_key", "addr_key"])
add("dev_distinct_cards", "distinct cards seen earlier with this device key", "all history", ["device_key", "card_key"], "pending")

not_features = ["TransactionID", "split", "isFraud"]
print("in spec but not in core:", [c for c in spec if c not in core.columns])
print("in core but not in spec:", [c for c in core.columns if c not in spec and c not in not_features])

for name in spec:
    spec[name]["dtype"] = str(core[name].dtype)

with open("/kaggle/working/feature_spec.yaml", "w") as f:
    yaml.safe_dump(spec, f, sort_keys=False)

# %%
import os
for f in sorted(os.listdir("/kaggle/working")):
    print(f"{f:30s} {os.path.getsize('/kaggle/working/' + f) / 1e6:8.1f} MB")

# %%
# %% [markdown]
# # SENTINEL Layer 3A (P2): EDA that saves its outputs
# Paste these cells at the END of the l3a_data_and_features.py notebook
# (after the feature-building cells), because they use `df` and `core`.
# Every plot and table is saved to /kaggle/working/reports/eda_l3a/

# %%
# %% 1. Setup (train split only for EDA, so we never look at val/test when deciding things)
import os
 
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
 
OUT = "/kaggle/working/reports/eda_l3a"
os.makedirs(OUT, exist_ok=True)
 
train = df[df["split"] == "train"].copy()
notes = []   # key numbers collected for the report
print(train.shape)

# %%
# %% 2. Overall fraud rate and has_identity
notes.append(f"Fraud rate overall: {df['isFraud'].mean():.4f} ({int(df['isFraud'].sum())} of {len(df)})")
notes.append(f"Rows with identity info: {df['has_identity'].mean():.4f}")
 
by_split = df.groupby("split").agg(rows=("TransactionID", "count"),
                                   fraud_rate=("isFraud", "mean"),
                                   first_time=("TransactionDT", "min"),
                                   last_time=("TransactionDT", "max"))
by_split.to_csv(f"{OUT}/split_summary.csv")
print(by_split)

# %%
# %% 3. Fraud rate per week
train["week"] = train["TransactionDT"] // (86400 * 7)
weekly = train.groupby("week")["isFraud"].mean()
 
plt.figure(figsize=(7, 4))
weekly.plot(marker="o", title="Fraud rate per week (train)")
plt.ylabel("fraud rate")
plt.tight_layout()
plt.savefig(f"{OUT}/fraud_rate_per_week.png", dpi=120)
plt.show()
weekly.to_csv(f"{OUT}/fraud_rate_per_week.csv")

# %%
# %% 4. Missing values by column group
groups = {
    "C": [c for c in df.columns if c.startswith("C") and c[1:].isdigit()],
    "D": [c for c in df.columns if c.startswith("D") and c[1:].isdigit()],
    "M": [c for c in df.columns if c.startswith("M") and c[1:].isdigit()],
    "V": [c for c in df.columns if c.startswith("V") and c[1:].isdigit()],
    "id": [c for c in df.columns if c.startswith("id_")],
}
rows = [{"group": g, "n_columns": len(cols), "avg_missing": train[cols].isna().mean().mean()}
        for g, cols in groups.items()]
missing_groups = pd.DataFrame(rows)
missing_groups.to_csv(f"{OUT}/missing_by_group.csv", index=False)
print(missing_groups)
 
# V columns that share the same missing percentage (candidates to group later)
v_missing = train[groups["V"]].isna().mean().round(3)
v_missing.value_counts().sort_index().to_csv(f"{OUT}/v_columns_by_missing_rate.csv")

# %%
# %% 5. Transaction amount per class (log scale)
plt.figure(figsize=(7, 4))
np.log1p(train.loc[train["isFraud"] == 0, "TransactionAmt"]).hist(bins=50, alpha=0.6, label="not fraud", density=True)
np.log1p(train.loc[train["isFraud"] == 1, "TransactionAmt"]).hist(bins=50, alpha=0.6, label="fraud", density=True)
plt.legend()
plt.title("log(1 + amount) by class (train)")
plt.tight_layout()
plt.savefig(f"{OUT}/amount_by_class.png", dpi=120)
plt.show()
 
cents = train.assign(is_round=(train["TransactionAmt"] % 1 == 0)).groupby("is_round")["isFraud"].agg(["count", "mean"])
cents.to_csv(f"{OUT}/round_amount_fraud_rate.csv")
print(cents)

# %%
# %% 6. Fraud rate by hour and by day of week (reference start is unknown, so these are relative)
train["hour"] = (train["TransactionDT"] // 3600) % 24
train["dayofweek"] = (train["TransactionDT"] // 86400) % 7
 
fig, axes = plt.subplots(2, 1, figsize=(7, 7))
train.groupby("hour")["isFraud"].mean().plot(kind="bar", ax=axes[0], title="Fraud rate by hour (relative)")
train.groupby("dayofweek")["isFraud"].mean().plot(kind="bar", ax=axes[1], title="Fraud rate by day of week (relative)")
plt.tight_layout()
plt.savefig(f"{OUT}/fraud_by_hour_and_day.png", dpi=120)
plt.show()

# %%
# %% 7. Fraud rate by category (count + rate, saved to one CSV)
tables = []
for col in ["ProductCD", "card4", "card6", "DeviceType", "has_identity"]:
    t = train.groupby(col)["isFraud"].agg(["count", "mean"]).reset_index()
    t.insert(0, "column", col)
    t = t.rename(columns={col: "value"})
    tables.append(t)
 
top_emails = train["P_emaildomain"].value_counts().head(10).index
t = (train[train["P_emaildomain"].isin(top_emails)]
     .groupby("P_emaildomain")["isFraud"].agg(["count", "mean"]).reset_index())
t.insert(0, "column", "P_emaildomain")
tables.append(t.rename(columns={"P_emaildomain": "value"}))
 
category_table = pd.concat(tables, ignore_index=True)
category_table.to_csv(f"{OUT}/fraud_rate_by_category.csv", index=False)
print(category_table.round(4))

# %%
# %% 8. Warm rate: share of test rows whose card key was seen in train
train_cards = set(df.loc[df["split"] == "train", "card_key"])
test_part = df[df["split"] == "test"]
warm_rate = test_part["card_key"].isin(train_cards).mean()
notes.append(f"Warm rate (test rows whose card_key was seen in train): {warm_rate:.4f}")
print("Warm rate:", round(warm_rate, 4))

# %%
# %% 9. Adversarial validation on ALL columns (can a model tell train from test?)
def adversarial(frame, drop_cols):
    adv = frame[frame["split"].isin(["train", "test"])]
    y = (adv["split"] == "test").astype(int)
    X = adv.drop(columns=drop_cols).copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        elif X[col].dtype == object:
            X[col] = X[col].astype("category")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    m.fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, m.predict_proba(X_va)[:, 1])
    imp = pd.Series(m.booster_.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return auc, imp
 
raw_drop = ["TransactionID", "TransactionDT", "isFraud", "split",
            "card_key", "card_base_key", "addr_key", "device_key"]
auc_all, imp_all = adversarial(df, raw_drop)
imp_all.head(30).to_csv(f"{OUT}/adversarial_all_columns_top30.csv")
notes.append(f"Adversarial validation AUC, all raw columns: {auc_all:.3f}")
print("Adversarial AUC (all columns):", round(auc_all, 3))
print(imp_all.head(15))

# %%
# %% 10. Adversarial validation on the Core features only
auc_core, imp_core = adversarial(core, ["TransactionID", "isFraud", "split"])
imp_core.head(30).to_csv(f"{OUT}/adversarial_core_top30.csv")
notes.append(f"Adversarial validation AUC, Core features: {auc_core:.3f}")
print("Adversarial AUC (Core features):", round(auc_core, 3))
print(imp_core.head(10))

# %%
# %% 11. Browser version drift check (why version numbers were removed)
top_train = df.loc[df["split"] == "train", "id_31"].value_counts().head(8)
top_test = df.loc[df["split"] == "test", "id_31"].value_counts().head(8)
pd.concat([top_train.rename("train"), top_test.rename("test")], axis=1).to_csv(f"{OUT}/id_31_top_values_train_vs_test.csv")
print(top_train, "\n", top_test)

# %%
# %% 12. Leakage audit: correlation of each numeric column with isFraud (train only)
numeric = train.select_dtypes("number").drop(columns=["isFraud"])
numeric = numeric.loc[:, numeric.std() > 0]          # skip constant columns (avoids divide-by-zero)
corr = numeric.corrwith(train["isFraud"]).abs().sort_values(ascending=False)
corr.head(30).to_csv(f"{OUT}/leakage_audit_top_correlations.csv")
notes.append(f"Highest |correlation| with isFraud: {corr.index[0]} = {corr.iloc[0]:.3f}")
print(corr.head(10))

# %%
# %% 13. Write a short summary file and list everything that was saved
with open(f"{OUT}/eda_key_numbers.txt", "w") as f:
    f.write("\n".join(notes))
print("\n".join(notes))
 
for name in sorted(os.listdir(OUT)):
    print(f"{name:50s} {os.path.getsize(f'{OUT}/{name}') / 1e3:8.1f} KB")
