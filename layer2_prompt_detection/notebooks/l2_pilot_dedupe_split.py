#0. Setup
import subprocess, sys, os, glob, json, re, unicodedata
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasketch"], check=False)

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

SEED = 42
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
INPUT_GLOB = os.environ.get("INPUT_GLOB", "/kaggle/input/**/pilot_raw.parquet")

DROP_SOURCES = {"mosscap"}   # audit: about a third of sampled Mosscap rows were not real attacks
AUDIT_N = 100            # rows per source to read by hand
MIN_TOKENS = 5           # handbook: minimum 5 tokens
JACCARD = 0.8            # handbook: near-duplicate threshold
NUM_PERM = 128
SHINGLE_K = 3            # word 3-grams
MAX_PER_CLUSTER = 2      # keep at most this many rows per near-duplicate cluster

TRAIN_POS_TARGET = 3200  # rough size of the balanced train positives (about 80% of a 4k pilot)
POS_CAP_FRAC = 0.30      # default: no source above 30% of train positives
SOURCE_CAP_FRAC = {}     # per-source caps override, e.g. {"gandalf": 0.20}
HARD_SOURCES = {"dolly", "oasst1"}
HARD_FRAC = 0.25         # at least this share of each length bin's negatives are hard negatives
# Benign text that USES attack vocabulary ("please ignore my last email", "follow the instructions", password notices).
# The baseline showed attack words are almost absent from the benign side, so the model gets no examples of benign use.
KW_RE = r"ignore|disregard|instruction|password|previous|secret|override|system prompt|confidential"
KW_FRAC = 0.5            # up to this share of each length bin's negatives, if that many exist

rng = np.random.RandomState(SEED)

#1. Load
paths = glob.glob(INPUT_GLOB, recursive=True)
assert paths, "pilot_raw.parquet not found: check Add Input and the printed paths"
print("loading", paths[0])
df = pd.read_parquet(paths[0])
df = df[~df["source"].isin(DROP_SOURCES)].reset_index(drop=True)
print(df.shape)
print(df.groupby(["label", "source"]).size())

#2. Audit sample (read these by hand BEFORE trusting any source)
audit = pd.concat(
    [g.sample(min(AUDIT_N, len(g)), random_state=SEED) for _, g in df.groupby("source")]
)
audit = audit[["source", "label", "text"]].assign(verdict="", note="")
audit.to_csv(f"{OUT}/audit_sample.csv", index=False)
print("saved audit_sample.csv:", len(audit), "rows")

def show(source, start=0, n=15, width=400):
    """Print rows for manual reading. repr() makes odd characters visible."""
    sub = audit[audit.source == source].iloc[start:start + n]
    for i, (_, r) in enumerate(sub.iterrows(), start):
        tag = "ATTACK" if r.label == 1 else "BENIGN"
        print(f"[{i}] ({tag}) {r.text[:width]!r}\n")

# Example: show("gandalf"); show("tensor_trust", start=15)
# For each row decide: correct / wrong / ambiguous. Wrong-rate per source = label noise estimate.

#3. Normalize (for dedupe only) and basic filters
def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"\s+", " ", t).strip()

# Mask source-identifying tokens (handbook: mask emails/phones/IDs; here also the company name and
# Enron's quoted-printable artifacts, which the baseline used as a source shortcut).
def mask(t, src):
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]", t)
    if src == "enron":
        t = t.replace("=\n", "").replace("=20", " ")
        t = re.sub(r"(?i)enron", "[ORG]", t)
    return t
df["text"] = [mask(t, s_) for t, s_ in zip(df["text"], df["source"])]

# Tag glitch-style attacks (gibberish / repeated symbols) so they can be reported as their own slice.
import zlib
def glitch(t):
    t = t.strip()
    alpha = sum(c.isalpha() for c in t) / max(1, len(t))
    comp = len(zlib.compress(t.encode("utf8"))) / max(1, len(t.encode("utf8")))
    return (alpha < 0.3) or (len(t) >= 100 and comp < 0.15) or (" " not in t and 15 < len(t) < 60 and alpha < 0.8)
df["subtype"] = np.where((df["label"] == 1) & df["text"].map(glitch), "glitch", "")
print("glitch share of positives by source:")
print(df[df.label == 1].groupby("source")["subtype"].apply(lambda x: round((x == "glitch").mean(), 3)))

n0 = len(df)
df["norm"] = df["text"].map(norm)
df["n_tokens"] = df["norm"].str.split().str.len()
df = df[df["n_tokens"] >= MIN_TOKENS]
n1 = len(df)
df = df.drop_duplicates("norm").reset_index(drop=True)
n2 = len(df)
print(f"start {n0} -> after min-length {n1} -> after exact dedupe {n2}")

#4. MinHash-LSH near-duplicate clusters
def signature(t):
    t = t[:1500]            # character 5-grams: works for non-English and unspaced obfuscated text
    m = MinHash(num_perm=NUM_PERM)
    for i in range(max(1, len(t) - 5 + 1)):
        m.update(t[i:i + 5].encode("utf8"))
    return m

sigs = [signature(t) for t in df["norm"]]
lsh = MinHashLSH(threshold=JACCARD, num_perm=NUM_PERM)
for i, m in enumerate(sigs):
    lsh.insert(str(i), m)

parent = list(range(len(df)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for i, m in enumerate(sigs):
    for j in lsh.query(m):
        union(i, int(j))

df["cluster_id"] = [find(i) for i in range(len(df))]
sizes = df.groupby("cluster_id").size()
print("clusters:", len(sizes), "| rows:", len(df))
print("cluster size quantiles:", sizes.quantile([0.5, 0.9, 0.99, 1.0]).to_dict())

# 5. Cluster report, drop label-conflict clusters, cap cluster size
# Largest clusters: look at them. This is where "thousands of variants of one trick" shows up.
for cid in sizes.sort_values(ascending=False).index[:5]:
    g = df[df.cluster_id == cid]
    print(f"cluster size {len(g)} | sources {g.source.value_counts().to_dict()}")
    print("   e.g.", repr(g.text.iloc[0][:200]), "\n")

# Clusters containing BOTH labels = conflicting labels (or an attack hidden in benign text). Drop them.
nlab = df.groupby("cluster_id")["label"].nunique()
mixed = nlab[nlab > 1].index
print("mixed-label clusters:", len(mixed), "| rows dropped:", df.cluster_id.isin(mixed).sum())
mixed_rows = df[df.cluster_id.isin(mixed)]
print(mixed_rows.groupby(["source", "label"]).size())  # inspect a few by hand
df = df[~df.cluster_id.isin(mixed)]

rows_before = df.groupby("source").size().rename("rows_before")
df = (df.sample(frac=1, random_state=SEED)
        .groupby("cluster_id", group_keys=False).head(MAX_PER_CLUSTER)
        .reset_index(drop=True))
rows_after = df.groupby("source").size().rename("rows_after")
report = pd.concat([rows_before, rows_after], axis=1)
report["pct_removed"] = (100 * (1 - report.rows_after / report.rows_before)).round(1)
print(report)
report.to_csv(f"{OUT}/dedupe_report.csv")

# 6. Cluster-aware split 80/10/10 (clusters never straddle a boundary)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, rest_idx = next(gss.split(df, groups=df["cluster_id"]))
train, rest = df.iloc[tr_idx], df.iloc[rest_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
v_idx, t_idx = next(gss2.split(rest, groups=rest["cluster_id"]))
val, test = rest.iloc[v_idx], rest.iloc[t_idx]

for a, b, name in [(train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")]:
    assert set(a.cluster_id).isdisjoint(set(b.cluster_id)), f"cluster leak {name}"
print("no cluster straddles a split")
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name, len(d), d.label.value_counts().to_dict())

# 7. Balance the TRAIN split only: source cap on positives, length-matched negatives, hard-negative floor
def cap_for(src):
    return int(SOURCE_CAP_FRAC.get(src, POS_CAP_FRAC) * TRAIN_POS_TARGET)

pos = pd.concat(
    [g.sample(min(cap_for(src), len(g)), random_state=SEED)
     for src, g in train[train.label == 1].groupby("source")]
)
print("train positives by source (caps:", {x: cap_for(x) for x in train[train.label == 1].source.unique()}, "):")
print(pos.source.value_counts())

inner = np.unique(np.quantile(pos["n_tokens"], np.linspace(0.1, 0.9, 9)))
pos_bin = np.digitize(pos["n_tokens"], inner)
neg_pool = train[train.label == 0].copy()
neg_pool["bin"] = np.digitize(neg_pool["n_tokens"], inner)

def take(avail, want):
    """Pick negatives for one length bin: keyword-bearing benign text first, then hard negatives, then random."""
    kw = avail[avail.text.str.lower().str.contains(KW_RE, regex=True)]
    n_kw = min(len(kw), int(round(KW_FRAC * want)))
    k = kw.sample(n_kw, random_state=SEED) if n_kw else kw.iloc[:0]
    avail = avail.drop(k.index)
    hard = avail[avail.source.isin(HARD_SOURCES)]
    n_hard = min(len(hard), max(0, int(round(HARD_FRAC * want)) - len(k)))
    h = hard.sample(n_hard, random_state=SEED) if n_hard else hard.iloc[:0]
    rest_ = avail.drop(h.index)
    r = rest_.sample(min(want - len(k) - n_hard, len(rest_)), random_state=SEED)
    return pd.concat([k, h, r])

pos = pos.assign(bin=pos_bin)
pos_keep, negs, shortfall = [], [], 0
for b in range(len(inner) + 1):
    pos_b = pos[pos.bin == b]
    got = take(neg_pool[neg_pool.bin == b], len(pos_b))
    shortfall += len(pos_b) - len(got)
    # if a length bin lacks negatives, drop the extra positives so classes stay balanced AND length-matched
    pos_keep.append(pos_b.sample(len(got), random_state=SEED) if len(got) < len(pos_b) else pos_b)
    negs.append(got)
pos = pd.concat(pos_keep).drop(columns="bin")
neg = pd.concat(negs).drop(columns="bin")
print(f"balanced: {len(pos)} positives, {len(neg)} negatives | positives dropped for lack of length-matched negatives: {shortfall}")
print("positives by source after matching:")
print(pos.source.value_counts())
print("negatives by source:")
print(neg.source.value_counts())

train_bal = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# 8. Quick sanity checks
def len_auc(d):
    a = roc_auc_score(d["label"], d["n_tokens"])
    return max(a, 1 - a)   # direction does not matter, only how separable

print("length-only AUROC  train_bal:", round(len_auc(train_bal), 3),
      "| val:", round(len_auc(val), 3))
print("hard-negative share of train negatives:",
      round(neg.source.isin(HARD_SOURCES).mean(), 3))
print("train_bal label counts:", train_bal.label.value_counts().to_dict())
print("train negatives containing attack vocabulary:", int(neg.text.str.lower().str.contains(KW_RE, regex=True).sum()), "of", len(neg))
print("glitch share of train positives:", round((train_bal[train_bal.label == 1].subtype == "glitch").mean(), 3))
# Healthy: length AUROC near 0.5-0.65 on train_bal. If it is > 0.8, matching did not work well
# (check shortfall above) and length is still a shortcut.

# 9. Save
cols = ["text", "label", "source", "subtype", "cluster_id", "n_tokens"]
train_bal[cols].to_parquet(f"{OUT}/pilot_train.parquet", index=False)
val[cols].to_parquet(f"{OUT}/pilot_val.parquet", index=False)
test[cols].to_parquet(f"{OUT}/pilot_test.parquet", index=False)

# Benign train-split rows that were NOT used for training: clean extra negatives for calibration/thresholds.
# Drop any row whose cluster also appears in the balanced train set (near-duplicate siblings would look too easy).
extra = train[(train.label == 0) & (~train.index.isin(neg.index))]
extra = extra[~extra.cluster_id.isin(train_bal.cluster_id)]
extra[cols].to_parquet(f"{OUT}/pilot_extra_neg.parquet", index=False)
print("saved", len(extra), "extra benign rows (calibration only):", extra.source.value_counts().to_dict())

split_report = {
    "seed": SEED, "jaccard": JACCARD, "num_perm": NUM_PERM, "shingle_k": SHINGLE_K,
    "max_per_cluster": MAX_PER_CLUSTER, "min_tokens": MIN_TOKENS,
    "rows": {"raw": n0, "after_min_len": n1, "after_exact_dedupe": n2, "after_cluster_cap": len(df)},
    "train_balanced": {"n": len(train_bal), "by_source": {f"{l}/{src}": int(c) for (l, src), c in train_bal.groupby(["label", "source"]).size().items()}},
    "val": {"n": len(val), "label_counts": {str(k): int(v) for k, v in val.label.value_counts().items()}},
    "test": {"n": len(test), "label_counts": {str(k): int(v) for k, v in test.label.value_counts().items()}},
}
json.dump(split_report, open(f"{OUT}/split_report.json", "w"), indent=2, default=str)
print("saved train/val/test parquet + reports to", OUT)

