# %% 0. Setup
import os, glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve,
                             accuracy_score, confusion_matrix)

SEED = 42
INPUT_GLOB = os.environ.get("INPUT_GLOB", "/kaggle/input/**/pilot_train.parquet")
paths = glob.glob(INPUT_GLOB, recursive=True)
assert paths, "pilot_train.parquet not found: add the dedupe/split notebook as an input"
D = os.path.dirname(paths[0])
print("data dir:", D)
train = pd.read_parquet(f"{D}/pilot_train.parquet")
val = pd.read_parquet(f"{D}/pilot_val.parquet")
test = pd.read_parquet(f"{D}/pilot_test.parquet")
for n, d in [("train", train), ("val", val), ("test", test)]:
    print(n, len(d), d.label.value_counts().to_dict())

POS_SOURCES = sorted(train[train.label == 1].source.unique())
NEG_SOURCES = sorted(train[train.label == 0].source.unique())

def recall_at_fpr(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))

def pr_auc_at_prior(y, s, ratio=100):
    """PR-AUC after re-weighting negatives so positives:negatives = 1:ratio (deployment-like prior)."""
    y = np.asarray(y)
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    w = np.where(y == 1, 1.0, ratio * n_pos / n_neg)
    return float(average_precision_score(y, s, sample_weight=w))

def make_vec(kind):
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                           lowercase=False, max_features=300_000)

def fit_lr(train_df, kind="word", C=4.0):
    vec = make_vec(kind)
    X = vec.fit_transform(train_df["text"])
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)
    clf.fit(X, train_df["label"])
    return vec, clf

def score(vec, clf, df):
    return clf.predict_proba(vec.transform(df["text"]))[:, 1]

# %% 1. Source-predictor test: can the text alone reveal WHICH dataset it came from?
# Each source here has exactly one label, so source identity = label. High accuracy means the label
# can be recovered by recognizing dataset style. It cannot tell you the model is cheating, but it
# tells you how separable the sources are; the leave-one-source-out test below is the stronger check.
vec_s = make_vec("word")
Xs = vec_s.fit_transform(train["text"])
clf_s = LogisticRegression(C=4.0, max_iter=2000, random_state=SEED).fit(Xs, train["source"])
pred = clf_s.predict(vec_s.transform(val["text"]))
print("source accuracy on val:", round(accuracy_score(val["source"], pred), 3))
print(pd.crosstab(val["source"], pred, rownames=["true"], colnames=["pred"]))
# Read it: which sources get confused? Confusion BETWEEN a positive and a negative source is
# the interesting case (e.g. mosscap vs dolly), since those are the rows the label can't be read from style alone.

# %% 2. Baselines: word TF-IDF and char TF-IDF + logistic regression
results, models = [], {}
for kind in ["word", "char"]:
    vec, clf = fit_lr(train, kind)
    models[kind] = (vec, clf)
    sv, st = score(vec, clf, val), score(vec, clf, test)
    tau = float(np.quantile(sv[val.label.values == 0], 0.99))   # val FPR = 1%
    for name, d, s in [("val", val, sv), ("test", test, st)]:
        results.append({
            "model": kind, "split": name,
            "PR-AUC": round(average_precision_score(d.label, s), 4),
            "PR-AUC@1:100": round(pr_auc_at_prior(d.label, s), 4),
            "ROC-AUC": round(roc_auc_score(d.label, s), 4),
            "recall@FPR0.5%": round(recall_at_fpr(d.label, s, 0.005), 4),
            "recall@FPR1%": round(recall_at_fpr(d.label, s, 0.01), 4),
            "test FPR at val-tau(1%)": round(float((s[d.label.values == 0] >= tau).mean()), 4) if name == "test" else None,
        })
res = pd.DataFrame(results)
print(res.to_string(index=False))
# Near-perfect scores (>0.99) = the corpus teaches lexical shortcuts. Look at section 3 before celebrating.

# %% 3. Top n-grams (word model): what is it actually keying on?
vec, clf = models["word"]
names = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
print("TOP BENIGN n-grams:  ", list(names[order[:30]]))
print("\nTOP ATTACK n-grams:  ", list(names[order[::-1][:30]]))
# Red flags: words that are dataset artifacts rather than attack behaviour (e.g. 'gandalf', 'password',
# 'access', 'level', product words, email headers). Those mean the model recognizes the source.

# %% 4. Leave-one-source-out: train WITHOUT one source, test on it (word model)
# Positive source held out -> recall on its attacks. Negative source held out -> false-positive rate on it.
rows = []
vl = pd.concat([val, test])
for s in POS_SOURCES + NEG_SOURCES:
    tr = train[train.source != s]
    v, c = fit_lr(tr, "word")
    other_val_neg = score(v, c, val[(val.label == 0) & (val.source != s)])
    tau = float(np.quantile(other_val_neg, 0.99))               # FPR 1% on the sources still in play
    held = vl[vl.source == s]
    sc = score(v, c, held)
    lab = int(held.label.iloc[0])
    rows.append({"held_out": s, "label": lab, "n": len(held),
                 "metric": "recall" if lab == 1 else "FPR",
                 "value": round(float((sc >= tau).mean()), 3)})
loso = pd.DataFrame(rows)
print(loso.to_string(index=False))
# Healthy: recall stays reasonably high for held-out attack sources, FPR stays near 1% for held-out benign sources.
# A source whose recall collapses (e.g. < 0.3) is one the model only knew by memorizing its style.

# %% 5. Per-source breakdown at the val-chosen threshold (word model)
vec, clf = models["word"]
sv, st = score(vec, clf, val), score(vec, clf, test)
tau = float(np.quantile(sv[val.label.values == 0], 0.99))
t = test.assign(score=st, hit=(st >= tau))
print("test, threshold from val FPR=1%  (attack sources: hit = recall | benign sources: hit = false-positive rate)")
print(t.groupby(["label", "source"])["hit"].agg(["mean", "size"]).round(3))

# %% 6. Error peek: feed these into your label guide, not into synthetic data
t = t.assign(text_preview=t.text.str.slice(0, 240))
fn = t[(t.label == 1) & (~t.hit)].sort_values("score").head(15)
fp = t[(t.label == 0) & (t.hit)].sort_values("score", ascending=False).head(15)
print("=== missed attacks (false negatives) ===")
for _, r in fn.iterrows():
    print(f"[{r.source}] {r.score:.2f} {r.text_preview!r}\n")
print("=== benign flagged (false positives) ===")
for _, r in fp.iterrows():
    print(f"[{r.source}] {r.score:.2f} {r.text_preview!r}\n")

# %% 7. Save the results table for the report
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
res.to_csv(f"{OUT}/baseline_results.csv", index=False)
loso.to_csv(f"{OUT}/leave_one_source_out_tfidf.csv", index=False)
print("saved baseline_results.csv and leave_one_source_out_tfidf.csv")