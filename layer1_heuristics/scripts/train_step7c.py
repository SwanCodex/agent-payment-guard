"""Step 7c: weighted training on ALL train rows. LR (char n-grams) vs LightGBM (features) vs average.
Model choices are made on val_natural PR-AUC. Test split untouched."""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_step6 as S                                              # noqa: E402
import train_step7b as B                                             # noqa: E402
from sentinel_l1.features.builder import build_base, MODEL_FEATURES  # noqa: E402
from sentinel_l1.features.lexicon import Lexicon, load_brands        # noqa: E402
from sentinel_l1.features.tld_risk import TldRisk, oof_tld_risk      # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
CACHE = S.MODELS / "feat_cache_full.pkl"       # delete if the feature code changes
HOST = ["host_len", "host_hyphens", "host_digits", "host_labels", "max_label_len",
        "n_subdomains", "host_digit_ratio", "entropy_host", "vowel_ratio_host"]
PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=200,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
              random_state=S.SEED, verbose=-1)


def get_data(ext_n, cap):
    if CACHE.exists():
        print("using cached features:", CACHE)
        return pd.read_pickle(CACHE)
    lex = Lexicon(load_brands(S.MODELS / "brands.json"))
    tr = pd.read_csv(S.DATA / "train.csv", low_memory=False)
    sets = {"val_matched": pd.read_csv(S.DATA / "val_matched.csv", low_memory=False),
            "val_natural": pd.read_csv(S.DATA / "val.csv", low_memory=False)}
    ext = pd.read_csv(S.DATA / "external_phiusiil.csv", low_memory=False)
    if len(ext) > ext_n:
        ext = ext.sample(ext_n, random_state=S.SEED)
    sets["external_phiusiil"] = ext
    Xtr, tlds = build_base(tr.url.tolist(), lex)
    ytr = tr.label.values
    Xtr["tld_risk"] = oof_tld_risk(tlds, ytr, tr.reg_domain.values)
    tld = TldRisk().fit(tlds, ytr)
    X, Y, urls = {}, {}, {"train": tr.url.tolist()}
    for name, d in sets.items():
        Xs, t = build_base(d.url.tolist(), lex)
        Xs["tld_risk"] = tld.transform(t)
        X[name], Y[name], urls[name] = Xs[MODEL_FEATURES], d.label.values, d.url.tolist()
    w = B.balanced_weights(np.array([B.depth_bin(u) for u in tr.url]), ytr, cap)
    data = {"Xtr": Xtr[MODEL_FEATURES], "ytr": ytr, "w": w, "X": X, "Y": Y, "urls": urls}
    pd.to_pickle(data, CACHE)
    return data


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(a):
    import lightgbm as lgb
    d = get_data(a.ext_n, a.cap)
    Xtr, ytr, w, X, Y, urls = d["Xtr"], d["ytr"], d["w"], d["X"], d["Y"], d["urls"]
    names = list(X)
    P = {}                                               # model name -> {set: probabilities}

    # ---- LR on char n-grams, weighted
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 20,
                           alternate_sign=False, norm=None, lowercase=False)
    tf = TfidfTransformer(sublinear_tf=True)
    Htr = tf.fit_transform(hv.transform([S.strip_scheme(u) for u in urls["train"]]))
    H = {k: tf.transform(hv.transform([S.strip_scheme(u) for u in urls[k]])) for k in names}
    best = None
    for C in (10.0, 30.0, 100.0):
        lr = LogisticRegression(C=C, max_iter=3000, solver="liblinear").fit(Htr, ytr, sample_weight=w)
        s = average_precision_score(Y["val_natural"], lr.predict_proba(H["val_natural"])[:, 1])
        print(f"  LR C={C}: val_natural PR-AUC {s:.4f}")
        if best is None or s > best[0]:
            best = (s, C, lr)
    print(f"LR chosen C={best[1]}")
    P["LR n-grams"] = {k: best[2].predict_proba(H[k])[:, 1] for k in names}

    # ---- LightGBM on engineered features, weighted
    for label, feats in (("LightGBM all feats", MODEL_FEATURES),
                         ("LightGBM no host feats", [f for f in MODEL_FEATURES if f not in HOST])):
        clf = lgb.LGBMClassifier(**PARAMS)
        clf.fit(Xtr[feats], ytr, sample_weight=w,
                eval_set=[(X["val_matched"][feats], Y["val_matched"])], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        print(f"{label}: best iteration {clf.best_iteration_}")
        P[label] = {k: clf.predict_proba(X[k][feats])[:, 1] for k in names}

    # ---- simple averages (logit space)
    for g in ("LightGBM all feats", "LightGBM no host feats"):
        P[f"avg LR + {g}"] = {k: 1 / (1 + np.exp(-(0.5 * logit(P["LR n-grams"][k]) + 0.5 * logit(P[g][k]))))
                              for k in names}

    rows = {}
    for model, per in P.items():
        for k in names:
            m = S.metrics(Y[k], per[k])
            rows[(model, k)] = {"ROC": m["ROC-AUC"], "PR": m["PR-AUC"],
                                "recall@0.5%FPR": m["recall@FPR0.5%"], "recall@1%FPR": m["recall@FPR1%"]}
    df = pd.DataFrame(rows).T.round(3)
    df.index.names = ["model", "eval set"]
    out = ["# Step 7c: weighted training on all train rows", "",
           f"Depth-balanced weights (cap {a.cap}). Choices made on val_natural PR-AUC. Test split not used.", "",
           df.to_markdown(), ""]
    (S.REPORT / "step7c_results.md").write_text("\n".join(out), encoding="utf-8")
    print("\n" + df.to_markdown())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-n", type=int, default=40000)
    ap.add_argument("--cap", type=float, default=20.0)
    main(ap.parse_args())