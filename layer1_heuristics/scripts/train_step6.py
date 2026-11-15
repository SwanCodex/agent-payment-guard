"""Step 6: features, hashed char n-gram baseline, first LightGBM. No test split is touched."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_l1.features.builder import build_base, MODEL_FEATURES        # noqa: E402
from sentinel_l1.features.lexicon import Lexicon, make_brand_list, save_brands   # noqa: E402
from sentinel_l1.features.tld_risk import TldRisk, oof_tld_risk            # noqa: E402

DATA = ROOT / "data" / "processed" / "url_dataset"
MODELS = ROOT / "data" / "interim" / "l1_models"        # git-ignored scratch; real artifacts come in Step 9
REPORT = L1_DIR / "reports"
SEED = 42
MODELS.mkdir(parents=True, exist_ok=True)


def strip_scheme(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", u.lower())[:200]


def recall_at_fpr(y, p, fprs=(0.001, 0.005, 0.01)):
    fpr, tpr, _ = roc_curve(y, p)
    return [float(np.interp(x, fpr, tpr)) for x in fprs]


def metrics(y, p):
    r = recall_at_fpr(y, p)
    return {"n": len(y), "mal_share": float(np.mean(y)), "PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p), "recall@FPR0.1%": r[0], "recall@FPR0.5%": r[1],
            "recall@FPR1%": r[2]}


def main(a):
    brands = make_brand_list(ROOT / "data" / "raw" / "tranco" / "top-1m.csv", top=500)
    save_brands(brands, MODELS / "brands.json")
    lex = Lexicon(brands)
    print(f"brand list: {len(brands)} labels")

    tr = pd.read_csv(DATA / "train_matched.csv", low_memory=False)
    sets = {"val_matched": pd.read_csv(DATA / "val_matched.csv", low_memory=False),
            "val_natural": pd.read_csv(DATA / "val.csv", low_memory=False)}
    ext = pd.read_csv(DATA / "external_phiusiil.csv", low_memory=False)
    if len(ext) > a.ext_n:
        ext = ext.sample(a.ext_n, random_state=SEED)
    sets["external_phiusiil"] = ext
    print({k: len(v) for k, v in sets.items()}, "train", len(tr))

    # ---- features
    Xtr, tlds_tr = build_base(tr.url.tolist(), lex)
    ytr = tr.label.values
    Xtr["tld_risk"] = oof_tld_risk(tlds_tr, ytr, tr.reg_domain.values)
    tld = TldRisk().fit(tlds_tr, ytr)
    tld.save(MODELS / "tld_risk.json")
    X, Y = {}, {}
    for name, d in sets.items():
        Xs, t = build_base(d.url.tolist(), lex)
        Xs["tld_risk"] = tld.transform(t)
        X[name], Y[name] = Xs[MODEL_FEATURES], d.label.values
    Xtr = Xtr[MODEL_FEATURES]
    print("features:", len(MODEL_FEATURES))

    results = {}

    # ---- baseline: logistic regression on hashed char n-grams (scheme and www removed)
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 20,
                           alternate_sign=False, norm=None, lowercase=False)
    tf = TfidfTransformer(sublinear_tf=True)
    Htr = tf.fit_transform(hv.transform([strip_scheme(u) for u in tr.url]))
    H = {k: tf.transform(hv.transform([strip_scheme(u) for u in d.url])) for k, d in sets.items()}
    best = None
    for C in (0.3, 1.0, 3.0):
        lr = LogisticRegression(C=C, max_iter=2000, solver="liblinear").fit(Htr, ytr)
        s = average_precision_score(Y["val_matched"], lr.predict_proba(H["val_matched"])[:, 1])
        print(f"  LR C={C}: val_matched PR-AUC {s:.4f}")
        if best is None or s > best[0]:
            best = (s, C, lr)
    lr = best[2]
    for k in sets:
        results[("LR char n-grams", k)] = metrics(Y[k], lr.predict_proba(H[k])[:, 1])

    # ---- LightGBM on the engineered features
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                             min_child_samples=40, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, verbose=-1)
    clf.fit(Xtr, ytr, eval_set=[(X["val_matched"], Y["val_matched"])], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    print(f"LightGBM best iteration: {clf.best_iteration_}")
    for k in sets:
        results[("LightGBM", k)] = metrics(Y[k], clf.predict_proba(X[k])[:, 1])
    clf.booster_.save_model(str(MODELS / "lgbm_step6.txt"))

    imp = pd.Series(clf.booster_.feature_importance(importance_type="gain"), index=MODEL_FEATURES)
    imp = (imp / imp.sum()).sort_values(ascending=False)

    # ---- report
    df = pd.DataFrame(results).T
    df.index.names = ["model", "eval set"]
    df = df.round(4)
    out = ["# Step 6 results (first models, untuned)", "",
           "Training set: depth-matched train. Early stopping and the LR constant were chosen on val_matched.",
           "Test split not used.", "", df.to_markdown(), "",
           "## LightGBM feature importance (share of total gain)", "",
           imp.head(20).round(4).to_frame("gain share").to_markdown(), ""]
    (REPORT / "step6_results.md").write_text("\n".join(out), encoding="utf-8")
    print(df.to_markdown())
    print("\nTOP FEATURES\n" + imp.head(15).round(4).to_frame("gain share").to_markdown())
    print(f"\nwrote {REPORT / 'step6_results.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-n", type=int, default=40000, help="rows of the external PhiUSIIL set to score")
    main(ap.parse_args())