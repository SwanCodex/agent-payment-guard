"""Step 7a: feature-group ablation with a more regularized LightGBM. Test split untouched."""
import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_step6 as S                                              # noqa: E402  (paths, metrics)
from sentinel_l1.features.builder import build_base, MODEL_FEATURES  # noqa: E402
from sentinel_l1.features.lexicon import Lexicon, load_brands, LEXICON_FEATURES   # noqa: E402
from sentinel_l1.features.tld_risk import TldRisk, oof_tld_risk      # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
CACHE = S.MODELS / "feat_cache.pkl"        # delete this file if the feature code changes

GROUPS = {
    "host": ["host_len", "host_hyphens", "host_digits", "host_labels", "max_label_len",
             "n_subdomains", "host_digit_ratio", "entropy_host", "vowel_ratio_host"],
    "path": ["path_len", "query_len", "path_depth", "n_query_params", "entropy_path",
             "hex_ratio_pathquery", "double_slash_in_path", "is_homepage"],
    "urlwide": ["url_len", "n_dots", "n_hyphens", "n_digits", "n_special", "digit_ratio", "longest_token"],
    "flags": ["has_at", "has_port", "is_ip_host", "has_punycode", "is_shortener"],
    "lexicon": list(LEXICON_FEATURES),
    "tld": ["tld_risk"],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(MODEL_FEATURES), "feature groups out of sync"

PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=200,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
              random_state=S.SEED, verbose=-1)


def get_data(ext_n):
    if CACHE.exists():
        print("using cached features:", CACHE)
        return pd.read_pickle(CACHE)
    lex = Lexicon(load_brands(S.MODELS / "brands.json"))
    tr = pd.read_csv(S.DATA / "train_matched.csv", low_memory=False)
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
    X, Y = {}, {}
    for name, d in sets.items():
        Xs, t = build_base(d.url.tolist(), lex)
        Xs["tld_risk"] = tld.transform(t)
        X[name], Y[name] = Xs[MODEL_FEATURES], d.label.values
    data = {"Xtr": Xtr[MODEL_FEATURES], "ytr": ytr, "X": X, "Y": Y}
    pd.to_pickle(data, CACHE)
    return data


def main(a):
    import lightgbm as lgb
    d = get_data(a.ext_n)
    Xtr, ytr, X, Y = d["Xtr"], d["ytr"], d["X"], d["Y"]

    variants = {"all features": MODEL_FEATURES}
    for g, cols in GROUPS.items():
        variants[f"drop {g}"] = [f for f in MODEL_FEATURES if f not in cols]
    variants["drop path_depth+is_homepage"] = [f for f in MODEL_FEATURES if f not in ("path_depth", "is_homepage")]
    for g, cols in GROUPS.items():
        variants[f"only {g}"] = list(cols)

    rows = {}
    for name, feats in variants.items():
        clf = lgb.LGBMClassifier(**PARAMS)
        clf.fit(Xtr[feats], ytr, eval_set=[(X["val_matched"][feats], Y["val_matched"])],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        m = {k: S.metrics(Y[k], clf.predict_proba(X[k][feats])[:, 1]) for k in X}
        rows[name] = {"n_feat": len(feats), "best_iter": clf.best_iteration_,
                      "ROC matched": m["val_matched"]["ROC-AUC"],
                      "ROC natural": m["val_natural"]["ROC-AUC"],
                      "ROC external": m["external_phiusiil"]["ROC-AUC"],
                      "PR matched": m["val_matched"]["PR-AUC"],
                      "recall@0.5%FPR matched": m["val_matched"]["recall@FPR0.5%"]}
        print(f"{name:32s} iters {clf.best_iteration_:4d}  ROC matched {rows[name]['ROC matched']:.3f}  "
              f"natural {rows[name]['ROC natural']:.3f}  external {rows[name]['ROC external']:.3f}")
    df = pd.DataFrame(rows).T.round(3)
    df["n_feat"] = df["n_feat"].astype(int)
    df["best_iter"] = df["best_iter"].astype(int)
    out = ["# Step 7a: feature-group ablation", "",
           "LightGBM (regularized: num_leaves 15, min_child_samples 200, lr 0.03, lambda_l2 10), trained on depth-matched train,",
           "early stopping on val_matched. Test split not used.", "", df.to_markdown(), ""]
    (S.REPORT / "step7_ablation.md").write_text("\n".join(out), encoding="utf-8")
    print("\n" + df.to_markdown())
    print(f"\nwrote {S.REPORT / 'step7_ablation.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-n", type=int, default=40000)
    main(ap.parse_args())