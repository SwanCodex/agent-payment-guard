"""Step 7b: does training on ALL train rows with depth-balanced weights beat the depth-matched subset?
Model: logistic regression on hashed char n-grams (scheme and www removed). Test split untouched."""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_step6 as S                                              # noqa: E402

N_BINS = 6


def depth_bin(url):
    try:
        n = len([x for x in urlsplit(url if "://" in url else "http://" + url).path.split("/") if x])
    except ValueError:
        n = 0
    return min(n, N_BINS - 1)


def balanced_weights(bins, y, cap):
    """Per depth bin, give both classes the same total weight (so depth carries no class prior)."""
    w = np.ones(len(y))
    for b in range(N_BINS):
        for c in (0, 1):
            m = (bins == b) & (y == c)
            n_bc, n_b = m.sum(), ((bins == b)).sum()
            if n_bc:
                w[m] = 0.5 * n_b / n_bc
    w = np.minimum(w, cap)
    return w * len(w) / w.sum()


def main(a):
    full = pd.read_csv(S.DATA / "train.csv", low_memory=False)
    matched = pd.read_csv(S.DATA / "train_matched.csv", low_memory=False)
    sets = {"val_matched": pd.read_csv(S.DATA / "val_matched.csv", low_memory=False),
            "val_natural": pd.read_csv(S.DATA / "val.csv", low_memory=False)}
    ext = pd.read_csv(S.DATA / "external_phiusiil.csv", low_memory=False)
    if len(ext) > a.ext_n:
        ext = ext.sample(a.ext_n, random_state=S.SEED)
    sets["external_phiusiil"] = ext
    Y = {k: d.label.values for k, d in sets.items()}

    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 20,
                           alternate_sign=False, norm=None, lowercase=False)

    def run(train_df, weights, label):
        tf = TfidfTransformer(sublinear_tf=True)
        Htr = tf.fit_transform(hv.transform([S.strip_scheme(u) for u in train_df.url]))
        H = {k: tf.transform(hv.transform([S.strip_scheme(u) for u in d.url])) for k, d in sets.items()}
        best = None
        for C in (1.0, 3.0, 10.0):
            lr = LogisticRegression(C=C, max_iter=3000, solver="liblinear")
            lr.fit(Htr, train_df.label.values, sample_weight=weights)
            s = average_precision_score(Y["val_matched"], lr.predict_proba(H["val_matched"])[:, 1])
            print(f"  {label} C={C}: val_matched PR-AUC {s:.4f}")
            if best is None or s > best[0]:
                best = (s, C, lr)
        lr = best[2]
        res = {}
        for k in sets:
            m = S.metrics(Y[k], lr.predict_proba(H[k])[:, 1])
            res[k] = m
        return best[1], res

    rows = {}
    for label, df, w in (("A matched subset", matched, None),
                         ("B all rows, depth-balanced weights", full,
                          balanced_weights(np.array([depth_bin(u) for u in full.url]), full.label.values, a.cap))):
        C, res = run(df, w, label)
        for k, m in res.items():
            rows[(label, k)] = {"C": C, "n_train": len(df), "ROC": m["ROC-AUC"], "PR": m["PR-AUC"],
                                "recall@0.1%FPR": m["recall@FPR0.1%"], "recall@0.5%FPR": m["recall@FPR0.5%"],
                                "recall@1%FPR": m["recall@FPR1%"]}
    df = pd.DataFrame(rows).T.round(3)
    df.index.names = ["training set", "eval set"]
    text = ["# Step 7b: matched subset vs all rows with depth-balanced weights (LR on char n-grams)", "",
            f"Weight cap {a.cap}. Test split not used.", "", df.to_markdown(), ""]
    (S.REPORT / "step7b_results.md").write_text("\n".join(text), encoding="utf-8")
    print("\n" + df.to_markdown())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-n", type=int, default=40000)
    ap.add_argument("--cap", type=float, default=20.0, help="max per-sample weight before rescaling")
    main(ap.parse_args())