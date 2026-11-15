"""Step 5: EDA on the training split, path-depth shortcut check, depth-matched resampling."""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
from sklearn.tree import DecisionTreeClassifier                   # noqa: E402

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_l1.features.lexical import url_features, FEATURE_NAMES   # noqa: E402

DATA = ROOT / "data" / "processed" / "url_dataset"
REPORT = L1_DIR / "reports" / "eda_l1"
REPORT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_BINS = 6                       # path depth 0,1,2,3,4,5+
BASE_COLS = ["url", "label", "source", "first_seen", "reg_domain", "split"]
KEY = ["url_len", "host_len", "path_depth", "query_len", "digit_ratio",
       "entropy_host", "n_subdomains", "is_https", "is_homepage"]


def load(split):
    d = pd.read_csv(DATA / f"{split}.csv", low_memory=False)
    d["first_seen"] = pd.to_datetime(d["first_seen"], utc=True, errors="coerce")
    return d


def featurize(d):
    F = pd.DataFrame([url_features(u) for u in d["url"]], index=d.index)
    out = pd.concat([d, F], axis=1)
    out["depth_bin"] = out["path_depth"].clip(upper=N_BINS - 1).astype(int)
    return out


def single_feature_auc(tr, va, feats):
    rows = []
    leaf = max(5, len(tr) // 1000)
    for f in feats:
        t = DecisionTreeClassifier(max_depth=4, min_samples_leaf=leaf, random_state=SEED)
        t.fit(tr[[f]], tr["label"])
        rows.append((f, roc_auc_score(va["label"], t.predict_proba(va[[f]])[:, 1])))
    return pd.DataFrame(rows, columns=["feature", "auroc"]).set_index("feature")


def match_depth(d):
    """Per path-depth bin keep equal numbers of benign and malicious rows."""
    keep, rows = [], []
    for b in range(N_BINS):
        pos = d[(d.depth_bin == b) & (d.label == 1)]
        neg = d[(d.depth_bin == b) & (d.label == 0)]
        k = min(len(pos), len(neg))
        rows.append({"depth_bin": b, "malicious": len(pos), "benign": len(neg), "kept_each": k})
        if k:
            keep += [pos.sample(k, random_state=SEED), neg.sample(k, random_state=SEED)]
    if not keep:
        sys.exit("no path-depth bin contains both classes: cannot match")
    m = pd.concat(keep).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return m, pd.DataFrame(rows).set_index("depth_bin")


def md(df, **kw):
    return df.to_markdown(**kw)


def main():
    tr = featurize(load("train"))
    va = featurize(load("val"))        # val is used for shortcut checks; test is never touched here
    print(f"train {len(tr)} rows, val {len(va)} rows")
    S = []                              # report sections
    key_print = []

    # --- balance
    bal = tr.groupby(["source", "label"]).size().unstack(fill_value=0)
    pos = tr[(tr.label == 1) & tr.first_seen.notna()]
    by_year = pos.groupby(pos.first_seen.dt.strftime("%Y")).size().rename("malicious rows")
    last12 = pos[pos.first_seen >= pos.first_seen.max() - pd.DateOffset(months=12)]
    by_month = last12.groupby(last12.first_seen.dt.strftime("%Y-%m")).size().rename("malicious rows")
    S += ["## Class balance (train)", "", md(bal), "",
          f"Malicious share overall: {tr.label.mean()*100:.1f}%", "",
          "Timestamped malicious rows per year:", "", md(by_year.to_frame()), "",
          "Last 12 months:", "", md(by_month.to_frame()), ""]

    # --- distributions
    rows = []
    for f in KEY:
        r = {"feature": f}
        for lab in (0, 1):
            s = tr.loc[tr.label == lab, f]
            r[f"L{lab} mean"], r[f"L{lab} median"], r[f"L{lab} p90"] = s.mean(), s.median(), s.quantile(.9)
        rows.append(r)
    dist = pd.DataFrame(rows).set_index("feature").round(3)
    src = tr.groupby(["source", "label"]).agg(
        n=("url", "size"), homepage_share=("is_homepage", "mean"), mean_depth=("path_depth", "mean"),
        median_url_len=("url_len", "median"), https_share=("is_https", "mean")).round(3)
    hist = pd.crosstab([tr.source, tr.label], tr.depth_bin, normalize="index").round(3)
    hist.columns = [f"depth {c}" if c < N_BINS - 1 else f"depth {c}+" for c in hist.columns]
    S += ["## Feature distributions by class (train)", "", md(dist), "",
          "## By source and label", "", md(src), "",
          "## Path-depth distribution by source and label (share of rows)", "", md(hist), ""]
    key_print += ["BY SOURCE AND LABEL", md(src), "PATH-DEPTH SHARE", md(hist)]

    # --- domains / infrastructure
    mal = tr[tr.label == 1]
    topdom = mal.reg_domain.value_counts().head(10)
    both = set(mal.reg_domain) & set(tr[tr.label == 0].reg_domain)
    srcs = {s: set(tr[tr.source == s].reg_domain) for s in tr.source.unique()}
    ov = [f"- {a} / {b}: {len(srcs[a] & srcs[b])} shared registered domains"
          for i, a in enumerate(sorted(srcs)) for b in sorted(srcs)[i + 1:] if srcs[a] & srcs[b]]
    S += ["## Domains and shared infrastructure (train)", "",
          f"- Top-10 registered domains hold {topdom.sum()/len(mal)*100:.1f}% of malicious rows:", "",
          md(topdom.rename("rows").to_frame()), "",
          f"- Registered domains present in BOTH classes: {len(both)}", *ov, ""]

    # --- shortcut check, natural data
    nat = single_feature_auc(tr, va, FEATURE_NAMES)
    S += ["## Single-feature classifiers (tree fit on train, AUROC on val), natural data", "",
          "A high value means the dataset itself may be teaching a shortcut.", "",
          md(nat.sort_values("auroc", ascending=False).round(3)), ""]

    # --- depth matching
    trm, tbl_tr = match_depth(tr)
    vam, tbl_va = match_depth(va)
    mat = single_feature_auc(trm, vam, FEATURE_NAMES)
    cmp_ = pd.concat([nat.rename(columns={"auroc": "natural"}),
                      mat.rename(columns={"auroc": "depth-matched"})], axis=1)
    cmp_ = cmp_.sort_values("natural", ascending=False).round(3)
    lost_pos = 1 - trm.label.sum() / max(tr.label.sum(), 1)
    lost_neg = 1 - (len(trm) - trm.label.sum()) / max((tr.label == 0).sum(), 1)
    S += ["## After path-depth matching (benign resampled to the malicious depth histogram)", "",
          "Train, rows per depth bin:", "", md(tbl_tr), "",
          f"Train rows kept: {len(trm)} of {len(tr)}. Malicious rows lost: {lost_pos*100:.1f}%, "
          f"benign rows lost: {lost_neg*100:.1f}%.", "",
          "Single-feature AUROC before and after matching (val):", "", md(cmp_), ""]
    key_print += ["MATCHING TABLE (train)", md(tbl_tr),
                  f"kept {len(trm)}/{len(tr)}; malicious lost {lost_pos*100:.1f}%, benign lost {lost_neg*100:.1f}%",
                  "SINGLE-FEATURE AUROC, top 12", md(cmp_.head(12))]

    cols = BASE_COLS
    trm[cols].to_csv(DATA / "train_matched.csv", index=False)
    vam[cols].to_csv(DATA / "val_matched.csv", index=False)

    # --- plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for a, d, t in ((ax[0], tr, "natural"), (ax[1], trm, "depth-matched")):
        h = pd.crosstab(d.depth_bin, d.label, normalize="columns")
        h.plot.bar(ax=a, rot=0)
        a.set_title(f"path depth, train, {t}")
        a.legend(["benign", "malicious"])
        a.set_xlabel("path depth (5 = 5+)")
    fig.tight_layout()
    fig.savefig(REPORT / "path_depth_hist.png", dpi=110)
    plt.close(fig)
    fig, a = plt.subplots(figsize=(6, 3.5))
    for lab, name in ((0, "benign"), (1, "malicious")):
        a.hist(tr.loc[tr.label == lab, "url_len"].clip(upper=300), bins=60, alpha=.55, density=True, label=name)
    a.set_xlabel("URL length (clipped at 300)")
    a.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "url_len_hist.png", dpi=110)
    plt.close(fig)

    S += ["## Notes", "",
          "- Only the train split is profiled; val is used for the shortcut check; test is untouched.",
          "- Matching equalizes benign and malicious counts per path-depth bin, so path depth carries no class information in the matched sets.",
          "- Other features (URL length etc.) may still differ between classes for legitimate reasons or because of source artifacts; see the table above.",
          "", "Figures: path_depth_hist.png, url_len_hist.png", ""]
    head = ["# Layer 1 URL EDA", "", f"Built: {date.today()}", ""]
    (REPORT / "eda_summary.md").write_text("\n".join(head + S), encoding="utf-8")
    print("\n\n".join(key_print))
    print(f"\nwrote {REPORT / 'eda_summary.md'}, train_matched.csv, val_matched.csv")


if __name__ == "__main__":
    main()