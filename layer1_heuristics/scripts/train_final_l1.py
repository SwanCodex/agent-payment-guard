"""Step 8: final URL ensemble, combiner, calibration, thresholds, conformal, ONE pass on the test split.
Base models: weighted LR (char n-grams) + weighted LightGBM (no host features), averaged in logit space.
Calibration, combiner and thresholds are fit on val_natural. The test split is scored at the end only (never used for any choice)."""
import argparse
import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_step6 as S                                              # noqa: E402
import train_step7c as C7                                            # noqa: E402
from sentinel_l1.features.builder import build_base, MODEL_FEATURES  # noqa: E402
from sentinel_l1.features.lexicon import Lexicon, load_brands, save_brands   # noqa: E402
from sentinel_l1.features.tld_risk import TldRisk, tld_of            # noqa: E402
from sentinel_l1.rules.r06_lookalike import check_r06                # noqa: E402
from sentinel_l1.rules.r07_punycode import check_r07                 # noqa: E402
from sentinel_l1.rules.r10_shortener import check_r10                # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
LR_C = 100.0
FEATS = [f for f in MODEL_FEATURES if f not in C7.HOST and f != "is_shortener"]
FLAG_NAMES = ["R06", "R07", "R10"]          # diagnostics only: the combiner uses z alone (see report)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def rule_flags(urls):
    return np.array([[bool(check_r06(u, None)), bool(check_r07(u)), bool(check_r10(u, None))]
                     for u in urls], dtype=float)


def ece(y, p, bins=15):
    o = np.argsort(p)
    y, p = np.asarray(y)[o], np.asarray(p)[o]
    chunks = np.array_split(np.arange(len(p)), bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def pick_thresholds(y, p, fpr_target=0.005, fnr_target=0.01):
    pb, pm = p[y == 0], p[y == 1]
    uniq = np.unique(p)
    tau_block = None
    for t in uniq:                                         # smallest t with FPR(p >= t) <= target
        if (pb >= t).mean() <= fpr_target:
            tau_block = float(t)
            break
    if tau_block is None:
        tau_block = float(min(uniq.max() + 1e-6, 1.0))
    tau_allow = None
    for t in uniq[::-1]:                                   # largest t with FNR(p <= t) <= target
        if (pm <= t).mean() <= fnr_target:
            tau_allow = float(t)
            break
    if tau_allow is None:
        tau_allow = 0.0
    return tau_allow, tau_block


def tau_allow_for(y, p, fnr):
    pm, uniq = p[y == 1], np.unique(p)
    for t in uniq[::-1]:
        if (pm <= t).mean() <= fnr:
            return float(t)
    return 0.0


def conformal_qhat(y, p, alpha=0.1):
    s = np.where(y == 1, 1 - p, p)
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(s)[k - 1])


def operating_point(y, p, tau_allow, tau_block, qhat):
    y = np.asarray(y)
    blocked, allowed = p >= tau_block, p <= tau_allow
    gray = ~blocked & ~allowed
    in_set1 = (1 - p) <= qhat            # class 1 in the prediction set when its score is conforming
    in_set0 = p <= qhat
    abst = (in_set0.astype(int) + in_set1.astype(int)) != 1      # empty or two-class set = not confident
    cover = np.where(y == 1, in_set1, in_set0).mean()
    return {"n": len(y), "FPR@block": float(blocked[y == 0].mean()), "recall@block": float(blocked[y == 1].mean()),
            "FNR@allow": float(allowed[y == 1].mean()), "benign allowed": float(allowed[y == 0].mean()),
            "benign gray": float(gray[y == 0].mean()), "abstain rate": float(abst.mean()),
            "conformal coverage": float(cover)}


def main(a):
    import lightgbm as lgb
    art = S.ROOT / "artifacts" / "layer1" / "url_ensemble" / a.version
    if art.exists() and any(art.iterdir()) and not a.force:
        sys.exit(f"{art} already exists: versions are immutable. Bump --version, or use --force while developing.")
    art.mkdir(parents=True, exist_ok=True)

    d = C7.get_data(a.ext_n, 20.0)
    Xtr, ytr, w, X, Y, urls = d["Xtr"], d["ytr"], d["w"], d["X"], d["Y"], d["urls"]
    lex = Lexicon(load_brands(S.MODELS / "brands.json"))
    tld = TldRisk().fit([tld_of(u) for u in urls["train"]], ytr)       # same map the cached features used

    # test features (first and only use of the test split)
    te = pd.read_csv(S.DATA / "test.csv", low_memory=False)
    Xte, t_te = build_base(te.url.tolist(), lex)
    Xte["tld_risk"] = tld.transform(t_te)
    X["test"], Y["test"], urls["test"] = Xte[MODEL_FEATURES], te.label.values, te.url.tolist()
    names = list(X)

    # ---- base model 1: weighted LR on hashed char n-grams
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 20,
                           alternate_sign=False, norm=None, lowercase=False)
    tf = TfidfTransformer(sublinear_tf=True)
    Htr = tf.fit_transform(hv.transform([S.strip_scheme(u) for u in urls["train"]]))
    lr = LogisticRegression(C=LR_C, max_iter=3000, solver="liblinear").fit(Htr, ytr, sample_weight=w)
    p_lr = {k: lr.predict_proba(tf.transform(hv.transform([S.strip_scheme(u) for u in urls[k]])))[:, 1]
            for k in names}
    np.savez_compressed(art / "lr_ngram.npz", idf=tf.idf_.astype(np.float32),
                        coef=lr.coef_[0].astype(np.float32), intercept=np.float32(lr.intercept_[0]),
                        C=np.float32(LR_C))

    # ---- base model 2: weighted LightGBM without host features
    clf = lgb.LGBMClassifier(**C7.PARAMS)
    clf.fit(Xtr[FEATS], ytr, sample_weight=w, eval_set=[(X["val_matched"][FEATS], Y["val_matched"])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    print(f"LightGBM best iteration {clf.best_iteration_}")
    clf.booster_.save_model(str(art / "model.txt"), num_iteration=clf.best_iteration_)
    p_gbm = {k: clf.predict_proba(X[k][FEATS])[:, 1] for k in names}
    z = {k: 0.5 * (logit(p_lr[k]) + logit(p_gbm[k])) for k in names}

    # ---- rule flags: diagnostics only (val_natural). The combiner uses z alone.
    yv = Y["val_natural"]
    Fv = rule_flags(urls["val_natural"])
    flag_counts = {n: (int(Fv[yv == 0, i].sum()), int(Fv[yv == 1, i].sum())) for i, n in enumerate(FLAG_NAMES)}
    print("rule flag counts on val_natural (benign / malicious):", flag_counts)
    comb = LogisticRegression(C=1.0, max_iter=1000).fit(z["val_natural"].reshape(-1, 1), yv)
    sc = {k: comb.decision_function(z[k].reshape(-1, 1)) for k in names}
    coefs = {"z": round(float(comb.coef_[0][0]), 4)}
    print("combiner (Platt on z):", coefs, "intercept", round(float(comb.intercept_[0]), 4))
    (art / "combiner.json").write_text(json.dumps(
        {"features": ["z"], "coef": comb.coef_[0].tolist(), "intercept": float(comb.intercept_[0])}),
        encoding="utf-8")

    # ---- isotonic calibration on val_natural
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(sc["val_natural"], yv)
    (art / "calibration.json").write_text(json.dumps(
        {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}),
        encoding="utf-8")
    pc = {k: iso.predict(sc[k]) for k in names}

    # ---- thresholds and conformal quantile (val_natural)
    tau_allow, tau_block = pick_thresholds(yv, pc["val_natural"])
    qhat = conformal_qhat(yv, pc["val_natural"])
    ok = tau_allow < tau_block
    print(f"tau_allow {tau_allow:.4f}  tau_block {tau_block:.4f}  qhat {qhat:.4f}  ordering ok: {ok}")
    op_val = operating_point(yv, pc["val_natural"], tau_allow, tau_block, qhat)
    (art / "thresholds.json").write_text(json.dumps(
        {"tau_allow": tau_allow, "tau_block": tau_block, "qhat": qhat,
         "fpr_at_block": op_val["FPR@block"], "fnr_at_allow": op_val["FNR@allow"],
         "val_split": "val_natural"}), encoding="utf-8")
    TldRisk.save(tld, art / "tld_risk.json")
    save_brands(load_brands(S.MODELS / "brands.json"), art / "brands.json")

    # ---- evaluation (test split scored here; nothing above uses it)
    rows = {}
    for k in ("val_natural", "val_matched", "external_phiusiil", "test"):
        for label, s in (("ensemble z only", z[k]), ("combiner + calibrated", pc[k])):
            m = S.metrics(Y[k], s)
            rows[(label, k)] = {"ROC": m["ROC-AUC"], "PR": m["PR-AUC"], "recall@0.1%FPR": m["recall@FPR0.1%"],
                                "recall@0.5%FPR": m["recall@FPR0.5%"], "recall@1%FPR": m["recall@FPR1%"],
                                "ECE": ece(Y[k], pc[k]) if label.startswith("comb") else np.nan}
    perf = pd.DataFrame(rows).T.round(4)
    perf.index.names = ["scores", "eval set"]
    ops = pd.DataFrame({k: operating_point(Y[k], pc[k], tau_allow, tau_block, qhat)
                        for k in ("val_natural", "val_matched", "external_phiusiil", "test")}).T.round(4)

    tl = pd.DataFrame({"tld": [tld_of(u) for u in urls["test"]], "y": Y["test"], "p": pc["test"]})
    tl["blocked"] = tl.p >= tau_block
    per_tld = (tl.groupby("tld").apply(lambda g: pd.Series({
        "n": len(g), "malicious share": g.y.mean(),
        "FPR@block": g.blocked[g.y == 0].mean() if (g.y == 0).any() else np.nan,
        "recall@block": g.blocked[g.y == 1].mean() if (g.y == 1).any() else np.nan}), include_groups=False)
        .query("n >= 300").sort_values("n", ascending=False).head(12).round(3))

    # allow-side trade-off: how many benign URLs can be confidently allowed at looser FNR targets
    trade = []
    for fnr in (0.01, 0.02, 0.05, 0.10):
        ta = tau_allow_for(yv, pc["val_natural"], fnr)
        yt, pt = Y["test"], pc["test"]
        trade.append({"target FNR (val)": fnr, "tau_allow": round(ta, 4),
                      "val benign allowed": round(float((pc["val_natural"][yv == 0] <= ta).mean()), 3),
                      "test FNR": round(float((pt[yt == 1] <= ta).mean()), 3),
                      "test benign allowed": round(float((pt[yt == 0] <= ta).mean()), 3)})
    trade = pd.DataFrame(trade)

    # drift: malicious rows by quarter of first-seen date
    def by_quarter(csv, p, y):
        d = pd.read_csv(csv, usecols=["first_seen"], low_memory=False)
        fs = pd.to_datetime(d["first_seen"], utc=True, errors="coerce")
        q = fs.dt.year.astype("Int64").astype(str) + "Q" + fs.dt.quarter.astype("Int64").astype(str)
        m = (y == 1) & fs.notna().values
        g = pd.DataFrame({"q": q[m].values, "blocked": (p[m] >= tau_block), "p": p[m]})
        return g.groupby("q").agg(n=("p", "size"), recall_at_block=("blocked", "mean"),
                                  mean_p=("p", "mean")).round(3)
    q_val = by_quarter(S.DATA / "val.csv", pc["val_natural"], Y["val_natural"])
    q_test = by_quarter(S.DATA / "test.csv", pc["test"], Y["test"])

    out = ["# Layer 1 URL ensemble: evaluation report", "", f"Built {date.today()}, version {a.version}.", "",
           "Base models: weighted LR on char n-grams + weighted LightGBM (no host features), logit average;",
           "calibration: Platt on z, then isotonic; thresholds and conformal qhat fit on val_natural.",
           "Rule flags R06/R07/R10 are NOT inputs to the combiner (see notes); they act through the decision policy.",
           "Selection of C, features and early stopping used the validation splits. Test-split history: see the notes at the end.", "",
           f"LightGBM best iteration {clf.best_iteration_}; combiner coefficients {coefs}.", "",
           "## Discrimination", "", perf.to_markdown(), "",
           f"## Operating point (tau_allow {tau_allow:.4f}, tau_block {tau_block:.4f}, qhat {qhat:.4f})", "",
           ops.to_markdown(), "",
           "`benign gray` is the share of benign URLs this layer sends to the gray zone (LLM Arbiter budget).", "",
           "`abstain` = conformal prediction set empty or two-class (size != 1).", "",
           "## Allow-side trade-off (how many benign URLs can be confidently allowed)", "", trade.to_markdown(index=False), "",
           "## Malicious recall at tau_block by first-seen quarter", "", "val_natural:", "", q_val.to_markdown(), "",
           "test:", "", q_test.to_markdown(), "",
           "## Test split by TLD (n >= 300)", "", per_tld.to_markdown(), "",
           "## History and notes", "",
           f"- Rule flag counts on val_natural (benign / malicious): {flag_counts}.",
           "- A first run kept URL-shortener hosts in the data and fed R06/R07/R10 to the combiner. Validation flag counts then showed",
           "  R10 firing on 318 malicious and 0 benign URLs (sampling artifact: benign shorteners were never sampled), and the combiner",
           "  gave it a large weight. That run's test numbers were ROC 0.892, PR 0.652, recall@0.5%FPR 0.179.",
           "- This version removes shortener-host rows from all splits (R10 stays a gray rule), drops is_shortener from LightGBM and",
           "  uses z alone in the combiner. The test split was therefore scored twice; both runs are disclosed here.", "",
           "## Not done / limitations", "",
           "- No HTML/page model; R03 disabled; no char-CNN.",
           "- R06 uses a 20-brand seed list, not Tranco top-N.",
           "- Benign data is Tranco homepages plus Common Crawl pages; malicious is PhishTank. External PhiUSIIL is homepage-heavy.",
           "- Thresholds and calibration use the natural val mix (about 15% malicious), not a deployment prior.", ""]
    (S.REPORT / "l1_eval_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n" + perf.to_markdown())
    print("\nOPERATING POINT\n" + ops.to_markdown())
    print("\nALLOW-SIDE TRADE-OFF\n" + trade.to_markdown(index=False))
    print("\nTEST BY QUARTER\n" + q_test.to_markdown())
    print("\nTEST BY TLD\n" + per_tld.to_markdown())
    print(f"\nartifacts -> {art}\nreport -> {S.REPORT / 'l1_eval_report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--force", action="store_true", help="overwrite an existing version directory (development only)")
    ap.add_argument("--ext-n", type=int, default=40000)
    main(ap.parse_args())