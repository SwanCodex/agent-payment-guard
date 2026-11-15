"""READ-ONLY diagnostics for the Layer 1 URL model. Modifies no file.
  python scripts/diagnose_l1_url.py [URL ...]        explain individual URLs (components, contributions)
  python scripts/diagnose_l1_url.py --groups         false-positive rate on benign validation/test rows by source and depth"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_l1.models.url_ensemble import GBM_FEATURES, UrlEnsemble, default_dir, strip_scheme   # noqa: E402

DATA = ROOT / "data" / "processed" / "url_dataset"
DEFAULT_URLS = [
    "https://www.google.com/",
    "https://www.google.com/search?q=sentinel+payment+guard",
    "https://accounts.google.com/signin/v2/identifier?service=mail&continue=https%3A%2F%2Fmail.google.com%2Fmail%2F",
    "https://pay.google.com/gp/w/u/0/home/paymentmethods",
    "https://www.paypal.com/signin",
    "https://en.wikipedia.org/wiki/Phishing",
    "http://secure-paypal-login.verify-account.example.top/signin/update.php?id=123",
]


def lr_contributions(m, url, k=8):
    """Per n-gram contribution to the LR decision (they sum to decision - intercept)."""
    txt = strip_scheme(url)
    X = m.lr.hv.transform([txt])
    idx, cnt = X.indices, X.data
    if len(idx) == 0:
        return [], 0.0
    v = (1.0 + np.log(cnt)) * m.lr.idf[idx]
    c_idx = dict(zip(idx.tolist(), (v * m.lr.coef[idx] / np.sqrt(v @ v)).tolist()))
    fh = FeatureHasher(n_features=2 ** 20, input_type="string", alternate_sign=False)
    seen, named = set(), []
    for g in m.lr.hv.build_analyzer()(txt):
        i = int(fh.transform([[g]]).indices[0])
        if g not in seen and i in c_idx:
            seen.add(g)
            named.append((g, c_idx[i]))
    named.sort(key=lambda t: -abs(t[1]))
    return named[:k], sum(c_idx.values())


def explain(m, url):
    r = m.score(url)
    print(f"\n{url}")
    print(f"  p={r['p']:.4f}  z={r['z']:.3f}  p_lr={r['p_lr']:.4f}  p_gbm={r['p_gbm']:.4f}  "
          f"gray={r['gray']}  abstain={r['abstain']}  (tau_allow={m.tau_allow:.4f}, tau_block={m.tau_block:.4f})")
    print(f"  at/above tau_block: {r['p'] >= m.tau_block}")
    row = np.array([m._gbm_row(url)])
    contrib = m.booster.predict(row, pred_contrib=True)[0]
    names = GBM_FEATURES + ["(bias)"]
    top = sorted(zip(names, contrib, list(row[0]) + [np.nan]), key=lambda t: -abs(t[1]))[:8]
    print("  LightGBM top contributions (log-odds):")
    for n, c, val in top:
        print(f"    {n:24s} {c:+.3f}   value={val}")
    grams, tot = lr_contributions(m, url)
    print(f"  LR decision {m.lr.decision(url):+.3f} = intercept {m.lr.b:+.3f} + n-grams {tot:+.3f}; top n-grams:")
    for g, c in grams:
        print(f"    {g!r:18s} {c:+.3f}")


def depth_bin(url):
    try:
        n = len([x for x in urlsplit(url if "://" in url else "http://" + url).path.split("/") if x])
    except ValueError:
        n = 0
    return min(n, 5)


def groups(m):
    rows = []
    for name in ("val", "test"):
        p = DATA / f"{name}.csv"
        if not p.exists():
            print(f"missing {p}: skipping {name}")
            continue
        d = pd.read_csv(p, low_memory=False)
        d = d[d.label == 0].copy()
        d["p"] = [m.score(u)["p"] for u in d.url]
        d["split"], d["depth"] = name, [depth_bin(u) for u in d.url]
        rows.append(d)
    if not rows:
        sys.exit("no data files found: the group analysis needs the processed CSVs")
    d = pd.concat(rows)
    d["blocked"], d["p>=0.5"] = d.p >= m.tau_block, d.p >= 0.5
    g = d.groupby(["source", "depth"]).agg(n=("p", "size"), at_tau_block=("blocked", "mean"),
                                            at_p50=("p>=0.5", "mean"), mean_p=("p", "mean")).round(4)
    print("\nBENIGN rows: share at or above tau_block, by source and path depth (5 = 5+)")
    print(g.to_string())
    print(f"\noverall benign share at/above tau_block: {d.blocked.mean():.4f} (n={len(d)})")
    print("\nTop 20 highest-scoring benign URLs:")
    print(d.sort_values("p", ascending=False).head(20)[["source", "split", "p", "url"]].to_string(index=False))


def main(a):
    m = UrlEnsemble(default_dir())
    print("model", m.version)
    if a.groups:
        groups(m)
    for u in (a.urls or ([] if a.groups else DEFAULT_URLS)):
        explain(m, u)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--groups", action="store_true")
    main(ap.parse_args())