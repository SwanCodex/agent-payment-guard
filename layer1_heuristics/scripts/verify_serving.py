"""Serving-path verification: reproduces the evaluation numbers through UrlEnsemble, domain-capped metrics, latency."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_contracts import L1In                                    # noqa: E402
from sentinel_l1.api import run_l1                                     # noqa: E402
from sentinel_l1.models.url_ensemble import UrlEnsemble, default_dir   # noqa: E402

DATA = ROOT / "data" / "processed" / "url_dataset"
REPORT = L1_DIR / "reports"
# combiner + calibrated rows from the training run's report (reports/l1_eval_report.md)
EXPECTED = {"val": {"ROC": 0.9375, "PR": 0.7999, "recall@0.5%FPR": 0.4461},
            "test": {"ROC": 0.8958, "PR": 0.6438, "recall@0.5%FPR": 0.1818}}
TOL = 0.003
CAP = 5


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {"ROC": roc_auc_score(y, p), "PR": average_precision_score(y, p),
            "recall@0.5%FPR": float(np.interp(0.005, fpr, tpr))}


def main():
    m = UrlEnsemble(default_dir())
    print("loaded", m.version, "| tau_allow", m.tau_allow, "tau_block", m.tau_block)
    scores, dfs = {}, {}
    for name, f in (("val", "val.csv"), ("test", "test.csv")):
        d = pd.read_csv(DATA / f, low_memory=False)
        t0 = time.perf_counter()
        scores[name] = np.array([m.score(u)["p"] for u in d.url])
        dfs[name] = d
        print(f"scored {name}: {len(d)} urls in {time.perf_counter() - t0:.0f}s")

    lines = ["# Serving verification", "", "## Serving path reproduces the evaluation numbers", "",
             f"Tolerance {TOL}.", "", "| split | metric | serving | expected | ok |", "|---|---|---|---|---|"]
    all_ok = True
    for name in ("val", "test"):
        got = metrics(dfs[name].label.values, scores[name])
        for k, exp in EXPECTED[name].items():
            ok = abs(got[k] - exp) <= TOL
            all_ok &= ok
            lines.append(f"| {name} | {k} | {got[k]:.4f} | {exp:.4f} | {'PASS' if ok else 'FAIL'} |")

    lines += ["", f"## Domain-capped metrics (at most {CAP} rows per registered domain and class)", "",
              "| split | rows | ROC | PR | recall@0.5%FPR |", "|---|---|---|---|---|"]
    for name in ("val", "test"):
        d = dfs[name].assign(p=scores[name]).sample(frac=1, random_state=42)
        c = d.groupby(["reg_domain", "label"]).head(CAP)
        g = metrics(c.label.values, c.p.values)
        lines.append(f"| {name} | {len(c)} | {g['ROC']:.4f} | {g['PR']:.4f} | {g['recall@0.5%FPR']:.4f} |")

    urls = dfs["test"].url.sample(min(2000, len(dfs["test"])), random_state=1).tolist()
    for u in urls[:50]:
        m.score(u)
    lat = []
    for u in urls:
        t0 = time.perf_counter()
        m.score(u)
        lat.append((time.perf_counter() - t0) * 1000)
    lat2 = []
    for u in urls[:500]:
        t0 = time.perf_counter()
        run_l1(L1In(request_id="v", url=u))
        lat2.append((time.perf_counter() - t0) * 1000)
    q = lambda v, x: float(np.percentile(v, x))                        # noqa: E731
    lines += ["", "## Latency (single URL, one thread)", "", "| path | p50 ms | p95 ms | p99 ms |", "|---|---|---|---|",
              f"| UrlEnsemble.score | {q(lat,50):.2f} | {q(lat,95):.2f} | {q(lat,99):.2f} |",
              f"| run_l1 (rules + model) | {q(lat2,50):.2f} | {q(lat2,95):.2f} | {q(lat2,99):.2f} |",
              "", "Budget: URL-only model p95 <= 20 ms.", "", f"Overall parity check: {'PASS' if all_ok else 'FAIL'}", ""]
    (REPORT / "serving_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()