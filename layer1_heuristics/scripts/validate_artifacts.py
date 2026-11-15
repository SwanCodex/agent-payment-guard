"""Validate a Layer 1 artifact bundle (handbook 5.2): file schemas, threshold ordering, calibration
monotonicity, p in [0,1], 100-row smoke predict. Exit code 1 if any check fails."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_contracts import L1In                                    # noqa: E402
from sentinel_l1.api import run_l1                                     # noqa: E402
from sentinel_l1.models.url_ensemble import GBM_FEATURES, UrlEnsemble  # noqa: E402

REQUIRED = ["model.txt", "lr_ngram.npz", "combiner.json", "calibration.json", "thresholds.json",
            "feature_spec.yaml", "model_card.md", "eval_report.md", "manifest.json", "tld_risk.json", "brands.json"]
CARD_HEADINGS = ["## Intended use", "## Data", "## Metrics", "## Limitations"]
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main(a):
    art = ROOT / "artifacts" / "layer1" / "url_ensemble" / a.version
    for f in REQUIRED:
        p = art / f
        check(f"file present: {f}", p.exists() and p.stat().st_size > 0)
    if not all(ok for _, ok, _ in checks):
        return report()

    cal = json.loads((art / "calibration.json").read_text(encoding="utf-8"))
    x, y = np.array(cal["x"]), np.array(cal["y"])
    check("calibration: method is isotonic", cal.get("method") == "isotonic")
    check("calibration: x and y same length (>= 2)", len(x) == len(y) >= 2)
    check("calibration: x strictly increasing", bool(np.all(np.diff(x) > 0)))
    check("calibration: y non-decreasing (monotone)", bool(np.all(np.diff(y) >= -1e-12)))
    check("calibration: y within [0, 1]", bool(y.min() >= 0 and y.max() <= 1))

    th = json.loads((art / "thresholds.json").read_text(encoding="utf-8"))
    need = {"tau_allow", "tau_block", "qhat", "fpr_at_block", "fnr_at_allow", "val_split"}
    check("thresholds: required keys", need <= set(th), f"missing {need - set(th)}")
    if need <= set(th):
        check("thresholds: 0 <= tau_allow < tau_block <= 1", 0 <= th["tau_allow"] < th["tau_block"] <= 1.0)
        check("thresholds: qhat within [0, 1]", 0 <= th["qhat"] <= 1)
        check("thresholds: fpr_at_block <= 0.5%", th["fpr_at_block"] <= 0.005 + 1e-9, str(th["fpr_at_block"]))
        check("thresholds: fnr_at_allow <= 1%", th["fnr_at_allow"] <= 0.01 + 1e-9, str(th["fnr_at_allow"]))

    spec = yaml.safe_load((art / "feature_spec.yaml").read_text(encoding="utf-8"))
    feats = spec.get("features", [])
    check("feature_spec: keys per feature", all({"name", "definition", "window", "serving_available"} <= set(f) for f in feats))
    check("feature_spec: names match serving feature order", [f["name"] for f in feats] == GBM_FEATURES)
    check("feature_spec: all serving_available", all(f["serving_available"] for f in feats))

    man = json.loads((art / "manifest.json").read_text(encoding="utf-8"))
    check("manifest: required keys", {"git_sha", "seed", "data_manifest_sha256", "packages", "files"} <= set(man))
    stale = [n for n, h in man.get("files", {}).items() if not (art / n).exists() or sha256(art / n) != h]
    check("manifest: file hashes match (re-run package_bundle.py after changing files)", not stale, str(stale))

    card = (art / "model_card.md").read_text(encoding="utf-8")
    check("model_card: required headings", all(h in card for h in CARD_HEADINGS),
          str([h for h in CARD_HEADINGS if h not in card]))

    d = pd.read_csv(ROOT / "data" / "processed" / "url_dataset" / "val.csv", low_memory=False).head(100)
    m = UrlEnsemble(art)
    p1 = np.array([m.score(u)["p"] for u in d.url])
    p2 = np.array([m.score(u)["p"] for u in d.url])
    check("smoke: 100 predictions in [0, 1] and finite", bool(np.all(np.isfinite(p1)) and p1.min() >= 0 and p1.max() <= 1))
    check("smoke: deterministic", bool(np.array_equal(p1, p2)))
    outs = [run_l1(L1In(request_id="v", url=u)) for u in d.url[:20]]
    check("smoke: run_l1 returns valid LayerOut", all(o.layer == "L1" and 0 <= o.score <= 1 and o.verdict is not None for o in outs))
    return report()


def report():
    width = max(len(n) for n, _, _ in checks)
    for n, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {n.ljust(width)}  {detail if not ok else ''}")
    bad = [n for n, ok, _ in checks if not ok]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0")
    sys.exit(main(ap.parse_args()))