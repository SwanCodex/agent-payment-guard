"""Step 9b: write feature_spec.yaml, eval_report.md and manifest.json into the artifact bundle."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from importlib import metadata
from pathlib import Path

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_l1.models.url_ensemble import GBM_FEATURES            # noqa: E402

# name -> (definition, source columns). No double quotes inside definitions.
DEFS = {
    "url_len": ("length of the URL string after scheme normalisation", "url"),
    "path_len": ("length of the URL path (at least 1: a bare host has path /)", "url"),
    "query_len": ("length of the query string", "url"),
    "n_dots": ("number of dots in the URL", "url"),
    "n_hyphens": ("number of hyphens in the URL", "url"),
    "n_digits": ("number of digit characters in the URL", "url"),
    "n_special": ("count of the characters @ ? & = _ % ~ $ ! * + , ;", "url"),
    "digit_ratio": ("digits divided by URL length", "url"),
    "entropy_path": ("Shannon entropy (bits) of the path string", "url"),
    "longest_token": ("length of the longest alphanumeric run in the URL", "url"),
    "hex_ratio_pathquery": ("share of hex characters among alphanumerics in path plus query", "url"),
    "path_depth": ("number of non-empty path segments", "url"),
    "n_query_params": ("number of &-separated query parameters", "url"),
    "has_at": ("1 if the URL contains @", "url"),
    "has_port": ("1 if the URL has an explicit port", "url"),
    "is_ip_host": ("1 if the host is an IP address", "url"),
    "has_punycode": ("1 if any host label starts with xn--", "url"),
    "is_homepage": ("1 if the path is / and there is no query", "url"),
    "double_slash_in_path": ("1 if the path contains //", "url"),
    "n_risky_host": ("count of login/verify/secure/update/account/wallet/invoice/pay in the host", "url"),
    "n_risky_path": ("count of the same words in path plus query", "url"),
    "brand_in_subdomain": ("distinct Tranco top-500 brand labels found in the subdomain part, excluding the site own label", "url"),
    "brand_in_path": ("distinct brand labels found in path plus query, excluding the site own label", "url"),
    "brand_in_domain_label": ("distinct brand labels contained in, but not equal to, the registered-domain label", "url"),
    "tld_risk": ("smoothed malicious rate of the public suffix; fit on train only (out-of-fold for training rows); unseen suffix gets the prior", "url"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                    # noqa: BLE001
        return None


def main(a):
    art = ROOT / "artifacts" / "layer1" / "url_ensemble" / a.version
    if not art.exists():
        sys.exit(f"bundle not found: {art}")

    missing = [f for f in GBM_FEATURES if f not in DEFS]
    if missing:
        sys.exit(f"no definition for features: {missing}")
    lines = [f"# Feature spec for the LightGBM part of url_ensemble {a.version}.",
             "# The LR part uses hashed char 2-5-grams of the URL (scheme and www. removed); see model_card.md.",
             "model: url_ensemble", f"version: {a.version}", "features:"]
    for f in GBM_FEATURES:
        d, src = DEFS[f]
        lines += [f"  - name: {f}", f'    definition: "{d}"', "    window: none",
                  f"    source_columns: [{src}]", "    dtype: float", "    serving_available: true"]
    (art / "feature_spec.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    parts = []
    for name in ("l1_eval_report.md", "serving_verification.md"):
        p = L1_DIR / "reports" / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8"))
        else:
            print(f"warning: reports/{name} not found")
    (art / "eval_report.md").write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8")

    pk = {}
    for n in ("scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "tldextract"):
        try:
            pk[n] = metadata.version(n)
        except Exception:                                # noqa: BLE001
            pk[n] = None
    dm = L1_DIR / "reports" / "data_manifest.json"
    manifest = {
        "model": "url_ensemble", "version": a.version, "created": str(date.today()),
        "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "seed": 42, "python": platform.python_version(), "packages": pk,
        "data_manifest_sha256": sha256(dm) if dm.exists() else None,
        "training_scripts": ["scripts/build_url_dataset.py", "scripts/collect_cc_urls.py",
                             "scripts/eda_url.py", "scripts/train_final_l1.py"],
        "files": {p.name: sha256(p) for p in sorted(art.iterdir())
                  if p.is_file() and p.name != "manifest.json"},
    }
    (art / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote feature_spec.yaml, eval_report.md, manifest.json ->", art)
    if not (art / "model_card.md").exists():
        print("reminder: model_card.md is not in the bundle yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.1.0")
    main(ap.parse_args())