"""Read-only measurement script for STEP 5: Root cause of the Google anomaly."""
import sys
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]

from sentinel_l1.models.url_ensemble import UrlEnsemble, default_dir, strip_scheme
from sentinel_l1.rules._domain import split_host, host_of

DATA = ROOT / "data" / "processed" / "url_dataset"

GENUINE_URLS = [
    "https://www.google.com/",
    "https://accounts.google.com/signin",
    "https://pay.google.com/",
    "https://www.paypal.com/signin",
    "https://www.paypal.com/myaccount/transfer",
    "https://www.microsoft.com/",
    "https://login.microsoftonline.com/",
    "https://www.apple.com/",
    "https://appleid.apple.com/",
    "https://www.amazon.com/",
    "https://www.amazon.com/gp/css/order-history",
    "https://github.com/",
    "https://github.com/login",
    "https://stripe.com/",
    "https://dashboard.stripe.com/login",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://www.netflix.com/login",
    "https://www.chase.com/",
    "https://www.dropbox.com/login",
    "https://www.linkedin.com/feed/",
]

PHISHING_URLS = [
    "http://secure-paypal-login.verify-account.example.top/signin/update.php?id=123",
    "http://paypa1.com/login",
    "http://www.paypall.com/signin",
    "http://paypal-security-verification.com/webscr/login",
    "http://google-account-recovery.phish.cc/",
    "http://accounts.google.com-security.net/signin",
    "http://drive.google.com.user-access.info/sharing",
    "http://microsoft-office-verify.com/auth",
    "http://login.microsoftonline.com.account-update.org/",
    "http://apple-id-verify.support-apple.me/",
    "http://appleid.apple.com.icloud-security.top/",
    "http://amazon-order-confirmation.suspicious.top/orders",
    "http://signin-amazon.com-security.net/ap/signin",
    "http://github-security-alert.org/login",
    "http://stripe-billing-portal.secure-pay.top/invoice",
    "http://netflix-billing-update.info/youraccount",
    "http://chase-bank-verify.online-banking.biz/secure",
    "http://dropbox-file-share.download-service.cc/dl",
    "http://linkedin-member-verify.com/login",
    "http://xn--pypal-4ve.com/signin",
]


def part_a_and_b():
    print("=" * 80)
    print("PART A: Rows whose host contains 'google' and 'paypal' by split and label")
    print("=" * 80)
    splits = {}
    for s in ["train", "val", "test"]:
        p = DATA / f"{s}.csv"
        if not p.exists():
            print(f"Missing {p}")
            continue
        df = pd.read_csv(p, low_memory=False)
        splits[s] = df

    rows = []
    for s, df in splits.items():
        hosts = df["url"].apply(host_of)
        for brand in ["google", "paypal"]:
            sub = df[hosts.str.contains(brand, case=False, na=False)]
            counts = sub["label"].value_counts().to_dict()
            rows.append({
                "split": s,
                "brand_in_host": brand,
                "benign (label=0)": counts.get(0, 0),
                "malicious (label=1)": counts.get(1, 0),
                "total": len(sub),
            })
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 80)
    print("PART B: Registered domains google.com and paypal.com location and counts")
    print("=" * 80)
    reg_rows = []
    for s, df in splits.items():
        hosts = df["url"].apply(host_of)
        reg_domains = hosts.apply(lambda h: split_host(h)[2])
        for brand_domain in ["google.com", "paypal.com"]:
            sub = df[reg_domains == brand_domain]
            counts = sub["label"].value_counts().to_dict()
            reg_rows.append({
                "split": s,
                "registered_domain": brand_domain,
                "benign (label=0)": counts.get(0, 0),
                "malicious (label=1)": counts.get(1, 0),
                "total": len(sub),
            })
    print(pd.DataFrame(reg_rows).to_string(index=False))


def part_c(m):
    print("\n" + "=" * 80)
    print("PART C: 20 Genuine Brand URLs vs 20 Lookalike/Phishing URLs")
    print("=" * 80)

    def evaluate_list(url_list, kind):
        res = []
        for u in url_list:
            sc = m.score(u)
            status = "STEP_UP" if sc["p"] >= m.tau_block else ("GRAY" if sc["gray"] else "ALLOW")
            res.append({
                "type": kind,
                "url": u[:65],
                "p": round(sc["p"], 4),
                "p_lr": round(sc["p_lr"], 4),
                "p_gbm": round(sc["p_gbm"], 4),
                "status": status,
            })
        return res

    df_gen = pd.DataFrame(evaluate_list(GENUINE_URLS, "GENUINE"))
    df_phish = pd.DataFrame(evaluate_list(PHISHING_URLS, "PHISHING"))
    combined = pd.concat([df_gen, df_phish], ignore_index=True)
    print(combined.to_string(index=False))


def part_d(m):
    print("\n" + "=" * 80)
    print("PART D: LR n-gram contributions split by host part vs path part for Google URLs")
    print("=" * 80)
    google_urls = [
        "https://www.google.com/",
        "https://www.google.com/search?q=sentinel+payment+guard",
        "https://accounts.google.com/signin/v2/identifier?service=mail&continue=https%3A%2F%2Fmail.google.com%2Fmail%2F",
        "https://pay.google.com/gp/w/u/0/home/paymentmethods",
    ]

    fh = FeatureHasher(n_features=2**20, input_type="string", alternate_sign=False)
    analyzer = m.lr.hv.build_analyzer()

    for url in google_urls:
        txt = strip_scheme(url)
        # Split txt into host and path/query
        if "/" in txt:
            host_part, path_part = txt.split("/", 1)
            path_part = "/" + path_part
        else:
            host_part, path_part = txt, ""

        X = m.lr.hv.transform([txt])
        idx, cnt = X.indices, X.data
        if len(idx) == 0:
            continue
        v = (1.0 + np.log(cnt)) * m.lr.idf[idx]
        norm = np.sqrt(float(v @ v))
        c_idx = dict(zip(idx.tolist(), (v * m.lr.coef[idx] / norm).tolist()))

        host_contrib = 0.0
        path_contrib = 0.0
        boundary_contrib = 0.0

        for g in analyzer(txt):
            i = int(fh.transform([[g]]).indices[0])
            if i in c_idx:
                val = c_idx[i]
                in_host = g in host_part
                in_path = g in path_part
                if in_host and not in_path:
                    host_contrib += val
                elif in_path and not in_host:
                    path_contrib += val
                else:
                    boundary_contrib += val

        tot = sum(c_idx.values())
        print(f"\nURL: {url}")
        print(f"  Stripped: {txt}")
        print(f"  Intercept: {m.lr.b:+.4f} | Total n-gram contrib: {tot:+.4f} | Decision: {m.lr.decision(url):+.4f}")
        print(f"  Host part: '{host_part}' -> contrib: {host_contrib:+.4f}")
        print(f"  Path part: '{path_part}' -> contrib: {path_contrib:+.4f}")
        print(f"  Spanning / boundary / ambiguous -> contrib: {boundary_contrib:+.4f}")


def main():
    m = UrlEnsemble(default_dir())
    part_a_and_b()
    part_c(m)
    part_d(m)


if __name__ == "__main__":
    main()
