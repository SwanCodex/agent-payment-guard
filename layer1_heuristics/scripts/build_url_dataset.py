"""Build the Layer 1 URL dataset: load sources, clean, dedupe, domain-grouped split."""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]
from sentinel_l1.rules._domain import split_host      # noqa: E402
from sentinel_l1.rules.r10_shortener import SHORTENERS    # noqa: E402

RAW = ROOT / "data" / "raw"
URLDIR = RAW / "url"
OUT = ROOT / "data" / "processed" / "url_dataset"
REPORT = L1_DIR / "reports" / "eda_l1"
MANIFEST = L1_DIR / "reports" / "data_manifest.json"
SEED = 42
FRACS = (0.70, 0.15, 0.15)
# PhiUSIIL: label 1 = legitimate, 0 = phishing (CONFIRM on the UCI dataset page)
PHIUSIIL_LEGIT_LABEL = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ts(x):
    try:
        t = pd.to_datetime(x, utc=True)
        return None if pd.isna(t) else t
    except Exception:
        return None


# ---------------------------------------------------------------- loaders
def load_phishtank():
    p = URLDIR / "phishtank" / "online-valid.csv"
    if not p.exists():
        return None, p
    rows = []
    with p.open(encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("verified") or "yes").lower() != "yes":
                continue
            rows.append((r["url"], 1, "phishtank", ts(r.get("submission_time"))))
    return rows, p


def load_openphish():
    p = URLDIR / "openphish" / "feed.txt"
    if not p.exists():
        return None, p
    # the feed has no per-URL timestamps: a snapshot date is NOT a first-seen date
    rows = [(l.strip(), 1, "openphish", None)
            for l in p.open(encoding="utf-8", errors="replace") if l.strip()]
    return rows, p


def load_urlhaus():
    p = URLDIR / "urlhaus" / "urlhaus.csv"
    if not p.exists():
        return None, p
    header, data = None, []
    for line in p.open(encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            cand = s.lstrip("# ").strip().lower().replace("(utc)", "").replace(" ", "")
            if "dateadded" in cand and "url" in cand and "," in cand:
                header = cand.split(",")
            continue
        data.append(s)
    if header is None:
        sys.exit("URLhaus: could not find the commented header line in " + str(p))
    i_url, i_dt = header.index("url"), header.index("dateadded")
    rows = []
    for rec in csv.reader(data):
        if len(rec) > max(i_url, i_dt):
            rows.append((rec[i_url], 1, "urlhaus", ts(rec[i_dt])))
    return rows, p


def load_phiusiil():
    p = URLDIR / "phiusiil" / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if not p.exists():
        return None, p
    df = pd.read_csv(p, usecols=["URL", "label"])
    rows = [(u, 0 if int(l) == PHIUSIIL_LEGIT_LABEL else 1, "phiusiil", None)
            for u, l in zip(df["URL"], df["label"])]
    return rows, p


def load_tranco(n, rnd):
    p = RAW / "tranco" / "top-1m.csv"
    if not p.exists():
        return None, p
    doms = []
    for line in p.open(encoding="utf-8"):
        _, _, d = line.strip().partition(",")
        if d:
            doms.append(d)
    doms = rnd.sample(doms, min(n, len(doms)))
    return [(f"https://{d}/", 0, "tranco", None) for d in doms], p


def load_cc_urls():
    out, files = [], list((ROOT / "data" / "interim" / "l1_eval").glob("commoncrawl*.jsonl"))
    for p in files:
        for line in p.open(encoding="utf-8"):
            u = json.loads(line).get("url")
            if u:
                out.append((u, 0, "commoncrawl", None))
    return (out or None), (files[0] if files else None)


# ---------------------------------------------------------------- cleaning
def normalize(u):
    u = (u or "").strip()
    if not u or any(c.isspace() for c in u) or len(u) > 2048:
        return None
    if "://" not in u:
        u = "http://" + u
    try:
        s = urlsplit(u)
        host = (s.hostname or "").lower()
        if s.scheme.lower() not in ("http", "https") or not host:
            return None
        netloc = host + (f":{s.port}" if s.port else "")
    except ValueError:
        return None
    return urlunsplit((s.scheme.lower(), netloc, s.path or "/", s.query, "")), host


_DIG = re.compile(r"[0-9]+")
_LONG = re.compile(r"[A-Za-z0-9_-]{16,}")


def path_skeleton(url):
    segs = []
    for seg in urlsplit(url).path.split("/"):
        seg = _LONG.sub("X", seg)
        segs.append(_DIG.sub("0", seg))
    return "/".join(segs)


def reg_domain(host):
    reg = split_host(host)[2]
    return reg or host            # IP hosts / no public suffix: the host itself


# ---------------------------------------------------------------- splitting
def assign_splits(df, rnd):
    g = df.groupby("reg_domain").agg(n=("url", "size"), t=("first_seen", "min")).reset_index()
    stamped = g[g["t"].notna()].sort_values("t")
    unstamped = g[g["t"].isna()]
    assign = {}
    # timestamped domains: ordered by earliest first-seen, cut at row-count fractions
    total, run = stamped["n"].sum(), 0
    c1, c2 = FRACS[0], FRACS[0] + FRACS[1]
    for d, n in zip(stamped["reg_domain"], stamped["n"]):
        frac = run / total if total else 0
        assign[d] = "train" if frac < c1 else ("val" if frac < c2 else "test")
        run += n
    # domains without a real first-seen date: seeded random, same proportions
    for d in unstamped["reg_domain"]:
        r = rnd.random()
        assign[d] = "train" if r < c1 else ("val" if r < c2 else "test")
    df["split"] = df["reg_domain"].map(assign)
    return df


def main(a):
    rnd = random.Random(SEED)
    loaders = [("phishtank", load_phishtank), ("openphish", load_openphish),
               ("urlhaus", load_urlhaus), ("phiusiil", load_phiusiil),
               ("tranco", lambda: load_tranco(a.tranco_n, rnd)), ("commoncrawl", load_cc_urls)]
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    recs = []
    skip = {x.strip() for x in a.skip.split(",") if x.strip()}
    for name, fn in loaders:
        if name in skip:
            print(f"[skip] {name}: excluded with --skip")
            continue
        rows, path = fn()
        if rows is None:
            print(f"[skip] {name}: file not found ({path})")
            continue
        if len(rows) > a.cap and name not in ("tranco", "commoncrawl", "phiusiil"):
            rows = rnd.sample(rows, a.cap)
        print(f"[load] {name}: {len(rows)} rows")
        man[f"url_{name}"] = {"name": f"url_{name}", "file": str(path.name), "rows_loaded": len(rows),
                              "sha256": sha256(path), "date": str(date.today()), "seed": SEED}
        recs += [(*r, name) for r in rows]

    if not recs:
        sys.exit("No source files found under data/raw. See the download instructions.")

    # normalize
    clean, dropped = [], 0
    for url, label, source, t, _ in recs:
        n = normalize(url)
        if n is None:
            dropped += 1
            continue
        clean.append({"url": n[0], "host": n[1], "label": label, "source": source, "first_seen": t})
    df = pd.DataFrame(clean)
    print(f"normalized: {len(df)} kept, {dropped} dropped (invalid)")

    # exact duplicates: conflicting labels -> drop both; else keep earliest
    nlab = df.groupby("url")["label"].nunique()
    conflict = set(nlab[nlab > 1].index)
    df = df[~df["url"].isin(conflict)]
    df = df.sort_values("first_seen", na_position="last").drop_duplicates("url", keep="first")
    print(f"exact-duplicate removal: {len(df)} left ({len(conflict)} conflicting URLs dropped)")

    # template duplicates: same registered domain + path skeleton + label
    df["reg_domain"] = df["host"].map(reg_domain)
    n_before = len(df)
    df = df[~df["reg_domain"].isin(SHORTENERS)]
    print(f"URL-shortener hosts removed: {n_before - len(df)} rows")
    df["skel"] = df["url"].map(path_skeleton)
    df = df.drop_duplicates(["reg_domain", "skel", "label"], keep="first")
    print(f"template-duplicate removal: {len(df)} left")

    df = assign_splits(df.drop(columns=["skel"]), rnd).reset_index(drop=True)

    # hard guarantee: no registered domain in more than one split
    assert (df.groupby("reg_domain")["split"].nunique() == 1).all(), "domain leaked across splits"

    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    cols = ["url", "label", "source", "first_seen", "reg_domain", "split"]
    for s in ("train", "val", "test"):
        df[df["split"] == s][cols].to_csv(OUT / f"{s}.csv", index=False)

    lines = ["# Layer 1 URL dataset: split summary", "", f"Built: {date.today()}  seed={SEED}", "",
             "## Rows by source and label", "", df.groupby(["source", "label"]).size().unstack(fill_value=0).to_markdown(), "",
             "## Rows by split and label", "", df.groupby(["split", "label"]).size().unstack(fill_value=0).to_markdown(), "",
             "## First-seen range of timestamped rows per split", ""]
    t = df[df["first_seen"].notna()].groupby("split")["first_seen"].agg(["min", "max", "count"])
    lines += [t.to_markdown() if len(t) else "(no timestamped rows)", "",
              "## Checks", "", "- registered domains in more than one split: **0** (asserted)",
              f"- total rows: {len(df)}, registered domains: {df['reg_domain'].nunique()}", "",
              "## Known limitations", "",
              "- Only PhishTank and URLhaus carry per-URL first-seen dates; those domains are split in time order.",
              "- Benign rows (Tranco, Common Crawl, PhiUSIIL) have no first-seen date and are split randomly by domain,"
              " so the time split covers malicious rows only.",
              "- OpenPhish is a snapshot (no first-seen dates).", ""]
    (REPORT / "split_summary.md").write_text("\n".join(lines), encoding="utf-8")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print("\n".join(lines[:22]))
    print(f"\nwrote {OUT}  and  {REPORT / 'split_summary.md'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=60000, help="max rows per malicious feed")
    ap.add_argument("--tranco-n", type=int, default=100000)
    ap.add_argument("--skip", default="", help="comma-separated source names to leave out")
    main(ap.parse_args())