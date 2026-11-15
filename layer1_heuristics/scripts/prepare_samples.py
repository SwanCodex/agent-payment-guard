"""Build small real-data evaluation samples for Layer 1 rule measurement."""
import argparse
import hashlib
import json
import random
import tarfile
from datetime import date
from pathlib import Path

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "interim" / "l1_eval"
MANIFEST = L1_DIR / "reports" / "data_manifest.json"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
SEED = 42


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(entry):
    data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    data[entry["name"]] = entry
    MANIFEST.write_text(json.dumps(data, indent=2))


def write_jsonl(name, rows):
    p = OUT / f"{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows -> {p}")


def prep_tranco(n):
    src = RAW / "tranco" / "top-1m.csv"
    domains = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            _, _, d = line.strip().partition(",")
            if d:
                domains.append(d)
    sample = random.Random(SEED).sample(domains, min(n, len(domains)))
    write_jsonl("tranco", [{"url": f"https://{d}/"} for d in sample])
    record({"name": "tranco", "source": "https://tranco-list.eu/",
            "date": str(date.today()), "rows_total": len(domains),
            "rows_sampled": len(sample), "sha256": sha256(src), "seed": SEED})


def prep_enron(n, total_guess=517_000):
    src = RAW / "enron" / "enron_mail_20150507.tar.gz"
    rnd = random.Random(SEED)
    p = min(1.0, n / total_guess * 1.2)
    rows = []
    with tarfile.open(src, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or rnd.random() > p:
                continue
            raw = tf.extractfile(m).read().decode("latin-1", errors="replace")
            raw = raw.replace("\r\n", "\n")
            _, _, body = raw.partition("\n\n")
            body = body.strip()[:5000]
            if len(body.split()) >= 5:
                rows.append({"text": body})
    rnd.shuffle(rows)
    rows = rows[:n]
    write_jsonl("enron", rows)
    record({"name": "enron", "source": "https://www.cs.cmu.edu/~enron/",
            "date": str(date.today()), "rows_sampled": len(rows),
            "sha256": sha256(src), "seed": SEED})


def prep_amazon(n, category):
    from datasets import load_dataset
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{category}",
                      split="full", streaming=True, trust_remote_code=True)
    ds = ds.shuffle(seed=SEED, buffer_size=10_000)
    rows = []
    for ex in ds:
        t = (ex.get("text") or "").strip()
        if len(t.split()) >= 5:
            rows.append({"text": t[:5000]})
        if len(rows) >= n:
            break
    write_jsonl("amazon", rows)
    record({"name": "amazon", "source": "McAuley-Lab/Amazon-Reviews-2023 (Hugging Face)",
            "category": category, "date": str(date.today()),
            "rows_sampled": len(rows), "seed": SEED})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tranco", "enron", "amazon"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--category", default="All_Beauty")
    a = ap.parse_args()
    if a.what == "tranco":
        prep_tranco(a.n)
    elif a.what == "enron":
        prep_enron(a.n)
    else:
        prep_amazon(a.n, a.category)