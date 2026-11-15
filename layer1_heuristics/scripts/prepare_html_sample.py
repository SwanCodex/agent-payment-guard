"""Sample real e-commerce-like HTML pages straight from Common Crawl WARC files."""
import argparse
import gzip
import json
import random
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests
from warcio.archiveiterator import ArchiveIterator

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
OUT = ROOT / "data" / "interim" / "l1_eval"
MANIFEST = L1_DIR / "reports" / "data_manifest.json"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "sentinel-l1-research/0.1 (student project; contact: "+ os.environ.get("CC_CONTACT", "unset") + ")"}
SEED = 42
SHOP_WORDS = ("add to cart", "add-to-cart", "add to basket", "add to bag", "buy now", "checkout")
MAX_PER_HOST = 3


def newest_crawl():
    r = requests.get("https://index.commoncrawl.org/collinfo.json", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()[0]["id"]


def warc_paths(crawl_id):
    url = f"https://data.commoncrawl.org/crawl-data/{crawl_id}/warc.paths.gz"
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    return gzip.decompress(r.content).decode().split()


def sample_file(path, per_file, max_mb, shop_only, rnd, host_counts, out):
    got = 0
    url = "https://data.commoncrawl.org/" + path
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        try:
            for rec in ArchiveIterator(r.raw):
                if rec.rec_type != "response" or rec.http_headers is None:
                    continue
                if rec.http_headers.get_statuscode() != "200":
                    continue
                if "html" not in (rec.http_headers.get_header("Content-Type") or "").lower():
                    continue
                raw = rec.content_stream().read()
                if len(raw) < 2000:
                    continue
                html = raw[:400_000].decode("utf-8", errors="replace")
                low = html.lower()
                shop = any(w in low for w in SHOP_WORDS)
                if shop_only and not shop:
                    continue
                if rnd.random() > 0.5:
                    continue
                uri = rec.rec_headers.get_header("WARC-Target-URI") or ""
                host = urlparse(uri).hostname or ""
                if host_counts.get(host, 0) >= MAX_PER_HOST:
                    continue
                host_counts[host] = host_counts.get(host, 0) + 1
                out.write(json.dumps({"url": uri, "html": html, "shop_like": shop},
                                     ensure_ascii=False) + "\n")
                out.flush()
                got += 1
                if got >= per_file or r.raw.tell() > max_mb * 1_000_000:
                    break
        except Exception as e:
            print("  stopped early on this file:", type(e).__name__, e)
    return got


def main(files, per_file, max_mb, shop_only):
    crawl_id = newest_crawl()
    paths = warc_paths(crawl_id)
    print(f"crawl {crawl_id}: {len(paths)} WARC files available")
    rnd = random.Random(SEED)
    chosen = rnd.sample(paths, files)
    host_counts, total = {}, 0
    path = OUT / "commoncrawl_html.jsonl"
    try:
        with path.open("w", encoding="utf-8") as out:
            for i, p in enumerate(chosen, 1):
                n = sample_file(p, per_file, max_mb, shop_only, rnd, host_counts, out)
                total += n
                print(f"file {i}/{files}: +{n} pages (total {total})")
    finally:
        data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        data["commoncrawl_html"] = {
            "name": "commoncrawl_html", "source": "Common Crawl WARC files",
            "crawl": crawl_id, "date": str(date.today()), "warc_files": chosen,
            "rows_sampled": total, "max_pages_per_host": MAX_PER_HOST,
            "selection": "HTML 200 responses; shop-like = contains cart/checkout keywords",
            "shop_only": shop_only, "seed": SEED}
        MANIFEST.write_text(json.dumps(data, indent=2))
        print(f"saved {total} pages -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--per-file", type=int, default=150)
    ap.add_argument("--max-mb", type=int, default=200, help="max download per WARC file")
    ap.add_argument("--any-page", action="store_true", help="keep non-shop pages too")
    a = ap.parse_args()
    main(a.files, a.per_file, a.max_mb, not a.any_page)