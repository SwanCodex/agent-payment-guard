"""Collect real benign URLs (HTML 200 responses) from Common Crawl WARC files.
Stores URLs only, no page bodies. Deep-path benign URLs for the Layer 1 URL model."""
import argparse
import gzip
import json
import random
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import requests
from warcio.archiveiterator import ArchiveIterator

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
OUT = ROOT / "data" / "interim" / "l1_eval"
MANIFEST = L1_DIR / "reports" / "data_manifest.json"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "sentinel-l1-research/0.1 (student project; contact: "+ os.environ.get("CC_CONTACT", "unset") + ")"}
SEED = 7
MAX_PER_HOST = 3


def newest_crawl():
    r = requests.get("https://index.commoncrawl.org/collinfo.json", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()[0]["id"]


def warc_paths(crawl_id):
    r = requests.get(f"https://data.commoncrawl.org/crawl-data/{crawl_id}/warc.paths.gz",
                     headers=UA, timeout=120)
    r.raise_for_status()
    return gzip.decompress(r.content).decode().split()


def iter_html_urls(stream, stop=None):
    for rec in ArchiveIterator(stream):
        if stop and stop():
            return
        if rec.rec_type != "response" or rec.http_headers is None:
            continue
        if rec.http_headers.get_statuscode() != "200":
            continue
        if "html" not in (rec.http_headers.get_header("Content-Type") or "").lower():
            continue
        uri = rec.rec_headers.get_header("WARC-Target-URI")
        if uri:
            yield uri


def take_urls(stream, host_counts, per_file, out, stop=None):
    got = 0
    try:
        for uri in iter_html_urls(stream, stop):
            host = urlsplit(uri).hostname or ""
            if not host or host_counts.get(host, 0) >= MAX_PER_HOST:
                continue
            host_counts[host] = host_counts.get(host, 0) + 1
            out.write(json.dumps({"url": uri}) + "\n")
            got += 1
            if got >= per_file:
                break
    except Exception as e:
        print("  stopped early on this file:", type(e).__name__, e)
    return got


def main(files, per_file, max_mb):
    crawl_id = newest_crawl()
    paths = warc_paths(crawl_id)
    chosen = random.Random(SEED).sample(paths, files)
    print(f"crawl {crawl_id}: {len(paths)} WARC files, using {files}")
    host_counts, total = {}, 0
    path = OUT / "commoncrawl_urls.jsonl"
    try:
        with path.open("w", encoding="utf-8") as out:
            for i, p in enumerate(chosen, 1):
                with requests.get("https://data.commoncrawl.org/" + p, headers=UA,
                                  stream=True, timeout=120) as r:
                    r.raise_for_status()
                    stop = lambda: r.raw.tell() > max_mb * 1_000_000      # noqa: E731
                    n = take_urls(r.raw, host_counts, per_file, out, stop)
                out.flush()
                total += n
                print(f"file {i}/{files}: +{n} urls (total {total}, hosts {len(host_counts)})")
    finally:
        data = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        data["commoncrawl_urls"] = {
            "name": "commoncrawl_urls", "source": "Common Crawl WARC files", "crawl": crawl_id,
            "date": str(date.today()), "warc_files": chosen, "rows_sampled": total,
            "max_urls_per_host": MAX_PER_HOST, "selection": "HTTP 200 text/html responses, URL only",
            "seed": SEED}
        MANIFEST.write_text(json.dumps(data, indent=2))
        print(f"saved {total} urls -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=16)
    ap.add_argument("--per-file", type=int, default=12000)
    ap.add_argument("--max-mb", type=int, default=120, help="max download per WARC file")
    a = ap.parse_args()
    main(a.files, a.per_file, a.max_mb)