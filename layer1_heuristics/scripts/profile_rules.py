"""Per-rule latency on real HTML pages."""
import json
import sys
import time
from pathlib import Path

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]

from sentinel_contracts import L1In                                  # noqa: E402
from sentinel_l1.rules import run_rules                              # noqa: E402
from sentinel_l1.rules.r01_unicode import check_r01                  # noqa: E402
from sentinel_l1.rules.r02_hidden_text import check_r02              # noqa: E402
from sentinel_l1.rules.r04_iban import check_r04                     # noqa: E402
from sentinel_l1.rules.r05_payment_change import check_r05           # noqa: E402
from sentinel_l1.rules.r06_lookalike import check_r06                # noqa: E402
from sentinel_l1.rules.r07_punycode import check_r07                 # noqa: E402
from sentinel_l1.rules.r08_link_mismatch import check_r08            # noqa: E402
from sentinel_l1.rules.r09_encoded_blobs import check_r09            # noqa: E402
from sentinel_l1.rules.r10_shortener import check_r10                # noqa: E402

RULES = {
    "R01": lambda h, u: check_r01(h),
    "R02": lambda h, u: check_r02(h),
    "R04": lambda h, u: check_r04(h, None),
    "R05": lambda h, u: check_r05(h),
    "R06": lambda h, u: check_r06(u, None),
    "R07": lambda h, u: check_r07(u),
    "R08": lambda h, u: check_r08(h),
    "R09": lambda h, u: check_r09(h),
    "R10": lambda h, u: check_r10(u, None),
}

rows = [json.loads(l) for l in (ROOT / "data/interim/l1_eval/commoncrawl_html.jsonl")
        .open(encoding="utf-8")]
times = {k: [] for k in RULES}
total = []
sizes = sorted(len(r["html"]) for r in rows)
for r in rows:
    h, u = r["html"], r["url"]
    for k, fn in RULES.items():
        t0 = time.perf_counter()
        fn(h, u)
        times[k].append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    run_rules(L1In(request_id="p", url=u, page_content=h))
    total.append((time.perf_counter() - t0) * 1000)

def p(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]

print(f"pages {len(rows)}, html size median {sizes[len(sizes)//2]} chars, p95 {sizes[int(.95*len(sizes))]}")
print(f"{'rule':5} {'p50 ms':>8} {'p95 ms':>8}")
for k, v in times.items():
    print(f"{k:5} {p(v,.5):8.3f} {p(v,.95):8.3f}")
print(f"{'ALL':5} {p(total,.5):8.3f} {p(total,.95):8.3f}   (run_rules, HTML pages only)")