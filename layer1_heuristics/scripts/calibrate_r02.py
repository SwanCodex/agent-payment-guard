"""Compute hidden-text word-count percentiles on real benign pages."""
import json
import sys
from pathlib import Path

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]

from sentinel_l1.rules.r02_hidden_text import hidden_text    # noqa: E402
from sentinel_l1.rules import config as C                    # noqa: E402

src = ROOT / "data" / "interim" / "l1_eval" / "commoncrawl_html.jsonl"
counts = []
with src.open(encoding="utf-8") as f:
    for line in f:
        counts.append(len(hidden_text(json.loads(line)["html"]).split()))
if not counts:
    sys.exit("no pages found")
counts.sort()
n = len(counts)

def pct(q):
    return counts[min(n - 1, int(q * n))]

print(f"pages: {n}")
print(f"pages with any hidden text: {sum(c > 0 for c in counts)/n*100:.1f}%")
for q in (0.50, 0.90, 0.95, 0.99, 0.995):
    print(f"p{q*100:g}: {pct(q)} words")
print(f"max: {counts[-1]} words")
print(f"share >= current N ({C.R02_MIN_HIDDEN_WORDS}): {sum(c >= C.R02_MIN_HIDDEN_WORDS for c in counts)/n*100:.2f}%")