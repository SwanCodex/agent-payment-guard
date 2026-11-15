"""Measure Layer 1 rule hit rates on real benign samples and write rule cards."""
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

L1_DIR = Path(__file__).resolve().parents[1]
ROOT = L1_DIR.parent
sys.path[:0] = [str(L1_DIR), str(ROOT / "contracts")]

from sentinel_contracts import L1In                      # noqa: E402
from sentinel_l1.rules import run_rules, config as C     # noqa: E402

EVAL = ROOT / "data" / "interim" / "l1_eval"
CARDS = L1_DIR / "reports" / "rule_cards"
LOCAL_HITS = ROOT / "data" / "interim" / "l1_eval_hits.json"   # git-ignored on purpose
CARDS.mkdir(parents=True, exist_ok=True)

# rule -> (what it detects, severity, threshold, row kinds it applies to)
RULES = {
    "R01": ("Zero-width/bidi control chars; mixed-script words", "step_up",
            "deterministic character classes", {"text", "url"}),
    "R02": ("Hidden text in HTML above N words", "step_up",
            f"N = {C.R02_MIN_HIDDEN_WORDS} words", {"html"}),
    "R03": ("PDF text layer vs rendered OCR divergence", "step_up",
            f"missing fraction >= {C.R03_MISSING_FRACTION}, min {C.R03_MIN_MISSING_TOKENS} tokens",
            {"pdf"}),
    "R04": ("IBAN structural validity (mod-97, country length)", "step_up",
            "deterministic", {"text"}),
    "R05": ("Payment-detail-change language", "gray", "regex set (6 patterns)", {"text"}),
    "R06": ("Lookalike domain vs protected brand list", "step_up/gray",
            f"edit distance <= {C.R06_MAX_EDIT} (brand label >= {C.R06_MIN_BRAND_LEN_EDIT} chars), homoglyph skeleton",
            {"url"}),
    "R07": ("Punycode / IDN host", "step_up (mixed script) / gray",
            "deterministic", {"url"}),
    "R08": ("Link text domain differs from href domain", "gray", "deterministic", {"html"}),
    "R09": ("Encoded blob that decodes to readable text", "gray",
            f"decoded text >= {C.R09_MIN_DECODED_CHARS} printable chars", {"text"}),
    "R10": ("URL shortener / redirect chain", "gray",
            f"shortener list; redirects > {C.R10_MAX_REDIRECTS}", {"url"}),
}

NOTES = {
    "R01": ["Joiners (U+200C/U+200D) allowed in emoji/non-Latin text; BOM at offset 0 ignored.",
            "Mixed-script = adjacent Latin/Greek/Cyrillic letters. Other scripts not covered; whole-script lookalikes are R06's job.",
            "Benign data: Enron, Amazon Reviews, Tranco hostnames."],
    "R02": ["N = 1276 words = p99.5 of hidden-text words on 1,200 real shop-like Common Crawl pages (CC-MAIN-2026-43).",
            "45.3% of pages have some hidden text; 15+ hidden words are forwarded to L2 WITHOUT a flag.",
            "Inline-style hiding only (CSS classes not resolved). p99.5 on 1,200 pages is the 6th-highest page: noisy estimate.",
            "'Shop-like' is a keyword proxy (cart/checkout phrases)."],
    "R03": ["Implemented but DISABLED (config.R03_ENABLED = False): no real invoice PDFs were available.",
            "Must be measured on real PDFs (e.g. DocILE, data-use agreement needed) before enabling."],
    "R04": ["Only strings whose first two letters are an IBAN country code are checked.",
            "Tightened after 17/10,000 benign hits (room names, reference codes). Non-IBAN payee accounts are out of scope."],
    "R05": ["0 benign hits shows the rule is quiet, not that it detects attacks: recall is not measured here.",
            "Every hit is forwarded to L2. Runs on extracted visible+hidden text, not HTML attributes."],
    "R06": ["Brand list is a 20-name seed; Step 4 swaps in Tranco top-N and this card must be re-measured.",
            "Edit distance 1 for brand labels < 8 chars, 2 for >= 8; transposition counts as 1; brand country sites excluded.",
            "Remaining benign hits reviewed manually: stirpe.co, stripw.top, paypac.in (typosquat-like), netflix.net, linked.earth."],
    "R07": ["Mixed-script IDN hosts only. Single-script IDNs are legitimate (30/20,000 hits before the change)."],
    "R08": ["Restricted to displayed domains on the protected-brand list, excluding same-owner domains.",
            "Unrestricted version: 21/1,200 benign pages (1.75%): affiliate/partner/citation links are normal on the web."],
    "R09": ["Only blobs that decode to readable text are flagged. 0 hits shows quietness, not recall.",
            "Runs on extracted text, not HTML attributes or scripts."],
    "R10": ["Shortener list + redirect count supplied by the fetcher (no network in rules).",
            "0/20,000 on a random Tranco sample is NOT informative about real traffic; the real rate comes from shadow-mode runs."],
}

def wilson_upper(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c + m) / d


def main():
    files = sorted(EVAL.glob("*.jsonl"))
    if not files:
        sys.exit(f"No samples in {EVAL}. Run prepare_samples.py first.")
    stats = defaultdict(lambda: defaultdict(lambda: {"n": 0, "hits": 0}))
    examples = defaultdict(list)
    lat = []

    for path in files:
        ds = path.stem
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if "html" in row:
                    kind = "html"
                    req = L1In(request_id="eval", url=row.get("url"), page_content=row["html"])
                else:
                    kind = "url" if "url" in row else "text"
                    req = L1In(request_id="eval", url=row.get("url"), page_content=row.get("text"))
                t0 = time.perf_counter()
                res = run_rules(req)
                lat.append((time.perf_counter() - t0) * 1000)
                hit = {}
                for fl in res.flags:
                    hit.setdefault(fl.rule_id, fl)
                for rid, info in RULES.items():
                    if kind in info[3]:
                        s = stats[rid][ds]
                        s["n"] += 1
                        if rid in hit:
                            s["hits"] += 1
                            if len(examples[rid]) < 20:
                                examples[rid].append({"dataset": ds, "evidence": hit[rid].evidence})

    lat.sort()
    p50, p95 = lat[len(lat) // 2], lat[int(len(lat) * 0.95)]
    LOCAL_HITS.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {"generated": str(date.today()), "latency_ms": {"p50": round(p50, 3), "p95": round(p95, 3)},
               "rules": {}}
    print(f"\n{'rule':5} {'n':>7} {'hits':>6} {'rate%':>7} {'upper95%':>9}")
    for rid, (title, sev, thr, _) in RULES.items():
        rows = stats.get(rid, {})
        n = sum(v["n"] for v in rows.values())
        k = sum(v["hits"] for v in rows.values())
        rate = k / n if n else float("nan")
        up = wilson_upper(k, n)
        summary["rules"][rid] = {"n": n, "hits": k, "rate": rate, "upper95": up}
        print(f"{rid:5} {n:7d} {k:6d} {rate*100:7.3f} {up*100:9.3f}")

        L = [f"# Rule card {rid}", "",
             f"- Detects: {title}", f"- Severity: {sev}", f"- Threshold: {thr}",
             f"- Measured on: {date.today()}", ""]
        if n == 0:
            L += ["## Benign hit rate", "", "**NOT MEASURED YET. This rule must not ship.**", ""]
        else:
            status = "OK" if up <= 0.01 else "REVIEW (upper bound above 1%)"
            L += ["## Benign hit rate (real data)", "",
                  "| dataset | n | hits | rate |", "|---|---|---|---|"]
            for ds, v in rows.items():
                L.append(f"| {ds} | {v['n']} | {v['hits']} | {v['hits']/v['n']*100:.3f}% |")
            L += ["", f"- Total: {k}/{n} = {rate*100:.3f}% (95% upper bound {up*100:.3f}%)",
                  f"- Status: **{status}**",
                  "- Hit examples are kept locally only (real emails), not committed.", ""]
        if "url" in RULES[rid][3] and rid != "R01":
            L += ["## Labelled precision/recall", "", "Pending (needs PhishTank data, Step 4).", ""]
        if rid in NOTES:
            L += ["## Notes and limitations", ""] + [f"- {x}" for x in NOTES[rid]] + [""]
        (CARDS / f"{rid}.md").write_text("\n".join(L), encoding="utf-8")

    (CARDS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nrules latency  p50={p50:.3f} ms  p95={p95:.3f} ms  (budget: p95 <= 5 ms)")
    print(f"cards -> {CARDS}\nlocal hit examples -> {LOCAL_HITS}")


if __name__ == "__main__":
    main()