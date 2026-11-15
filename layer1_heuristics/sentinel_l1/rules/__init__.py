import re
from sentinel_contracts import L1In
from .base import RuleFlag, RuleResult
from ._html import parse_html
from .r01_unicode import check_r01
from .r02_hidden_text import check_r02
from .r03_pdf_divergence import check_r03
from .r04_iban import check_r04
from .r05_payment_change import check_r05
from .r06_lookalike import check_r06
from .r07_punycode import check_r07
from .r08_link_mismatch import check_r08
from .r09_encoded_blobs import check_r09
from .r10_shortener import check_r10

_HTML = re.compile(r"<\s*(html|body|div|a|p|span|script|style)\b", re.I)

def run_rules(req: L1In) -> RuleResult:
    res = RuleResult()
    raw = req.page_content or ""
    payload = req.payload or {}
    html = payload.get("html") or (raw if _HTML.search(raw) else None)
    parsed = parse_html(html) if html else None      # parse ONCE

    # text rules run on extracted text (visible + hidden), not on raw markup/scripts
    if parsed is not None:
        text = parsed.visible + "\n" + parsed.hidden
        if raw and html is not raw:
            text = raw + "\n" + text
    else:
        text = raw

    def add(out):
        flags, fwd = out
        res.flags += flags
        res.forwarded += fwd

    res.flags += check_r01(raw)                      # cheap, scans raw (incl. attributes)
    if html and html is not raw:
        res.flags += check_r01(html)
    res.flags += check_r01(req.url or "")
    add(check_r02(parsed))
    add(check_r03(payload.get("pdf_b64")))
    res.flags += check_r04(text, payload.get("payee_account"))
    add(check_r05(text))
    res.flags += check_r06(req.url, payload.get("payee_domain"))
    res.flags += check_r07(req.url)
    res.flags += check_r08(parsed)
    add(check_r09(text))
    res.flags += check_r10(req.url, payload.get("redirect_count"))
    return res