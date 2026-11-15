import re
from typing import List
from .base import RuleFlag
from ._domain import host_of, split_host
from ._html import parse_html
from .r06_lookalike import BRANDS

_DOMAINISH = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)", re.I)

# Registered domains of the protected brands (same list R06 uses; Step 4 swaps in Tranco top-N).
_BRAND_REG = {split_host(b)[2] for b in BRANDS}

# Domains owned by the same company: a link from brand text to these is normal.
_SAME_OWNER = {
    "amazon.com": {"amzn.to", "a.co"},
    "google.com": {"goo.gl", "g.co", "youtube.com", "youtu.be"},
    "facebook.com": {"fb.com", "fb.me", "instagram.com", "whatsapp.com"},
    "instagram.com": {"instagr.am", "facebook.com"},
    "linkedin.com": {"lnkd.in", "microsoft.com"},
    "paypal.com": {"paypal.me"},
}

def check_r08(src) -> List[RuleFlag]:
    """src: a ParsedHTML (preferred), a raw HTML string, or None."""
    if not src:
        return []
    parsed = parse_html(src) if isinstance(src, str) else src
    for href, text in parsed.anchors:
        href = href.strip()
        if not href.lower().startswith(("http://", "https://")):
            continue
        m = _DOMAINISH.match(text.strip())
        if not m:
            continue
        shown = split_host(m.group(1).lower())[2]
        real = split_host(host_of(href))[2]
        if not shown or not real or shown == real:
            continue
        if shown not in _BRAND_REG:
            continue                       # only brand impersonation is in scope
        if real in _SAME_OWNER.get(shown, ()):
            continue
        return [RuleFlag("R08", "L1:R08_LINK_TEXT_MISMATCH", "gray",
                         f"link text shows {shown} but href goes to {real}")]
    return []