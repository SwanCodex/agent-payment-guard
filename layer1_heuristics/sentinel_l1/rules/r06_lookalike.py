from typing import List, Optional
from .base import RuleFlag
from . import config
from ._domain import host_of, split_host

# Seed list. Step 4 replaces it with the Tranco top-N brand list.
BRANDS = ["paypal.com", "amazon.com", "google.com", "microsoft.com", "apple.com",
          "netflix.com", "facebook.com", "instagram.com", "linkedin.com", "dropbox.com",
          "stripe.com", "chase.com", "wellsfargo.com", "bankofamerica.com", "ebay.com",
          "walmart.com", "docusign.com", "adobe.com", "whatsapp.com", "coinbase.com"]

_HOMO = {"0": "o", "1": "l", "3": "e", "5": "s", "$": "s", "|": "l", "!": "i",
         "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
         "ӏ": "l", "ο": "o", "α": "a", "ι": "i", "ν": "v"}

# country-code suffixes often abused in phishing: same-name hits stay flagged on these
_ABUSED_CC = {"tk", "ml", "ga", "cf", "gq", "cc", "co", "ws", "to", "pw"}

def skeleton(s: str) -> str:
    s = s.lower()
    s = "".join(_HOMO.get(c, c) for c in s)
    return s.replace("rn", "m").replace("vv", "w")

def edit_distance(a: str, b: str) -> int:
    """Optimal string alignment: an adjacent swap counts as one edit."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]

def _decode_label(label: str) -> str:
    if label.startswith("xn--"):
        try:
            return label.encode("ascii").decode("idna")
        except UnicodeError:
            return label
    return label

_BRAND_SPLIT = [(split_host(b)[0], b) for b in BRANDS]   # (label, registered domain)

def check_r06(url: Optional[str], payee_domain: Optional[str]) -> List[RuleFlag]:
    flags: List[RuleFlag] = []
    for raw in (host_of(url), (payee_domain or "").lower()):
        if not raw:
            continue
        label, suffix, reg = split_host(raw)
        label = _decode_label(label)
        if not label:
            continue
        for b_label, b_reg in _BRAND_SPLIT:
            if reg == b_reg:
                break                                  # genuine brand domain
            hit = None
            if label == b_label:
                last = suffix.split(".")[-1]
                if len(last) == 2 and last not in _ABUSED_CC:
                    continue                           # the brand's own country site
                hit = "same name, different TLD"
            elif skeleton(label) == skeleton(b_label):
                hit = "homoglyph skeleton match"
            elif len(b_label) >= config.R06_MIN_BRAND_LEN_EDIT:
                limit = (config.R06_MAX_EDIT if len(b_label) >= config.R06_SHORT_LABEL_LEN
                         else config.R06_MAX_EDIT_SHORT)
                if edit_distance(label, b_label) <= limit:
                    hit = f"edit distance <= {limit}"
            if hit:
                sev = "gray" if hit.startswith("same name") else "step_up"
                flags.append(RuleFlag("R06", "L1:R06_LOOKALIKE_DOMAIN", sev,
                                      f"{reg} resembles {b_reg} ({hit})"))
                break
    return flags