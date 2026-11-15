from typing import List, Optional
from .base import RuleFlag
from . import config
from ._domain import host_of, split_host

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
              "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy"}

def check_r10(url: Optional[str], redirect_count: Optional[int]) -> List[RuleFlag]:
    flags: List[RuleFlag] = []
    host = host_of(url)
    if host and split_host(host)[2] in SHORTENERS:
        flags.append(RuleFlag("R10", "L1:R10_URL_SHORTENER", "gray", f"shortener host: {host}"))
    if redirect_count is not None and redirect_count > config.R10_MAX_REDIRECTS:
        flags.append(RuleFlag("R10", "L1:R10_LONG_REDIRECT_CHAIN", "gray",
                              f"{redirect_count} redirects"))
    return flags