import re
from typing import List, Tuple
from .base import RuleFlag

_PATTERNS = [
    r"new\s+bank\s+(?:details|account)",
    r"(?:bank|account|payment)\s+details\s+(?:have\s+)?(?:changed|been\s+updated|updated)",
    r"account\s+(?:number\s+)?has\s+changed",
    r"updated\s+(?:bank|payment|wire)\s+instructions",
    r"change\s+(?:of|in)\s+bank",
    r"please\s+(?:update|use)\s+(?:the\s+)?(?:new|following)\s+(?:bank|account)",
]
_RX = re.compile("|".join(_PATTERNS), re.I)

def check_r05(text: str) -> Tuple[List[RuleFlag], List[dict]]:
    flags, fwd = [], []
    m = _RX.search(text or "")
    if m:
        a, b = max(0, m.start() - 100), min(len(text), m.end() + 100)
        flags.append(RuleFlag("R05", "L1:R05_PAYMENT_DETAIL_CHANGE", "gray",
                              f"payment-change phrase: '{m.group()}'", [m.start(), m.end()]))
        fwd.append({"source": "R05", "text": text[a:b]})
    return flags, fwd