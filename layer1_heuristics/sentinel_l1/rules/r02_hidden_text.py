from typing import List, Optional, Tuple
from .base import RuleFlag
from . import config
from ._html import ParsedHTML, parse_html

def hidden_text(html: Optional[str]) -> str:
    """Used by scripts/calibrate_r02.py."""
    return parse_html(html).hidden if html else ""

def check_r02(src) -> Tuple[List[RuleFlag], List[dict]]:
    """src: a ParsedHTML (preferred), a raw HTML string, or None."""
    if isinstance(src, str):
        src = parse_html(src)
    text = src.hidden if src else ""
    n = len(text.split())
    flags: List[RuleFlag] = []
    fwd: List[dict] = []
    if n >= config.R02_FORWARD_MIN_WORDS:
        fwd.append({"source": "R02", "text": text[:8000]})
    if n >= config.R02_MIN_HIDDEN_WORDS:
        flags.append(RuleFlag("R02", "L1:R02_HIDDEN_TEXT", "step_up",
                              f"{n} words of hidden text: '{text[:150]}'"))
    return flags, fwd