import base64
import re
from typing import List, Optional, Tuple
from urllib.parse import unquote
from .base import RuleFlag
from . import config

_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_HEX = re.compile(r"\b(?:[0-9a-fA-F]{2}){20,}\b")
_URLENC = re.compile(r"(?:%[0-9a-fA-F]{2}){8,}")

def _printable(s: str) -> bool:
    if len(s) < config.R09_MIN_DECODED_CHARS:
        return False
    ok = sum(1 for c in s if c.isprintable() or c.isspace())
    return ok / len(s) >= 0.9

def _try_b64(s: str) -> Optional[str]:
    try:
        raw = base64.b64decode(s + "=" * (-len(s) % 4))
        out = raw.decode("utf-8")
        return out if _printable(out) else None
    except Exception:
        return None

def _try_hex(s: str) -> Optional[str]:
    try:
        out = bytes.fromhex(s).decode("utf-8")
        return out if _printable(out) else None
    except Exception:
        return None

def _try_url(s: str) -> Optional[str]:
    out = unquote(s)
    return out if _printable(out) else None

def check_r09(text: str) -> Tuple[List[RuleFlag], List[dict]]:
    flags, fwd = [], []
    for kind, rx, fn in (("base64", _B64, _try_b64), ("hex", _HEX, _try_hex),
                         ("urlenc", _URLENC, _try_url)):
        for m in rx.finditer(text or ""):
            decoded = fn(m.group())
            if decoded:
                flags.append(RuleFlag("R09", "L1:R09_ENCODED_BLOB", "gray",
                                      f"{kind} blob decodes to text: '{decoded[:120]}'",
                                      [m.start(), m.end()]))
                fwd.append({"source": "R09", "text": decoded[:2000]})
                break
    return flags, fwd