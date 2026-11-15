import re
from typing import List
from .base import RuleFlag

_CONTROL = re.compile("[\u200b-\u200f\u2060\ufeff\u202a-\u202e\u2066-\u2069]")
_JOINERS = {"\u200c", "\u200d"}       # legitimate in emoji sequences, Persian, Indic scripts

_L = "A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f"   # Latin letters
_G = "\u0370-\u03ff"                                   # Greek
_C = "\u0400-\u04ff"                                   # Cyrillic
# two adjacent letters from different scripts = a mixed-script word
_MIX = re.compile(f"[{_L}][{_G}{_C}]|[{_G}{_C}][{_L}]|[{_G}][{_C}]|[{_C}][{_G}]")
_NONLATIN = re.compile(f"[{_G}{_C}]")                  # cheap prefilter for the mixed-script scan

def _ascii_alnum(c: str) -> bool:
    return c.isascii() and c.isalnum()

def check_r01(text: str) -> List[RuleFlag]:
    flags: List[RuleFlag] = []
    if not text:
        return flags

    for m in _CONTROL.finditer(text):
        i, ch = m.start(), m.group()
        if ch == "\ufeff" and i == 0:
            continue                      # byte-order mark at the start
        if ch in _JOINERS:
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not (prev and nxt and _ascii_alnum(prev) and _ascii_alnum(nxt)):
                continue                  # emoji sequence / non-Latin text
        flags.append(RuleFlag("R01", "L1:R01_HIDDEN_CONTROL_CHAR", "step_up",
                              f"U+{ord(ch):04X} at offset {i}", [i, i + 1]))
        break

    if _NONLATIN.search(text):
        m = _MIX.search(text)
        if m:
            a, b = m.start(), m.end()
            while a > 0 and text[a - 1].isalpha():
                a -= 1
            while b < len(text) and text[b].isalpha():
                b += 1
            flags.append(RuleFlag("R01", "L1:R01_MIXED_SCRIPT", "step_up",
                                  f"mixed scripts in '{text[a:b][:60]}'", [a, b]))
    return flags