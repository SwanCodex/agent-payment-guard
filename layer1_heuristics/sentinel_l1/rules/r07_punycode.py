from typing import List, Optional
from urllib.parse import urlparse
import unicodedata
from .base import RuleFlag

def _scripts(s: str) -> set:
    out = set()
    for c in s:
        if c.isalpha():
            try:
                out.add(unicodedata.name(c).split()[0])
            except ValueError:
                pass
    return out & {"LATIN", "CYRILLIC", "GREEK"}

def check_r07(url: Optional[str]) -> List[RuleFlag]:
    """Flag IDN hosts that mix Latin/Cyrillic/Greek in one label (handbook: mixed-script).
    Single-script IDNs are legitimate; whole-script lookalikes are caught by R06."""
    if not url:
        return []
    host = (urlparse(url if "://" in url else "//" + url).hostname or "").lower()
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                decoded = label.encode("ascii").decode("idna")
            except UnicodeError:
                continue
            if len(_scripts(decoded)) > 1:
                return [RuleFlag("R07", "L1:R07_PUNYCODE_HOST", "step_up",
                                 f"mixed-script host label '{label}' decodes to '{decoded}'")]
    return []