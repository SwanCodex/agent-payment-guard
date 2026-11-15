import re
from typing import List, Optional
from .base import RuleFlag

# Verify this table against the official ISO 13616 registry before final use.
IBAN_LEN = {
 "AL":28,"AD":24,"AT":20,"AZ":28,"BH":22,"BY":28,"BE":16,"BA":20,"BR":29,"BG":22,
 "CR":22,"HR":21,"CY":28,"CZ":24,"DK":18,"DO":28,"EG":29,"SV":28,"EE":20,"FO":18,
 "FI":18,"FR":27,"GE":22,"DE":22,"GI":23,"GR":27,"GL":18,"GT":28,"HU":28,"IS":26,
 "IQ":23,"IE":22,"IL":23,"IT":27,"JO":30,"KZ":20,"XK":20,"KW":30,"LV":21,"LB":28,
 "LI":21,"LT":20,"LU":20,"MT":31,"MR":27,"MU":30,"MD":24,"MC":27,"ME":22,"NL":18,
 "MK":19,"NO":15,"PK":24,"PS":29,"PL":28,"PT":25,"QA":29,"RO":24,"LC":32,"SM":27,
 "SA":24,"RS":22,"SC":31,"SK":24,"SI":19,"ES":24,"SE":24,"CH":21,"TL":23,"TN":24,
 "TR":26,"UA":29,"AE":23,"GB":22,"VA":22,"VG":24,
}
_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")

def _compact(raw: str) -> str:
    return raw.replace(" ", "").upper()

def has_iban_country(raw: str) -> bool:
    return _compact(raw)[:2] in IBAN_LEN

def iban_valid(raw: str) -> bool:
    s = _compact(raw)
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", s):
        return False
    if IBAN_LEN.get(s[:2]) != len(s):
        return False
    rearranged = s[4:] + s[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)  # A=10 ... Z=35
    return int(digits) % 97 == 1

def check_r04(text: Optional[str], payee_account: Optional[str]) -> List[RuleFlag]:
    flags: List[RuleFlag] = []
    p = (payee_account or "").strip()
    # only IBAN-shaped payees: non-IBAN accounts (e.g. US numbers) are out of scope
    if p and has_iban_country(p) and not iban_valid(p):
        flags.append(RuleFlag("R04", "L1:R04_IBAN_INVALID", "step_up",
                              f"payee account fails IBAN check: {p[:40]}"))
    for m in _CANDIDATE.finditer(text or ""):
        if has_iban_country(m.group()) and not iban_valid(m.group()):
            flags.append(RuleFlag("R04", "L1:R04_IBAN_INVALID", "step_up",
                                  f"IBAN-like string fails check: {m.group()[:40]}",
                                  [m.start(), m.end()]))
            break
    return flags