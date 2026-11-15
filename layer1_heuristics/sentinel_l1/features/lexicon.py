"""Brand / risky-word features. Brand list comes from Tranco top-N and is saved with the model."""
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urlsplit

from ..rules._domain import split_host

RISKY_WORDS = ("login", "verify", "secure", "update", "account", "wallet", "invoice", "pay")
_RISKY = re.compile("|".join(RISKY_WORDS))
LEXICON_FEATURES = ["n_risky_host", "n_risky_path", "brand_in_subdomain",
                    "brand_in_path", "brand_in_domain_label"]


def make_brand_list(tranco_csv: Path, top: int = 500, min_len: int = 4) -> List[str]:
    labels = set()
    with Path(tranco_csv).open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= top:
                break
            _, _, d = line.strip().partition(",")
            label = split_host(d.lower())[0] if d else ""
            if len(label) >= min_len and label.isalnum():
                labels.add(label)
    return sorted(labels)


def save_brands(brands: Iterable[str], path: Path) -> None:
    Path(path).write_text(json.dumps(sorted(set(brands))), encoding="utf-8")


def load_brands(path: Path) -> List[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Lexicon:
    def __init__(self, brands: Iterable[str]):
        self.brands = sorted(set(brands), key=len, reverse=True)
        self.rx = re.compile("|".join(map(re.escape, self.brands))) if self.brands else None

    def features(self, url: str) -> Dict[str, float]:
        u = (url or "").strip().lower()
        if "://" not in u:
            u = "http://" + u
        try:
            s = urlsplit(u)
            host = s.hostname or ""
        except ValueError:
            s, host = urlsplit("http://invalid/"), ""
        label, _suffix, reg = split_host(host) if host else ("", "", "")
        sub = host[: -len(reg)] if reg and host.endswith(reg) else ""
        pathq = (s.path or "") + "?" + (s.query or "")
        if self.rx is not None:
            in_sub = set(self.rx.findall(sub)) - {label}
            in_path = set(self.rx.findall(pathq)) - {label}
            in_label = {b for b in set(self.rx.findall(label)) if b != label}
        else:
            in_sub = in_path = in_label = set()
        return {
            "n_risky_host": float(len(_RISKY.findall(host))),
            "n_risky_path": float(len(_RISKY.findall(pathq))),
            "brand_in_subdomain": float(len(in_sub)),
            "brand_in_path": float(len(in_path)),
            "brand_in_domain_label": float(len(in_label)),
        }