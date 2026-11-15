"""Smoothed per-TLD malicious rate. Fit on TRAIN only; out-of-fold values for the training rows."""
import json
from pathlib import Path
from typing import Dict, Sequence
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ..rules._domain import split_host


def tld_of(url: str) -> str:
    u = (url or "").strip()
    if "://" not in u:
        u = "http://" + u
    try:
        host = (urlsplit(u).hostname or "").lower()
    except ValueError:
        return "(none)"
    return (split_host(host)[1] if host else "") or "(none)"


class TldRisk:
    def __init__(self, smoothing: float = 20.0):
        self.m = smoothing
        self.prior = 0.5
        self.rates: Dict[str, float] = {}

    def fit(self, tlds: Sequence[str], y: Sequence[int]) -> "TldRisk":
        y = np.asarray(y, dtype=float)
        self.prior = float(y.mean())
        g = pd.DataFrame({"t": list(tlds), "y": y}).groupby("t")["y"].agg(["sum", "count"])
        self.rates = {t: float((r["sum"] + self.m * self.prior) / (r["count"] + self.m))
                      for t, r in g.iterrows()}
        return self

    def transform(self, tlds: Sequence[str]) -> np.ndarray:
        return np.array([self.rates.get(t, self.prior) for t in tlds], dtype=float)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"m": self.m, "prior": self.prior, "rates": self.rates}),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "TldRisk":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        o = cls(d["m"])
        o.prior, o.rates = d["prior"], d["rates"]
        return o


def oof_tld_risk(tlds, y, groups, n_splits: int = 5, smoothing: float = 20.0) -> np.ndarray:
    """Training-row values computed without the row's own fold (grouped by registered domain)."""
    tlds, y = np.asarray(tlds), np.asarray(y)
    out = np.zeros(len(y), dtype=float)
    for tr_idx, te_idx in GroupKFold(n_splits=n_splits).split(tlds, y, groups):
        m = TldRisk(smoothing).fit(tlds[tr_idx], y[tr_idx])
        out[te_idx] = m.transform(tlds[te_idx])
    return out