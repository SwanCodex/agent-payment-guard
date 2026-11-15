"""Serving scorer for the Layer 1 URL ensemble. Loads an artifact bundle
(artifacts/layer1/url_ensemble/<version>) produced by scripts/train_final_l1.py."""
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..features.builder import LEXICAL_USED
from ..features.lexical import url_features
from ..features.lexicon import LEXICON_FEATURES, Lexicon, load_brands
from ..features.tld_risk import TldRisk, tld_of

# must stay identical to the training script (scripts/train_step7c.py HOST, train_final_l1.py FEATS)
_HOST = ["host_len", "host_hyphens", "host_digits", "host_labels", "max_label_len",
         "n_subdomains", "host_digit_ratio", "entropy_host", "vowel_ratio_host"]
GBM_FEATURES: List[str] = [f for f in LEXICAL_USED + LEXICON_FEATURES + ["tld_risk"]
                           if f not in _HOST and f != "is_shortener"]

_LOGIT_CLIP = 1e-6


def strip_scheme(u: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", u.lower())[:200]


def _logit(p: float) -> float:
    p = min(max(p, _LOGIT_CLIP), 1.0 - _LOGIT_CLIP)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    x = max(min(x, 30.0), -30.0)
    return 1.0 / (1.0 + math.exp(-x))


class LRScorer:
    """Hashed char n-gram TF-IDF logistic regression, re-implemented with numpy from saved weights."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.idf = d["idf"].astype(np.float64)
        self.coef = d["coef"].astype(np.float64)
        self.b = float(d["intercept"])
        self.hv = HashingVectorizer(analyzer="char", ngram_range=(2, 5), n_features=2 ** 20,
                                    alternate_sign=False, norm=None, lowercase=False)

    def decision(self, url: str) -> float:
        X = self.hv.transform([strip_scheme(url)])
        idx, cnt = X.indices, X.data
        if len(idx) == 0:
            return self.b
        v = (1.0 + np.log(cnt)) * self.idf[idx]          # sublinear tf * idf
        n = math.sqrt(float(v @ v))
        if n == 0.0:
            return self.b
        return float(v @ self.coef[idx]) / n + self.b


def default_dir(version: str = "0.1.0") -> Path:
    env = os.environ.get("SENTINEL_L1_URL_MODEL_DIR")
    if env:
        return Path(env)
    root = Path(__file__).resolve().parents[3]            # repository root
    return root / "artifacts" / "layer1" / "url_ensemble" / version


class UrlEnsemble:
    def __init__(self, art_dir):
        import lightgbm as lgb
        d = Path(art_dir)
        self.version = d.name
        self.lr = LRScorer(d / "lr_ngram.npz")
        self.booster = lgb.Booster(model_file=str(d / "model.txt"))
        if self.booster.num_feature() != len(GBM_FEATURES):
            raise ValueError(f"LightGBM expects {self.booster.num_feature()} features, "
                             f"serving builds {len(GBM_FEATURES)}")
        self.lex = Lexicon(load_brands(d / "brands.json"))
        self.tld = TldRisk.load(d / "tld_risk.json")
        comb = json.loads((d / "combiner.json").read_text(encoding="utf-8"))
        self.c_coef, self.c_b = float(comb["coef"][0]), float(comb["intercept"])
        cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
        self.cal_x, self.cal_y = np.array(cal["x"]), np.array(cal["y"])
        th = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        self.tau_allow, self.tau_block, self.qhat = th["tau_allow"], th["tau_block"], th["qhat"]

    def _gbm_row(self, url: str) -> List[float]:
        f = url_features(url)
        f.update(self.lex.features(url))
        f["tld_risk"] = float(self.tld.transform([tld_of(url)])[0])
        return [f[k] for k in GBM_FEATURES]

    def score(self, url: str) -> Dict[str, float]:
        p_lr = _sigmoid(self.lr.decision(url))
        p_gbm = float(self.booster.predict(np.array([self._gbm_row(url)]))[0])
        z = 0.5 * (_logit(p_lr) + _logit(p_gbm))
        s = self.c_coef * z + self.c_b
        p = float(np.interp(s, self.cal_x, self.cal_y))     # same as isotonic predict with clipping
        in1, in0 = (1.0 - p) <= self.qhat, p <= self.qhat
        return {"p": p, "z": z, "p_lr": p_lr, "p_gbm": p_gbm,
                "gray": self.tau_allow < p < self.tau_block,
                "abstain": (int(in0) + int(in1)) != 1}