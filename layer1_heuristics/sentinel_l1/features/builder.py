"""Model feature matrix: lexical + lexicon + TLD risk. is_https is excluded on purpose (see EDA)."""
from typing import Sequence

import pandas as pd

from .lexical import FEATURE_NAMES, url_features
from .lexicon import LEXICON_FEATURES, Lexicon
from .tld_risk import tld_of

EXCLUDED = {"is_https"}
LEXICAL_USED = [f for f in FEATURE_NAMES if f not in EXCLUDED]
MODEL_FEATURES = LEXICAL_USED + LEXICON_FEATURES + ["tld_risk"]


def build_base(urls: Sequence[str], lexicon: Lexicon):
    """Everything except tld_risk (which needs a fitted TldRisk). Returns (DataFrame, tld list)."""
    rows = []
    for u in urls:
        f = url_features(u)
        d = {k: f[k] for k in LEXICAL_USED}
        d.update(lexicon.features(u))
        rows.append(d)
    return pd.DataFrame(rows, columns=LEXICAL_USED + LEXICON_FEATURES), [tld_of(u) for u in urls]