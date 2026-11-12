"""Pure-numpy calibration helpers. Must match the training notebook (l2_deberta_finetune, block 1)."""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def calibrated(z, T: float, delta: float):
    """Raw margin z (attack logit minus benign logit) -> probability at the deployment prior.

    The model was trained on a balanced split, so the prior-shift offset `delta`
    (= log(deploy_odds) - log(train_odds), -4.60 for a 1:99 deployment prior) is added after temperature scaling:
        p = sigmoid(z / T + delta)
    """
    return sigmoid(np.asarray(z, dtype=float) / T + delta)


def verdict_from_p(p: float, tau_block: float, tau_allow: float) -> str:
    """BLOCK is checked first, so when tau_allow >= tau_block (the two operating points cross) there is no GRAY."""
    if p >= tau_block:
        return "BLOCK"
    if p <= tau_allow:
        return "ALLOW"
    return "GRAY"
