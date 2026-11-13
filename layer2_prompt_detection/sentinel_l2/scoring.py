"""Contract-free scoring core. api.py maps the result onto the team's LayerOut.

Keeping this separate means the model logic does not change if the shared schemas do.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .calibration import calibrated, verdict_from_p
from .evidence import top_spans
from .models.bundle import Bundle, load_bundle
from .models.scorer import TorchScorer

_runtime: dict = {}


def configure(bundle: Optional[Bundle] = None, scorer=None) -> None:
    """Install a bundle and/or scorer explicitly (tests, or a service that preloads at startup)."""
    if bundle is not None:
        _runtime["bundle"] = bundle
    if scorer is not None:
        _runtime["scorer"] = scorer


def reset() -> None:
    _runtime.clear()


def get_runtime():
    if "bundle" not in _runtime:
        _runtime["bundle"] = load_bundle()
    if "scorer" not in _runtime:
        _runtime["scorer"] = TorchScorer(_runtime["bundle"])
    return _runtime["bundle"], _runtime["scorer"]


@dataclass
class L2Result:
    z: float                         # raw margin
    p: float                         # calibrated probability at the deployment prior
    zone: str                        # "ALLOW" | "GRAY" | "BLOCK"
    tau_block: float
    tau_allow: float
    evidence: list = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    empty: bool = False


def score_text(text: str) -> L2Result:
    t0 = time.perf_counter()
    bundle, scorer = get_runtime()
    base = dict(tau_block=bundle.tau_block, tau_allow=bundle.tau_allow, versions=bundle.versions)

    if not text or not text.strip():
        return L2Result(z=float("-inf"), p=0.0, zone="ALLOW", empty=True,
                        latency_ms=(time.perf_counter() - t0) * 1000, **base)

    z = float(scorer.score([text])[0])
    p = min(1.0, max(0.0, float(calibrated(z, bundle.T, bundle.delta))))
    zone = verdict_from_p(p, bundle.tau_block, bundle.tau_allow)
    evidence = top_spans(text, scorer, z) if zone != "ALLOW" else []   # spans cost extra passes: only when read
    return L2Result(z=z, p=p, zone=zone, evidence=evidence,
                    latency_ms=(time.perf_counter() - t0) * 1000, **base)
