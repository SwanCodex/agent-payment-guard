"""Layer 2 entrypoint required by the contract: run_l2(L2In) -> LayerOut."""
from __future__ import annotations

import time
from typing import Optional

try:                                    # the team's contract package (the original stub imported from the package root)
    from sentinel_contracts import L2In, LayerOut  # type: ignore
except ImportError:                     # handbook section 5.1 mirror, used when sentinel_contracts is not installed
    from sentinel_l2._contracts_fallback import L2In, LayerOut

from sentinel_l2.calibration import calibrated, verdict_from_p
from sentinel_l2.evidence import top_spans
from sentinel_l2.models.bundle import Bundle, load_bundle
from sentinel_l2.models.scorer import TorchScorer

_runtime: dict = {}


def configure(bundle: Optional[Bundle] = None, scorer=None) -> None:
    """Install a bundle and/or scorer explicitly (used by tests and by services that preload at startup)."""
    if bundle is not None:
        _runtime["bundle"] = bundle
    if scorer is not None:
        _runtime["scorer"] = scorer


def reset() -> None:
    _runtime.clear()


def _get_runtime():
    if "bundle" not in _runtime:
        _runtime["bundle"] = load_bundle()
    if "scorer" not in _runtime:
        _runtime["scorer"] = TorchScorer(_runtime["bundle"])
    return _runtime["bundle"], _runtime["scorer"]


def run_l2(request: L2In) -> LayerOut:
    """Score one untrusted text chunk. Raises on model or bundle errors; the router must map errors to STEP_UP."""
    t0 = time.perf_counter()
    bundle, scorer = _get_runtime()
    text = request.text or ""

    if not text.strip():
        return LayerOut(layer="L2", p=0.0, abstain=False, flags=[],
                        evidence=[{"kind": "note", "note": "empty text"}],
                        model_versions=bundle.versions, latency_ms=(time.perf_counter() - t0) * 1000)

    z = float(scorer.score([text])[0])
    p = min(1.0, max(0.0, float(calibrated(z, bundle.T, bundle.delta))))
    verdict = verdict_from_p(p, bundle.tau_block, bundle.tau_allow)

    evidence = [{"kind": "score", "z": round(z, 3), "p": p, "verdict": verdict,
                 "tau_block": bundle.tau_block, "tau_allow": bundle.tau_allow}]
    if verdict != "ALLOW":              # spans cost extra forward passes, so only when someone will look at them
        evidence += top_spans(text, scorer, z)

    return LayerOut(
        layer="L2", p=p,
        abstain=False,                  # TODO: split-conformal abstention once qhat is in thresholds.json
        flags=[], evidence=evidence, model_versions=bundle.versions,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )