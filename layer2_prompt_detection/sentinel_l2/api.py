"""Layer 2 entrypoint: run_l2(L2In) -> LayerOut, written against the repo's contracts/sentinel_contracts.

Install the contracts first (from the repo root):  pip install -e contracts

Mapping onto the contract's LayerOut:
    score        calibrated probability p in [0, 1] (rare-attack prior, see model card)
    verdict      BLOCK if p >= tau_block; ALLOW if p <= tau_allow; the band in between becomes STEP_UP
    reason_codes ["L2:INJECTION_HIGH"] or ["L2:INJECTION_UNCERTAIN"]; ["SYSTEM:MODEL_UNAVAILABLE"] on failure
    metadata     zone ("ALLOW" | "GRAY" | "BLOCK"), z, p, thresholds, evidence spans, model versions.
                 The decision engine should read metadata["zone"]: GRAY is an LLM Arbiter candidate, not a human step-up.
Failures fail closed (STEP_UP), as the handbook requires; run_l2 never raises.
"""
from __future__ import annotations

import logging
import time

try:
    from sentinel_contracts import L2In, LayerOut, Verdict
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("sentinel_contracts is not installed. From the repo root run: pip install -e contracts") from exc

from .scoring import configure, reset, score_text   # noqa: F401  (configure/reset re-exported for tests)

log = logging.getLogger(__name__)

MODEL_NAME = "deberta-v3-base-injection"
_VERDICT = {"ALLOW": Verdict.ALLOW, "GRAY": Verdict.STEP_UP, "BLOCK": Verdict.BLOCK}
_REASONS = {"ALLOW": [], "GRAY": ["L2:INJECTION_UNCERTAIN"], "BLOCK": ["L2:INJECTION_HIGH"]}


def run_l2(request: L2In) -> LayerOut:
    t0 = time.perf_counter()
    source_type = (request.payload or {}).get("source_type")
    try:
        res = score_text(request.prompt or "")
    except Exception as exc:                                 # fail closed: an unscored chunk goes to a human
        log.exception("L2 scoring failed; returning STEP_UP")
        return LayerOut(
            layer="L2", score=0.5, verdict=Verdict.STEP_UP, reason_codes=["SYSTEM:MODEL_UNAVAILABLE"],
            model_name=MODEL_NAME, model_version=None,
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={"zone": "ERROR", "error": f"{type(exc).__name__}: {exc}"[:300], "source_type": source_type},
        )

    reasons = list(_REASONS[res.zone]) + (["L2:EMPTY_PROMPT"] if res.empty else [])
    return LayerOut(
        layer="L2", score=res.p, verdict=_VERDICT[res.zone], reason_codes=reasons,
        model_name=MODEL_NAME, model_version=res.versions.get("l2_deberta"),
        latency_ms=(time.perf_counter() - t0) * 1000,
        metadata={
            "zone": res.zone, "z": None if res.empty else round(res.z, 3), "p": res.p,
            "tau_block": res.tau_block, "tau_allow": res.tau_allow, "abstain": False,
            "evidence": res.evidence, "model_versions": res.versions, "source_type": source_type,
        },
    )


def warmup() -> float:
    """Load the bundle and run one dummy inference so the first real request is not slow. Returns seconds taken.

    Call it once at service start (for example in a FastAPI lifespan or startup hook).
    """
    t0 = time.perf_counter()
    score_text("warm-up text for the first inference")
    return time.perf_counter() - t0
