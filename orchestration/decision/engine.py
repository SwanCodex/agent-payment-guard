
from typing import List

from sentinel_contracts import LayerOut, SentinelResponse, Verdict


UNAVAILABLE_CODE = "SYSTEM:MODEL_UNAVAILABLE"


def calculate_risk_score(outputs: List[LayerOut]) -> float:
    if not outputs:
        return 0.0

    return max(output.score for output in outputs)


def decide_verdict(outputs: List[LayerOut]) -> Verdict:
    if not outputs:
        return Verdict.STEP_UP

    # Explicit blocks always take priority.
    if any(output.verdict == Verdict.BLOCK for output in outputs):
        return Verdict.BLOCK

    # Unavailable layers must never silently approve.
    if any(UNAVAILABLE_CODE in output.reason_codes for output in outputs):
        return Verdict.STEP_UP

    if any(output.verdict == Verdict.STEP_UP for output in outputs):
        return Verdict.STEP_UP

    return Verdict.ALLOW


def build_response(
    request_id: str,
    outputs: List[LayerOut],
) -> SentinelResponse:
    risk_score = calculate_risk_score(outputs)
    verdict = decide_verdict(outputs)

    reason_codes = []
    for output in outputs:
        reason_codes.extend(output.reason_codes)

    return SentinelResponse(
        request_id=request_id,
        verdict=verdict,
        risk_score=risk_score,
        reason_codes=reason_codes,
        layer_outputs=outputs,
    )
