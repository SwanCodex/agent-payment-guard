from typing import List

from sentinel_contracts import LayerOut, SentinelResponse, Verdict


def calculate_risk_score(outputs: List[LayerOut]) -> float:
    """
    Combine layer scores into a single risk score.

    Current bootstrap implementation uses the maximum layer score.
    This will be replaced/tuned when the real models are integrated.
    """
    if not outputs:
        return 0.0

    return max(output.score for output in outputs)


def decide_verdict(risk_score: float) -> Verdict:
    """
    Bootstrap decision thresholds.

    These are temporary values and are NOT final model thresholds.
    """
    if risk_score >= 0.80:
        return Verdict.BLOCK

    if risk_score >= 0.50:
        return Verdict.STEP_UP

    return Verdict.ALLOW


def build_response(
    request_id: str,
    outputs: List[LayerOut],
) -> SentinelResponse:
    risk_score = calculate_risk_score(outputs)
    verdict = decide_verdict(risk_score)

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
