from .schemas import L1In, L2In, L3In, LayerOut, Verdict


def run_l1_stub(request: L1In) -> LayerOut:
    return LayerOut(
        layer="L1",
        score=0.0,
        verdict=Verdict.ALLOW,
        reason_codes=["SYSTEM:MODEL_UNAVAILABLE"],
        model_name="l1-stub",
        model_version="0.1.0",
    )


def run_l2_stub(request: L2In) -> LayerOut:
    return LayerOut(
        layer="L2",
        score=0.0,
        verdict=Verdict.ALLOW,
        reason_codes=["SYSTEM:MODEL_UNAVAILABLE"],
        model_name="l2-stub",
        model_version="0.1.0",
    )


def run_l3a_stub(request: L3In) -> LayerOut:
    return LayerOut(
        layer="L3A",
        score=0.0,
        verdict=Verdict.ALLOW,
        reason_codes=["SYSTEM:MODEL_UNAVAILABLE"],
        model_name="l3a-stub",
        model_version="0.1.0",
    )


def run_l3b_stub(request: L3In) -> LayerOut:
    return LayerOut(
        layer="L3B",
        score=0.0,
        verdict=Verdict.ALLOW,
        reason_codes=["SYSTEM:MODEL_UNAVAILABLE"],
        model_name="l3b-stub",
        model_version="0.1.0",
    )
