from sentinel_contracts import L2In, LayerOut
from sentinel_contracts.stubs import run_l2_stub


def run_l2(request: L2In) -> LayerOut:
    return run_l2_stub(request)
