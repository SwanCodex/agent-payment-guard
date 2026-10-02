from sentinel_contracts import L1In, LayerOut
from sentinel_contracts.stubs import run_l1_stub


def run_l1(request: L1In) -> LayerOut:
    return run_l1_stub(request)
