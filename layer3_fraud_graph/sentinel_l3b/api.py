from sentinel_contracts import L3In, LayerOut
from sentinel_contracts.stubs import run_l3b_stub


def run_l3b(request: L3In) -> LayerOut:
    return run_l3b_stub(request)
