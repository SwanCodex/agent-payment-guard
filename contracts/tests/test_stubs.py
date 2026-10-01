import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel_contracts import L1In, L2In, L3In
from sentinel_contracts.stubs import (
    run_l1_stub,
    run_l2_stub,
    run_l3a_stub,
    run_l3b_stub,
)


def test_l1_stub():
    result = run_l1_stub(L1In(request_id="test"))

    assert result.layer == "L1"
    assert result.verdict.value == "ALLOW"
    assert result.model_name == "l1-stub"


def test_l2_stub():
    result = run_l2_stub(
        L2In(
            request_id="test",
            prompt="hello",
        )
    )

    assert result.layer == "L2"
    assert result.model_name == "l2-stub"


def test_l3a_stub():
    result = run_l3a_stub(L3In(request_id="test"))

    assert result.layer == "L3A"
    assert result.model_name == "l3a-stub"


def test_l3b_stub():
    result = run_l3b_stub(L3In(request_id="test"))

    assert result.layer == "L3B"
    assert result.model_name == "l3b-stub"
