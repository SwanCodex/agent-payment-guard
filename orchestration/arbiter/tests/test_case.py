import pytest
from sentinel_contracts import LayerOut, Verdict

from orchestration.arbiter import ArbiterCase, LayerSummary, build_case


def _lo(layer, score, verdict, zone=None, spans=()):
    return LayerOut(layer=layer, score=score, verdict=verdict, reason_codes=[f"{layer}:X"],
                    metadata={"zone": zone, "evidence": [{"kind": "span", "text": t} for t in spans]})


def test_builds_from_real_layer_outs_and_limits_evidence():
    outs = [_lo("L1", 0.2, Verdict.STEP_UP), _lo("L2", 0.03, Verdict.STEP_UP, "GRAY", ["a" * 500, "b", "c", "d"])]
    case = build_case("pay_invoice", outs, amount=40.0, extra_evidence=["b"])
    assert set(case.layers) == {"L1", "L2"} and case.layers["L2"].zone == "GRAY"
    assert case.layers["L2"].verdict == "STEP_UP"
    assert len(case.evidence) == 3 and all(len(e) <= 300 for e in case.evidence)
    assert case.evidence.count("b") == 1                       # de-duplicated


def test_case_rejects_oversized_evidence():
    with pytest.raises(ValueError):
        ArbiterCase(action="x", layers={}, evidence=["e"] * 4)
    with pytest.raises(ValueError):
        ArbiterCase(action="x", layers={}, evidence=["e" * 301])


def test_cache_key_is_stable_and_sensitive():
    a = ArbiterCase(action="x", layers={"L2": LayerSummary(0.1, "ALLOW")}, evidence=["e"])
    b = ArbiterCase(action="x", layers={"L2": LayerSummary(0.1, "ALLOW")}, evidence=["e"])
    c = ArbiterCase(action="x", layers={"L2": LayerSummary(0.1, "ALLOW")}, evidence=["f"])
    assert a.cache_key() == b.cache_key() != c.cache_key()
