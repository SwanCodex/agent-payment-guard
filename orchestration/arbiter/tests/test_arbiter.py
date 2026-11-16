import time

import pytest
from sentinel_contracts import Verdict

from orchestration.arbiter import Arbiter, ArbiterCase, ArbiterPolicy, LayerSummary

from arbiter_fakes import FakeClient, reply  # noqa: E402


def make(client, **policy):
    return Arbiter(client, ArbiterPolicy(**policy))


def test_block_passes_through(case):
    r = make(FakeClient(reply("BLOCK", ["APPROVAL_BYPASS"], 0.9))).arbitrate(case)
    assert r.verdict == Verdict.BLOCK and r.reason_codes == ["APPROVAL_BYPASS"] and r.source == "llm"
    assert r.needs_human_label is True


def test_allow_is_honored_only_up_to_the_policy_amount(case):
    ok = make(FakeClient(reply("ALLOW", ["BENIGN_PATTERN"])), allow_max_amount=100).arbitrate(case)
    assert ok.verdict == Verdict.ALLOW and ok.source == "llm"
    over = make(FakeClient(reply("ALLOW", ["BENIGN_PATTERN"])), allow_max_amount=10).arbitrate(case)       # amount is 40
    assert over.verdict == Verdict.STEP_UP and "POLICY:ARBITER_ALLOW_AMOUNT_CAP" in over.reason_codes and over.source == "policy"


def test_default_policy_never_honors_allow(case):
    assert make(FakeClient(reply("ALLOW"))).arbitrate(case).verdict == Verdict.STEP_UP


def test_unknown_amount_counts_as_over_the_cap(case):
    case.amount = None
    r = make(FakeClient(reply("ALLOW")), allow_max_amount=1e9).arbitrate(case)
    assert r.verdict == Verdict.STEP_UP


@pytest.mark.parametrize("raw", ["garbage", "", '{"verdict":"ALLOW"}', "Sure! " + reply("ALLOW")])
def test_bad_output_fails_closed(case, raw):
    r = make(FakeClient(raw), allow_max_amount=1e9).arbitrate(case)
    assert r.verdict == Verdict.STEP_UP and r.reason_codes == ["SYSTEM:ARBITER_BAD_OUTPUT"] and r.source == "fail_closed"


def test_client_exception_fails_closed(case):
    r = make(FakeClient(exc=RuntimeError("boom"))).arbitrate(case)
    assert r.verdict == Verdict.STEP_UP and r.reason_codes == ["SYSTEM:ARBITER_ERROR"]


def test_timeout_fails_closed_without_waiting_for_the_model(case):
    t0 = time.perf_counter()
    r = make(FakeClient(reply("ALLOW"), delay=0.6), timeout_s=0.05).arbitrate(case)
    assert time.perf_counter() - t0 < 0.4
    assert r.verdict == Verdict.STEP_UP and r.reason_codes == ["SYSTEM:ARBITER_TIMEOUT"]


def test_identical_cases_hit_the_cache(case):
    client = FakeClient(reply("BLOCK", ["INSTRUCTION_IN_DATA"]))
    arb = make(client)
    first, second = arb.arbitrate(case), arb.arbitrate(case)
    assert client.calls == 1 and second.source == "cache" and second.verdict == first.verdict


def test_failures_are_not_cached(case):
    client = FakeClient("garbage")
    arb = make(client)
    arb.arbitrate(case); arb.arbitrate(case)
    assert client.calls == 2


def test_different_cases_are_not_confused(case):
    client = FakeClient(reply("BLOCK"))
    arb = make(client)
    arb.arbitrate(case)
    other = ArbiterCase(action="refund", layers=case.layers, evidence=["different"])
    arb.arbitrate(other)
    assert client.calls == 2


def test_per_session_cap_stops_calls_before_the_model(case):
    client = FakeClient(reply("BLOCK"))
    arb = make(client, max_calls_per_session=2)
    cases = [ArbiterCase(action="pay", layers=case.layers, evidence=[f"e{i}"]) for i in range(4)]
    results = [arb.arbitrate(c, session_id="s1") for c in cases]
    assert client.calls == 2
    assert [r.reason_codes for r in results[2:]] == [["SYSTEM:ARBITER_CAP_REACHED"]] * 2
    assert arb.arbitrate(cases[3], session_id="s2").source == "llm"        # another session is unaffected


def test_cache_hits_do_not_use_up_the_session_cap(case):
    client = FakeClient(reply("BLOCK"))
    arb = make(client, max_calls_per_session=1)
    for _ in range(5):
        assert arb.arbitrate(case, session_id="s").verdict == Verdict.BLOCK
    assert client.calls == 1


def test_audit_hook_sees_every_result_and_cannot_break_the_verdict(case):
    seen = []
    arb = Arbiter(FakeClient(reply("BLOCK")), on_case=lambda c, r: seen.append(r))
    arb.arbitrate(case)
    assert len(seen) == 1 and seen[0].needs_human_label

    def explode(c, r):
        raise RuntimeError("audit store down")
    assert Arbiter(FakeClient(reply("BLOCK")), on_case=explode).arbitrate(case).verdict == Verdict.BLOCK


def test_prompt_given_to_the_model_wraps_evidence(case):
    client = FakeClient(reply("BLOCK"))
    make(client).arbitrate(case)
    system, user, timeout_s = client.last
    assert "BEGIN UNTRUSTED" in user and "bank details have changed" in user and timeout_s == 3.0
