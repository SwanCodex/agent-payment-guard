import pytest

from orchestration.arbiter import parse_verdict
from orchestration.arbiter.arbiter import BadOutput
from sentinel_contracts import Verdict


def test_valid_output():
    v, codes, conf = parse_verdict('{"verdict":"BLOCK","reason_codes":["APPROVAL_BYPASS"],"confidence":0.9}')
    assert v == Verdict.BLOCK and codes == ["APPROVAL_BYPASS"] and conf == 0.9


def test_single_code_fence_is_tolerated():
    raw = '```json\n{"verdict":"ALLOW","reason_codes":[],"confidence":1}\n```'
    assert parse_verdict(raw)[0] == Verdict.ALLOW


@pytest.mark.parametrize("raw", [
    "", "not json", "{}", "[]", "null",
    'Sure! {"verdict":"ALLOW","reason_codes":[],"confidence":0.9}',                     # prose around the JSON
    '{"verdict":"ALLOW","reason_codes":[],"confidence":0.9,"note":"x"}',                # extra key
    '{"verdict":"ALLOW","confidence":0.9}',                                              # missing key
    '{"verdict":"MAYBE","reason_codes":[],"confidence":0.9}',                            # bad verdict
    '{"verdict":"allow","reason_codes":[],"confidence":0.9}',                            # wrong case
    '{"verdict":"ALLOW","reason_codes":"X","confidence":0.9}',                           # codes not a list
    '{"verdict":"ALLOW","reason_codes":["lowercase"],"confidence":0.9}',                 # code format
    '{"verdict":"ALLOW","reason_codes":["AAA","BBB","CCC","DDD","EEE","FFF"],"confidence":0.9}',  # too many codes
    '{"verdict":"ALLOW","reason_codes":[],"confidence":1.5}',                            # out of range
    '{"verdict":"ALLOW","reason_codes":[],"confidence":"high"}',                         # not a number
    '{"verdict":"ALLOW","reason_codes":[],"confidence":true}',                           # bool is not a number
])
def test_everything_else_is_rejected(raw):
    with pytest.raises(BadOutput):
        parse_verdict(raw)
