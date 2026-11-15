from sentinel_contracts import L1In, Verdict
from sentinel_l1.api import run_l1
from sentinel_l1.rules.r04_iban import iban_valid

def test_iban_valid_and_invalid():
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")

def test_clean_input_allows():
    out = run_l1(L1In(request_id="t1", url="https://example.com", page_content="hello"))
    assert out.verdict == Verdict.ALLOW and 0.0 <= out.score <= 1.0

def test_zero_width_flags():
    out = run_l1(L1In(request_id="t2", page_content="pay\u200bnow"))
    assert "L1:R01_HIDDEN_CONTROL_CHAR" in out.reason_codes

def test_mixed_script_flags():
    out = run_l1(L1In(request_id="t3", page_content="p\u0430ypal"))  # Cyrillic a
    assert "L1:R01_MIXED_SCRIPT" in out.reason_codes

def test_bad_iban_payee():
    out = run_l1(L1In(request_id="t4", payload={"payee_account": "GB82WEST12345698765433"}))
    assert out.verdict == Verdict.STEP_UP

def test_punycode_host():
    out = run_l1(L1In(request_id="t5", url="http://xn--pypal-4ve.com"))
    assert "L1:R07_PUNYCODE_HOST" in out.reason_codes
    