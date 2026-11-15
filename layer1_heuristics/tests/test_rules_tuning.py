from sentinel_contracts import L1In
from sentinel_l1.api import run_l1

def codes(**kw):
    return run_l1(L1In(request_id="t", **kw)).reason_codes

def test_r01_emoji_zwj_not_flagged():
    assert "L1:R01_HIDDEN_CONTROL_CHAR" not in codes(page_content="great \U0001F469\u200d\U0001F4BB product")

def test_r01_zwj_between_ascii_letters_flagged():
    assert "L1:R01_HIDDEN_CONTROL_CHAR" in codes(page_content="pay\u200dnow")

def test_r04_non_iban_codes_ignored():
    assert "L1:R04_IBAN_INVALID" not in codes(page_content="Room EB34 DATA CENTER floor 2")
    assert "L1:R04_IBAN_INVALID" not in codes(page_content="ref HB2986082267X972194X50117X")

def test_r04_us_account_payee_ignored():
    assert "L1:R04_IBAN_INVALID" not in codes(payload={"payee_account": "123456789012"})

def test_r06_country_site_not_flagged():
    for u in ("https://www.google.in", "https://paypal.jp", "https://google.com.pg"):
        assert "L1:R06_LOOKALIKE_DOMAIN" not in codes(url=u)

def test_r06_transposition_flagged():
    assert "L1:R06_LOOKALIKE_DOMAIN" in codes(url="http://stirpe.com")

def test_r06_decoded_idn_lookalike_flagged():
    host = "p\u0430ypal".encode("idna").decode()       # Latin p + Cyrillic a
    assert "L1:R06_LOOKALIKE_DOMAIN" in codes(url=f"http://{host}.com")

def test_r07_single_script_idn_not_flagged():
    host = "\u043c\u043e\u0438\u043f\u0435\u0441\u043d\u0438".encode("idna").decode()
    assert "L1:R07_PUNYCODE_HOST" not in codes(url=f"https://{host}.ru")

def test_r08_non_brand_mismatch_not_flagged():
    html = '<a href="http://brainitworks.com/x">nurse.org</a>'
    assert "L1:R08_LINK_TEXT_MISMATCH" not in codes(page_content=html)

def test_r08_same_owner_not_flagged():
    html = '<a href="https://amzn.to/abc">amazon.com</a>'
    assert "L1:R08_LINK_TEXT_MISMATCH" not in codes(page_content=html)

def test_html_unclosed_tag_does_not_leak_hidden_state():
    vis = "visible " * 40
    html = f'<html><body><div style="display:none"><p>secret</div><p>{vis}</p></body></html>'
    out = run_l1(L1In(request_id="t", page_content=html))
    assert not out.metadata["forwarded"]
    assert "L1:R02_HIDDEN_TEXT" not in out.reason_codes