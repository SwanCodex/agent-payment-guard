import base64
from unittest.mock import MagicMock, patch
import pytest

from sentinel_contracts import SentinelRequest, SentinelResponse, Verdict, LayerOut
from orchestration.api.guard_service import process_request
from orchestration.api.l1_context import build_l1_input
import sentinel_l1.rules as rules_pkg
import sentinel_l1.rules.config as rules_config


# ==============================================================================
# Task 2: Context Forwarding & Validation Integration Tests
# ==============================================================================

def test_l1_context_r02_hidden_text_step_up():
    # Hidden-text HTML above the R02 threshold (>=1276 words) in page_content ->
    # L1:R02_HIDDEN_TEXT in response.reason_codes, final STEP_UP
    hidden_words = " ".join(["secret"] * 1300)
    html_content = f"<html><body><div>Visible</div><div style='display:none'>{hidden_words}</div></body></html>"

    req = SentinelRequest(
        request_id="ctx-r02",
        metadata={"page_content": html_content},
    )
    resp = process_request(req)
    assert resp.verdict == Verdict.STEP_UP
    assert "L1:R02_HIDDEN_TEXT" in resp.reason_codes
    assert "page_content" in resp.metadata["l1_context"]["forwarded"]


def test_l1_context_r05_payment_detail_change():
    # Payment-change phrase in page_content -> R05 code present,
    # forwarded items present in the L1 output metadata
    content = "Invoice approved. Please note our new bank details for future wire transfers."
    req = SentinelRequest(
        request_id="ctx-r05",
        metadata={"page_content": content},
    )
    resp = process_request(req)
    assert "L1:R05_PAYMENT_DETAIL_CHANGE" in resp.reason_codes
    l1_out = next(o for o in resp.layer_outputs if o.layer == "L1")
    assert "forwarded" in l1_out.metadata
    assert any(item["source"] == "R05" for item in l1_out.metadata["forwarded"])


def test_l1_context_r04_invalid_iban_in_transaction():
    # Invalid IBAN in transaction["payee_account"] -> R04 code, STEP_UP
    # German IBAN with invalid checksum (ends in 001 instead of 000)
    req = SentinelRequest(
        request_id="ctx-r04",
        transaction={"payee_account": "DE89370400440532013001"},
    )
    resp = process_request(req)
    assert resp.verdict == Verdict.STEP_UP
    assert "L1:R04_IBAN_INVALID" in resp.reason_codes
    assert "payee_account" in resp.metadata["l1_context"]["forwarded"]


def test_l1_context_r08_brand_mismatch_link():
    # Brand-mismatch link in page_html -> R08 code
    html = '<p>Check out our site at <a href="http://phishing-attacker.com/login">paypal.com</a></p>'
    req = SentinelRequest(
        request_id="ctx-r08",
        metadata={"page_html": html},
    )
    resp = process_request(req)
    assert "L1:R08_LINK_TEXT_MISMATCH" in resp.reason_codes
    assert "page_html" in resp.metadata["l1_context"]["forwarded"]


def test_l1_context_r09_encoded_blob_in_page_content():
    # Base64 blob that decodes to text in page_content -> R09 code
    # 44 chars base64 decoding to printable string
    # "UGF5bWVudCBpbnN0cnVjdGlvbnMgZm9yIGFjY291bnQ=" -> "Payment instructions for account"
    blob = "UGF5bWVudCBpbnN0cnVjdGlvbnMgZm9yIGFjY291bnQ="
    text = f"Notice: {blob} please process"
    req = SentinelRequest(
        request_id="ctx-r09",
        metadata={"page_content": text},
    )
    resp = process_request(req)
    assert "L1:R09_ENCODED_BLOB" in resp.reason_codes
    assert "page_content" in resp.metadata["l1_context"]["forwarded"]


def test_l1_context_r10_redirect_count():
    # redirect_count=5 -> R10 long-chain code; True, -1, 999 and "5" rejected
    req_valid = SentinelRequest(
        request_id="ctx-r10-valid",
        metadata={"redirect_count": 5},
    )
    resp_valid = process_request(req_valid)
    assert "L1:R10_LONG_REDIRECT_CHAIN" in resp_valid.reason_codes
    assert "redirect_count" in resp_valid.metadata["l1_context"]["forwarded"]

    # Rejected cases: True, -1, 999, "5"
    for invalid_val in [True, -1, 999, "5"]:
        req_bad = SentinelRequest(
            request_id="ctx-r10-bad",
            url="https://example.com",
            metadata={"redirect_count": invalid_val},
        )
        resp_bad = process_request(req_bad)
        assert "L1:R10_LONG_REDIRECT_CHAIN" not in resp_bad.reason_codes
        assert "redirect_count" in resp_bad.metadata["l1_context"]["rejected"]
        assert "redirect_count" not in resp_bad.metadata["l1_context"]["forwarded"]


def test_l1_context_r03_invoice_pdf_b64():
    # invoice_pdf_b64: with R03 disabled (default) the value reaches run_rules
    # but R03 returns nothing (use a spy on sentinel_l1.rules.check_r03);
    # with R03_ENABLED monkeypatched on and a mocked OCR path, the code appears.
    dummy_b64 = base64.b64encode(b"%PDF-1.4 test invoice").decode("ascii")

    # Part 1: Default (R03 disabled) -> reaches run_rules and check_r03, returns []
    with patch("layer1_heuristics.sentinel_l1.rules.check_r03", wraps=rules_pkg.check_r03) as spy_check, \
         patch("sentinel_l1.rules.check_r03", wraps=rules_pkg.check_r03):
        req = SentinelRequest(
            request_id="ctx-r03-disabled",
            metadata={"invoice_pdf_b64": dummy_b64},
        )
        resp = process_request(req)
        spy_check.assert_called_once_with(dummy_b64)
        assert "L1:R03_PDF_TEXT_DIVERGENCE" not in resp.reason_codes

    # Part 2: R03_ENABLED monkeypatched on with mocked divergence
    mock_divergence_flag = [
        rules_pkg.RuleFlag(
            rule_id="R03",
            code="L1:R03_PDF_TEXT_DIVERGENCE",
            severity="step_up",
            evidence="12/20 text-layer tokens not visible in render",
        )
    ]
    with patch.object(rules_config, "R03_ENABLED", True), \
         patch("layer1_heuristics.sentinel_l1.rules.config.R03_ENABLED", True), \
         patch("layer1_heuristics.sentinel_l1.rules.check_r03", return_value=(mock_divergence_flag, [{"source": "R03", "text": "secret"}])), \
         patch("sentinel_l1.rules.check_r03", return_value=(mock_divergence_flag, [{"source": "R03", "text": "secret"}])):
        req_enabled = SentinelRequest(
            request_id="ctx-r03-enabled",
            metadata={"invoice_pdf_b64": dummy_b64},
        )
        resp_enabled = process_request(req_enabled)
        assert "L1:R03_PDF_TEXT_DIVERGENCE" in resp_enabled.reason_codes
        assert resp_enabled.verdict == Verdict.STEP_UP


def test_context_only_request_no_url():
    # Context-only request (no URL) runs L1 with url_model status "not_applicable"
    req = SentinelRequest(
        request_id="ctx-only",
        metadata={"page_content": "Invoice #INV-2026-991 payment confirmed."},
    )
    resp = process_request(req)
    l1_out = next(o for o in resp.layer_outputs if o.layer == "L1")
    assert l1_out.verdict == Verdict.ALLOW
    assert l1_out.metadata["url_model"]["status"] == "not_applicable"
    assert l1_out.metadata["url_model"]["reason"] is None
    assert "p" not in l1_out.metadata["url_model"]
    assert l1_out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
    assert l1_out.metadata["score_components"]["url_model_policy"] is None


def test_unknown_and_invalid_keys_dropped_and_rejected():
    # Unknown keys, a nested "payload" key in metadata, non-str page_content,
    # oversize values: absent from the L1In passed to run_l1 (assert on the actual object)
    # and listed under rejected.
    req = SentinelRequest(
        request_id="ctx-reject",
        url="https://example.com/pay",
        metadata={
            "unknown_metadata_key": "some_value",
            "payload": {"injected_field": "exploit"},
            "page_content": 12345,  # non-str
            "page_html": "x" * 1_000_001,  # oversized
            "invoice_pdf_b64": "invalid!@#base64",  # invalid charset
        },
    )

    captured_l1_in = []

    def mock_run_l1(l1_in):
        captured_l1_in.append(l1_in)
        return LayerOut(
            layer="L1",
            score=0.0,
            verdict=Verdict.ALLOW,
            metadata={"url_model": {"status": "ok"}},
        )

    with patch("orchestration.api.guard_service.run_l1", side_effect=mock_run_l1):
        resp = process_request(req)

    assert len(captured_l1_in) == 1
    actual_l1_in = captured_l1_in[0]

    # Assert absent from L1In passed to run_l1
    assert actual_l1_in.page_content is None
    assert actual_l1_in.payload == {}
    assert "unknown_metadata_key" not in actual_l1_in.payload
    assert "payload" not in actual_l1_in.payload
    assert "injected_field" not in actual_l1_in.payload

    # Assert listed under rejected with reason codes
    rejected = resp.metadata["l1_context"]["rejected"]
    assert rejected["unknown_metadata_key"] == "UNKNOWN_FIELD"
    assert rejected["payload"] == "DISALLOWED_FIELD"
    assert rejected["page_content"] == "TYPE_MISMATCH"
    assert rejected["page_html"] == "OVERSIZED"
    assert rejected["invoice_pdf_b64"] == "INVALID_CHARSET"
    assert resp.metadata["l1_context"]["forwarded"] == []


def test_no_url_and_no_valid_context_does_not_run_l1():
    # If no URL is provided and no valid context field exists (only rejected keys),
    # L1 should not run at all.
    req = SentinelRequest(
        request_id="ctx-norun",
        metadata={"unknown_key": "junk"},
    )
    resp = process_request(req)
    assert not any(o.layer == "L1" for o in resp.layer_outputs)
    assert resp.metadata["l1_context"]["forwarded"] == []
    assert resp.metadata["l1_context"]["rejected"]["unknown_key"] == "UNKNOWN_FIELD"
