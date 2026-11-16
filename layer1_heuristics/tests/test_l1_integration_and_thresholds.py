import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock, patch
import pytest

from sentinel_contracts import L1In, LayerOut, SentinelRequest, SentinelResponse, Verdict
from sentinel_l1.api import run_l1, get_url_model, reset_model_cache
from sentinel_l1.models.url_ensemble import UrlEnsemble, IntegrityError
from sentinel_l1.rules.base import RuleFlag, RuleResult
from orchestration.api.guard_service import process_request


# ==============================================================================
# Helper Mock Model
# ==============================================================================
class MockUrlEnsemble:
    def __init__(self, p_to_return=0.5, tau_allow=0.008197, tau_block=0.833333):
        self.version = "0.1.0-mock"
        self.tau_allow = tau_allow
        self.tau_block = tau_block
        self.p_to_return = p_to_return

    def score(self, url: str):
        p = self.p_to_return
        return {
            "p": p,
            "z": 0.0,
            "p_lr": p,
            "p_gbm": p,
            "gray": self.tau_allow < p < self.tau_block,
            "abstain": self.tau_allow < p < self.tau_block,
        }


# ==============================================================================
# Task 1 & 4: Score vs Verdict & Exact Threshold Boundaries
# ==============================================================================
def test_threshold_below_tau_allow():
    mock_model = MockUrlEnsemble(p_to_return=0.005)
    with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
        out = run_l1(L1In(request_id="t_allow", url="https://safe-url.com"))
        assert out.verdict == Verdict.ALLOW
        # Model contribution is min(p, 0.49) = 0.005
        assert abs(out.score - 0.005) < 1e-4
        assert "L1:URL_MODEL_HIGH_RISK" not in out.reason_codes
        assert out.metadata["url_model"]["status"] == "ok"
        assert out.metadata["url_model"]["p"] == 0.005
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["url_model_policy"] == pytest.approx(0.005, abs=1e-4)


def test_threshold_in_gray_zone_allows():
    # Gray zone (tau_allow < p < tau_block): verdict ALLOW, no L1:URL_MODEL_HIGH_RISK,
    # gray True, raw p unchanged in metadata
    p_val = 0.500
    mock_model = MockUrlEnsemble(p_to_return=p_val)
    with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
        out = run_l1(L1In(request_id="t_gray", url="https://example.com"))
        assert out.verdict == Verdict.ALLOW
        assert "L1:URL_MODEL_HIGH_RISK" not in out.reason_codes
        assert out.metadata["url_model"]["status"] == "ok"
        assert out.metadata["url_model"]["gray"] is True
        assert out.metadata["url_model"]["abstain"] is True
        assert out.metadata["url_model"]["p"] == 0.500
        # Score contribution capped at min(p, 0.49) = 0.49
        assert abs(out.score - 0.49) < 1e-4
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["url_model_policy"] == pytest.approx(0.49, abs=1e-4)


def test_threshold_exact_tau_block_boundary():
    # Exactly at tau_block boundary: triggers STEP_UP, but NOT BLOCK. Score mapped to 0.60
    mock_model = MockUrlEnsemble(p_to_return=0.833333)
    with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
        out = run_l1(L1In(request_id="t_bound", url="https://suspicious-site.com"))
        assert out.verdict == Verdict.STEP_UP
        assert "L1:URL_MODEL_HIGH_RISK" in out.reason_codes
        assert abs(out.score - 0.60) < 1e-4
        assert out.metadata["url_model"]["p"] == 0.833333
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["url_model_policy"] == 0.60


def test_threshold_above_tau_block_score_semantics():
    # At p = 1.0 (extreme risk): L1 score must be 0.60, url_model.p must stay 1.0
    mock_model = MockUrlEnsemble(p_to_return=1.0)
    with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
        out = run_l1(L1In(request_id="t_phish", url="https://accounts.google.com/signin"))
        assert out.verdict == Verdict.STEP_UP
        assert out.verdict != Verdict.BLOCK
        assert "L1:URL_MODEL_HIGH_RISK" in out.reason_codes
        assert abs(out.score - 0.60) < 1e-4
        assert out.metadata["url_model"]["p"] == 1.0
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["rule_flags"] == 0.0
        assert out.metadata["score_components"]["url_model_policy"] == 0.60


def test_end_to_end_google_url_step_up_not_block():
    # Test through process_request() with real engine:
    # Mock URL model returning p=1.0 and p=0.9844 for https://www.google.com/
    # With L2/L3 returning ALLOW, final SentinelResponse MUST NOT BE BLOCK, and MUST BE STEP_UP.
    for p_val in [1.0, 0.9844]:
        mock_model = MockUrlEnsemble(p_to_return=p_val)
        with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")), \
             patch("layer1_heuristics.sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
            req = SentinelRequest(
                request_id=f"orch-google-{p_val}",
                url="https://www.google.com/",
            )
            resp = process_request(req)
            assert resp.verdict == Verdict.STEP_UP, f"Expected STEP_UP for p={p_val}, got {resp.verdict}"
            assert resp.verdict != Verdict.BLOCK
            # Risk score must be <= 0.60, not exceeding BLOCK thresholds
            assert resp.risk_score == pytest.approx(0.60, abs=1e-3)
            assert "L1:URL_MODEL_HIGH_RISK" in resp.reason_codes
            assert resp.metadata["risk_score_semantics"] == "max of per-layer policy scores; not a calibrated probability"
            l1_out = next(o for o in resp.layer_outputs if o.layer == "L1")
            assert l1_out.metadata["url_model"]["p"] == pytest.approx(p_val, abs=1e-4)


# ==============================================================================
# Task 1 & 4: Model Failure Handling, Regression & Metadata Schemas
# ==============================================================================
def test_metadata_url_model_schema_when_no_url():
    out = run_l1(L1In(request_id="t_no_url"))
    um = out.metadata["url_model"]
    assert um["status"] == "not_applicable"
    assert um["reason"] is None
    assert um["version"] is None
    assert "p" not in um
    assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
    assert out.metadata["score_components"] == {"rule_flags": 0.0, "url_model_policy": None}


def test_model_unavailable_fail_closed_step_up():
    with patch("sentinel_l1.api.get_url_model", return_value=(None, "directory_not_found", None)):
        out = run_l1(L1In(request_id="t_unavail", url="https://example.com"))
        assert out.verdict == Verdict.STEP_UP
        assert "SYSTEM:MODEL_UNAVAILABLE" in out.reason_codes
        assert abs(out.score - 0.60) < 1e-4
        assert out.metadata["url_model"]["status"] == "unavailable"
        assert out.metadata["url_model"]["reason"] == "directory_not_found"
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["url_model_policy"] == 0.60


def test_model_scoring_exception_error():
    mock_model = MagicMock()
    mock_model.version = "0.1.0-mock"
    mock_model.score.side_effect = RuntimeError("Inference tensor crash")
    with patch("sentinel_l1.api.get_url_model", return_value=(mock_model, None, "0.1.0-mock")):
        out = run_l1(L1In(request_id="t_crash", url="https://example.com"))
        assert out.verdict == Verdict.STEP_UP
        assert "SYSTEM:MODEL_UNAVAILABLE" in out.reason_codes
        assert abs(out.score - 0.60) < 1e-4
        assert out.metadata["url_model"]["status"] == "error"
        assert "RuntimeError" in out.metadata["url_model"]["reason"]
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["url_model_policy"] == 0.60


def test_regression_no_fail_open_override():
    # Setting SENTINEL_L1_ON_MODEL_FAILURE to allow/true/1/ALLOW MUST STILL return STEP_UP (fail-closed)
    for env_val in ["allow", "ALLOW", "true", "1", "yes"]:
        with patch.dict(os.environ, {"SENTINEL_L1_ON_MODEL_FAILURE": env_val}), \
             patch("sentinel_l1.api.get_url_model", return_value=(None, "missing_weights", None)):
            out = run_l1(L1In(request_id="t_regression_env", url="https://example.com"))
            assert out.verdict == Verdict.STEP_UP, f"Expected STEP_UP with env={env_val}, got {out.verdict}"
            assert "SYSTEM:MODEL_UNAVAILABLE" in out.reason_codes
            assert out.score == 0.60
            assert out.metadata["url_model"]["status"] == "unavailable"


def test_model_load_cooldown_and_recovery():
    reset_model_cache()
    # First attempt: load fails
    with patch("sentinel_l1.models.url_ensemble.UrlEnsemble", side_effect=FileNotFoundError("missing bundle")):
        m1, err1, v1 = get_url_model()
        assert m1 is None
        assert "missing bundle" in err1

    # Second attempt immediately: returns cached failure without attempting reload
    with patch("sentinel_l1.models.url_ensemble.UrlEnsemble", side_effect=AssertionError("Should not be called")):
        m2, err2, v2 = get_url_model()
        assert m2 is None
        assert err2 == err1

    # Simulate cooldown expiry (30s) and recovery when bundle is available
    mock_ok = MockUrlEnsemble()
    with patch("sentinel_l1.api._MODEL_LAST_TRY", time.monotonic() - 35), \
         patch("sentinel_l1.models.url_ensemble.UrlEnsemble", return_value=mock_ok):
        m3, err3, v3 = get_url_model()
        assert m3 is not None
        assert err3 is None

    reset_model_cache()


# ==============================================================================
# Task 5: Artifact Integrity at Load
# ==============================================================================
def test_artifact_integrity_hash_mismatch_in_temp_dir():
    # Copy bundle to a temp dir, tamper 1 byte of model.txt, verify UrlEnsemble raises IntegrityError
    from sentinel_l1.models.url_ensemble import default_dir
    bundle_dir = default_dir()
    if not os.path.isdir(bundle_dir):
        pytest.skip("Default bundle dir not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy real bundle files
        for item in os.listdir(bundle_dir):
            s = os.path.join(bundle_dir, item)
            d = os.path.join(tmpdir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)

        # Corrupt model.txt
        model_path = os.path.join(tmpdir, "model.txt")
        with open(model_path, "a", encoding="utf-8") as f:
            f.write("\n# tampered line\n")

        with pytest.raises(IntegrityError) as exc_info:
            UrlEnsemble(tmpdir)
        assert "integrity_mismatch: model.txt" in str(exc_info.value)


# ==============================================================================
# Deterministic BLOCK Verification
# ==============================================================================
def test_deterministic_block_remains_functional():
    # When a deterministic rule engine flag has severity == 'block', L1 MUST BLOCK
    block_rule_result = RuleResult(
        flags=[
            RuleFlag(
                rule_id="R99",
                code="L1:CUSTOM_HARD_BLOCK",
                severity="block",
                evidence="Critical deterministic violation detected",
            )
        ]
    )
    with patch("sentinel_l1.api.run_rules", return_value=block_rule_result):
        out = run_l1(L1In(request_id="t_det_block", url="https://example.com"))
        assert out.verdict == Verdict.BLOCK
        assert "L1:CUSTOM_HARD_BLOCK" in out.reason_codes
        assert out.score >= 0.95
        assert out.metadata["score_semantics"] == "policy_contribution_not_a_probability"
        assert out.metadata["score_components"]["rule_flags"] == 0.95


def test_orchestration_deterministic_block_hard_blocks_payment():
    # If L1 has a deterministic BLOCK, the overall orchestration response MUST BLOCK
    block_rule_result = RuleResult(
        flags=[
            RuleFlag(
                rule_id="R04",
                code="L1:SANCTIONED_IBAN_BLOCK",
                severity="block",
                evidence="IBAN matches known sanctioned list",
            )
        ]
    )
    with patch("sentinel_l1.api.run_rules", return_value=block_rule_result), \
         patch("layer1_heuristics.sentinel_l1.api.run_rules", return_value=block_rule_result):
        req = SentinelRequest(
            request_id="orch-det-block",
            url="https://example.com",
        )
        resp = process_request(req)
        assert resp.verdict == Verdict.BLOCK
        assert "L1:SANCTIONED_IBAN_BLOCK" in resp.reason_codes
        assert resp.risk_score >= 0.95


def test_orchestration_unavailable_model_steps_up():
    # If L1 model is unavailable, orchestration must step up, never silently allow
    with patch("sentinel_l1.api.get_url_model", return_value=(None, "bundle_missing", None)), \
         patch("layer1_heuristics.sentinel_l1.api.get_url_model", return_value=(None, "bundle_missing", None)):
        req = SentinelRequest(
            request_id="orch-unavail",
            url="https://example.com",
        )
        resp = process_request(req)
        assert "SYSTEM:MODEL_UNAVAILABLE" in resp.reason_codes
        assert resp.verdict == Verdict.STEP_UP
