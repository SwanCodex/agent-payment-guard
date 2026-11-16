import pytest

from sentinel_contracts import SentinelResponse, Verdict, LayerOut

from mock_agent.client import (
    parse_sentinel_response,
    SentinelClientError,
)


def test_parse_valid_sentinel_response():
    payload = {
        "request_id": "REQ-001",
        "verdict": "STEP_UP",
        "risk_score": 0.75,
        "reason_codes": ["L3A:FRAUD_RISK_REVIEW"],
        "layer_outputs": [
            {
                "layer": "L3A",
                "score": 0.75,
                "verdict": "STEP_UP",
                "reason_codes": ["L3A:FRAUD_RISK_REVIEW"],
                "model_name": "test-model",
                "model_version": "1.0",
                "latency_ms": 12.5,
                "metadata": {"status": "evaluated"},
            }
        ],
        "metadata": {"risk_score_semantics": "policy_score"},
    }

    response = parse_sentinel_response(payload)

    assert isinstance(response, SentinelResponse)
    assert response.verdict == Verdict.STEP_UP
    assert response.risk_score == 0.75
    assert len(response.layer_outputs) == 1
    assert isinstance(response.layer_outputs[0], LayerOut)
    assert response.layer_outputs[0].verdict == Verdict.STEP_UP


def test_parse_block_response():
    payload = {
        "request_id": "REQ-002",
        "verdict": "BLOCK",
        "risk_score": 0.95,
    }

    response = parse_sentinel_response(payload)

    assert response.verdict == Verdict.BLOCK
    assert response.layer_outputs == []


def test_reject_invalid_verdict():
    payload = {
        "request_id": "REQ-003",
        "verdict": "UNKNOWN",
        "risk_score": 0.5,
    }

    with pytest.raises(SentinelClientError):
        parse_sentinel_response(payload)


def test_reject_missing_request_id():
    payload = {
        "verdict": "ALLOW",
        "risk_score": 0.1,
    }

    with pytest.raises(SentinelClientError):
        parse_sentinel_response(payload)
