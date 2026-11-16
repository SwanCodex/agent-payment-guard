import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sentinel_contracts import (
    PaymentProposal,
    SentinelRequest,
    SentinelResponse,
    Verdict,
    LayerOut,
)

from mock_agent.transaction_profile import build_transaction_profile


class SentinelClientError(RuntimeError):
    pass


def parse_sentinel_response(result: dict) -> SentinelResponse:
    try:
        outputs = [
            LayerOut(
                layer=item["layer"],
                score=float(item["score"]),
                verdict=Verdict(item["verdict"]),
                reason_codes=item.get("reason_codes", []),
                model_name=item.get("model_name"),
                model_version=item.get("model_version"),
                latency_ms=item.get("latency_ms"),
                metadata=item.get("metadata", {}),
            )
            for item in result.get("layer_outputs", [])
        ]

        return SentinelResponse(
            request_id=result["request_id"],
            verdict=Verdict(result["verdict"]),
            risk_score=float(result["risk_score"]),
            reason_codes=result.get("reason_codes", []),
            layer_outputs=outputs,
            metadata=result.get("metadata", {}),
        )

    except (KeyError, TypeError, ValueError) as exc:
        raise SentinelClientError(
            "SENTINEL returned an invalid response."
        ) from exc


class SentinelClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def evaluate_payment(
        self,
        proposal: PaymentProposal,
        prompt: str,
        url: str = None,
    ) -> SentinelResponse:

        if not isinstance(proposal, PaymentProposal):
            raise TypeError("proposal must be a PaymentProposal")

        transaction = build_transaction_profile(
            user_id=proposal.user_id,
            amount=proposal.amount,
            currency=proposal.currency,
            recipient_id=proposal.recipient_id,
            reference=proposal.reference,
        )

        sentinel_request = SentinelRequest(
            request_id=proposal.request_id,
            agent_id=proposal.agent_id,
            user_id=proposal.user_id,
            transaction_id=proposal.proposal_id,
            prompt=prompt,
            url=url,
            transaction=transaction,
            metadata={
                "proposal_id": proposal.proposal_id,
                "payment_fingerprint": proposal.fingerprint,
                "simulation": True,
            },
        )

        payload = {
            "request_id": sentinel_request.request_id,
            "agent_id": sentinel_request.agent_id,
            "user_id": sentinel_request.user_id,
            "transaction_id": sentinel_request.transaction_id,
            "prompt": sentinel_request.prompt,
            "url": sentinel_request.url,
            "transaction": sentinel_request.transaction,
            "metadata": sentinel_request.metadata,
        }

        request = Request(
            url=f"{self.base_url}/v1/guard",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SentinelClientError(
                f"SENTINEL returned HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise SentinelClientError(
                f"Could not connect to SENTINEL: {exc}"
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SentinelClientError(
                "SENTINEL returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise SentinelClientError("SENTINEL response must be a JSON object.")

        return parse_sentinel_response(result)
