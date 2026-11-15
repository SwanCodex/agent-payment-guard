
from sentinel_contracts import SentinelRequest
from orchestration.api.guard_service import process_request

request = SentinelRequest(
    request_id="sentinel-demo-001",
    agent_id="demo-agent",
    user_id="demo-user",
    transaction_id="demo-txn-001",
    prompt="Pay the approved invoice of INR 150.75.",
    url="https://example.com/invoice",
    transaction={
        "amount": 150.75,
        "event_time": 1000,
        "product_cd": "W",
        "card_network": "visa",
        "card_type": "debit",
        "payer_email_domain": "gmail.com",
        "recipient_email_domain": "gmail.com",
        "billing_addr1": 300.0,
        "billing_addr2": 87.0,
        "dist1": None,
        "device_type": None,
        "card_key": "test-card-001",
        "addr_key": "300.0_87.0",
        "device_key": "missing",
    },
)

response = process_request(request)

print("\n--- SENTINEL Orchestration Test ---")
print("Request ID:", response.request_id)
print("Overall verdict:", response.verdict.value)
print("Risk score:", response.risk_score)
print("Reason codes:", response.reason_codes)

print("\n--- Individual Layer Results ---")
for layer in response.layer_outputs:
    print(
        layer.layer,
        "| Verdict:", layer.verdict.value,
        "| Score:", round(layer.score, 4),
        "| Model:", layer.model_name,
        "| Reasons:", layer.reason_codes,
    )
