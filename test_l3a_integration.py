
from sentinel_contracts import L3In
from orchestration.integrations.l3a import run_l3a

request = L3In(
    request_id="test-l3a-001",
    transaction_features={
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

result = run_l3a(request)

print("\n--- L3A Integration Test ---")
print("Layer:", result.layer)
print("Verdict:", result.verdict.value)
print("Risk score:", result.score)
print("Reason codes:", result.reason_codes)
print("Model:", result.model_name)
print("Version:", result.model_version)
print("Latency:", result.latency_ms)
print("Metadata:", result.metadata)
