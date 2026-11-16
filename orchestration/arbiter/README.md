# LLM Arbiter (third opinion, gray zone only)

Owner: P1. Called by the decision engine (P2) only when no layer is confidently BLOCK or ALLOW.

```python
from orchestration.arbiter import AnthropicClient, Arbiter, ArbiterPolicy, build_case

arbiter = Arbiter(
    AnthropicClient(),                                   # needs ANTHROPIC_API_KEY in the environment
    ArbiterPolicy(allow_max_amount=50.0),                # an LLM ALLOW is honored up to this amount; default 0 = never
    on_case=audit_store.log_arbiter_case,                # every case is stored for human labeling
)
case = build_case("pay_invoice", [l1_out, l2_out, l3_out], action_args=args, amount=args["amount"],
                  txn_summary="first payment to this payee")
result = arbiter.arbitrate(case, session_id=request.agent_id or request.request_id)
# result.verdict is a sentinel_contracts.Verdict; result.reason_codes, result.confidence, result.source
```

Guarantees (all tested with a fake client): no tools, one call, 3 s hard timeout, strict JSON (anything else becomes
STEP_UP), result cached by case hash (failures are never cached), per-session call cap, ALLOW downgraded to STEP_UP above
the policy amount or when the amount is unknown, evidence wrapped in a per-call random boundary and sanitized,
`arbitrate` never raises.

Run the tests from the repo root: `python -m pytest orchestration/arbiter/tests -q`
Not covered by tests: a real API call. Try one with `ANTHROPIC_API_KEY` set before relying on it.
