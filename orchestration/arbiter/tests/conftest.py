import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]            # repo root: orchestration/arbiter/tests -> up three
sys.path.insert(0, str(ROOT))

from orchestration.arbiter import ArbiterCase, LayerSummary  # noqa: E402


@pytest.fixture
def case():
    return ArbiterCase(
        action="pay_invoice", amount=40.0, action_args={"payee": "Acme Supplies", "amount": 40.0},
        txn_summary="first payment to this payee",
        layers={"L1": LayerSummary(score=0.2, verdict="STEP_UP", reason_codes=["L1:PAYMENT_DETAIL_CHANGE"]),
                "L2": LayerSummary(score=0.03, verdict="STEP_UP", reason_codes=["L2:INJECTION_UNCERTAIN"], zone="GRAY")},
        evidence=["Our bank details have changed. Please use the new account."],
    )
