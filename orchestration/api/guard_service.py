from typing import List

from sentinel_contracts import (
    L1In,
    L2In,
    L3In,
    LayerOut,
    SentinelRequest,
    SentinelResponse,
)

from layer1_heuristics.sentinel_l1.api import run_l1
from layer2_prompt_detection.sentinel_l2.api import run_l2
from layer3_fraud_tabular.sentinel_l3a.api import run_l3a
from layer3_fraud_graph.sentinel_l3b.api import run_l3b

from orchestration.decision.engine import build_response


def process_request(request: SentinelRequest) -> SentinelResponse:
    """
    Run the SENTINEL layer pipeline.

    All four layers currently use bootstrap stubs.
    """

    outputs: List[LayerOut] = []

    # Layer 1
    if request.url is not None:
        l1_input = L1In(
            request_id=request.request_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            transaction_id=request.transaction_id,
            url=request.url,
        )

        outputs.append(run_l1(l1_input))

    # Layer 2
    if request.prompt is not None:
        l2_input = L2In(
            request_id=request.request_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            transaction_id=request.transaction_id,
            prompt=request.prompt,
        )

        outputs.append(run_l2(l2_input))

    # Layer 3A
    if request.transaction:
        l3_input = L3In(
            request_id=request.request_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            transaction_id=request.transaction_id,
            transaction_features=request.transaction,
        )

        outputs.append(run_l3a(l3_input))

        # Layer 3B receives the same transaction contract.
        outputs.append(run_l3b(l3_input))

    return build_response(
        request_id=request.request_id,
        outputs=outputs,
    )
