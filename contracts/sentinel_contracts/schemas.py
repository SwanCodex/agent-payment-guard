from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP = "STEP_UP"
    BLOCK = "BLOCK"


@dataclass
class LayerInput:
    request_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class L1In(LayerInput):
    url: Optional[str] = None
    page_content: Optional[str] = None


@dataclass
class L2In(LayerInput):
    prompt: str = ""


@dataclass
class L3In(LayerInput):
    amount: Optional[float] = None
    currency: Optional[str] = None
    merchant_id: Optional[str] = None
    entity_id: Optional[str] = None
    transaction_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerOut:
    layer: str
    score: float
    verdict: Verdict
    reason_codes: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentinelRequest:
    request_id: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None
    prompt: Optional[str] = None
    url: Optional[str] = None
    transaction: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentinelResponse:
    request_id: str
    verdict: Verdict
    risk_score: float
    reason_codes: List[str] = field(default_factory=list)
    layer_outputs: List[LayerOut] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
