"""Arbiter data types. The repo's contracts have no Arbiter schema yet, so they live here (small and dataclass-based,
matching the style of sentinel_contracts). If P4 adds them to contracts/, import from there instead."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from sentinel_contracts import Verdict
except ImportError as exc:                                   # pragma: no cover
    raise ImportError("sentinel_contracts is not installed. From the repo root run: pip install -e contracts") from exc

MAX_EVIDENCE_ITEMS = 3
MAX_EVIDENCE_CHARS = 300
MAX_TXN_CHARS = 500


@dataclass
class LayerSummary:
    score: Optional[float]
    verdict: str                                    # "ALLOW" | "STEP_UP" | "BLOCK"
    reason_codes: List[str] = field(default_factory=list)
    zone: Optional[str] = None                      # L2 reports "GRAY" for the uncertain band


@dataclass
class ArbiterCase:
    """Compact input for the third-opinion LLM (handbook 2.5). Evidence is untrusted text."""
    action: str                                     # e.g. "pay_invoice"
    layers: Dict[str, LayerSummary]
    action_args: Dict[str, Any] = field(default_factory=dict)
    amount: Optional[float] = None                  # used by the policy cap on ALLOW
    txn_summary: Optional[str] = None
    evidence: List[str] = field(default_factory=list)

    def __post_init__(self):
        if len(self.evidence) > MAX_EVIDENCE_ITEMS:
            raise ValueError(f"at most {MAX_EVIDENCE_ITEMS} evidence excerpts")
        if any(len(e) > MAX_EVIDENCE_CHARS for e in self.evidence):
            raise ValueError(f"each evidence excerpt must be at most {MAX_EVIDENCE_CHARS} characters")
        if self.txn_summary is not None and len(self.txn_summary) > MAX_TXN_CHARS:
            raise ValueError(f"txn_summary must be at most {MAX_TXN_CHARS} characters")

    def cache_key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class ArbiterResult:
    verdict: Verdict
    reason_codes: List[str]
    confidence: float
    source: str                                     # "llm" | "cache" | "fail_closed" | "policy"
    latency_ms: float
    needs_human_label: bool = True                  # handbook: every Arbiter case goes to the audit store for labeling
    raw: Optional[str] = None                       # truncated model output, for debugging
