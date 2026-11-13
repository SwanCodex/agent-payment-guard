"""Ingress router: POST /v1/scan_content endpoint.

Owner: P1.
Handbook specification (Sections 2.3, 5.1, 10.1):
- Called for every untrusted item the agent reads (web pages, invoices, emails, messages).
- Runs Layer 1 (heuristics / URL counterparty risk) and Layer 2 (malicious prompt detector).
- Stores scan results under caller-generated content_id in a thread-safe ContentStore.
- Stored L2 scores are retrieved by POST /v1/authorize_action via context_refs:
    p_L2 = max over content_id in context_refs
- Supports 'enforce' (default) and 'shadow' deployment modes.
- Fail closed: unhandled errors return STEP_UP; endpoints never raise unhandled 500s.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

try:
    from sentinel_contracts import L1In, L2In, LayerOut, Verdict
except ImportError:
    from sentinel_l2._contracts_fallback import L2In, LayerOut, Verdict  # fallback if uninstalled
    from dataclasses import dataclass as _dc

    @_dc
    class L1In:  # type: ignore
        request_id: str
        agent_id: Optional[str] = None
        user_id: Optional[str] = None
        transaction_id: Optional[str] = None
        url: Optional[str] = None
        page_content: Optional[str] = None
        payload: Dict[str, Any] = field(default_factory=dict)

# Import layer entrypoints
try:
    from sentinel_l1.api import run_l1
except ImportError:
    try:
        from layer1_heuristics.sentinel_l1.api import run_l1
    except ImportError:
        from sentinel_contracts.stubs import run_l1_stub as run_l1

try:
    from sentinel_l2.api import run_l2
except ImportError:
    from layer2_prompt_detection.sentinel_l2.api import run_l2

log = logging.getLogger(__name__)

router = APIRouter(tags=["ingress"])


# ============================================================================
# Schemas
# ============================================================================

class ScanContentRequest(BaseModel):
    content_id: str = Field(..., description="Caller-generated unique content identifier")
    source_type: str = Field("untrusted_content", description="Origin: email, web_page, invoice, chat, etc.")
    text: Optional[str] = Field(None, description="Raw text content to inspect for malicious prompts")
    url: Optional[str] = Field(None, description="Counterparty URL to inspect with L1 heuristics")
    html: Optional[str] = Field(None, description="Raw HTML if web page was fetched")
    pdf_b64: Optional[str] = Field(None, description="Base64 encoded PDF if applicable")
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    mode: str = Field("enforce", description="Deployment mode: 'enforce' or 'shadow'")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScanContentResponse(BaseModel):
    content_id: str
    verdict: Verdict
    risk_score: float
    p_l2: float
    zone: str = Field(..., description="L2 decision zone: ALLOW, GRAY, BLOCK")
    reason_codes: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float
    mode: str = "enforce"
    l1_score: Optional[float] = None
    l2_score: Optional[float] = None
    stored: bool = True


# ============================================================================
# Content Store (for context_refs reuse in /v1/authorize_action)
# ============================================================================

@dataclass
class ContentScanRecord:
    content_id: str
    source_type: str
    verdict: Verdict
    risk_score: float
    p_l2: float
    zone: str
    reason_codes: List[str]
    evidence: List[Dict[str, Any]]
    timestamp: float
    l1_score: Optional[float] = None
    l2_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentStore:
    """Thread-safe LRU store of scanned content scores and evidence."""

    def __init__(self, capacity: int = 10000):
        self._store: "OrderedDict[str, ContentScanRecord]" = OrderedDict()
        self._lock = threading.Lock()
        self._capacity = capacity

    def put(self, record: ContentScanRecord) -> None:
        with self._lock:
            self._store[record.content_id] = record
            self._store.move_to_end(record.content_id)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def get(self, content_id: str) -> Optional[ContentScanRecord]:
        with self._lock:
            rec = self._store.get(content_id)
            if rec is not None:
                self._store.move_to_end(content_id)
            return rec

    def get_max_l2(self, content_ids: Iterable[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Handbook Section 2.4: p_L2 = max over content_id in context_refs."""
        max_p = 0.0
        all_evidence: List[Dict[str, Any]] = []
        with self._lock:
            for cid in content_ids:
                rec = self._store.get(cid)
                if rec is not None:
                    if rec.p_l2 > max_p:
                        max_p = rec.p_l2
                    if rec.evidence:
                        all_evidence.extend(rec.evidence)
        return max_p, all_evidence

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


# Default singleton content store
content_store = ContentStore()


# ============================================================================
# Core Ingress Scanning Logic
# ============================================================================

def scan_content_internal(request: ScanContentRequest) -> ScanContentResponse:
    """Run L1 and L2 on content, store result under content_id, and return aggregated verdict."""
    t0 = time.perf_counter()

    try:
        # 1. Run Layer 1 if URL or text/HTML is provided
        l1_out: Optional[LayerOut] = None
        if request.url is not None or request.html is not None or request.text is not None:
            l1_in = L1In(
                request_id=request.content_id,
                agent_id=request.agent_id,
                user_id=request.user_id,
                url=request.url,
                page_content=request.html or request.text,
                payload={
                    "source_type": request.source_type,
                    "text": request.text,
                    "html": request.html,
                    "pdf_b64": request.pdf_b64,
                },
            )
            try:
                l1_out = run_l1(l1_in)
            except Exception:
                log.exception("L1 execution failed during content scan; failing closed to STEP_UP")
                l1_out = LayerOut(
                    layer="L1",
                    score=0.5,
                    verdict=Verdict.STEP_UP,
                    reason_codes=["SYSTEM:L1_ERROR"],
                    model_name="l1",
                    latency_ms=0.0,
                )

        # 2. Run Layer 2 on text (or HTML fallback)
        l2_text = request.text or request.html or ""
        l2_in = L2In(
            request_id=request.content_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            prompt=l2_text,
            payload={"source_type": request.source_type},
        )
        try:
            l2_out = run_l2(l2_in)
        except Exception:
            log.exception("L2 execution failed during content scan; failing closed to STEP_UP")
            l2_out = LayerOut(
                layer="L2",
                score=0.5,
                verdict=Verdict.STEP_UP,
                reason_codes=["SYSTEM:L2_ERROR"],
                model_name="deberta-v3-base-injection",
                latency_ms=0.0,
                metadata={"zone": "ERROR"},
            )

        # 3. Combine layer results according to handbook decision policy
        p_l2 = float(l2_out.score)
        zone_l2 = str((l2_out.metadata or {}).get("zone", "ALLOW"))
        evidence = list((l2_out.metadata or {}).get("evidence", []))

        reason_codes: List[str] = []
        if l1_out and l1_out.reason_codes:
            reason_codes.extend(l1_out.reason_codes)
        if l2_out and l2_out.reason_codes:
            reason_codes.extend(l2_out.reason_codes)

        # Risk score is maximum observed risk across active layers
        l1_score = float(l1_out.score) if l1_out is not None else 0.0
        risk_score = max(l1_score, p_l2)

        # Verdict calculation
        if (l1_out and l1_out.verdict == Verdict.BLOCK) or l2_out.verdict == Verdict.BLOCK:
            verdict = Verdict.BLOCK
        elif (l1_out and l1_out.verdict == Verdict.STEP_UP) or l2_out.verdict == Verdict.STEP_UP:
            verdict = Verdict.STEP_UP
        else:
            verdict = Verdict.ALLOW

        latency_ms = (time.perf_counter() - t0) * 1000

        # In shadow mode, the system records everything but returns ALLOW for non-disruptive monitoring
        reported_verdict = verdict
        if request.mode == "shadow":
            reported_verdict = Verdict.ALLOW

        # 4. Save record in ContentStore under caller-generated content_id
        record = ContentScanRecord(
            content_id=request.content_id,
            source_type=request.source_type,
            verdict=verdict,
            risk_score=risk_score,
            p_l2=p_l2,
            zone=zone_l2,
            reason_codes=reason_codes,
            evidence=evidence,
            timestamp=time.time(),
            l1_score=l1_score,
            l2_score=p_l2,
            metadata={"mode": request.mode, "raw_verdict": verdict.value},
        )
        content_store.put(record)

        return ScanContentResponse(
            content_id=request.content_id,
            verdict=reported_verdict,
            risk_score=risk_score,
            p_l2=p_l2,
            zone=zone_l2,
            reason_codes=reason_codes,
            evidence=evidence,
            latency_ms=latency_ms,
            mode=request.mode,
            l1_score=l1_score,
            l2_score=p_l2,
            stored=True,
        )

    except Exception as exc:
        log.exception("Unexpected error in /v1/scan_content; failing closed to STEP_UP")
        latency_ms = (time.perf_counter() - t0) * 1000
        return ScanContentResponse(
            content_id=request.content_id,
            verdict=Verdict.STEP_UP,
            risk_score=0.5,
            p_l2=0.5,
            zone="ERROR",
            reason_codes=["SYSTEM:SCAN_CONTENT_ERROR"],
            evidence=[{"kind": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}],
            latency_ms=latency_ms,
            mode=request.mode,
            stored=False,
        )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/v1/scan_content", response_model=ScanContentResponse)
def scan_content(request: ScanContentRequest) -> ScanContentResponse:
    return scan_content_internal(request)


@router.get("/v1/scanned_content/{content_id}")
def get_scanned_content(content_id: str):
    rec = content_store.get(content_id)
    if rec is None:
        return {"content_id": content_id, "found": False}
    return {
        "content_id": rec.content_id,
        "found": True,
        "source_type": rec.source_type,
        "verdict": rec.verdict.value,
        "risk_score": rec.risk_score,
        "p_l2": rec.p_l2,
        "zone": rec.zone,
        "reason_codes": rec.reason_codes,
        "evidence": rec.evidence,
        "timestamp": rec.timestamp,
    }

