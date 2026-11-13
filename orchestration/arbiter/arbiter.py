"""LLM Arbiter: a third opinion for the gray zone only.

Rules from the handbook (2.5): no tools, temperature 0, one call, 3 s timeout, cache by case hash, per-session cap,
strict JSON output, anything else becomes STEP_UP, and an ALLOW is honored only up to a policy amount.
Every path that cannot produce a clean answer fails closed to STEP_UP; `arbitrate` never raises.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

from .client import LLMClient
from .prompt import build_messages
from .schemas import ArbiterCase, ArbiterResult, Verdict

log = logging.getLogger(__name__)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,39}$")
_VERDICTS = {"ALLOW": Verdict.ALLOW, "STEP_UP": Verdict.STEP_UP, "BLOCK": Verdict.BLOCK}


class BadOutput(ValueError):
    pass


@dataclass
class ArbiterPolicy:
    allow_max_amount: float = 0.0        # an LLM ALLOW is honored only up to this amount; 0 = never (safest default)
    timeout_s: float = 3.0
    max_calls_per_session: int = 20
    cache_size: int = 1024


def parse_verdict(raw: str) -> Tuple[Verdict, List[str], float]:
    """Strict: one JSON object with exactly verdict, reason_codes, confidence. A single code fence is tolerated."""
    s = (raw or "").strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        raise BadOutput(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict) or set(obj) != {"verdict", "reason_codes", "confidence"}:
        raise BadOutput("wrong keys")
    if obj["verdict"] not in _VERDICTS:
        raise BadOutput("bad verdict")
    codes = obj["reason_codes"]
    if not isinstance(codes, list) or len(codes) > 5 or not all(isinstance(c, str) and _CODE.match(c) for c in codes):
        raise BadOutput("bad reason_codes")
    conf = obj["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
        raise BadOutput("bad confidence")
    return _VERDICTS[obj["verdict"]], list(codes), float(conf)


class Arbiter:
    def __init__(self, client: LLMClient, policy: Optional[ArbiterPolicy] = None,
                 on_case: Optional[Callable[[ArbiterCase, ArbiterResult], None]] = None):
        self.client, self.policy, self.on_case = client, policy or ArbiterPolicy(), on_case
        self._cache: "OrderedDict[str, ArbiterResult]" = OrderedDict()
        self._calls: dict = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="arbiter")

    # ---- helpers -------------------------------------------------------------------------------------------
    def _fail(self, code: str, t0: float, raw: Optional[str] = None) -> ArbiterResult:
        return ArbiterResult(verdict=Verdict.STEP_UP, reason_codes=[code], confidence=0.0, source="fail_closed",
                             latency_ms=(time.perf_counter() - t0) * 1000, raw=(raw or None) and raw[:200])

    def _finish(self, case: ArbiterCase, res: ArbiterResult) -> ArbiterResult:
        if self.on_case is not None:
            try:
                self.on_case(case, res)
            except Exception:                                  # an audit-store problem must not change the verdict
                log.exception("arbiter audit hook failed")
        return res

    # ---- main entry ----------------------------------------------------------------------------------------
    def arbitrate(self, case: ArbiterCase, session_id: str = "default") -> ArbiterResult:
        t0 = time.perf_counter()
        key = case.cache_key()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return self._finish(case, replace(hit, source="cache", latency_ms=(time.perf_counter() - t0) * 1000))
            if self._calls.get(session_id, 0) >= self.policy.max_calls_per_session:
                return self._finish(case, self._fail("SYSTEM:ARBITER_CAP_REACHED", t0))
            self._calls[session_id] = self._calls.get(session_id, 0) + 1

        system, user, _ = build_messages(case)
        future = self._pool.submit(self.client.complete, system, user, timeout_s=self.policy.timeout_s)
        try:
            raw = future.result(timeout=self.policy.timeout_s)
        except FutureTimeout:
            future.cancel()
            return self._finish(case, self._fail("SYSTEM:ARBITER_TIMEOUT", t0))
        except Exception:
            log.exception("arbiter LLM call failed")
            return self._finish(case, self._fail("SYSTEM:ARBITER_ERROR", t0))

        try:
            verdict, codes, conf = parse_verdict(raw)
        except BadOutput:
            return self._finish(case, self._fail("SYSTEM:ARBITER_BAD_OUTPUT", t0, raw))

        source = "llm"
        if verdict == Verdict.ALLOW:
            ok = case.amount is not None and case.amount <= self.policy.allow_max_amount
            if not ok:                                         # unknown amount counts as over the cap
                verdict, codes, source = Verdict.STEP_UP, (codes + ["POLICY:ARBITER_ALLOW_AMOUNT_CAP"])[:6], "policy"
        res = ArbiterResult(verdict=verdict, reason_codes=codes, confidence=conf, source=source,
                            latency_ms=(time.perf_counter() - t0) * 1000, raw=raw[:200])
        with self._lock:                                       # only real answers are cached, never failures
            self._cache[key] = res
            while len(self._cache) > self.policy.cache_size:
                self._cache.popitem(last=False)
        return self._finish(case, res)
