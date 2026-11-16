"""Builds an ArbiterCase from the repo's LayerOut objects."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .schemas import MAX_EVIDENCE_CHARS, MAX_EVIDENCE_ITEMS, ArbiterCase, LayerSummary


def _evidence_texts(layer_out) -> List[str]:
    out = []
    for e in (getattr(layer_out, "metadata", None) or {}).get("evidence", []) or []:
        if isinstance(e, dict) and e.get("kind") == "span" and e.get("text"):
            out.append(str(e["text"]))
    return out


def build_case(action: str, layer_outs: Iterable, *, action_args: Optional[Dict[str, Any]] = None,
               amount: Optional[float] = None, txn_summary: Optional[str] = None,
               extra_evidence: Optional[List[str]] = None) -> ArbiterCase:
    layers, evidence = {}, []
    for lo in layer_outs:
        verdict = getattr(lo.verdict, "value", lo.verdict)
        layers[lo.layer] = LayerSummary(
            score=lo.score, verdict=str(verdict), reason_codes=list(lo.reason_codes or [])[:5],
            zone=(getattr(lo, "metadata", None) or {}).get("zone"),
        )
        evidence += _evidence_texts(lo)
    evidence += list(extra_evidence or [])

    seen, picked = set(), []
    for t in evidence:
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            picked.append(t[:MAX_EVIDENCE_CHARS])
        if len(picked) == MAX_EVIDENCE_ITEMS:
            break
    return ArbiterCase(action=action, layers=layers, action_args=dict(action_args or {}), amount=amount,
                       txn_summary=(txn_summary[:500] if txn_summary else None), evidence=picked)
