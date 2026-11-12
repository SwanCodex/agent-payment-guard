"""Local copy of the handbook's section 5.1 schemas.

Used only when `sentinel_contracts` is not installed. When contracts/ is frozen, api.py imports the real ones
and this file can be deleted. Keep field names identical to the handbook.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Flag(BaseModel):
    code: str
    severity: Literal["gray", "step_up", "block"]
    evidence: str                      # <= 300 chars
    span: Optional[list[int]] = None


class LayerOut(BaseModel):
    layer: Literal["L1", "L2", "L3"]
    p: Optional[float]                 # calibrated P(risky); None if the layer emitted only flags
    abstain: bool                      # conformal set contains both classes
    flags: list[Flag] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    model_versions: dict[str, str]
    latency_ms: float


class L2In(BaseModel):
    source_type: str
    text: str
