"""Loads and validates an L2 artifact bundle (calibration.json, thresholds.json, manifest.json, model.pt)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BUNDLE_REL = Path("artifacts") / "layer2" / "deberta_v3_base_injection" / "0.1.0"


@dataclass(frozen=True)
class Bundle:
    dir: Optional[Path]
    T: float
    delta: float
    tau_block: float
    tau_allow: float
    model_name: str = "microsoft/deberta-v3-base"
    version: str = "0.1.0"
    max_len: int = 256

    @property
    def versions(self) -> dict[str, str]:
        return {"l2_deberta": self.version, "base_model": self.model_name}


def default_bundle_dir() -> Path:
    env = os.environ.get("SENTINEL_L2_BUNDLE")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        cand = parent / BUNDLE_REL
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"bundle not found: set SENTINEL_L2_BUNDLE or run from a checkout containing {BUNDLE_REL}")


def load_bundle(path: Optional[os.PathLike] = None) -> Bundle:
    d = Path(path) if path else default_bundle_dir()
    cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    thr = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}

    if "T" not in cal or "delta" not in cal:
        raise ValueError("calibration.json must contain 'T' and 'delta' (prior-shift offset); see model card section 1")
    for k in ("tau_block", "tau_allow"):
        if k not in thr or not (0.0 <= float(thr[k]) <= 1.0):
            raise ValueError(f"thresholds.json: '{k}' missing or outside [0, 1]")
    if float(cal["T"]) <= 0:
        raise ValueError("calibration.json: T must be positive")
    if not ((d / "model.pt").exists() or (d / "model_fp16.pt").exists()):
        raise FileNotFoundError(f"no model.pt in {d}")

    return Bundle(
        dir=d, T=float(cal["T"]), delta=float(cal["delta"]),
        tau_block=float(thr["tau_block"]), tau_allow=float(thr["tau_allow"]),
        model_name=man.get("model", "microsoft/deberta-v3-base"),
        version=d.name, max_len=int(man.get("max_len", 256)),
    )
