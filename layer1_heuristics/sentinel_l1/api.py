import json
import threading
import time
from pathlib import Path
from typing import Optional, Tuple
from sentinel_contracts import L1In, LayerOut, Verdict
from .rules import run_rules

# Interim placeholders for rule severities (the contract has no separate flags/p fields yet; see RFC note).
_FLAG_SCORE = {"gray": 0.30, "step_up": 0.60, "block": 0.95}

_MODEL_LOCK = threading.Lock()
_MODEL = None
_MODEL_LAST_TRY = 0.0
_MODEL_ERROR: Optional[str] = None
_MODEL_VERSION: Optional[str] = None
COOLDOWN_SECONDS = 30.0


def reset_model_cache():
    """Testing helper to reset loaded model cache and cooldown."""
    global _MODEL, _MODEL_LAST_TRY, _MODEL_ERROR, _MODEL_VERSION
    with _MODEL_LOCK:
        _MODEL = None
        _MODEL_LAST_TRY = 0.0
        _MODEL_ERROR = None
        _MODEL_VERSION = None


def get_url_model(force_reload: bool = False):
    """Load the URL ensemble once with thread-safety and cooldown on failures."""
    global _MODEL, _MODEL_LAST_TRY, _MODEL_ERROR, _MODEL_VERSION
    with _MODEL_LOCK:
        now = time.monotonic()
        if _MODEL is not None and not force_reload:
            return _MODEL, None, _MODEL_VERSION

        if not force_reload and (now - _MODEL_LAST_TRY < COOLDOWN_SECONDS) and _MODEL_ERROR is not None:
            return None, _MODEL_ERROR, _MODEL_VERSION

        _MODEL_LAST_TRY = now
        try:
            from .models.url_ensemble import UrlEnsemble, default_dir
            d = default_dir()
            if not d.exists():
                _MODEL_ERROR = f"artifact directory not found: {d}"
                _MODEL = None
                _MODEL_VERSION = None
                return None, _MODEL_ERROR, None

            # Attempt to read version from manifest if available
            try:
                m_path = d / "manifest.json"
                if m_path.exists():
                    _MODEL_VERSION = json.loads(m_path.read_text(encoding="utf-8")).get("version", d.name)
                else:
                    _MODEL_VERSION = d.name
            except Exception:
                _MODEL_VERSION = d.name

            _MODEL = UrlEnsemble(d)
            _MODEL_VERSION = _MODEL.version
            _MODEL_ERROR = None
            return _MODEL, None, _MODEL_VERSION
        except Exception as e:
            _MODEL_ERROR = str(e)
            _MODEL = None
            return None, _MODEL_ERROR, _MODEL_VERSION


def run_l1(request: L1In) -> LayerOut:
    t0 = time.perf_counter()
    res = run_rules(request)
    flags = res.flags
    codes = {f.code for f in flags}
    meta = {"flags": [vars(f) for f in flags], "forwarded": res.forwarded}

    p = None
    model_contrib = None
    model_status = "not_applicable"
    model_reason = None
    model_version = None
    model = None

    if request.url:
        model, load_err, ver = get_url_model()
        model_version = ver
        if model is None:
            model_status = "unavailable"
            model_reason = load_err
            codes.add("SYSTEM:MODEL_UNAVAILABLE")
            model_contrib = 0.60
        else:
            try:
                r = model.score(request.url)
                p = r["p"]
                model_status = "ok"
                model_reason = None
                model_version = model.version
                meta["url_model"] = {
                    "status": "ok",
                    "reason": None,
                    "version": model.version,
                    "p": round(p, 6),
                    "z": round(r["z"], 4),
                    "gray": r["gray"],
                    "abstain": r["abstain"],
                    "tau_allow": model.tau_allow,
                    "tau_block": model.tau_block,
                }
                if p >= model.tau_block:
                    codes.add("L1:URL_MODEL_HIGH_RISK")
                    model_contrib = 0.60
                else:
                    model_contrib = min(p, 0.49)
            except Exception as e:
                model_status = "error"
                model_reason = repr(e)
                codes.add("SYSTEM:MODEL_UNAVAILABLE")
                model_contrib = 0.60
                p = None
    else:
        meta["url_model"] = {
            "status": "not_applicable",
            "reason": None,
            "version": None,
        }

    if model_status in ("unavailable", "error"):
        meta["url_model"] = {
            "status": model_status,
            "reason": model_reason,
            "version": model_version,
        }

    block = any(f.severity == "block" for f in flags)
    failure_forces_step_up = model_status in ("unavailable", "error")

    step_up = (
        any(f.severity == "step_up" for f in flags)
        or (p is not None and model is not None and p >= model.tau_block)
        or failure_forces_step_up
    )
    verdict = Verdict.BLOCK if block else (Verdict.STEP_UP if step_up else Verdict.ALLOW)

    flag_score = max((_FLAG_SCORE[f.severity] for f in flags), default=0.0)
    if model_contrib is not None:
        score = max(flag_score, model_contrib)
    else:
        score = flag_score

    meta["score_semantics"] = "policy_contribution_not_a_probability"
    meta["score_components"] = {
        "rule_flags": round(flag_score, 4),
        "url_model_policy": round(model_contrib, 4) if model_contrib is not None else None,
    }

    return LayerOut(
        layer="L1",
        score=score,
        verdict=verdict,
        reason_codes=sorted(codes),
        model_name="l1-rules+url-ensemble" if (p is not None and model_status == "ok") else "l1-rules",
        model_version=model_version or "0.2.0",
        latency_ms=(time.perf_counter() - t0) * 1000,
        metadata=meta,
    )
