import time
from sentinel_contracts import L1In, LayerOut, Verdict
from .rules import run_rules

# Interim placeholders for rule severities (the contract has no separate flags/p fields yet; see RFC note).
_FLAG_SCORE = {"gray": 0.30, "step_up": 0.60, "block": 0.95}

_MODEL = None
_MODEL_STATE = {"tried": False, "error": None}


def get_url_model():
    """Load the URL ensemble once. If artifacts or lightgbm are missing, fall back to rules only
    and say so in the output metadata (never silently)."""
    if not _MODEL_STATE["tried"]:
        _MODEL_STATE["tried"] = True
        global _MODEL
        try:
            from .models.url_ensemble import UrlEnsemble, default_dir
            d = default_dir()
            if d.exists():
                _MODEL = UrlEnsemble(d)
            else:
                _MODEL_STATE["error"] = f"artifact directory not found: {d}"
        except Exception as e:                      # noqa: BLE001
            _MODEL_STATE["error"] = repr(e)
    return _MODEL


def run_l1(request: L1In) -> LayerOut:
    t0 = time.perf_counter()
    res = run_rules(request)
    flags = res.flags
    codes = {f.code for f in flags}
    meta = {"flags": [vars(f) for f in flags], "forwarded": res.forwarded}

    p, model = None, None
    if request.url:
        model = get_url_model()
        if model is None:
            meta["url_model"] = f"unavailable: {_MODEL_STATE['error']}"
        else:
            try:
                r = model.score(request.url)
                p = r["p"]
                meta["url_model"] = {"version": model.version, "p": round(p, 6), "z": round(r["z"], 4),
                                     "gray": r["gray"], "abstain": r["abstain"],
                                     "tau_allow": model.tau_allow, "tau_block": model.tau_block}
                if p >= model.tau_block:
                    codes.add("L1:URL_MODEL_HIGH_RISK")
            except Exception as e:                  # noqa: BLE001
                meta["url_model"] = f"error: {e!r}"
                p = None

    block = any(f.severity == "block" for f in flags) or (p is not None and p >= model.tau_block)
    step_up = any(f.severity == "step_up" for f in flags)
    verdict = Verdict.BLOCK if block else (Verdict.STEP_UP if step_up else Verdict.ALLOW)
    flag_score = max((_FLAG_SCORE[f.severity] for f in flags), default=0.0)
    score = max(flag_score, p) if p is not None else flag_score

    return LayerOut(
        layer="L1",
        score=score,
        verdict=verdict,
        reason_codes=sorted(codes),
        model_name="l1-rules+url-ensemble" if p is not None else "l1-rules",
        model_version=model.version if p is not None else "0.2.0",
        latency_ms=(time.perf_counter() - t0) * 1000,
        metadata=meta,
    )