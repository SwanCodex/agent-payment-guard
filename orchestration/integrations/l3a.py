
import json
import logging
import threading
import time
from pathlib import Path

from sentinel_contracts import L3In, LayerOut, Verdict
from layer3_fraud_tabular.sentinel_l3a.api import L3AScorer


logger = logging.getLogger(__name__)

BUNDLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "layer3_tabular"
    / "lgbm_core"
    / "1.0.0"
)

REQUIRED_FIELDS = (
    "amount",
    "event_time",
    "card_key",
    "addr_key",
    "device_key",
)

SCORER = None
THRESHOLDS = None
LOAD_ERROR = None
LOAD_LOCK = threading.Lock()


def _load_scorer():
    global SCORER, THRESHOLDS, LOAD_ERROR

    if SCORER is not None:
        return SCORER

    with LOAD_LOCK:
        if SCORER is not None:
            return SCORER

        try:
            if not BUNDLE_DIR.exists():
                raise FileNotFoundError(
                    f"L3A bundle not found: {BUNDLE_DIR}"
                )

            with open(BUNDLE_DIR / "thresholds.json", encoding="utf-8") as f:
                thresholds = json.load(f)

            tau_allow = float(thresholds["tau_allow"])
            tau_block = float(thresholds["tau_block"])

            if not 0 <= tau_allow < tau_block <= 1:
                raise ValueError("Invalid L3A thresholds")

            scorer = L3AScorer(
                str(BUNDLE_DIR),
                str(BUNDLE_DIR),
            )

            SCORER = scorer
            THRESHOLDS = thresholds
            LOAD_ERROR = None

            return SCORER

        except Exception as exc:
            LOAD_ERROR = str(exc)
            logger.exception("Unable to load L3A scorer")
            raise


def _unavailable(reason: str, latency_ms: float = 0.0) -> LayerOut:
    return LayerOut(
        layer="L3A",
        score=0.0,
        verdict=Verdict.STEP_UP,
        reason_codes=["SYSTEM:MODEL_UNAVAILABLE", reason],
        model_name="l3a-lightgbm-ensemble",
        model_version="1.0.0",
        latency_ms=latency_ms,
        metadata={
            "status": "unavailable",
            "bundle": "lgbm_core/1.0.0",
        },
    )


def run_l3a(request: L3In) -> LayerOut:
    started = time.perf_counter()

    txn = request.transaction_features or {}

    missing = [
        field for field in REQUIRED_FIELDS
        if field not in txn or txn[field] is None
    ]

    if missing:
        return _unavailable(
            "L3A:REQUIRED_FEATURES_MISSING:" + ",".join(missing)
        )

    try:
        scorer = _load_scorer()

        # Validate basic numeric inputs before model inference.
        amount = float(txn["amount"])
        event_time = float(txn["event_time"])

        if amount < 0:
            return _unavailable("L3A:INVALID_AMOUNT")

        if not amount < float("inf"):
            return _unavailable("L3A:INVALID_AMOUNT")

        if not event_time < float("inf"):
            return _unavailable("L3A:INVALID_EVENT_TIME")

        if not all(
            isinstance(txn[key], str) and txn[key].strip()
            for key in ("card_key", "addr_key", "device_key")
        ):
            return _unavailable("L3A:INVALID_ENTITY_KEY")

        # Do not mutate entity history during inference.
        result = scorer.score(txn, explain=False, commit=False)

        probability = float(result["p"])
        raw_probability = float(result["p_raw"])

        if not 0 <= probability <= 1:
            raise ValueError("Calibrated probability outside [0, 1]")

        if not 0 <= raw_probability <= 1:
            raise ValueError("Raw probability outside [0, 1]")

        tau_allow = float(THRESHOLDS["tau_allow"])
        tau_block = float(THRESHOLDS["tau_block"])

        if probability >= tau_block:
            verdict = Verdict.BLOCK
        elif probability >= tau_allow:
            verdict = Verdict.STEP_UP
        else:
            verdict = Verdict.ALLOW

        reason_codes = [
            "L3A:FRAUD_RISK_HIGH"
            if verdict == Verdict.BLOCK
            else "L3A:FRAUD_RISK_REVIEW"
            if verdict == Verdict.STEP_UP
            else "L3A:LOW_MODEL_RISK"
        ]

        latency_ms = (time.perf_counter() - started) * 1000

        return LayerOut(
            layer="L3A",
            score=probability,
            verdict=verdict,
            reason_codes=reason_codes,
            model_name="l3a-lightgbm-ensemble",
            model_version="1.0.0",
            latency_ms=latency_ms,
            metadata={
                "status": "evaluated",
                "p_calibrated": probability,
                "p_raw": raw_probability,
                "warm": result["warm"],
                "threshold_allow": tau_allow,
                "threshold_block": tau_block,
                "history_committed": False,
                "bundle": "lgbm_core/1.0.0",
            },
        )

    except Exception:
        logger.exception("L3A inference failed")
        return _unavailable(
            "L3A:INFERENCE_FAILED",
            (time.perf_counter() - started) * 1000,
        )
