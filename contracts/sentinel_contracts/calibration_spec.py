from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationSpec:
    method: str
    calibrated: bool
    validation_split: str
    score_min: float = 0.0
    score_max: float = 1.0
