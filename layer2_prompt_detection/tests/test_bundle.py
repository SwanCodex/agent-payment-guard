import json

import pytest

from sentinel_l2.models.bundle import load_bundle


def _write(d, cal=None, thr=None, model=True):
    (d / "calibration.json").write_text(json.dumps(cal if cal is not None else {"method": "temperature+prior_shift", "T": 0.624, "delta": -4.6}))
    (d / "thresholds.json").write_text(json.dumps(thr if thr is not None else {"tau_allow": 0.0201, "tau_block": 0.05}))
    (d / "manifest.json").write_text(json.dumps({"model": "microsoft/deberta-v3-base", "max_len": 256}))
    if model:
        (d / "model.pt").write_bytes(b"x")


def test_loads_a_valid_bundle(tmp_path):
    _write(tmp_path)
    b = load_bundle(tmp_path)
    assert (b.T, b.delta, b.tau_block, b.tau_allow, b.max_len) == (0.624, -4.6, 0.05, 0.0201, 256)
    assert b.versions["base_model"] == "microsoft/deberta-v3-base"


def test_missing_prior_shift_offset_is_rejected(tmp_path):
    _write(tmp_path, cal={"T": 0.624})
    with pytest.raises(ValueError, match="delta"):
        load_bundle(tmp_path)


@pytest.mark.parametrize("thr", [{"tau_allow": 0.02}, {"tau_allow": 0.02, "tau_block": 1.5}])
def test_bad_thresholds_are_rejected(tmp_path, thr):
    _write(tmp_path, thr=thr)
    with pytest.raises(ValueError):
        load_bundle(tmp_path)


def test_missing_weights_are_rejected(tmp_path):
    _write(tmp_path, model=False)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path)


def test_crossed_thresholds_are_accepted(tmp_path):
    _write(tmp_path, thr={"tau_allow": 0.0986, "tau_block": 0.0212})
    b = load_bundle(tmp_path)
    assert b.tau_allow > b.tau_block
