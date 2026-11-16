import numpy as np

from sentinel_l2.calibration import calibrated, sigmoid, verdict_from_p


def test_formula():
    z, T, delta = 2.0, 0.624, -4.6
    assert np.isclose(calibrated(z, T, delta), 1 / (1 + np.exp(-(z / T + delta))))


def test_zero_margin_gives_prior_shift_only():
    assert np.isclose(calibrated(0.0, 0.7, -4.6), sigmoid(-4.6))


def test_monotone_in_z():
    zs = np.linspace(-10, 10, 200)
    p = calibrated(zs, 0.62, -4.6)
    assert np.all(np.diff(p) >= 0) and p.min() >= 0 and p.max() <= 1


def test_extreme_values_do_not_overflow():
    assert calibrated(1e6, 0.6, -4.6) == 1.0 or calibrated(1e6, 0.6, -4.6) > 0.999999
    assert calibrated(-1e6, 0.6, -4.6) < 1e-6


def test_verdicts_ordered_thresholds():
    assert verdict_from_p(0.9, 0.05, 0.02) == "BLOCK"
    assert verdict_from_p(0.03, 0.05, 0.02) == "GRAY"
    assert verdict_from_p(0.01, 0.05, 0.02) == "ALLOW"


def test_verdicts_crossed_thresholds_have_no_gray_and_block_wins():
    # tau_allow >= tau_block (seeds 2 and 3 of the pilot)
    assert verdict_from_p(0.05, 0.0212, 0.0986) == "BLOCK"
    assert verdict_from_p(0.02, 0.0212, 0.0986) == "ALLOW"
    assert all(verdict_from_p(p, 0.0212, 0.0986) != "GRAY" for p in np.linspace(0, 1, 101))
