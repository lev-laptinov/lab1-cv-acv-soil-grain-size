import numpy as np
import pytest

from soilgsd.metric import (
    LOG_WIDTHS,
    d_value,
    emd,
    emd_per_sample,
    is_valid_curve,
    repair_cummax,
    repair_isotonic,
    shift_curve,
)

UNIFORM = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])


def test_perfect_prediction_is_zero():
    assert emd(UNIFORM, UNIFORM) == 0.0


def test_range_is_0_to_500():
    # All mass below 0.002 mm vs all mass above 63 mm: the curves differ by 100 % everywhere.
    fine = np.full(11, 100.0)
    coarse = np.r_[np.zeros(10), 100.0]
    assert LOG_WIDTHS.sum() == pytest.approx(5.0)
    assert emd(fine, coarse) == pytest.approx(500.0)


def test_constant_offset_costs_offset_times_five_decades():
    y = np.r_[np.full(10, 40.0), 100.0]
    assert emd(y, y + np.r_[np.full(10, 10.0), 0.0]) == pytest.approx(50.0)


def test_per_sample_then_mean():
    a, b = UNIFORM, np.clip(UNIFORM + 5, 0, 100)
    per = emd_per_sample(np.stack([a, a]), np.stack([a, b]))
    assert per[0] == 0 and emd(np.stack([a, a]), np.stack([a, b])) == pytest.approx(per[1] / 2)


def test_assignment_exercise_7():
    # Exercise 7 uses a 3-point grid; check the same formula by hand on it.
    d = np.log10([0.1, 1, 10])
    p, t = np.array([20, 50, 100]), np.array([40, 60, 100])
    assert (np.abs(p - t)[:-1] @ np.diff(d)) == pytest.approx(30.0)


def test_two_x_scale_error_costs_about_30():
    # A soil whose whole curve lies well inside the sieve range: a 2x scale error moves the
    # curve by log10(2) decades, and the area between the curves is 100 % * 0.301.
    y = np.array([0, 0, 0, 0, 10, 40, 80, 100, 100, 100, 100], dtype=float)
    assert emd(y, shift_curve(y, 2.0)) == pytest.approx(100 * np.log10(2), rel=0.02)


def test_validity_and_repairs():
    bad = np.array([10, 35, 30, 80, 105, 100, 100, 100, 100, 100, 99], dtype=float)
    assert not is_valid_curve(bad)[0]
    for fix in (repair_cummax, repair_isotonic):
        assert is_valid_curve(fix(bad))[0]
    # Exercise 6: running max lifts 30 to 35, PAV pools (35, 30) to 32.5.
    assert repair_cummax(bad)[0, 2] == 35
    assert repair_isotonic(bad)[0, 1:3] == pytest.approx([32.5, 32.5])


def test_d50_log_interpolation():
    # Exercise 7.2 on the competition grid: 50 % reached between 0.2 mm (40 %) and 0.63 mm (60 %).
    y = np.array([0, 0, 0, 0, 40, 60, 100, 100, 100, 100, 100], dtype=float)
    expected = 10 ** (np.log10(0.2) + 0.5 * (np.log10(0.63) - np.log10(0.2)))
    assert d_value(y, 50)[0] == pytest.approx(expected)
    assert np.isnan(d_value(np.full(11, 100.0), 10)[0])  # >10 % clay: D10 is undefined
