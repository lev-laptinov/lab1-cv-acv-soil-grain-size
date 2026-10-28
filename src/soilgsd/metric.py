"""Competition metric (log-weighted EMD), curve validity repairs and D-value metrics."""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from soilgsd.data import SIEVES_MM

LOG_D = np.log10(SIEVES_MM)
LOG_WIDTHS = np.diff(LOG_D)  # 10 intervals, each ~0.5 decade, 5 decades in total


def emd_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Area between cumulative curves on the log10(d) axis, in %·decades.

    Left-point rule over the 10 intervals: sum_i |F_i - F^_i| * (log10 x_{i+1} - log10 x_i).
    The value at 200 mm is 100 for every valid curve, so it never contributes.
    """
    y_true = np.atleast_2d(np.asarray(y_true, dtype=float))
    y_pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    return np.abs(y_true[:, :-1] - y_pred[:, :-1]) @ LOG_WIDTHS


def emd_parts(y_true: np.ndarray, y_pred: np.ndarray, split_mm: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample EMD split at split_mm into the fine part (intervals below) and the coarse part.

    The two parts add up to emd_per_sample, so each one shows where on the curve the error is.
    """
    y_true = np.atleast_2d(np.asarray(y_true, dtype=float))
    y_pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    err = np.abs(y_true[:, :-1] - y_pred[:, :-1]) * LOG_WIDTHS
    k = int(np.searchsorted(SIEVES_MM, split_mm))
    return err[:, :k].sum(1), err[:, k:].sum(1)


def emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(emd_per_sample(y_true, y_pred).mean())


def is_valid_curve(y: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    y = np.atleast_2d(y)
    in_range = (y >= -atol).all(1) & (y <= 100 + atol).all(1)
    monotone = (np.diff(y, axis=1) >= -atol).all(1)
    ends_at_100 = np.abs(y[:, -1] - 100) <= atol
    return in_range & monotone & ends_at_100


def repair_cummax(y: np.ndarray) -> np.ndarray:
    """Clip to [0, 100], force 100 at 200 mm, then running maximum."""
    y = np.clip(np.atleast_2d(np.asarray(y, dtype=float)), 0, 100)
    y[:, -1] = 100.0
    return np.maximum.accumulate(y, axis=1)


def repair_isotonic(y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Clip, then the closest non-decreasing curve in (weighted) squared error (PAV)."""
    y = np.clip(np.atleast_2d(np.asarray(y, dtype=float)), 0, 100)
    y[:, -1] = 100.0
    iso = IsotonicRegression(y_min=0, y_max=100, increasing=True)
    x = np.arange(y.shape[1])
    out = np.stack([iso.fit_transform(x, row, sample_weight=weights) for row in y])
    out[:, -1] = 100.0  # PAV may pool the last point with smaller ones
    return np.maximum.accumulate(out, axis=1)


def d_value(y: np.ndarray, percent: float) -> np.ndarray:
    """Diameter (mm) at which `percent` of the mass passes, linear in log10(d).

    Returns NaN where the curve never reaches `percent` inside the sieve range
    (e.g. D10 of a soil with more than 10 % clay).
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    out = np.full(len(y), np.nan)
    for k, row in enumerate(y):
        above = np.nonzero(row >= percent)[0]
        if len(above) == 0 or above[0] == 0:
            continue
        j = above[0]
        lo, hi = row[j - 1], row[j]
        t = (percent - lo) / (hi - lo) if hi > lo else 0.0
        out[k] = 10 ** (LOG_D[j - 1] + t * (LOG_D[j] - LOG_D[j - 1]))
    return out


def d_value_errors(y_true: np.ndarray, y_pred: np.ndarray, percents=(10, 50, 60)) -> dict:
    """Mean |log10 D_p error| in decades, over samples where both D_p are defined."""
    res = {}
    for p in percents:
        dt, dp = d_value(y_true, p), d_value(y_pred, p)
        ok = ~np.isnan(dt) & ~np.isnan(dp)
        res[f"D{p}_log_err"] = float(np.abs(np.log10(dt[ok]) - np.log10(dp[ok])).mean()) if ok.any() else np.nan
        res[f"D{p}_n"] = int(ok.sum())
    return res


def shift_curve(y: np.ndarray, factor: float) -> np.ndarray:
    """Curve of the same soil seen at a wrong scale: every grain looks `factor`x larger.

    F'(d) = F(d / factor), evaluated by linear interpolation in log10(d); outside the
    sieve range the curve is clamped to 0 below and 100 above.
    """
    y = np.atleast_2d(np.asarray(y, dtype=float))
    src = LOG_D - np.log10(factor)
    return np.stack([np.interp(src, LOG_D, row, left=0.0, right=100.0) for row in y])
