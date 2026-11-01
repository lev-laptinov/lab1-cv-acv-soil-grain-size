"""Weighted average of model curves: weights on the simplex, fitted to minimise EMD on out-of-fold predictions.

A convex combination of valid curves is a valid curve (non-decreasing, in [0, 100], 100 at 200 mm), so the
ensemble needs no repair. EMD is a weighted L1 distance, so the best weights are the solution of a linear
programme: exact, with no learning rate or initialisation.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from soilgsd.data import TARGET_COLS
from soilgsd.metric import LOG_WIDTHS, emd_per_sample


def fit_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights w (w >= 0, sum 1) minimising the mean EMD of sum_m w_m * preds[m] against y.

    preds: (models, samples, 11), y: (samples, 11). Variables: w (models) and t >= |ensemble - y| for every
    sample and the 10 sieve intervals that EMD counts; minimise sum t * interval width / samples.
    """
    m, n = preds.shape[:2]
    p = preds[:, :, :-1].reshape(m, -1).T  # (n*10, m): the value at 200 mm is 100 for every curve
    target = y[:, :-1].ravel()
    k = len(target)
    cost = np.concatenate([np.zeros(m), np.tile(LOG_WIDTHS, n) / n])
    eye = np.eye(k)
    a_ub = np.vstack([np.hstack([p, -eye]), np.hstack([-p, -eye])])  # +-(P w - y) <= t
    b_ub = np.concatenate([target, -target])
    a_eq = np.concatenate([np.ones(m), np.zeros(k)])[None]
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=[(0, None)] * (m + k), method="highs")
    assert res.success, res.message
    w = np.clip(res.x[:m], 0, None)
    return w / w.sum()


def blend(preds: np.ndarray, w: np.ndarray) -> np.ndarray:
    """sum_m w_m * preds[m]; the 200 mm point is set to exactly 100 (floating-point sums give 99.999...)."""
    out = np.clip(np.tensordot(w, preds, axes=1), 0, 100)
    out[..., -1] = 100.0
    return out


def nested_cv(preds: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """Honest ensemble score: the weights of each fold are fitted on the other folds only.

    Returns the ensemble's out-of-fold curves and the weights used for every fold.
    """
    out = np.zeros_like(y)
    weights = {}
    for f in np.unique(folds):
        tr = folds != f
        w = fit_weights(preds[:, tr], y[tr])
        out[~tr] = blend(preds[:, ~tr], w)
        weights[int(f)] = w
    return out, pd.DataFrame(weights).T


def load_member(name: str, outputs, submissions, sample_ids: pd.Index, test_ids: pd.Index,
                repeat: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(OOF curves of `repeat` aligned to sample_ids, test curves aligned to test_ids) of one experiment.

    The test curves come from outputs/<name>/test_preds.csv, else submissions/<name>.csv.
    """
    oof = pd.read_csv(outputs / name / "oof.csv")
    oof = oof[oof["repeat"] == repeat].set_index("sample_id")
    missing = set(sample_ids) - set(oof.index)
    assert not missing, f"{name}: no OOF prediction for {sorted(missing)}"
    test_path = outputs / name / "test_preds.csv"
    if not test_path.exists():
        test_path = submissions / f"{name}.csv"
    test = pd.read_csv(test_path).set_index("sample_id")
    return oof.loc[sample_ids, TARGET_COLS].to_numpy(float), test.loc[test_ids, TARGET_COLS].to_numpy(float)


def error_correlation(preds: np.ndarray, y: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Pearson correlation of the members' per-sample EMD: low values mean they fail on different samples."""
    e = pd.DataFrame({n: emd_per_sample(y, p) for n, p in zip(names, preds, strict=True)})
    return e.corr()
