import numpy as np
from scipy.optimize import minimize

from soilgsd.ensemble import blend, fit_weights, nested_cv
from soilgsd.metric import emd_per_sample, is_valid_curve


def _curves(rng, n):
    y = np.sort(rng.uniform(0, 100, (n, 11)), axis=1)
    y[:, -1] = 100
    return y


def test_fit_weights_finds_the_member_that_equals_the_truth():
    rng = np.random.default_rng(0)
    y = _curves(rng, 12)
    preds = np.stack([_curves(rng, 12), y, _curves(rng, 12)])
    w = fit_weights(preds, y)
    assert np.allclose(w, [0, 1, 0], atol=1e-6)


def test_fit_weights_is_on_the_simplex_and_beats_any_other_mix():
    rng = np.random.default_rng(1)
    y, preds = _curves(rng, 20), np.stack([_curves(rng, 20) for _ in range(4)])
    w = fit_weights(preds, y)
    assert (w >= 0).all() and np.isclose(w.sum(), 1)
    assert is_valid_curve(blend(preds, w)).all()
    best = emd_per_sample(y, blend(preds, w)).mean()
    for _ in range(200):
        v = rng.dirichlet(np.ones(4))
        assert emd_per_sample(y, blend(preds, v)).mean() >= best - 1e-6
    # and a generic optimiser started from the uniform mix does not get lower either
    obj = lambda v: emd_per_sample(y, blend(preds, np.abs(v) / np.abs(v).sum())).mean()  # noqa: E731
    assert minimize(obj, np.ones(4) / 4, method="Nelder-Mead").fun >= best - 1e-6


def test_nested_cv_never_uses_the_held_out_fold():
    rng = np.random.default_rng(2)
    y = _curves(rng, 12)
    folds = np.repeat([0, 1, 2], 4)
    # member 0 is perfect on fold 0 only, member 1 elsewhere: fold 0 must be blended with weights fitted on folds 1-2
    good, bad = y.copy(), _curves(rng, 12)
    m0 = np.where((folds == 0)[:, None], good, bad)
    m1 = np.where((folds == 0)[:, None], bad, good)
    out, weights = nested_cv(np.stack([m0, m1]), y, folds)
    assert np.allclose(weights.loc[0].to_numpy(), [0, 1], atol=1e-6)
    assert np.allclose(out[folds == 0], bad[folds == 0], atol=1e-6)
