import numpy as np
import pandas as pd

from soilgsd.data import TARGET_COLS
from soilgsd.folds import curve_families, iter_folds, make_folds
from soilgsd.metric import repair_cummax, shift_curve

SAND = np.array([0, 1, 3, 10, 50, 90, 98, 100, 100, 100, 100.0])
FACTORS = [1 / 16, 1 / 4, 1, 4, 16, 64]  # 4x apart: ~60 EMD between different soils


def _labels() -> pd.DataFrame:
    """Six distinct soils (fine to gravel), each with a near-duplicate twin (~2 EMD away)."""
    rows = []
    for k, f in enumerate(FACTORS):
        y = shift_curve(SAND, f)[0]
        twin = repair_cummax(y + np.r_[np.full(10, 0.5), 0.0])[0]
        rows += [[f"S{k}a", *y], [f"S{k}b", *twin]]
    return pd.DataFrame(rows, columns=["sample_id", *TARGET_COLS])


def test_twins_form_one_family_and_distinct_soils_do_not():
    fam = curve_families(_labels(), emd_threshold=10)
    assert fam.nunique() == len(FACTORS)
    assert (fam.iloc[0::2].to_numpy() == fam.iloc[1::2].to_numpy()).all()


def test_every_sample_held_out_once_and_families_not_split():
    labels = _labels()
    folds = make_folds(labels, n_splits=3, n_repeats=2, seed=0, family_emd_threshold=10)
    for r in (0, 1):
        f = folds[folds["repeat"] == r]
        assert sorted(f["sample_id"]) == sorted(labels["sample_id"])
        assert (f.groupby("family")["fold"].nunique() == 1).all()
        for _, train, val in iter_folds(folds, r):
            assert not set(train) & set(val)
