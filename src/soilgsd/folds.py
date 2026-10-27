"""Group-aware folds: all photos of a sample, and all samples of one curve family, share a fold."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedGroupKFold

from soilgsd.data import TARGET_COLS
from soilgsd.metric import d_value, emd_per_sample


def d50_class(labels: pd.DataFrame) -> pd.Series:
    """fine: D50 < 0.063 mm (silt/clay), sand: 0.063-2 mm, gravel: > 2 mm."""
    d50 = d_value(labels[TARGET_COLS].to_numpy(), 50)
    cls = np.select([d50 < 0.063, d50 <= 2.0], ["fine", "sand"], "gravel")
    return pd.Series(cls, index=labels.index, name="d50_class")


def curve_families(labels: pd.DataFrame, emd_threshold: float) -> pd.Series:
    """Samples whose grading curves are all within `emd_threshold` of each other (complete linkage).

    Near-identical curves with consecutive ids (H366/H367/H371/H372) most likely come from one site.
    The test soils come from other sites, so such a family must not be split between train and
    validation, or neighbour-matching models get an optimistic CV.
    """
    y = labels[TARGET_COLS].to_numpy(float)
    dist = np.array([emd_per_sample(np.repeat(a[None], len(y), 0), y) for a in y])
    clusters = fcluster(linkage(squareform(dist, checks=False), "complete"), emd_threshold, "distance")
    # Name each family after its first sample id so that folds.csv is readable.
    first = labels.groupby(clusters)["sample_id"].transform("min")
    return pd.Series(first.to_numpy(), index=labels.index, name="family")


def make_folds(labels: pd.DataFrame, n_splits: int, n_repeats: int, seed: int,
               family_emd_threshold: float) -> pd.DataFrame:
    """One row per (sample, repeat) with the fold that holds the sample out.

    Splitting labelled samples (not photos) makes photo leakage impossible by construction;
    grouping by curve family keeps near-duplicate samples together; stratifying by D50 class
    keeps fine, sandy and gravelly soils in every fold.
    """
    strata = d50_class(labels)
    families = curve_families(labels, family_emd_threshold)
    rows = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for fold, (_, val) in enumerate(sgkf.split(labels, strata, groups=families)):
            for i in val:
                rows.append({"sample_id": labels.loc[i, "sample_id"], "repeat": r, "fold": fold,
                             "family": families[i], "d50_class": strata[i]})
    return pd.DataFrame(rows).sort_values(["repeat", "fold", "sample_id"]).reset_index(drop=True)


def iter_folds(folds: pd.DataFrame, repeat: int = 0):
    """Yield (fold, train_ids, val_ids) for one repeat."""
    f = folds[folds["repeat"] == repeat]
    for k in sorted(f["fold"].unique()):
        val = f.loc[f["fold"] == k, "sample_id"].tolist()
        train = f.loc[f["fold"] != k, "sample_id"].tolist()
        yield k, train, val
