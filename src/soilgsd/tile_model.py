"""Ridge regression from frozen tile features to the grading curve, aggregated to one curve per sample."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from soilgsd.data import TARGET_COLS
from soilgsd.metric import emd_per_sample, repair_cummax


def sample_weights(sample_ids: pd.Series) -> np.ndarray:
    """Every sample weighs the same in the loss, however many photos and tiles it has (mean 1)."""
    counts = sample_ids.map(sample_ids.value_counts()).to_numpy(float)
    w = 1 / counts
    return w * len(w) / w.sum()


def fit_predict(x_tr: np.ndarray, y_tr: np.ndarray, ids_tr: pd.Series, x_te: np.ndarray,
                alphas: list[float]) -> list[np.ndarray]:
    """Tile predictions on x_te for each alpha. Every training tile carries its sample's curve."""
    scaler = StandardScaler().fit(x_tr)
    a, b = scaler.transform(x_tr), scaler.transform(x_te)
    w = sample_weights(ids_tr)
    return [Ridge(alpha=alpha).fit(a, y_tr, sample_weight=w).predict(b) for alpha in alphas]


def aggregate(pred: np.ndarray, meta: pd.DataFrame, how: str) -> pd.DataFrame:
    """Tiles -> photo (mean or median per sieve) -> sample (mean over photos) -> valid curve."""
    df = pd.DataFrame(pred, columns=TARGET_COLS)
    df["sample_id"] = meta["sample_id"].to_numpy()
    df["path"] = meta["path"].to_numpy()
    per_photo = df.groupby(["sample_id", "path"])[TARGET_COLS].agg(how)
    per_sample = per_photo.groupby("sample_id").mean()
    return pd.DataFrame(repair_cummax(per_sample.to_numpy()), index=per_sample.index, columns=TARGET_COLS)


def choose_alpha(x: np.ndarray, meta: pd.DataFrame, y_samples: pd.DataFrame, families: pd.Series,
                 alphas: list[float], how: str, n_inner: int) -> float:
    """Alpha with the lowest sample-level EMD in an inner group CV on the training fold only."""
    groups = meta["sample_id"].map(families).to_numpy()
    y_tiles = y_samples.loc[meta["sample_id"]].to_numpy()
    total = np.zeros(len(alphas))
    for tr, va in GroupKFold(n_splits=n_inner).split(x, groups=groups):
        preds = fit_predict(x[tr], y_tiles[tr], meta["sample_id"].iloc[tr], x[va], alphas)
        for k, p in enumerate(preds):
            agg = aggregate(p, meta.iloc[va], how)
            total[k] += emd_per_sample(y_samples.loc[agg.index].to_numpy(), agg.to_numpy()).sum()
    return alphas[int(np.argmin(total))]


def cross_validate(x: np.ndarray, meta: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame,
                   alphas: list[float], how: str, n_inner: int) -> pd.DataFrame:
    """Out-of-fold sample curves for every repeat: rows (repeat, sample_id, alpha, 11 curve values)."""
    y_samples = labels.set_index("sample_id")[TARGET_COLS].astype(float)
    families = folds.drop_duplicates("sample_id").set_index("sample_id")["family"]
    out = []
    for (repeat, _fold), part in folds.groupby(["repeat", "fold"]):
        val_ids = set(part["sample_id"])
        is_val = meta["sample_id"].isin(val_ids).to_numpy()
        m_tr, m_va = meta[~is_val].reset_index(drop=True), meta[is_val].reset_index(drop=True)
        alpha = choose_alpha(x[~is_val], m_tr, y_samples, families, alphas, how, n_inner)
        y_tr = y_samples.loc[m_tr["sample_id"]].to_numpy()
        (pred,) = fit_predict(x[~is_val], y_tr, m_tr["sample_id"], x[is_val], [alpha])
        agg = aggregate(pred, m_va, how)
        agg.insert(0, "alpha", alpha)
        agg.insert(0, "repeat", repeat)
        out.append(agg.reset_index())
    return pd.concat(out, ignore_index=True)
