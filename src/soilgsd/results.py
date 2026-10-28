"""Scores of one experiment from its out-of-fold predictions, and the shared results table."""

import pandas as pd

from soilgsd.config import resolve
from soilgsd.data import TARGET_COLS
from soilgsd.metric import d_value_errors, emd_parts

RESULTS = resolve("reports/results.csv")


def summarise(oof: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame) -> dict:
    """Sample-level scores, averaged over repeats.

    emd_std_folds: std of the fold means (the "mean ± std across folds" of the results table);
    emd_std_repeats: std over repeats of the whole-CV mean (split noise; 0 with a single repeat).
    """
    y = labels.set_index("sample_id")[TARGET_COLS].astype(float)
    cls = folds.drop_duplicates("sample_id").set_index("sample_id")["d50_class"]
    per_repeat = []
    for _, part in oof.groupby("repeat"):
        yt, yp = y.loc[part["sample_id"]].to_numpy(), part[TARGET_COLS].to_numpy()
        fine, coarse = emd_parts(yt, yp)
        e = fine + coarse
        row = {"emd": e.mean(), "emd_fine_part": fine.mean(), "emd_coarse_part": coarse.mean(),
               "D50_log_err": d_value_errors(yt, yp)["D50_log_err"]}
        for c in ["fine", "sand", "gravel"]:
            row[f"emd_{c}"] = e[(cls.loc[part["sample_id"]] == c).to_numpy()].mean()
        row["emd_std_folds"] = pd.Series(e).groupby(part["fold"].to_numpy()).mean().std(ddof=0)
        per_repeat.append(row)
    pr = pd.DataFrame(per_repeat)
    res = pr.mean().round(2).to_dict()
    res["emd_std_repeats"] = round(float(pr["emd"].std(ddof=0)), 2)
    res["repeats"] = len(pr)
    return res


def save_result(res: dict) -> None:
    """Add or replace the row of this experiment (by name) in reports/results.csv."""
    old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
    if len(old):
        old = old[old["name"] != res["name"]]
    pd.concat([old, pd.DataFrame([res])], ignore_index=True).to_csv(RESULTS, index=False)
