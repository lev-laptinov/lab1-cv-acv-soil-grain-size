"""Run tile experiments (configs/exp/*.yaml) on the shared folds and add them to reports/results.csv."""

import argparse

import numpy as np
import pandas as pd
import yaml

from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.features import tile_features
from soilgsd.metric import d_value_errors, emd_parts
from soilgsd.tile_model import cross_validate
from soilgsd.tiles import soil_boxes

RESULTS = resolve("reports/results.csv")


def load_boxes(cfg: dict, index: pd.DataFrame) -> pd.DataFrame:
    """Soil box of every photo (train and test), computed once and kept in data/soil_boxes.csv."""
    path = resolve(cfg["data"]["soil_boxes"])
    if not path.exists():
        soil_boxes(index).to_csv(path, index=False)
    boxes = pd.read_csv(path)
    assert (boxes["path"] == index["path"]).all(), "soil_boxes.csv is stale: delete it and rerun"
    return boxes


def summarise(exp: dict, oof: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame, n_tiles: int) -> dict:
    """Mean over repeats of the sample-level scores; std over repeats shows the split noise."""
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
        per_repeat.append(row)
    pr = pd.DataFrame(per_repeat)
    res = {"name": exp["name"], "backbone": exp["backbone"], "tile_mm": exp["tile_mm"],
           "work_ppm": exp["work_ppm"], "agg": exp["agg"], "train_tiles": n_tiles,
           "alpha_median": float(np.median(oof["alpha"]))}
    res.update(pr.mean().round(2).to_dict())
    res["emd_std_repeats"] = round(float(pr["emd"].std(ddof=0)), 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="configs/exp/*.yaml")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    index = load_photo_index(cfg)
    labels = load_labels(cfg)
    folds = pd.read_csv(resolve(cfg["cv"]["folds_file"]))
    boxes = load_boxes(cfg, index)

    for exp_path in args.exps:
        exp = yaml.safe_load(open(resolve(exp_path)))
        feats, meta = tile_features(index, boxes, exp["backbone"], exp["tile_mm"], exp["work_ppm"], "train")
        oof = cross_validate(feats, meta, labels, folds, exp["alphas"], exp["agg"], exp["inner_splits"])
        out_dir = resolve(f"outputs/{exp['name']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out_dir / "oof.csv", index=False)

        res = summarise(exp, oof, labels, folds, len(feats))
        old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame()
        if len(old):
            old = old[old["name"] != res["name"]]
        pd.concat([old, pd.DataFrame([res])], ignore_index=True).to_csv(RESULTS, index=False)
        print(pd.Series(res).to_string(), "\n")


if __name__ == "__main__":
    main()
