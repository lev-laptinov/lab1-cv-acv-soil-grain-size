"""Run tile experiments (configs/exp/*.yaml) on the shared folds and add them to reports/results.csv."""

import argparse
import time

import numpy as np
import pandas as pd
import yaml

from soilgsd.config import load_config, resolve
from soilgsd.data import load_labels, load_photo_index
from soilgsd.features import tile_features
from soilgsd.results import save_result, summarise
from soilgsd.tile_model import cross_validate
from soilgsd.tiles import load_soil_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="configs/exp/*.yaml")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    index = load_photo_index(cfg)
    labels = load_labels(cfg)
    folds = pd.read_csv(resolve(cfg["cv"]["folds_file"]))
    boxes = load_soil_boxes(cfg, index)

    for exp_path in args.exps:
        exp = yaml.safe_load(open(resolve(exp_path)))
        feats, meta = tile_features(index, boxes, exp["backbone"], exp["tile_mm"], exp["work_ppm"], "train")
        start = time.time()
        oof = cross_validate(feats, meta, labels, folds, exp["alphas"], exp["agg"], exp["inner_splits"])
        out_dir = resolve(f"outputs/{exp['name']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out_dir / "oof.csv", index=False)

        res = {"name": exp["name"], "approach": "frozen+ridge", "backbone": exp["backbone"],
               "tile_mm": exp["tile_mm"], "work_ppm": exp["work_ppm"], "agg": exp["agg"],
               "train_tiles": len(feats), "alpha_median": float(np.median(oof["alpha"]))}
        res.update(summarise(oof, labels, folds))
        res["compute"] = f"cpu ridge {time.time() - start:.0f}s (features cached)"
        save_result(res)
        print(pd.Series(res).to_string(), "\n")


if __name__ == "__main__":
    main()
