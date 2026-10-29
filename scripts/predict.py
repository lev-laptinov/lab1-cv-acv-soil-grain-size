"""Inference: photo(s) of one soil sample + pixels per mm -> one valid grain-size curve.

    uv run python scripts/predict.py outputs/S1_mae_frozen/model.joblib --ppm 13.94 photo1.jpg photo2.jpg

The model file is written by scripts/make_submission.py (frozen backbone name, tiling and ridge).
"""

import argparse

import joblib
import numpy as np
import pandas as pd

from soilgsd.data import TARGET_COLS
from soilgsd.features import embed, load_backbone, pick_device
from soilgsd.tile_model import aggregate
from soilgsd.tiles import cut_tiles, load_photo, single_photo_box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="outputs/<experiment>/model.joblib")
    ap.add_argument("photos", nargs="+")
    ap.add_argument("--ppm", type=float, required=True, help="pixels per mm of the photos as stored")
    args = ap.parse_args()
    saved = joblib.load(args.model)
    exp, model = saved["exp"], saved["model"]
    device = pick_device()
    backbone = load_backbone(exp["backbone"], device)

    feats, paths = [], []
    for path in args.photos:
        im = load_photo(path)
        tiles = cut_tiles(im, args.ppm, single_photo_box(im, args.ppm), exp["tile_mm"], exp["work_ppm"])
        if not len(tiles):
            raise SystemExit(f"{path}: soil region smaller than one {exp['tile_mm']} mm tile")
        feats.append(embed(backbone, tiles, device))
        paths += [path] * len(tiles)
    meta = pd.DataFrame({"path": paths, "sample_id": "sample"})
    curve = aggregate(model.predict(np.concatenate(feats)), meta, exp["agg"]).loc["sample"]
    print(pd.DataFrame({"sieve_mm": TARGET_COLS, "percent_passing": curve.round(2).to_numpy()}).to_string(index=False))


if __name__ == "__main__":
    main()
