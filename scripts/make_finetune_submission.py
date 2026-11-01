"""Train a fine-tuning experiment on all training samples and write its test submission.

The recipe is the experiment's config unchanged (epochs, lr, bags, exclude_train), so nothing is tuned on
the test photos. --seeds N trains N models with different seeds and averages their test curves, as the
CV repeats average over splits.
"""

import argparse

import pandas as pd
import yaml

from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.features import pick_device
from soilgsd.finetune import Trainer
from soilgsd.metric import is_valid_curve, repair_cummax
from soilgsd.tile_model import aggregate
from soilgsd.tiles import load_soil_boxes, tile_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="configs/exp/*.yaml of a fine-tuning experiment")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = yaml.safe_load(open(resolve(args.exp)))
    index, labels = load_photo_index(cfg), load_labels(cfg)
    boxes = load_soil_boxes(cfg, index)
    device = pick_device()

    tiles, meta = tile_cache(index, boxes, exp["tile_mm"], exp["work_ppm"], "train")
    keep = ~meta["sample_id"].isin(exp.get("exclude_train", [])).to_numpy()
    m_tr = meta[keep].reset_index(drop=True)
    y = labels.set_index("sample_id").loc[m_tr["sample_id"], TARGET_COLS].to_numpy(float)
    test_tiles, test_meta = tile_cache(index, boxes, exp["tile_mm"], exp["work_ppm"], "test")

    preds = []
    for k in range(args.seeds):
        trainer = Trainer(exp, device, cfg["seed"] + 1000 + k)  # not one of the CV seeds
        log = trainer.fit(tiles[keep], y, m_tr["sample_id"], photo_ids=m_tr["path"])
        if exp.get("bag_tiles"):
            pred = trainer.predict_samples(test_tiles, test_meta)
        else:
            pred = aggregate(trainer.predict(test_tiles), test_meta, exp.get("agg", "mean"))
        preds.append(pred)
        print(f"seed {k}: last train EMD {log[-1]['train_emd']:.1f}", flush=True)
    mean = pd.concat(preds).groupby(level=0).mean()
    pred = pd.DataFrame(repair_cummax(mean.to_numpy()), index=mean.index, columns=TARGET_COLS)

    sub = pd.read_csv(resolve(cfg["data"]["sample_submission"]))[["sample_id"]].join(pred, on="sample_id")
    assert sub[TARGET_COLS].notna().all().all(), "a test sample has no prediction"
    assert is_valid_curve(sub[TARGET_COLS].to_numpy()).all()
    out_dir = resolve(f"outputs/{exp['name']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_dir / "test_preds.csv", index=False)
    path = resolve(f"submissions/{exp['name']}.csv")
    path.parent.mkdir(exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"{path}: {args.seeds} seeds, {int(keep.sum())} training tiles, {len(test_tiles)} test tiles")


if __name__ == "__main__":
    main()
