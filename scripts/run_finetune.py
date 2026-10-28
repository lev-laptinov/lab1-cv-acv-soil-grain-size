"""Fine-tuning experiments (configs/exp/*.yaml with approach: finetune) on the shared folds.

--smoke N trains N steps on the first training fold and reports speed and memory only: it checks
that a config runs before its hypothesis is committed, without producing any validation score.
"""

import argparse
import time

import pandas as pd
import torch
import yaml

from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.features import pick_device
from soilgsd.finetune import Trainer, cross_validate, peak_memory_gb
from soilgsd.results import save_result, summarise
from soilgsd.tiles import load_soil_boxes, tile_cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+", help="configs/exp/*.yaml")
    ap.add_argument("--smoke", type=int, default=0, help="train this many steps only, no validation")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    index = load_photo_index(cfg)
    labels = load_labels(cfg)
    folds = pd.read_csv(resolve(cfg["cv"]["folds_file"]))
    boxes = load_soil_boxes(cfg, index)
    device = pick_device()

    for exp_path in args.exps:
        exp = yaml.safe_load(open(resolve(exp_path)))
        tiles, meta = tile_cache(index, boxes, exp["tile_mm"], exp["work_ppm"], "train")
        if args.smoke:
            val = set(folds[(folds["repeat"] == 0) & (folds["fold"] == 0)]["sample_id"])
            keep = ~meta["sample_id"].isin(val).to_numpy()
            m_tr = meta[keep].reset_index(drop=True)
            y = labels.set_index("sample_id").loc[m_tr["sample_id"], TARGET_COLS].to_numpy(float)
            trainer = Trainer(exp, device, cfg["seed"])
            start = time.time()
            log = trainer.fit(tiles[keep], y, m_tr["sample_id"], max_steps=args.smoke)
            sec = time.time() - start
            n = args.smoke * exp["batch_size"]
            per_fold = sec / n * (keep.sum() // exp["batch_size"] * exp["batch_size"]) * exp["epochs"]
            print(f"{exp['name']} on {device}: {n / sec:.0f} tiles/s, train EMD after {args.smoke} steps "
                  f"{log[-1]['train_emd']:.1f}, peak memory {peak_memory_gb(device):.1f} GB, "
                  f"~{per_fold / 60:.0f} min per fold ({exp['epochs']} epochs)")
            continue

        start = time.time()
        oof, log = cross_validate(exp, tiles, meta, labels, folds, device, cfg["seed"])
        hours = (time.time() - start) / 3600
        out_dir = resolve(f"outputs/{exp['name']}")
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out_dir / "oof.csv", index=False)
        log.to_csv(out_dir / "train_log.csv", index=False)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(exp, f)

        res = {"name": exp["name"], "approach": "finetune", "backbone": exp["backbone"],
               "tile_mm": exp["tile_mm"], "work_ppm": exp["work_ppm"], "agg": exp.get("agg", "mean"),
               "train_tiles": len(tiles)}
        res.update(summarise(oof, labels, folds))
        gpu = torch.cuda.get_device_name() if device.type == "cuda" else device.type
        res["compute"] = f"{gpu} {hours:.2f} h, peak {peak_memory_gb(device):.1f} GB"
        save_result(res)
        print(pd.Series(res).to_string(), "\n")


if __name__ == "__main__":
    main()
