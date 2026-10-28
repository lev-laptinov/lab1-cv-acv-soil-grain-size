"""Cut and cache tiles, then cache frozen features of several backbones. Computes no metrics."""

import argparse

from soilgsd.config import load_config
from soilgsd.data import load_photo_index
from soilgsd.features import tile_features
from soilgsd.tiles import load_soil_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", required=True, help="timm model names")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--tile-mm", type=float, default=50)
    ap.add_argument("--work-ppm", type=float, default=4.5)
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    index = load_photo_index(cfg)
    boxes = load_soil_boxes(cfg, index)
    for split in args.splits:
        for name in args.backbones:
            feats, _ = tile_features(index, boxes, name, args.tile_mm, args.work_ppm, split)
            print(f"{split} {name}: {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
