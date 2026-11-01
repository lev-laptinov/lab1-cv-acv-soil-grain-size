import argparse
import functools
import time

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from part2_vit.metrics import curve_scores, sample_curves
from part2_vit.train import load_model
from part2_vit.wandb_logging import PROJECT
from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.tiles import load_soil_boxes, tile_cache


@functools.cache
def _api() -> wandb.Api:
    # One connection for all downloads: a new Api() per checkpoint re-verifies the login and can time out
    return wandb.Api(timeout=60)


def checkpoint(name: str, repeat: int, fold: int, entity: str | None, tries: int = 3) -> str:
    path = resolve(f"outputs/{name}/checkpoints/r{repeat}_f{fold}.pt")
    if not path.exists():
        assert entity, f"{path} is missing: pass --entity to download it from W&B"
        for attempt in range(tries):
            try:
                _api().artifact(f"{entity}/{PROJECT}/{name}-r{repeat}-f{fold}:latest").download(root=str(path.parent))
                break
            except Exception:  # W&B service busy / network hiccup: wait and retry
                if attempt == tries - 1:
                    raise
                time.sleep(10 * (attempt + 1))
    return str(path)


def channel_stats(tiles: np.ndarray, n: int = 500) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(tiles[np.linspace(0, len(tiles) - 1, n).astype(int)]).float().permute(0, 3, 1, 2) / 255
    return x.mean((2, 3)).mean(0).view(1, 3, 1, 1), x.std((2, 3)).mean(0).view(1, 3, 1, 1)


def remove_colour(x: torch.Tensor, mode: str, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if mode == "gray":
        return x.mean(1, keepdim=True).expand_as(x)
    m, s = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True) + 1e-6
    return ((x - m) / s * std + mean).clamp(0, 1)


@torch.no_grad()
def predict(model, tiles: np.ndarray, mode: str | None, mean, std, tta: bool, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(tiles), batch):
        x = torch.from_numpy(tiles[i:i + batch]).to(model.device).permute(0, 3, 1, 2).float() / 255
        if mode:
            x = remove_colour(x, mode, mean.to(x.device), std.to(x.device))
        with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
            out.append(model._predict((x - model.mean) / model.std, tta))
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="part2_vit/configs/*.yaml of a trained model")
    ap.add_argument("--mode", default="colournorm", choices=["colournorm", "gray"])
    ap.add_argument("--entity", help="W&B entity, to download missing checkpoints")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    base = load_config(args.config)
    cfg = yaml.safe_load(open(resolve(args.exp)))
    index, labels = load_photo_index(base), load_labels(base)
    folds = pd.read_csv(resolve(base["cv"]["folds_file"]))
    tiles, meta = tile_cache(index, load_soil_boxes(base, index), cfg["tile_mm"], cfg["work_ppm"], "train")
    mean, std = channel_stats(tiles)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    curves = {"original": [], args.mode: []}
    for repeat in cfg.get("checkpoint_repeats", [0]):
        for fold, part in folds[folds["repeat"] == repeat].groupby("fold"):
            model = load_model(args.exp, checkpoint(cfg["name"], repeat, int(fold), args.entity)).to(device)
            is_val = meta["sample_id"].isin(set(part["sample_id"])).to_numpy()
            m_va = meta[is_val].reset_index(drop=True)
            for mode in curves:
                pred = predict(model, tiles[is_val], None if mode == "original" else mode, mean, std, cfg.get("tta", True))
                curves[mode].append(sample_curves(pred, m_va, cfg.get("agg", "mean")).assign(repeat=repeat, fold=fold))

    cls = folds.drop_duplicates("sample_id").set_index("sample_id")["d50_class"]
    rows = []
    for mode, parts in curves.items():
        c = pd.concat(parts)
        row = {"model": cfg["name"], "input": mode, **curve_scores(c[TARGET_COLS], labels)}
        for k in ["fine", "sand", "gravel"]:
            row[f"emd_{k}"] = curve_scores(c[cls.loc[c.index] == k][TARGET_COLS], labels)["emd"]
        row["mean_pred_fines_%"] = c["0.063"].mean()
        rows.append(row)
    res = pd.DataFrame(rows).round(2)
    res.to_csv(resolve(f"outputs/{cfg['name']}/colour_test.csv"), index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
