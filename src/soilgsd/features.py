"""Frozen pretrained backbone: one feature vector per tile, cached on disk per tiling setup."""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

from soilgsd.config import resolve
from soilgsd.tiles import cut_tiles, load_photo


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_backbone(name: str, device: torch.device) -> torch.nn.Module:
    """Pretrained timm model without its classifier: the output is the pooled feature vector."""
    return timm.create_model(name, pretrained=True, num_classes=0).eval().to(device)


@torch.no_grad()
def embed(model: torch.nn.Module, tiles: np.ndarray, device: torch.device, batch: int = 64) -> np.ndarray:
    """(n, H, W, 3) uint8 tiles -> (n, d) float32 features, with the backbone's own normalisation."""
    mean = torch.tensor(model.pretrained_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(model.pretrained_cfg["std"], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(tiles), batch):
        x = torch.from_numpy(tiles[i:i + batch]).to(device).permute(0, 3, 1, 2).float() / 255
        out.append(model((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.num_features), np.float32)


def cache_path(split: str, backbone: str, tile_mm: float, work_ppm: float) -> Path:
    return resolve(f"data/cache/features/{split}_{backbone}_t{tile_mm:g}_p{work_ppm:g}.npz")


def tile_features(index: pd.DataFrame, boxes: pd.DataFrame, backbone: str, tile_mm: float,
                  work_ppm: float, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Features of every tile of every photo in `split`, and one row per tile (photo path, sample).

    Computed once per (split, backbone, tile_mm, work_ppm) and then read from data/cache/.
    """
    path = cache_path(split, backbone, tile_mm, work_ppm)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["feats"], pd.DataFrame({"path": z["path"], "sample_id": z["sample_id"]})

    device = pick_device()
    model = load_backbone(backbone, device)
    photos = index[index["split"] == split]
    feats, rows = [], []
    for i, r in photos.iterrows():
        b = boxes.loc[i]
        tiles = cut_tiles(load_photo(r["path"]), r["ppm"], (b["left"], b["top"], b["right"], b["bottom"]),
                          tile_mm, work_ppm)
        feats.append(embed(model, tiles, device))
        rows += [(r["path"], r["sample_id"])] * len(tiles)
    feats = np.concatenate(feats).astype(np.float32)
    meta = pd.DataFrame(rows, columns=["path", "sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feats=feats, path=meta["path"].to_numpy(str), sample_id=meta["sample_id"].to_numpy(str))
    return feats, meta
