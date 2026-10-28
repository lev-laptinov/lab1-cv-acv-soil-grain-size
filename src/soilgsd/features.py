"""Frozen pretrained backbone: one feature vector per tile, cached on disk per tiling setup."""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch

from soilgsd.config import resolve
from soilgsd.tiles import tile_cache


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Extra timm arguments per backbone. DINOv2 is published at 518 px; its position embeddings are
# interpolated to our 224 px tiles (16 x 16 patches of 14 px).
BACKBONE_KWARGS = {"vit_small_patch14_dinov2.lvd142m": {"img_size": 224}}


def load_backbone(name: str, device: torch.device) -> torch.nn.Module:
    """Pretrained timm model without its classifier: the output is the pooled feature vector."""
    model = timm.create_model(name, pretrained=True, num_classes=0, **BACKBONE_KWARGS.get(name, {}))
    return model.eval().to(device)


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

    tiles, meta = tile_cache(index, boxes, tile_mm, work_ppm, split)
    device = pick_device()
    feats = embed(load_backbone(backbone, device), tiles, device).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feats=feats, path=meta["path"].to_numpy(str), sample_id=meta["sample_id"].to_numpy(str))
    return feats, meta
