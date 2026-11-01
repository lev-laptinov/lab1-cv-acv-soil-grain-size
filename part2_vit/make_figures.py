"""Report figures for Part 2: maps of the same tiles across all models, and the learning curves.

    python -m part2_vit.make_figures --entity <wandb entity>

Each soil is shown with the fold model that never saw it (repeat 0). Writes to reports/part2_vit/figures/:
- maps_gradcam_fines.png: Grad-CAM of the predicted fines share F(0.063 mm), the same method for every model
- maps_rollout.png: attention rollout of our ViT models (Swin has no global attention to roll out)
- curves.png: training and validation EMD per epoch, from outputs/<name>/train_log.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.ticker import MaxNLocator

from part2_vit.colour_test import checkpoint
from part2_vit.train import load_model
from part2_vit.vit import VisionTransformer, attention_rollout
from part2_vit.wandb_logging import overlay
from soilgsd.config import load_config, resolve
from soilgsd.data import load_labels, load_photo_index
from soilgsd.tiles import load_soil_boxes, tile_cache

MODELS = ["V4_own_vit_tiny_scratch", "V5_own_vit_small_augreg", "V6_swin_tiny", "V7_own_vit_small_augreg_head",
          "V8_swin_tiny_head"]
# Failure cases (H126, F827, H616, H038) next to soils the models get right (H516, H183)
SOILS = ["H126", "F827", "H616", "H516", "H038", "H183"]
FINES = 3  # index of the 0.063 mm sieve in the curve
LABELS = {"V4_own_vit_tiny_scratch": "V4 ViT-Ti scratch", "V5_own_vit_small_augreg": "V5 ViT-S AugReg FT",
          "V6_swin_tiny": "V6 Swin-T FT", "V7_own_vit_small_augreg_head": "V7 ViT-S AugReg head",
          "V8_swin_tiny_head": "V8 Swin-T head"}


def fines_gradcam(model, x: torch.Tensor) -> torch.Tensor:
    """Grad-CAM of the predicted fines share on a 14 x 14 token / feature grid, for our ViT and for Swin."""
    bb = model.backbone
    if isinstance(bb, VisionTransformer):
        # Input of the last block: after it only [CLS] reaches the head, so patch tokens there have no gradient
        layer, p, g = bb.blocks[-1], bb.num_prefix_tokens, bb.patch_embed.grid_size
        hook_input, to_grid = True, lambda t: t[:, p:].reshape(len(t), g, g, -1)
    else:
        layer, hook_input, to_grid = bb.layers[2], False, lambda t: t
    acts = {}
    hook = layer.register_forward_hook(lambda mod, inp, out: acts.__setitem__("a", inp[0] if hook_input else out))
    with torch.enable_grad():
        out = model(x.detach().requires_grad_(True))
        (grad,) = torch.autograd.grad(out[:, FINES].sum(), acts["a"])
    hook.remove()
    a, grad = to_grid(acts["a"]), to_grid(grad)
    return torch.relu((grad.mean(dim=(1, 2), keepdim=True) * a).sum(dim=-1)).detach()


def pick_tiles(tiles: np.ndarray, meta: pd.DataFrame, soils: list[str]) -> np.ndarray:
    """The middle tile of the first photo of each soil (the same tile for every model)."""
    picks = []
    for s in soils:
        rows = meta.index[(meta["sample_id"] == s) & (meta["path"] == meta.loc[meta["sample_id"] == s, "path"].iloc[0])]
        picks.append(rows[len(rows) // 2])
    return tiles[picks]


def map_grid(tiles, soils, labels, cls, columns: dict, title: str, path) -> None:
    """Rows: soils; first column the tile, then one column per model with its map and predicted fines %."""
    fig, axes = plt.subplots(len(soils), len(columns) + 1, figsize=(2.3 * (len(columns) + 1), 2.5 * len(soils)))
    true_fines = labels.set_index("sample_id")["0.063"]
    for i, s in enumerate(soils):
        axes[i, 0].imshow(tiles[i])
        axes[i, 0].set_ylabel(f"{s} ({cls[s]})\ntrue fines {true_fines[s]:.0f}%", fontsize=9)
        for j, (name, (maps, fines)) in enumerate(columns.items(), start=1):
            axes[i, j].imshow(overlay(tiles[i], maps[i]))
            axes[i, j].set_xlabel(f"pred fines {fines[i]:.0f}%", fontsize=8)
            if i == 0:
                axes[i, j].set_title(LABELS.get(name, name), fontsize=9)
        if i == 0:
            axes[i, 0].set_title("tile (50 mm)", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def curves(models: list[str], path) -> None:
    """Validation EMD per fold (thin), its mean (thick) and training EMD (dashed, per tile) per epoch."""
    logs = {m: resolve(f"outputs/{m}/train_log.csv") for m in models}
    logs = {m: pd.read_csv(p) for m, p in logs.items() if p.exists()}
    if not logs:
        return
    fig, axes = plt.subplots(1, len(logs), figsize=(3.4 * len(logs), 3.2), sharey=True, squeeze=False)
    for ax, (m, log) in zip(axes[0], logs.items(), strict=True):
        log = log[log["repeat"] == 0]
        for _, f in log.groupby("fold"):
            ax.plot(f["epoch"] + 1, f["val_emd"], color="tab:blue", alpha=0.3, lw=1)
        mean = log.groupby("epoch")[["val_emd", "train_emd"]].mean()
        ax.plot(mean.index + 1, mean["val_emd"], color="tab:blue", lw=2.5, marker="o", ms=4, label="validation (per sample)")
        ax.plot(mean.index + 1, mean["train_emd"], color="tab:orange", lw=2, ls="--", marker="o", ms=3,
                label="training (per tile)")
        ax.set_title(LABELS.get(m, m), fontsize=9)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("EMD")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, help="config names in part2_vit/configs (or yaml paths)")
    ap.add_argument("--soils", nargs="+", default=SOILS)
    ap.add_argument("--entity", help="W&B entity, to download missing checkpoints")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    base = load_config(args.config)
    index, labels = load_photo_index(base), load_labels(base)
    folds = pd.read_csv(resolve(base["cv"]["folds_file"]))
    fold_of = folds[folds["repeat"] == 0].set_index("sample_id")["fold"]
    cls = folds.drop_duplicates("sample_id").set_index("sample_id")["d50_class"]
    tiles, meta = tile_cache(index, load_soil_boxes(base, index), 50, 4.5, "train")
    picked = pick_tiles(tiles, meta, args.soils)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = resolve("reports/part2_vit/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    gradcams, rollouts, names = {}, {}, []
    for m in args.models:
        cfg_path = m if m.endswith(".yaml") else f"part2_vit/configs/{m}.yaml"
        cfg = yaml.safe_load(open(resolve(cfg_path)))
        names.append(cfg["name"])
        try:  # only the folds that hold the chosen soils out
            ckpts = {f: checkpoint(cfg["name"], 0, f, args.entity) for f in sorted({int(fold_of[s]) for s in args.soils})}
        except Exception as e:  # a model without checkpoints is left out; the others are still drawn
            print(f"skipping {cfg['name']} in the maps: {type(e).__name__}: {e}", flush=True)
            continue
        models = {f: load_model(cfg_path, p).to(device) for f, p in ckpts.items()}
        g_maps, r_maps, fines = [], [], []
        for s, tile in zip(args.soils, picked, strict=True):
            model = models[int(fold_of[s])]
            x = model.prepare(torch.from_numpy(tile[None]).permute(0, 3, 1, 2).to(device))
            g_maps.append(fines_gradcam(model, x)[0].cpu())
            with torch.no_grad():
                fines.append(float(model(x)[0, FINES]))
            if isinstance(model.backbone, VisionTransformer):
                r_maps.append(attention_rollout(model.backbone, x)[0].cpu())
        gradcams[cfg["name"]] = (g_maps, fines)
        if r_maps:
            rollouts[cfg["name"]] = (r_maps, fines)
        del models

    if gradcams:
        map_grid(picked, args.soils, labels, cls, gradcams, "Grad-CAM of the predicted fines share (< 0.063 mm)",
                 out_dir / "maps_gradcam_fines.png")
    if rollouts:
        map_grid(picked, args.soils, labels, cls, rollouts, "Attention rollout (our ViT models)",
                 out_dir / "maps_rollout.png")
    curves(names, out_dir / "curves.png")
    print(f"figures in {out_dir}")


if __name__ == "__main__":
    main()
