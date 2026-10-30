import os

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from matplotlib import colormaps

from part2_vit.vit import VisionTransformer, attention_rollout
from soilgsd.config import resolve

PROJECT = "soilgsd-part2-vit"
SAVE_DIR = resolve("outputs/wandb")


def _mode(enabled: bool) -> str:
    if not enabled:
        return "disabled"
    return "online" if os.environ.get("WANDB_API_KEY") else "offline"


def fold_logger(cfg: dict, repeat: int, fold: int, enabled: bool = True) -> WandbLogger:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    return WandbLogger(project=PROJECT, group=cfg["name"], name=f"{cfg['name']}-r{repeat}-f{fold}", job_type="cv-fold",
                       config={**cfg, "repeat": repeat, "fold": fold}, save_dir=str(SAVE_DIR), mode=_mode(enabled))


def watch_weights(logger: WandbLogger, model: torch.nn.Module, log_freq: int = 100) -> None:
    """Histograms of every weight and its gradient."""
    logger.watch(model, log="all", log_freq=log_freq)


def log_checkpoint(logger: WandbLogger, path: str, cfg: dict, repeat: int, fold: int, scores: dict) -> None:
    art = wandb.Artifact(f"{cfg['name']}-r{repeat}-f{fold}", type="model", metadata={**scores, "repeat": repeat, "fold": fold})
    art.add_file(path)
    logger.experiment.log_artifact(art)


def gradcam(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor, out_index: int) -> torch.Tensor:
    """Grad-CAM of curve output out_index on the NHWC activations of `layer` (timm's Swin)."""
    acts = {}
    hook = layer.register_forward_hook(lambda mod, inp, out: acts.__setitem__("a", out))
    with torch.enable_grad():
        out = model(x)
        a = acts["a"]
        (grad,) = torch.autograd.grad(out[:, out_index].sum(), a)
    hook.remove()
    weights = grad.mean(dim=(1, 2), keepdim=True)
    return torch.relu((weights * a).sum(dim=-1)).detach()  # (B, H, W)


def overlay(tile: np.ndarray, heat: torch.Tensor) -> np.ndarray:
    """uint8 tile with the map (normalised per tile) on top."""
    h = torch.nn.functional.interpolate(heat[None, None].float(), size=tile.shape[:2], mode="bilinear")[0, 0]
    h = ((h - h.min()) / (h.max() - h.min() + 1e-8)).cpu().numpy()
    colour = colormaps["jet"](h)[..., :3] * 255
    return (0.55 * tile + 0.45 * colour).astype(np.uint8)


class AttentionMapLogger(L.Callback):
    """Maps of fixed validation tiles every `every` epochs: rollout for our ViT, Grad-CAM of fines % for Swin."""

    FINES = 3  # index of the 0.063 mm sieve in the curve

    def __init__(self, tiles: np.ndarray, captions: list[str], every: int = 5):
        self.tiles, self.captions, self.every = tiles, captions, every

    def on_validation_epoch_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        last = trainer.current_epoch == trainer.max_epochs - 1
        if trainer.sanity_checking or not (last or trainer.current_epoch % self.every == 0):
            return
        x = module.prepare(torch.from_numpy(self.tiles).permute(0, 3, 1, 2).to(module.device))
        backbone = module.backbone
        if isinstance(backbone, VisionTransformer):
            maps, kind = attention_rollout(backbone, x), "rollout"
        else:
            maps, kind = gradcam(module, backbone.layers[2], x, self.FINES), "gradcam_fines"
        images = [wandb.Image(overlay(t, m), caption=c) for t, m, c in zip(self.tiles, maps, self.captions, strict=True)]
        module.logger.experiment.log({f"maps/{kind}": images, "epoch": trainer.current_epoch})


def log_summary(cfg: dict, result: dict, oof: pd.DataFrame, submission: str | None, enabled: bool = True) -> None:
    """CV scores, OOF curves and the submission file of one experiment."""
    run = wandb.init(project=PROJECT, group=cfg["name"], name=f"{cfg['name']}-summary", job_type="summary",
                     config=cfg, dir=str(SAVE_DIR), mode=_mode(enabled), reinit="finish_previous")
    run.summary.update({k: v for k, v in result.items() if isinstance(v, int | float)})
    run.log({"oof": wandb.Table(dataframe=oof)})
    if submission:
        art = wandb.Artifact(f"{cfg['name']}-submission", type="submission")
        art.add_file(submission)
        run.log_artifact(art)
    run.finish()
