"""Fine-tuning a timm backbone on tiles: softmax over the 11 mass bins, EMD loss, sample-balanced batches."""

import time

import numpy as np
import pandas as pd
import timm
import torch
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch import nn

from soilgsd.data import TARGET_COLS
from soilgsd.features import BACKBONE_KWARGS
from soilgsd.metric import LOG_WIDTHS
from soilgsd.tile_model import aggregate


class CurveModel(nn.Module):
    """Backbone -> 11 logits -> softmax = mass share of each sieve bin -> cumsum = % passing.

    The output is a valid curve by construction: non-negative steps, and the last value
    (everything below 200 mm) is the sum of all shares, i.e. 100.
    """

    def __init__(self, backbone: str, pretrained: bool, drop_path: float = 0.0):
        super().__init__()
        kwargs = dict(BACKBONE_KWARGS.get(backbone, {}))
        if drop_path:
            kwargs["drop_path_rate"] = drop_path
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **kwargs)
        self.head = nn.Linear(self.backbone.num_features, len(TARGET_COLS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shares = torch.softmax(self.head(self.backbone(x)).float(), dim=1)
        return 100 * torch.cumsum(shares, dim=1)


def emd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The competition metric itself: |F - F^| weighted by the log10 width of each sieve interval."""
    w = torch.as_tensor(LOG_WIDTHS, dtype=pred.dtype, device=pred.device)
    return (torch.abs(pred[:, :-1] - target[:, :-1]) @ w).mean()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Augmentations that keep millimetres per pixel: rotations by 90 degrees, flips, mild colour.

    No random resized crops, zoom or blur: they change the apparent grain size and so the label.
    """
    k = torch.randint(0, 4, (len(x),), device=x.device)
    for r in range(1, 4):
        m = k == r
        if m.any():
            x[m] = torch.rot90(x[m], r, dims=(2, 3))
    flip = torch.rand(len(x), device=x.device) < 0.5
    x[flip] = torch.flip(x[flip], dims=(3,))
    brightness = 1 + 0.1 * (2 * torch.rand(len(x), 1, 1, 1, device=x.device) - 1)
    contrast = 1 + 0.1 * (2 * torch.rand(len(x), 1, 1, 1, device=x.device) - 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * contrast + mean * brightness).clamp(0, 1)


def freeze_bn_stats(model: nn.Module) -> None:
    """Keep BatchNorm in eval mode: running statistics stay the pretrained ones.

    A batch holds tiles of only a few physical samples, so its statistics describe those soils,
    not the data; normalising with them would shift the features with every batch.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def to_input(tiles: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(tiles).to(device).permute(0, 3, 1, 2).float() / 255


class Trainer:
    """One fine-tuning run on the training tiles of a fold; predicts curves for other tiles."""

    def __init__(self, exp: dict, device: torch.device, seed: int):
        torch.manual_seed(seed)
        self.exp, self.device = exp, device
        self.rng = np.random.default_rng(seed)
        self.model = CurveModel(exp["backbone"], exp.get("pretrained", True), exp.get("drop_path", 0.0)).to(device)
        cfg = self.model.backbone.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.amp = device.type == "cuda"  # fp16 on T4 / P100; MPS fp16 autocast is not reliable
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def fit(self, tiles: np.ndarray, y_tiles: np.ndarray, sample_ids: pd.Series, max_steps: int | None = None) -> list:
        """Train; returns one log row per epoch (mean training EMD, lr, seconds)."""
        exp = self.exp
        bs, epochs = exp["batch_size"], exp["epochs"]
        steps = len(tiles) // bs
        # Every sample is drawn equally often, however many photos and tiles it has.
        p = 1 / sample_ids.map(sample_ids.value_counts()).to_numpy(float)
        p /= p.sum()
        # LP-FT: a linear probe first, then everything. Each phase gets its own optimizer and warmup:
        # the backbone enters the second phase with empty Adam moments, so its first steps move every
        # weight by ~lr, and at the peak lr that erases the pretraining (the first S3 run collapsed so).
        head_only = exp.get("head_only_epochs", 0)
        phases = [(True, head_only), (False, epochs - head_only)] if head_only else [(False, epochs)]
        log, step, epoch = [], 0, 0
        for frozen, phase_epochs in phases:
            for prm in self.model.backbone.parameters():
                prm.requires_grad = not frozen
            opt = create_optimizer_v2(self.model.head if frozen else self.model, opt=exp["opt"], lr=exp["lr"],
                                      weight_decay=exp["weight_decay"],
                                      layer_decay=None if frozen else exp.get("layer_decay"))
            sched = CosineLRScheduler(opt, t_initial=phase_epochs * steps,
                                      warmup_t=int(exp.get("warmup_epochs", 0) * steps),
                                      warmup_lr_init=exp["lr"] * 0.01, lr_min=exp["lr"] * 0.01, t_in_epochs=False)
            phase_step = 0
            for _ in range(phase_epochs):
                self.model.train()
                if frozen:
                    self.model.backbone.eval()
                elif exp.get("bn", "frozen") == "frozen":
                    freeze_bn_stats(self.model)
                start, losses = time.time(), []
                order = self.rng.choice(len(tiles), size=steps * bs, p=p)
                for b in range(steps):
                    idx = np.sort(order[b * bs:(b + 1) * bs])
                    x = self._norm(augment(to_input(tiles[idx], self.device)))
                    y = torch.from_numpy(y_tiles[idx]).float().to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                        loss = emd_loss(self.model(x), y)
                    opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(opt)
                    self.scaler.update()
                    step += 1
                    phase_step += 1
                    sched.step_update(phase_step)
                    losses.append(loss.item())
                    if max_steps and step >= max_steps:
                        return log + [{"epoch": epoch, "train_emd": np.mean(losses), "steps": len(losses),
                                       "seconds": time.time() - start}]
                log.append({"epoch": epoch, "train_emd": float(np.mean(losses)), "lr": opt.param_groups[-1]["lr"],
                            "seconds": time.time() - start})
                epoch += 1
        return log

    @torch.no_grad()
    def predict(self, tiles: np.ndarray, batch: int = 128) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(tiles), batch):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                out.append(self.model(self._norm(to_input(tiles[i:i + batch], self.device))).float().cpu().numpy())
        return np.concatenate(out)


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return float("nan")


def cross_validate(exp: dict, tiles: np.ndarray, meta: pd.DataFrame, labels: pd.DataFrame, folds: pd.DataFrame,
                   device: torch.device, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold sample curves (same layout as the ridge runs) and a per-epoch training log."""
    y_samples = labels.set_index("sample_id")[TARGET_COLS].astype(float)
    repeats = exp.get("repeats", sorted(folds["repeat"].unique()))
    oof, logs = [], []
    for (repeat, fold), part in folds[folds["repeat"].isin(repeats)].groupby(["repeat", "fold"]):
        is_val = meta["sample_id"].isin(set(part["sample_id"])).to_numpy()
        m_tr, m_va = meta[~is_val].reset_index(drop=True), meta[is_val].reset_index(drop=True)
        trainer = Trainer(exp, device, seed + 100 * repeat + fold)
        start = time.time()
        log = trainer.fit(tiles[~is_val], y_samples.loc[m_tr["sample_id"]].to_numpy(), m_tr["sample_id"])
        agg = aggregate(trainer.predict(tiles[is_val]), m_va, exp.get("agg", "mean"))
        agg.insert(0, "fold", fold)
        agg.insert(0, "repeat", repeat)
        oof.append(agg.reset_index())
        for row in log:
            logs.append({"repeat": repeat, "fold": fold, **row})
        print(f"repeat {repeat} fold {fold}: {time.time() - start:.0f}s, last train EMD {log[-1]['train_emd']:.1f}",
              flush=True)
    return pd.concat(oof, ignore_index=True), pd.DataFrame(logs)
