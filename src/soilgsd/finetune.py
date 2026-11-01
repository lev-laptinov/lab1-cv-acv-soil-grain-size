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
from soilgsd.metric import LOG_WIDTHS, repair_cummax
from soilgsd.tile_model import aggregate


def load_hf_weights(model: nn.Module, spec: str) -> None:
    """Load a backbone state_dict from the Hugging Face Hub: spec = "<user>/<repo>/<file>.pth".

    Used for the MicroNet microscopy encoders (NASA, Stuckner et al. 2022): ImageNet weights trained further
    on ~100k labelled micrographs of materials (no soil). The file is read with weights_only=True, so it
    can hold tensors only, never code. ResNets use timm's (torchvision) key names; EfficientNets come in
    efficientnet_pytorch naming, which lists the same tensors in the same order as timm's tf_efficientnet,
    so they are matched by position and every shape must agree.
    """
    from huggingface_hub import hf_hub_download

    user, repo, fname = spec.split("/", 2)
    sd = torch.load(hf_hub_download(f"{user}/{repo}", fname), map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "_fc.")) and not k.endswith("num_batches_tracked")}
    own = {k: v for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    if set(sd) != set(own):
        assert len(sd) == len(own), f"{spec}: {len(sd)} tensors for a model with {len(own)}"
        sd = dict(zip(own, sd.values(), strict=True))
    bad = [k for k, v in sd.items() if v.shape != own[k].shape]
    assert not bad, f"{spec}: shape mismatch at {bad[:3]}"
    missing = model.load_state_dict(sd, strict=False).missing_keys
    assert all(k.endswith("num_batches_tracked") for k in missing), f"{spec}: not loaded {missing[:3]}"


class CurveModel(nn.Module):
    """Backbone -> 11 logits -> softmax = mass share of each sieve bin -> cumsum = % passing.

    The output is a valid curve by construction: non-negative steps, and the last value
    (everything below 200 mm) is the sum of all shares, i.e. 100.
    """

    def __init__(self, backbone: str, pretrained: bool, drop_path: float = 0.0, weights: str | None = None):
        super().__init__()
        kwargs = dict(BACKBONE_KWARGS.get(backbone, {}))
        if drop_path:
            kwargs["drop_path_rate"] = drop_path
        # External weights replace timm's own pretrained ones (the input normalisation stays ImageNet's).
        self.backbone = timm.create_model(backbone, pretrained=pretrained and not weights, num_classes=0, **kwargs)
        if weights:
            load_hf_weights(self.backbone, weights)
        self.head = nn.Linear(self.backbone.num_features, len(TARGET_COLS))

    def curve(self, feats: torch.Tensor) -> torch.Tensor:
        shares = torch.softmax(self.head(feats).float(), dim=1)
        return 100 * torch.cumsum(shares, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.curve(self.backbone(x))

    def forward_bags(self, x: torch.Tensor, bag_tiles: int) -> torch.Tensor:
        """(n_bags * bag_tiles, 3, H, W) -> one curve per bag from the mean of its tiles' features.

        The label describes the whole sample, not a tile: a 50 mm tile of a gravelly soil can hold only
        sand. Pooling the features of several tiles before the head lets the model weigh such tiles.
        """
        feats = self.backbone(x)
        return self.curve(feats.view(-1, bag_tiles, feats.shape[-1]).mean(1))


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
        self.model = CurveModel(exp["backbone"], exp.get("pretrained", True), exp.get("drop_path", 0.0),
                                exp.get("weights")).to(device)
        cfg = self.model.backbone.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.amp = device.type == "cuda"  # fp16 on T4 / P100; MPS fp16 autocast is not reliable
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _bags(self, sample_ids: pd.Series, photo_ids: pd.Series | None) -> dict:
        """Per sample: its tile positions and their draw weights (every photo of the sample weighs the same)."""
        s = sample_ids.to_numpy()
        ph = (photo_ids if photo_ids is not None else sample_ids).to_numpy()
        bags = {}
        for sid in np.unique(s):
            ix = np.flatnonzero(s == sid)
            _, inv, counts = np.unique(ph[ix], return_inverse=True, return_counts=True)
            w = 1 / counts[inv]
            bags[sid] = (ix, w / w.sum())
        return bags

    def _bag_batch(self, bags: dict, y_tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """bag_samples different samples, bag_tiles random tiles of each: tile positions (bag-major) and labels."""
        k, n = self.exp["bag_tiles"], self.exp["bag_samples"]
        sids = self.rng.choice(list(bags), size=n, replace=False)
        idx = [self.rng.choice(bags[s][0], size=k, replace=False, p=bags[s][1]) for s in sids]
        return np.concatenate(idx), np.stack([y_tiles[i[0]] for i in idx])

    def fit(self, tiles: np.ndarray, y_tiles: np.ndarray, sample_ids: pd.Series, max_steps: int | None = None,
            photo_ids: pd.Series | None = None) -> list:
        """Train; returns one log row per epoch (mean training EMD, lr, seconds).

        With bag_tiles in the config, a step sees bag_samples samples x bag_tiles tiles and the loss is taken
        on one pooled curve per sample (forward_bags); otherwise every tile is an example with its sample's curve.
        """
        exp = self.exp
        bag_k, epochs = exp.get("bag_tiles"), exp["epochs"]
        bs = tiles_per_step(exp)
        steps = len(tiles) // bs
        # Every sample is drawn equally often, however many photos and tiles it has.
        p = 1 / sample_ids.map(sample_ids.value_counts()).to_numpy(float)
        p /= p.sum()
        bags = self._bags(sample_ids, photo_ids) if bag_k else None
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
                    if bag_k:
                        idx, yb = self._bag_batch(bags, y_tiles)
                    else:
                        idx = np.sort(order[b * bs:(b + 1) * bs])
                        yb = y_tiles[idx]
                    x = self._norm(augment(to_input(tiles[idx], self.device)))
                    y = torch.from_numpy(yb).float().to(self.device)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                        pred = self.model.forward_bags(x, bag_k) if bag_k else self.model(x)
                        loss = emd_loss(pred, y)
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

    @torch.no_grad()
    def predict_samples(self, tiles: np.ndarray, meta: pd.DataFrame, batch: int = 128) -> pd.DataFrame:
        """Bag models: tile features -> mean per photo -> mean over photos -> one valid curve per sample.

        Photos weigh the same here as in training (_bags), however many tiles each one has.
        """
        self.model.eval()
        feats = []
        for i in range(0, len(tiles), batch):
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                feats.append(self.model.backbone(self._norm(to_input(tiles[i:i + batch], self.device))).float().cpu())
        df = pd.DataFrame(torch.cat(feats).numpy())
        df["sample_id"], df["path"] = meta["sample_id"].to_numpy(), meta["path"].to_numpy()
        per_sample = df.groupby(["sample_id", "path"]).mean().groupby("sample_id").mean()
        feats = torch.tensor(per_sample.to_numpy(), dtype=torch.float32, device=self.device)
        curves = self.model.curve(feats).cpu().numpy()
        return pd.DataFrame(repair_cummax(curves), index=per_sample.index, columns=TARGET_COLS)


def tiles_per_step(exp: dict) -> int:
    """Tiles in one training step: batch_size, or bag_samples x bag_tiles for bag models."""
    return exp["bag_samples"] * exp["bag_tiles"] if exp.get("bag_tiles") else exp["batch_size"]


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
        log = trainer.fit(tiles[~is_val], y_samples.loc[m_tr["sample_id"]].to_numpy(), m_tr["sample_id"],
                          photo_ids=m_tr["path"])
        if exp.get("bag_tiles"):
            agg = trainer.predict_samples(tiles[is_val], m_va)
        else:
            agg = aggregate(trainer.predict(tiles[is_val]), m_va, exp.get("agg", "mean"))
        agg.insert(0, "fold", fold)
        agg.insert(0, "repeat", repeat)
        oof.append(agg.reset_index())
        for row in log:
            logs.append({"repeat": repeat, "fold": fold, **row})
        print(f"repeat {repeat} fold {fold}: {time.time() - start:.0f}s, last train EMD {log[-1]['train_emd']:.1f}",
              flush=True)
    return pd.concat(oof, ignore_index=True), pd.DataFrame(logs)
