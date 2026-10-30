import argparse
import math
import time

import lightning as L
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor
from timm.optim import param_groups_layer_decay, param_groups_weight_decay
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from part2_vit.augment import d4_views, train_augment
from part2_vit.metrics import LOSSES, curve_scores, sample_curves
from part2_vit.vit import VisionTransformer, load_timm_weights
from part2_vit.wandb_logging import AttentionMapLogger, fold_logger, log_checkpoint, log_summary, watch_weights
from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.finetune import peak_memory_gb
from soilgsd.metric import is_valid_curve
from soilgsd.results import RESULTS, save_result, summarise
from soilgsd.tiles import load_soil_boxes, tile_cache


def build_backbone(cfg: dict) -> nn.Module:
    if cfg["backbone"] == "own_vit":
        model = VisionTransformer(**cfg["vit"], drop_path_rate=cfg["drop_path"])
        if cfg["init"] != "scratch":
            load_timm_weights(model, cfg["init"])
        return model
    return timm.create_model(cfg["backbone"], pretrained=True, num_classes=0, drop_path_rate=cfg["drop_path"])


def mean_bin_shares(curves: np.ndarray) -> np.ndarray:
    shares = np.diff(np.concatenate([np.zeros((len(curves), 1)), curves], axis=1), axis=1) / 100
    return shares.mean(axis=0)


class CurveRegressor(L.LightningModule):
    def __init__(self, cfg: dict, shares: np.ndarray, val_meta: pd.DataFrame, test_meta: pd.DataFrame,
                 labels: pd.DataFrame):
        super().__init__()
        self.cfg, self.val_meta, self.test_meta, self.labels = cfg, val_meta, test_meta, labels
        self.backbone = build_backbone(cfg)
        self.head = nn.Linear(self.backbone.num_features, len(TARGET_COLS))
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.log(torch.as_tensor(shares, dtype=torch.float32) + 1e-4))
        self.loss_fn = LOSSES[cfg.get("loss", "emd")]
        norm = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(norm["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(norm["std"]).view(1, 3, 1, 1), persistent=False)
        self.preds, self.history, self.val_curves, self.test_curves = [], [], None, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shares = torch.softmax(self.head(self.backbone(x)).float(), dim=1)
        return 100 * torch.cumsum(shares, dim=1)

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) -> normalised float."""
        return (x.float() / 255 - self.mean) / self.std

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = x.float() / 255
        if self.trainer.training:
            x, y = train_augment(x, y, self.cfg.get("aug", {}))
        return (x - self.mean) / self.std, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train/emd", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _predict(self, x: torch.Tensor, tta: bool) -> np.ndarray:
        views = d4_views(x) if tta else [x]
        return torch.stack([self(v) for v in views]).mean(0).float().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        # TTA only on the last epoch (the OOF predictions)
        last = self.current_epoch == self.trainer.max_epochs - 1
        self.preds.append(self._predict(batch[0], self.cfg.get("tta", True) and last))

    def on_validation_epoch_end(self):
        n = sum(len(p) for p in self.preds)
        curves = sample_curves(np.concatenate(self.preds), self.val_meta.iloc[:n], self.cfg.get("agg", "mean"))
        self.preds = []
        scores = curve_scores(curves, self.labels)
        self.log_dict({f"val/{k}": v for k, v in scores.items()})
        self.val_curves = curves

    def on_train_epoch_end(self):
        m = self.trainer.callback_metrics
        self.history.append({"epoch": self.current_epoch, "train_emd": float(m["train/emd_epoch"]),
                             "val_emd": float(m.get("val/emd", math.nan)),
                             "lr": self.trainer.optimizers[0].param_groups[-1]["lr"]})

    def test_step(self, batch, batch_idx):
        self.preds.append(self._predict(batch[0], self.cfg.get("tta", True)))

    def on_test_epoch_end(self):
        n = sum(len(p) for p in self.preds)
        self.test_curves = sample_curves(np.concatenate(self.preds), self.test_meta.iloc[:n], self.cfg.get("agg", "mean"))
        self.preds = []

    def configure_optimizers(self):
        c = self.cfg
        skip = self.backbone.no_weight_decay() if hasattr(self.backbone, "no_weight_decay") else ()
        if c.get("layer_decay"):
            groups = param_groups_layer_decay(self.backbone, c["weight_decay"], no_weight_decay_list=skip,
                                              layer_decay=c["layer_decay"])
        else:
            groups = param_groups_weight_decay(self.backbone, c["weight_decay"], no_weight_decay_list=skip)
        for g in groups:
            g["lr"] = c["lr"] * g.pop("lr_scale", 1.0)
        groups += param_groups_weight_decay(self.head, c["weight_decay"])
        opt = torch.optim.AdamW(groups, lr=c["lr"])
        total = self.trainer.estimated_stepping_batches
        warmup = int(c.get("warmup_epochs", 0) * total / c["epochs"])

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            t = min(1.0, (step - warmup) / max(1, total - warmup))
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def load_model(cfg_path: str, ckpt_path: str) -> CurveRegressor:
    """A saved fold model: model(model.prepare(uint8 tiles (B, 3, 224, 224))) -> (B, 11) curves."""
    cfg = yaml.safe_load(open(resolve(cfg_path)))
    model = CurveRegressor(cfg, np.full(len(TARGET_COLS), 1 / len(TARGET_COLS)), None, None, None)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


class Tiles(Dataset):

    def __init__(self, tiles: np.ndarray, targets: np.ndarray, idx: np.ndarray):
        self.tiles, self.targets, self.idx = tiles, targets, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return torch.from_numpy(self.tiles[j]).permute(2, 0, 1), torch.from_numpy(self.targets[j])


def train_loader(tiles, targets, idx, sample_ids: pd.Series, batch_size: int, seed: int) -> DataLoader:
    weights = 1 / sample_ids.map(sample_ids.value_counts()).to_numpy(float)
    n = len(idx) // batch_size * batch_size
    sampler = WeightedRandomSampler(weights, n, replacement=True, generator=torch.Generator().manual_seed(seed))
    return DataLoader(Tiles(tiles, targets, idx), batch_size=batch_size, sampler=sampler, drop_last=True)


def map_tiles(tiles: np.ndarray, meta: pd.DataFrame, idx: np.ndarray, k: int = 6) -> tuple[np.ndarray, list[str]]:
    m = meta.iloc[idx].reset_index()
    first = m.groupby("sample_id", sort=False)["path"].first()
    picks = []
    for sid, path in list(first.items())[:k]:
        rows = m.index[(m["sample_id"] == sid) & (m["path"] == path)]
        picks.append((rows[len(rows) // 2], sid))
    return tiles[idx[[p for p, _ in picks]]], [s for _, s in picks]


def run_fold(cfg, tiles, meta, targets, labels, test_tiles, test_meta, repeat, fold, val_ids, args, out_dir):
    seed = cfg.get("seed", 42) + 100 * repeat + fold
    L.seed_everything(seed, workers=True)
    is_val = meta["sample_id"].isin(val_ids).to_numpy()
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    y_samples = labels.set_index("sample_id").loc[sorted(set(meta["sample_id"][tr_idx])), TARGET_COLS].to_numpy(float)

    module = CurveRegressor(cfg, mean_bin_shares(y_samples), meta.iloc[va_idx].reset_index(drop=True), test_meta, labels)
    logger = fold_logger(cfg, repeat, fold, enabled=not args.smoke and not args.no_wandb)
    watch_weights(logger, module)
    map_x, map_caps = map_tiles(tiles, meta, va_idx)
    cuda = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=1 if args.smoke else cfg["epochs"], limit_train_batches=args.smoke or 1.0, accelerator="auto", devices=1,
        precision="16-mixed" if cuda else "32-true",
        logger=logger, callbacks=[LearningRateMonitor("step"), AttentionMapLogger(map_x, map_caps)],
        gradient_clip_val=1.0, num_sanity_val_steps=0, enable_checkpointing=False, log_every_n_steps=10,
        inference_mode=False,  # Grad-CAM needs gradients
        limit_val_batches=2 if args.smoke else 1.0, limit_test_batches=2 if args.smoke else 1.0,
    )
    val = DataLoader(Tiles(tiles, targets, va_idx), batch_size=128)
    test = DataLoader(Tiles(test_tiles, np.zeros((len(test_tiles), 11), np.float32), np.arange(len(test_tiles))),
                      batch_size=128)
    start = time.time()
    trainer.fit(module, train_loader(tiles, targets, tr_idx, meta["sample_id"].iloc[tr_idx], cfg["batch_size"], seed), val)
    trainer.test(module, test, verbose=False)
    seconds = time.time() - start

    if not args.smoke and repeat in cfg.get("checkpoint_repeats", [0]):
        path = out_dir / "checkpoints" / f"r{repeat}_f{fold}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(module.state_dict(), path)
        if not args.no_wandb:
            log_checkpoint(logger, str(path), cfg, repeat, fold, curve_scores(module.val_curves, labels))
    logger.experiment.finish()
    print(f"repeat {repeat} fold {fold}: {seconds:.0f}s, val EMD {curve_scores(module.val_curves, labels)['emd']:.1f}",
          flush=True)
    log = pd.DataFrame(module.history).assign(repeat=repeat, fold=fold, seconds=seconds / max(1, len(module.history)))
    oof = module.val_curves.reset_index().assign(repeat=repeat, fold=fold)
    return oof, log, module.test_curves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="part2_vit/configs/*.yaml")
    ap.add_argument("--repeats", type=int, nargs="+", help="override the repeats of the config")
    ap.add_argument("--smoke", type=int, default=0, help="train this many steps on one fold only, write nothing")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    base = load_config(args.config)
    cfg = yaml.safe_load(open(resolve(args.exp)))
    if args.repeats:
        cfg["repeats"] = args.repeats

    index, labels = load_photo_index(base), load_labels(base)
    folds = pd.read_csv(resolve(base["cv"]["folds_file"]))
    boxes = load_soil_boxes(base, index)
    tiles, meta = tile_cache(index, boxes, cfg["tile_mm"], cfg["work_ppm"], "train")
    test_tiles, test_meta = tile_cache(index, boxes, cfg["tile_mm"], cfg["work_ppm"], "test")
    targets = labels.set_index("sample_id").loc[meta["sample_id"], TARGET_COLS].to_numpy(np.float32)
    out_dir = resolve(f"outputs/{cfg['name']}")

    oofs, logs, tests = [], [], []
    start = time.time()
    for (repeat, fold), part in folds[folds["repeat"].isin(cfg["repeats"])].groupby(["repeat", "fold"]):
        repeat, fold = int(repeat), int(fold)  # numpy ints from groupby; torch's manual_seed rejects them
        oof, log, test = run_fold(cfg, tiles, meta, targets, labels, test_tiles, test_meta, repeat, fold,
                                  set(part["sample_id"]), args, out_dir)
        oofs.append(oof), logs.append(log), tests.append(test.reset_index().assign(repeat=repeat, fold=fold))
        if args.smoke:
            return
    hours = (time.time() - start) / 3600

    cols = ["sample_id", "repeat", "fold", *TARGET_COLS]
    outputs = {"oof.csv": pd.concat(oofs, ignore_index=True)[cols], "train_log.csv": pd.concat(logs, ignore_index=True),
               "test_preds.csv": pd.concat(tests, ignore_index=True)[cols]}  # test curves of every fold model
    compute = ""
    if args.repeats and (out_dir / "oof.csv").exists():  # --repeats extends an earlier run of this config
        for file, new in outputs.items():
            old = pd.read_csv(out_dir / file)
            outputs[file] = pd.concat([old[~old["repeat"].isin(args.repeats)], new], ignore_index=True)
        cfg["repeats"] = sorted(outputs["oof.csv"]["repeat"].unique().tolist())
        old = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=["name"])
        compute = "".join(f"{c} + " for c in old.loc[old["name"] == cfg["name"], "compute"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for file, df in outputs.items():
        df.to_csv(out_dir / file, index=False)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    oof, test_preds = outputs["oof.csv"], outputs["test_preds.csv"]

    # Submission: mean of the test curves of all fold models
    sub = pd.read_csv(resolve(base["data"]["sample_submission"]))[["sample_id"]]
    sub = sub.join(test_preds.groupby("sample_id")[TARGET_COLS].mean(), on="sample_id")
    assert sub[TARGET_COLS].notna().all().all() and is_valid_curve(sub[TARGET_COLS].to_numpy()).all()
    sub_path = resolve(f"submissions/{cfg['name']}.csv")
    sub_path.parent.mkdir(exist_ok=True)
    sub.to_csv(sub_path, index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    gpu = torch.cuda.get_device_name() if device.type == "cuda" else device.type
    backbone = cfg["backbone"] if cfg["backbone"] != "own_vit" else f"own_vit d{cfg['vit']['embed_dim']} ({cfg['init']})"
    res = {"name": cfg["name"], "approach": "finetune", "backbone": backbone, "tile_mm": cfg["tile_mm"],
           "work_ppm": cfg["work_ppm"], "agg": cfg.get("agg", "mean"), "train_tiles": len(tiles)}
    res.update(summarise(oof, labels, folds))
    res["compute"] = compute + f"{gpu} {hours:.2f} h, peak {peak_memory_gb(device):.1f} GB"
    save_result(res)
    log_summary(cfg, res, oof, str(sub_path), enabled=not args.no_wandb)
    print(pd.Series(res).to_string())


if __name__ == "__main__":
    main()
