"""Part 0 EDA: data inventory, physical scale per device, curve shapes and metric baselines.

Writes figures and reports/eda/summary.md; every number in the report comes from here.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgsd.config import load_config, resolve
from soilgsd.data import SIEVES_MM, TARGET_COLS, load_labels, load_photo_index
from soilgsd.folds import iter_folds
from soilgsd.metric import d_value_errors, emd_per_sample, shift_curve

OUT = resolve("reports/eda")
CLASS_COLORS = {"fine": "#2a6fdb", "sand": "#d9a400", "gravel": "#8a5a44"}
UNIFORM = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])


def plot_curves(labels: pd.DataFrame, classes: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (_, row), cls in zip(labels.iterrows(), classes, strict=True):
        ax.plot(SIEVES_MM, row[TARGET_COLS].to_numpy(float), color=CLASS_COLORS[cls], alpha=0.75, lw=1.2)
    for cls, c in CLASS_COLORS.items():
        ax.plot([], [], color=c, label=f"{cls} (n={int((classes == cls).sum())})")
    ax.set_xscale("log")
    ax.set_xlabel("grain diameter, mm (log)")
    ax.set_ylabel("% passing (cumulative)")
    ax.set_title("Training grading curves by D50 class")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=150)
    plt.close(fig)


def physical_crop(path: str, ppm: float, side_mm: float, out_px: int) -> Image.Image:
    """Centre crop of `side_mm` x `side_mm`, resampled to out_px: same mm per pixel for every photo."""
    im = ImageOps.exif_transpose(Image.open(resolve(path))).convert("RGB")
    side = int(round(side_mm * ppm))
    cx, cy = im.width // 2, im.height // 2
    crop = im.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
    return crop.resize((out_px, out_px), Image.Resampling.LANCZOS)


def plot_scale_sheet(index: pd.DataFrame, side_mm: float = 60, out_px: int = 270) -> None:
    """The same physical window (side_mm) from each device, next to a naive same-pixel crop."""
    picks = (index.sort_values("path").groupby("device").head(1)
             .sort_values(["split", "device"]).reset_index(drop=True))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6))
    naive_px = int(round(side_mm * picks["ppm"].min()))
    for j, row in picks.iterrows():
        axes[0, j].imshow(physical_crop(row["path"], row["ppm"], side_mm, out_px))
        axes[0, j].set_title(f"{row['device']}\n{row['split']}, {row['ppm']:.1f} px/mm", fontsize=8)
        # Naive: the same number of pixels from every photo, ignoring the physical scale.
        axes[1, j].imshow(physical_crop(row["path"], naive_px / side_mm, side_mm, out_px))
        axes[1, j].set_title(f"naive {naive_px}px = {naive_px / row['ppm']:.0f} mm", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel(f"{side_mm:.0f} mm window", fontsize=9)
    axes[1, 0].set_ylabel(f"{naive_px} px window", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "scale_sheet.png", dpi=130)
    plt.close(fig)


def plot_full_photos(index: pd.DataFrame) -> None:
    """Whole photos: what besides soil is in the frame (tray, rims, lamp)."""
    picks = index.sort_values("path").groupby("device").head(1).sort_values(["split", "device"])
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3))
    for ax, (_, row) in zip(axes, picks.iterrows(), strict=True):
        im = ImageOps.exif_transpose(Image.open(resolve(row["path"])))
        im.thumbnail((500, 500))
        ax.imshow(im)
        ax.set_title(f"{row['device']} ({row['field_w_mm']:.0f}x{row['field_h_mm']:.0f} mm)", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "full_photos.png", dpi=130)
    plt.close(fig)


def cv_constant_baselines(labels: pd.DataFrame, folds: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    """Image-free predictors scored with the same folds that every model will use."""
    y = labels.set_index("sample_id")[TARGET_COLS].astype(float)
    cls = pd.Series(classes.to_numpy(), index=y.index)
    results = {name: [] for name in ["uniform (Kaggle baseline)", "train mean", "train median",
                                     "oracle D50 class -> class median"]}
    d_errs = {name: [] for name in results}
    for r in sorted(folds["repeat"].unique()):
        per_name = {name: [] for name in results}
        for _, tr, va in iter_folds(folds, r):
            yt, yv = y.loc[tr].to_numpy(), y.loc[va].to_numpy()
            preds = {
                "uniform (Kaggle baseline)": np.tile(UNIFORM, (len(va), 1)),
                "train mean": np.tile(yt.mean(0), (len(va), 1)),
                # EMD is a weighted L1 on the curve, so the per-point median is the best constant.
                "train median": np.tile(np.median(yt, 0), (len(va), 1)),
                # Upper bound for a 3-way texture classifier: knows the true class of each sample.
                "oracle D50 class -> class median": np.stack([
                    np.median(y.loc[[s for s in tr if cls[s] == cls[v]]].to_numpy(), 0) for v in va]),
            }
            for name, p in preds.items():
                per_name[name].append((yv, p))
        for name, pairs in per_name.items():
            yv = np.concatenate([a for a, _ in pairs])
            p = np.concatenate([b for _, b in pairs])
            results[name].append(emd_per_sample(yv, p).mean())
            d_errs[name].append(d_value_errors(yv, p)["D50_log_err"])
    return pd.DataFrame({
        "CV EMD mean": {k: np.mean(v) for k, v in results.items()},
        "std over repeats": {k: np.std(v) for k, v in results.items()},
        "D50 |log10 err|": {k: np.mean(v) for k, v in d_errs.items()},
    }).round(2)


def scale_error_costs(labels: pd.DataFrame) -> pd.DataFrame:
    """EMD caused by a pure scale error on the true curves (grains look `factor`x larger)."""
    y = labels[TARGET_COLS].to_numpy(float)
    rows = {}
    for f in [1.25, 1.5, 2.0, 3.0, 4.0, 1 / 2, 1 / 4]:
        e = emd_per_sample(y, shift_curve(y, f))
        rows[f"x{f:g}"] = {"median": np.median(e), "min": e.min(), "max": e.max(),
                           "ideal 100*|log10 f|": 100 * abs(np.log10(f))}
    return pd.DataFrame(rows).T.round(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)
    OUT.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg)
    index = load_photo_index(cfg)
    folds = pd.read_csv(resolve(cfg["cv"]["folds_file"]))
    classes = labels["sample_id"].map(folds.drop_duplicates("sample_id").set_index("sample_id")["d50_class"])

    inventory = (index.groupby(["split", "device"])
                 .agg(photos=("path", "size"), samples=("sample_id", "nunique"),
                      stored_px=("width", lambda s: f"{s.max()}"), stored_scale=("stored_scale", "first"),
                      ppm=("ppm", "first"), field_mm=("field_w_mm", "mean")).round(2))
    per_sample = index.groupby(["split", "sample_id"]).size().groupby("split").describe()[["min", "50%", "max"]]
    devices_per_sample = index[index.split == "train"].groupby("sample_id")["device"].agg(lambda s: " + ".join(sorted(set(s))))

    plot_curves(labels, classes)
    plot_scale_sheet(index)
    plot_full_photos(index)
    baselines = cv_constant_baselines(labels, folds, classes)
    scale_costs = scale_error_costs(labels)

    y = labels[TARGET_COLS].to_numpy(float)
    fines = y[:, 3]  # % finer than 0.063 mm: below one pixel at 4.6 px/mm (0.22 mm)
    lines = [
        "# EDA summary (generated by scripts/eda.py)",
        "",
        "## Inventory and physical scale",
        inventory.to_markdown(),
        "",
        "Photos per sample:",
        per_sample.to_markdown(),
        "",
        "Device combinations of training samples:",
        devices_per_sample.value_counts().to_markdown(),
        "",
        "## Curves",
        f"D50 classes: {classes.value_counts().to_dict()}",
        f"% finer than 0.063 mm: median {np.median(fines):.1f}, range {fines.min():.1f}-{fines.max():.1f}",
        f"Samples with any mass > 63 mm: {int((y[:, 9] < 100).sum())}",
        f"Samples with mass > 20 mm (F(20) < 100): {int((y[:, 8] < 100).sum())}",
        "",
        "## Image-free baselines on data/folds.csv (6 folds x 3 repeats)",
        baselines.to_markdown(),
        "",
        "## Cost of a pure scale error on the true training curves",
        scale_costs.to_markdown(),
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
