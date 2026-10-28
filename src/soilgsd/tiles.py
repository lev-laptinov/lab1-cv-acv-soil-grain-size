"""Soil region, resampling to a common scale and fixed-size tiles in millimetres."""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from soilgsd.config import resolve

TILE_PX = 224  # network input size


def load_photo(path: str) -> Image.Image:
    """RGB photo in its displayed orientation (EXIF rotation applied)."""
    return ImageOps.exif_transpose(Image.open(resolve(path))).convert("RGB")


def _soil_run(profile: np.ndarray, ref: float, low: float, high: float) -> tuple[int, int]:
    """Longest run of soil-like values (between low*ref and high*ref) that contains the centre."""
    ok = (profile > low * ref) & (profile < high * ref)
    c = len(profile) // 2
    if not ok[c]:
        return 0, len(profile)
    a = c
    while a > 0 and ok[a - 1]:
        a -= 1
    b = c
    while b < len(profile) - 1 and ok[b + 1]:
        b += 1
    return a, b + 1


def detect_soil_box(im: Image.Image, low: float = 0.6, high: float = 1.5) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the soil in the tray as fractions of the frame size.

    Tray rims and the dark side strip are much darker than soil, the Samsung lamp is much brighter.
    The median brightness of each row (and column) is compared with the median of the frame centre;
    the soil is the run of rows (columns) around the centre that stays close to it. This fails when
    the centre is a single bright stone, so it is only used through soil_boxes().
    """
    scale = 400 / max(im.size)
    small = np.asarray(im.convert("L").resize((round(im.width * scale), round(im.height * scale))), float)
    h, w = small.shape
    ref = np.median(small[h // 3: 2 * h // 3, w // 3: 2 * w // 3])
    # Profiles over the middle half only, so a rim on the other axis does not drag the median.
    rows = np.median(small[:, w // 4: 3 * w // 4], axis=1)
    cols = np.median(small[h // 4: 3 * h // 4, :], axis=0)
    t, b = _soil_run(rows, ref, low, high)
    left, r = _soil_run(cols, ref, low, high)
    return left / w, t / h, r / w, b / h


def soil_boxes(index: pd.DataFrame, margin_mm: float = 5.0) -> pd.DataFrame:
    """Soil box in pixels for every photo: the median detected box of its device and orientation.

    Each camera sits at a fixed position over the tray, so the box is a property of the setup, not of
    the photo. The median over a group overrides the few photos where detection fails, and the box
    cannot depend on what the soil looks like. A margin keeps the rim's shadow out.
    """
    rows = []
    for _, r in index.iterrows():
        im = load_photo(r["path"])
        rows.append((im.width, im.height, *detect_soil_box(im)))
    boxes = pd.DataFrame(rows, columns=["w", "h", "l", "t", "r", "b"], index=index.index)
    group = index["device"] + np.where(boxes["w"] > boxes["h"], " landscape", " portrait")
    rel = boxes[["l", "t", "r", "b"]].groupby(group).transform("median")
    m = margin_mm * index["ppm"]
    out = pd.DataFrame({
        "path": index["path"],
        "group": group,
        "left": (rel["l"] * boxes["w"] + m).round().astype(int),
        "top": (rel["t"] * boxes["h"] + m).round().astype(int),
        "right": (rel["r"] * boxes["w"] - m).round().astype(int),
        "bottom": (rel["b"] * boxes["h"] - m).round().astype(int),
    })
    out["soil_w_mm"] = ((out["right"] - out["left"]) / index["ppm"]).round(1)
    out["soil_h_mm"] = ((out["bottom"] - out["top"]) / index["ppm"]).round(1)
    return out


def load_soil_boxes(cfg: dict, index: pd.DataFrame) -> pd.DataFrame:
    """Soil box of every photo (train and test), computed once and kept in data/soil_boxes.csv."""
    path = resolve(cfg["data"]["soil_boxes"])
    if not path.exists():
        soil_boxes(index).to_csv(path, index=False)
    boxes = pd.read_csv(path)
    assert (boxes["path"] == index["path"]).all(), "soil_boxes.csv is stale: delete it and rerun"
    return boxes


def cut_tiles(im: Image.Image, ppm: float, box: tuple[int, int, int, int], tile_mm: float,
              work_ppm: float, out_px: int = TILE_PX) -> np.ndarray:
    """Tiles of tile_mm x tile_mm covering the soil box, shape (n, out_px, out_px, 3).

    The box is first resampled from the photo's ppm to work_ppm, so one pixel is the same number
    of millimetres on every camera; only then is it cut. The grid covers the whole box: the fewest
    tiles that fit, spread evenly, so neighbours overlap slightly instead of dropping a strip of
    soil. Each tile is finally resized to the network input size (the same factor for every photo,
    so the physical scale stays common).
    """
    soil = im.crop(box)
    f = work_ppm / ppm
    soil = soil.resize((round(soil.width * f), round(soil.height * f)), Image.Resampling.LANCZOS)
    side = tile_mm * work_ppm  # may be fractional (50 mm at 2.25 px/mm = 112.5 px)
    px = round(side)
    xs = _starts(soil.width, side)
    ys = _starts(soil.height, side)
    tiles = [np.asarray(soil.crop((x, y, x + px, y + px)).resize((out_px, out_px), Image.Resampling.BICUBIC))
             for y in ys for x in xs]
    return np.stack(tiles) if tiles else np.zeros((0, out_px, out_px, 3), np.uint8)


def _starts(length: int, side: float) -> list[int]:
    """Left edges of the fewest tiles of `side` px that cover `length`, spread evenly.

    The count uses the exact side, so rounding 112.5 px down to 112 does not add a tile.
    """
    if length < round(side):
        return []
    n = int(np.ceil(length / side - 1e-9))
    return [int(v) for v in np.linspace(0, length - round(side), n).round()]


def tile_cache(index: pd.DataFrame, boxes: pd.DataFrame, tile_mm: float, work_ppm: float,
               split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """All tiles of a split as uint8 (n, 224, 224, 3) plus one row per tile (path, sample_id).

    Cutting is the slow part (decoding 12-48 MP JPEGs), so it runs once per setup and the result
    is kept in data/cache/tiles/; every model (frozen features or fine-tuning) reads the same tiles.
    """
    path = resolve(f"data/cache/tiles/{split}_t{tile_mm:g}_p{work_ppm:g}.npz")
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return z["tiles"], pd.DataFrame({"path": z["path"], "sample_id": z["sample_id"]})
    tiles, rows = [], []
    for i, r in index[index["split"] == split].iterrows():
        b = boxes.loc[i]
        t = cut_tiles(load_photo(r["path"]), r["ppm"], (b["left"], b["top"], b["right"], b["bottom"]),
                      tile_mm, work_ppm)
        tiles.append(t)
        rows += [(r["path"], r["sample_id"])] * len(t)
    tiles = np.concatenate(tiles)
    meta = pd.DataFrame(rows, columns=["path", "sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tiles=tiles, path=meta["path"].to_numpy(str), sample_id=meta["sample_id"].to_numpy(str))
    return tiles, meta
