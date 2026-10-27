"""Labels, camera table and the photo index (one row per photo with its physical scale)."""

import difflib
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from soilgsd.config import resolve

SIEVES_MM = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
TARGET_COLS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]


def norm_key(text: str) -> str:
    """Lower-case ASCII letters and digits only, so that file names and ids can be compared."""
    text = text.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def load_labels(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(resolve(cfg["data"]["train_labels"]))
    assert list(df.columns[1:]) == TARGET_COLS, df.columns
    return df


def load_cameras(cfg: dict) -> pd.DataFrame:
    cams = pd.read_csv(resolve(cfg["data"]["ppm_table"]))
    cams["key"] = cams["phone"].map(norm_key)
    return cams


def _match_device(stem_key: str, cams: pd.DataFrame) -> pd.Series:
    # Longest prefix wins: "motorolaedge60fusion..." must not match plain "motorolaedge".
    hits = cams[cams["key"].map(stem_key.startswith)]
    if hits.empty:
        raise ValueError(f"no camera in the ppm table matches {stem_key!r}")
    return hits.loc[hits["key"].str.len().idxmax()]


def _match_sample(name_key: str, ids: list[str]) -> str:
    keys = {norm_key(i): i for i in ids}
    if name_key in keys:
        return keys[name_key]
    # Archive extraction can mangle non-ASCII characters (e.g. "Münster" -> "M++nster");
    # fall back to the closest id and refuse ambiguous matches.
    close = difflib.get_close_matches(name_key, list(keys), n=2, cutoff=0.85)
    if len(close) != 1:
        raise ValueError(f"cannot map photo {name_key!r} to one sample id (candidates: {close})")
    return keys[close[0]]


def build_photo_index(cfg: dict) -> pd.DataFrame:
    """One row per photo: split, sample_id, device and the effective pixels per mm.

    ppm_updated.csv gives pixels per mm at each camera's native resolution. Some photos were
    stored downscaled, so the effective ppm is the native ppm scaled by the stored long side
    over the native long side.
    """
    cams = load_cameras(cfg)
    split_ids = {
        "train": load_labels(cfg)["sample_id"].tolist(),
        "test": pd.read_csv(resolve(cfg["data"]["sample_submission"]))["sample_id"].tolist(),
    }
    rows = []
    for split, folder in [("train", cfg["data"]["train_photos"]), ("test", cfg["data"]["test_photos"])]:
        for path in sorted(resolve(folder).iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem_key = norm_key(path.stem)
            cam = _match_device(stem_key, cams)
            # Drop the device prefix and the shot number ("_01" or " (2)") before matching the id.
            rest = re.sub(r"(_\d+|\s*\(\d+\))$", "", path.stem)
            rest_key = norm_key(rest)[len(cam["key"]):]
            sample_id = _match_sample(rest_key, split_ids[split])
            with Image.open(path) as im:
                w, h = im.size
                orientation = im.getexif().get(274, 1)
            scale = max(w, h) / max(cam["width"], cam["height"])
            rows.append({
                "path": str(path.relative_to(resolve("."))),
                "split": split,
                "sample_id": sample_id,
                "device": cam["phone"],
                "width": w,
                "height": h,
                "exif_orientation": orientation,
                "ppm_native": cam["ppm"],
                "stored_scale": round(scale, 4),
                "ppm": round(cam["ppm"] * scale, 4),
            })
    index = pd.DataFrame(rows)
    index["field_w_mm"] = (index[["width", "height"]].max(axis=1) / index["ppm"]).round(1)
    index["field_h_mm"] = (index[["width", "height"]].min(axis=1) / index["ppm"]).round(1)
    for split, ids in split_ids.items():
        missing = set(ids) - set(index.loc[index["split"] == split, "sample_id"])
        assert not missing, f"{split} samples without photos: {missing}"
    return index


def load_photo_index(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg["data"]["photo_index"])
    return pd.read_csv(path) if Path(path).exists() else build_photo_index(cfg)
