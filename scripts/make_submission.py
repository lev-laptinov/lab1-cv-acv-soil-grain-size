"""Fit a frozen-feature + ridge experiment on all training samples and write its test submission.

The alpha is the median of the alphas chosen inside CV (outputs/<name>/oof.csv), so the test model
uses nothing that was tuned on the test photos. The fitted scaler + ridge is saved for scripts/predict.py.
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.features import tile_features
from soilgsd.metric import is_valid_curve
from soilgsd.tile_model import aggregate, sample_weights
from soilgsd.tiles import load_soil_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="configs/exp/*.yaml of a frozen+ridge experiment")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = yaml.safe_load(open(resolve(args.exp)))
    index, labels = load_photo_index(cfg), load_labels(cfg)
    boxes = load_soil_boxes(cfg, index)
    out_dir = resolve(f"outputs/{exp['name']}")
    alpha = float(np.median(pd.read_csv(out_dir / "oof.csv")["alpha"]))

    x_tr, m_tr = tile_features(index, boxes, exp["backbone"], exp["tile_mm"], exp["work_ppm"], "train")
    y = labels.set_index("sample_id").loc[m_tr["sample_id"], TARGET_COLS].to_numpy(float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_tr, y, ridge__sample_weight=sample_weights(m_tr["sample_id"]))
    joblib.dump({"model": model, "exp": exp}, out_dir / "model.joblib")

    x_te, m_te = tile_features(index, boxes, exp["backbone"], exp["tile_mm"], exp["work_ppm"], "test")
    pred = aggregate(model.predict(x_te), m_te, exp["agg"])
    sub = pd.read_csv(resolve(cfg["data"]["sample_submission"]))[["sample_id"]]
    sub = sub.join(pred, on="sample_id")
    assert sub[TARGET_COLS].notna().all().all(), "a test sample has no prediction"
    assert is_valid_curve(sub[TARGET_COLS].to_numpy()).all()
    path = resolve(f"submissions/{exp['name']}.csv")
    path.parent.mkdir(exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"{path} (alpha {alpha:g}, {len(x_tr)} training tiles, {len(x_te)} test tiles)")


if __name__ == "__main__":
    main()
