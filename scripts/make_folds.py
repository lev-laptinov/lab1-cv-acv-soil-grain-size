"""Write data/photo_index.csv and data/folds.csv. Every experiment reads these two files."""

import argparse

from soilgsd.config import load_config, resolve
from soilgsd.data import build_photo_index, load_labels
from soilgsd.folds import make_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    cfg = load_config(ap.parse_args().config)

    index = build_photo_index(cfg)
    index.to_csv(resolve(cfg["data"]["photo_index"]), index=False)

    labels = load_labels(cfg)
    folds = make_folds(labels, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["seed"],
                       cfg["cv"]["family_emd_threshold"])
    folds.to_csv(resolve(cfg["cv"]["folds_file"]), index=False)

    # Sanity checks: every sample is held out exactly once per repeat, and no photo of a
    # validation sample can appear in training because the split is on sample_id.
    per_repeat = folds.groupby("repeat")["sample_id"].agg(["nunique", "size"])
    assert (per_repeat["nunique"] == len(labels)).all() and (per_repeat["size"] == len(labels)).all()
    assert (folds.groupby(["repeat", "family"])["fold"].nunique() == 1).all(), "a family was split"

    fams = folds[folds["repeat"] == 0].groupby("family")["sample_id"].agg(list)
    print("families:", fams[fams.map(len) > 1].to_dict())

    print(index.groupby(["split", "device"]).agg(photos=("path", "size"),
                                                 samples=("sample_id", "nunique"),
                                                 ppm=("ppm", "first")))
    r0 = folds[folds["repeat"] == 0]
    print(r0.groupby("fold")["d50_class"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
