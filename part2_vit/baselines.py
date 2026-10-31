"""Baselines that never look at a photo: CV scores on the shared folds and a submission each.

    python -m part2_vit.baselines

B0_train_mean    - the mean training curve for every sample
B1_train_median  - the per-sieve median training curve (the best constant on the training soils, since EMD is
                   a weighted L1 on the curve; on held-out soils it can be worse than the mean)
B2_random_curve  - the curve of a random training sample (seeded)
"""

import argparse

import numpy as np
import pandas as pd

from soilgsd.config import load_config, resolve
from soilgsd.data import TARGET_COLS, load_labels, load_photo_index
from soilgsd.metric import is_valid_curve
from soilgsd.results import save_result, summarise

RULES = {
    "B0_train_mean": lambda y, n, rng: np.tile(y.mean(0), (n, 1)),
    "B1_train_median": lambda y, n, rng: np.tile(np.median(y, 0), (n, 1)),
    "B2_random_curve": lambda y, n, rng: y[rng.integers(0, len(y), n)],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    base = load_config(args.config)
    labels = load_labels(base)
    y = labels.set_index("sample_id")[TARGET_COLS].astype(float)
    folds = pd.read_csv(resolve(base["cv"]["folds_file"]))
    index = load_photo_index(base)
    test_ids = index.loc[index["split"] == "test", "sample_id"].drop_duplicates().tolist()

    for name, rule in RULES.items():
        rng = np.random.default_rng(base["seed"])
        oof = []
        for (repeat, fold), part in folds.groupby(["repeat", "fold"]):
            train = y.drop(index=part["sample_id"]).to_numpy()
            pred = pd.DataFrame(rule(train, len(part), rng), columns=TARGET_COLS)
            oof.append(pred.assign(sample_id=part["sample_id"].to_numpy(), repeat=repeat, fold=fold))
        oof = pd.concat(oof, ignore_index=True)[["sample_id", "repeat", "fold", *TARGET_COLS]]
        out_dir = resolve(f"outputs/{name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        oof.to_csv(out_dir / "oof.csv", index=False)

        sub = pd.DataFrame(rule(y.to_numpy(), len(test_ids), rng), columns=TARGET_COLS)
        sub.insert(0, "sample_id", test_ids)
        assert is_valid_curve(sub[TARGET_COLS].to_numpy()).all()
        sub.to_csv(resolve(f"submissions/{name}.csv"), index=False)

        res = {"name": name, "approach": "baseline", "backbone": "none (no photo)", "train_tiles": 0}
        res.update(summarise(oof, labels, folds))
        res["compute"] = "cpu < 1 s"
        save_result(res)
        print(f"{name}: CV EMD {res['emd']:.1f} ± {res['emd_std_repeats']:.1f} (fine {res['emd_fine']:.1f}, "
              f"sand {res['emd_sand']:.1f}, gravel {res['emd_gravel']:.1f})")


if __name__ == "__main__":
    main()
