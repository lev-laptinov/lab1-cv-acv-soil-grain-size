#!/usr/bin/env bash
# Build build/kaggle_bundle/: the code, the folds and the cached tiles, uploaded as one private
# Kaggle dataset so that fine-tuning runs on a Kaggle GPU with exactly the local pipeline.
#   bash kaggle/bundle.sh && uvx kaggle datasets create -p build/kaggle_bundle --dir-mode zip   (first time)
#   bash kaggle/bundle.sh && uvx kaggle datasets version -p build/kaggle_bundle -m "update" --dir-mode zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/kaggle_bundle
rm -rf "$OUT" && mkdir -p "$OUT/repo/data" "$OUT/tiles"
cp -r pyproject.toml src scripts configs part2_vit "$OUT/repo/"
cp data/folds.csv data/photo_index.csv data/soil_boxes.csv "$OUT/repo/data/"
# Earlier runs, so that run_finetune.py --repeats can extend them on Kaggle
mkdir -p "$OUT/repo/reports" && cp reports/results.csv "$OUT/repo/reports/"
for f in outputs/*/oof.csv outputs/*/train_log.csv outputs/*/config.yaml; do
  mkdir -p "$OUT/repo/$(dirname "$f")" && cp "$f" "$OUT/repo/$f"
done
# Without a local tile cache the kernel cuts the tiles from the competition photos itself
cp data/cache/tiles/*.npz "$OUT/tiles/" 2>/dev/null || echo "no local tiles: the kernel will cut them"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "soilgsd-bundle", "id": "$USER/soilgsd-bundle", "licenses": [{"name": "other"}]}
JSON
du -sh "$OUT"
