# Soil grain-size distribution from photos

KSE AI401 Advanced Computer Vision, assignment 1: the Kaggle competition
[Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos),
solved with CNNs, ViTs and self-supervised encoders on one shared validation.

## Quick start

```bash
uv sync                                   # pinned environment from uv.lock
uv run kaggle competitions download soil-grain-size-from-photos -p data/raw
(cd data/raw && unzip -q soil-grain-size-from-photos.zip && rm soil-grain-size-from-photos.zip)
uv run python scripts/make_folds.py       # data/photo_index.csv, data/folds.csv
uv run python scripts/eda.py              # reports/eda/
uv run pytest -q                          # metric tests
```

## Layout

| Path | What |
|---|---|
| `configs/base.yaml` | paths, CV settings, seed; experiment configs override it |
| `src/soilgsd/data.py` | labels, camera table, photo index with the effective px/mm of every photo |
| `src/soilgsd/metric.py` | competition EMD, curve repairs, D10/D50/D60 errors, scale-shift helper |
| `src/soilgsd/folds.py` | group-aware folds: a sample's photos and a curve family's samples share a fold |
| `src/soilgsd/tiles.py` | soil box per photo, resampling to a common px/mm, tiles of fixed size in mm |
| `src/soilgsd/features.py` | frozen pretrained backbone, one cached feature vector per tile |
| `src/soilgsd/tile_model.py` | ridge on tile features, tile -> photo -> sample aggregation, CV loop |
| `configs/exp/` | one YAML per experiment, run with `scripts/run_tiles.py` |
| `data/folds.csv` | the folds used by **every** experiment (6 folds x 3 repeats, grouped by family, stratified by D50 class) |
| `reports/eda/` | EDA figures and `summary.md` |
| `HYPOTHESES.md` | hypotheses log, committed before each block of experiments |
| `docs/research-notes.md` | competition rules, forum findings, decisions to make |

## Part 0 findings so far

- **Training photos are stored downscaled** (Motorola Edge 1600 px instead of 4000, Samsung A52
  1599 px instead of 9248), so their effective scale is ~4.6 px/mm, not the 11.5 / 26.3 listed in
  `ppm_updated.csv`. Test photos (iPhone 14 / 16) are full-size: 13.9 / 19.5 px/mm. With the
  correction, every device sees a ~290–350 mm field, as expected for the fixed 21 cm camera distance.
  A pipeline that ignores this sees test grains 3–4x larger, which alone costs ~48–60 EMD.
- The common working scale can be at most ~4.5 px/mm (one pixel ≈ 0.22 mm) without upsampling the
  training photos, so silt and clay (< 0.063 mm) are never resolved as grains.
- Frames contain non-soil regions (tray rims, the lamp strip on Samsung photos) that differ by device.
- Image-free baselines on our folds: uniform 102.0, train mean 89.5; an oracle that only knows the
  D50 class (fine / sand / gravel) reaches 29.7, so getting the soil class right is most of the score.

## Validation

- The unit is the physical sample: all photos of a sample are in one fold, predictions are averaged
  to one curve per sample before scoring.
- Several training samples are near-duplicates with consecutive ids (H366/H367/H371/H372 are within
  3.5–5.3 EMD of each other), most likely one site. The test soils come from other sites, so samples
  whose curves are all within 10 EMD (complete linkage) form a **family** that is never split
  between train and validation. This gives 16 groups for 24 samples; single, complete and average
  linkage give the same families for thresholds 10–12.
- Folds: `StratifiedGroupKFold`, 6 folds x 3 repeats, groups = family, strata = D50 class.
  Results are reported as mean ± std over the 18 fold scores / 3 repeats.

Details: [`reports/eda/summary.md`](reports/eda/summary.md).

## Tile pipeline

1. **Soil box** (`soilgsd.tiles.soil_boxes`, cached in `data/soil_boxes.csv`). Rows and columns whose
   median brightness is close to the frame centre are soil; rims, the dark side strip and the lamp
   are not. Each camera is fixed over the tray, so every photo gets the median box of its device and
   orientation (this overrides the few photos where a bright stone in the centre fools the detector,
   and the box cannot depend on the soil), minus a 5 mm margin.
2. **Common scale.** The box is resampled from the photo's own px/mm to the working scale
   (`work_ppm`, 4.5 by default: the coarsest training photos, so nothing is upsampled).
3. **Tiles** of `tile_mm` x `tile_mm` covering the whole box (the fewest that fit, spread evenly, so
   neighbours overlap slightly), each resized to the 224 px network input.
4. **Features**: a frozen pretrained backbone (`resnet50.a1_in1k` from timm), one 2048-d vector per
   tile, cached in `data/cache/features/`.
5. **Model**: ridge regression from standardised features to the 11-point curve; every tile carries its
   sample's curve and every sample weighs the same. Alpha is chosen by an inner group CV on the
   training fold only, scored with sample-level EMD.
6. **Aggregation**: tiles -> photo (mean or median per sieve) -> sample (mean over photos) -> running
   max, clipped to [0, 100], 100 at 200 mm.

```bash
uv run python scripts/run_tiles.py configs/exp/T1_base.yaml   # adds a row to reports/results.csv
```

Results of every experiment: [`reports/results.csv`](reports/results.csv); hypotheses and outcomes:
[`HYPOTHESES.md`](HYPOTHESES.md).
