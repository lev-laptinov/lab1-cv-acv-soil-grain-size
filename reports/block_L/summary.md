# Block L: leave-one-device-out diagnostic (generated from outputs/L_*)

Each recipe is trained on the photos of one phone only and validated on the photos of each phone of the held-out
samples (same sample folds as every experiment, 3 repeats, EMD averaged over repeats per sample). The 21
samples with photos from both phones are compared. Shift penalty = EMD of the other-phone model minus EMD of the
own-phone model on the same validation photos, averaged over both directions; Wilcoxon over samples.

| recipe            | trained on Motorola: val Moto / val Samsung   | trained on Samsung: val Moto / val Samsung   |   shift penalty (other phone - own phone) |   p (penalty != 0) | penalty vs O7   |
|:------------------|:----------------------------------------------|:---------------------------------------------|------------------------------------------:|-------------------:|:----------------|
| O7_groupnorm      | 53.4 / 54.4                                   | 41.5 / 41.9                                  |                                      0.31 |               0.34 |                 |
| D1_gray           | 54.1 / 54.2                                   | 42.6 / 41.9                                  |                                      0.37 |               0.23 | +0.06 (p 0.52)  |
| D2_colour_strong  | 56.7 / 57.5                                   | 47.7 / 46.6                                  |                                      0.98 |               0.1  | +0.68 (p 0.10)  |
| D3_sharpness      | 56.7 / 56.9                                   | 44.5 / 43.7                                  |                                      0.49 |               0.36 | +0.18 (p 0.49)  |
| D4_gray_sharpness | 55.0 / 55.5                                   | 44.4 / 44.0                                  |                                      0.41 |               0.02 | +0.11 (p 0.76)  |

Per-run scores: reports/results.csv (rows L_*); logs: reports/block_L/logs/.
