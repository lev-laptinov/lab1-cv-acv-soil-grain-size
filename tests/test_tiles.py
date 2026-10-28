import numpy as np
import pandas as pd
from PIL import Image

from soilgsd.data import TARGET_COLS
from soilgsd.tile_model import aggregate, sample_weights
from soilgsd.tiles import _starts, cut_tiles


def test_starts_cover_the_whole_length():
    for length, side in [(225, 225), (660, 225), (1000, 90), (300, 225), (711, 112.5)]:
        xs = _starts(length, side)
        assert xs[0] == 0 and xs[-1] + round(side) == length
        assert all(b - a <= side + 1 for a, b in zip(xs, xs[1:], strict=False))  # no gaps beyond rounding
    assert len(_starts(450, 112.5)) == 4
    assert _starts(100, 225) == []


def test_tiles_have_the_same_physical_size_on_any_camera():
    # The same 100 x 50 mm field photographed at 4.5 and at 13.5 px/mm gives the same tile grid.
    for ppm in (4.5, 13.5):
        im = Image.new("RGB", (round(100 * ppm), round(50 * ppm)))
        tiles = cut_tiles(im, ppm, (0, 0, im.width, im.height), tile_mm=25, work_ppm=4.5, out_px=32)
        assert tiles.shape == (4 * 2, 32, 32, 3)


def test_every_sample_weighs_the_same():
    ids = pd.Series(["a"] * 10 + ["b"] * 30)
    w = sample_weights(ids)
    assert np.isclose(w[:10].sum(), w[10:].sum()) and np.isclose(w.mean(), 1)


def test_aggregate_mean_and_median_per_photo_then_mean_over_photos():
    meta = pd.DataFrame({"sample_id": ["s"] * 4, "path": ["p1", "p1", "p1", "p2"]})
    base = np.linspace(0, 100, 11)
    pred = np.stack([base, base, base + 30, base])  # one outlier tile in photo p1
    mean = aggregate(pred, meta, "mean").loc["s"].to_numpy()
    median = aggregate(pred, meta, "median").loc["s"].to_numpy()
    np.testing.assert_allclose(median, base)
    assert (mean[:-1] > base[:-1]).all() and mean[-1] == 100
    assert list(aggregate(pred, meta, "mean").columns) == TARGET_COLS
