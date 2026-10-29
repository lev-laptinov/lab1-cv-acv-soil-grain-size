import numpy as np
import pandas as pd
import torch

from soilgsd.finetune import CurveModel, Trainer, augment, emd_loss
from soilgsd.metric import emd, is_valid_curve


def test_emd_loss_is_the_competition_metric():
    rng = np.random.default_rng(0)
    y = np.sort(rng.uniform(0, 100, (4, 11)), axis=1)
    p = np.sort(rng.uniform(0, 100, (4, 11)), axis=1)
    y[:, -1] = p[:, -1] = 100
    assert np.isclose(emd_loss(torch.tensor(p), torch.tensor(y)).item(), emd(y, p))


def test_model_output_is_always_a_valid_curve():
    model = CurveModel("resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 3, 64, 64) * 5).numpy()
    assert is_valid_curve(out, atol=1e-3).all()


def test_augment_keeps_shape_range_and_content_scale():
    x = torch.rand(8, 3, 32, 32)
    a = augment(x.clone())
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    # Rotations and flips only permute pixels: the sorted values change by the colour jitter alone.
    assert torch.allclose(a.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=0.15)


def test_lpft_trains_head_first_then_the_whole_model():
    exp = {"backbone": "resnet18", "pretrained": False, "batch_size": 4, "epochs": 2, "head_only_epochs": 1,
           "opt": "adamw", "lr": 1e-3, "weight_decay": 0.0, "warmup_epochs": 0}
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 255, (8, 32, 32, 3), dtype=np.uint8)
    y = np.tile(np.linspace(5, 100, 11), (8, 1))
    trainer = Trainer(exp, torch.device("cpu"), seed=0)
    start = [p.detach().clone() for p in trainer.model.backbone.parameters()]
    log = trainer.fit(tiles, y, pd.Series(list("aabbccdd")))
    assert [row["epoch"] for row in log] == [0, 1]
    moved = [not torch.equal(a, b) for a, b in zip(start, trainer.model.backbone.parameters(), strict=True)]
    assert any(moved), "the backbone never trained after the linear-probe phase"
