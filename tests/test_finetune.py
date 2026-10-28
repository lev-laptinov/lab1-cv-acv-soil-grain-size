import numpy as np
import torch

from soilgsd.finetune import CurveModel, augment, emd_loss
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
