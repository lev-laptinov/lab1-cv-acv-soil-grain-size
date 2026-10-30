"""Training augmentations on GPU batches; all of them keep millimetres per pixel (grain size is the label)."""

import torch
import torch.nn.functional as F


def random_d4(x: torch.Tensor) -> torch.Tensor:
    k = torch.randint(0, 4, (len(x),), device=x.device)
    for r in range(1, 4):
        m = k == r
        if m.any():
            x[m] = torch.rot90(x[m], r, dims=(2, 3))
    flip = torch.rand(len(x), device=x.device) < 0.5
    x[flip] = torch.flip(x[flip], dims=(3,))
    return x


def colour_jitter(x: torch.Tensor, strength: float, white_balance: float) -> torch.Tensor:
    n = (len(x), 1, 1, 1)
    brightness = 1 + strength * (2 * torch.rand(n, device=x.device) - 1)
    contrast = 1 + strength * (2 * torch.rand(n, device=x.device) - 1)
    gain = 1 + white_balance * (2 * torch.rand(len(x), 3, 1, 1, device=x.device) - 1)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (((x - mean) * contrast + mean * brightness) * gain).clamp(0, 1)


def cutmix(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    size = x.shape[-1]
    side = int(size * torch.rand(1).sqrt())
    if side == 0:
        return x, y
    top, left = torch.randint(0, size - side + 1, (2,)).tolist()
    perm = torch.randperm(len(x), device=x.device)
    x = x.clone()
    x[:, :, top:top + side, left:left + side] = x[perm, :, top:top + side, left:left + side]
    lam = side * side / size**2
    return x, (1 - lam) * y + lam * y[perm]


def random_resized_crop(x: torch.Tensor, min_scale: float) -> torch.Tensor:
    size = x.shape[-1]
    out = torch.empty_like(x)
    for i in range(len(x)):
        side = int(size * (min_scale + (1 - min_scale) * torch.rand(1)).sqrt())
        top, left = torch.randint(0, size - side + 1, (2,)).tolist()
        crop = x[i:i + 1, :, top:top + side, left:left + side]
        out[i] = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)[0]
    return out


def train_augment(x: torch.Tensor, y: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    x = random_d4(x)
    x = colour_jitter(x, cfg.get("colour", 0.1), cfg.get("white_balance", 0.0))
    if cfg.get("rrc_min_scale"):
        x = random_resized_crop(x, cfg["rrc_min_scale"])
    if torch.rand(1).item() < cfg.get("cutmix_p", 0.0):
        x, y = cutmix(x, y)
    return x, y


def d4_views(x: torch.Tensor) -> list[torch.Tensor]:
    rots = [torch.rot90(x, r, dims=(2, 3)) for r in range(4)]
    return rots + [torch.flip(v, dims=(3,)) for v in rots]
