import numpy as np
import pandas as pd
import torch

from soilgsd.data import TARGET_COLS
from soilgsd.metric import LOG_WIDTHS, d_value_errors, emd_parts
from soilgsd.tile_model import aggregate


def emd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The competition metric: |F - F^| weighted by the log10 width of each sieve interval."""
    w = torch.as_tensor(LOG_WIDTHS, dtype=pred.dtype, device=pred.device)
    return (torch.abs(pred[:, :-1] - target[:, :-1]) @ w).mean()


def squared_emd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Squared EMD (Hou et al. 2017), divided by 100 to be on the scale of emd_loss."""
    w = torch.as_tensor(LOG_WIDTHS, dtype=pred.dtype, device=pred.device)
    return ((pred[:, :-1] - target[:, :-1]) ** 2 @ w).mean() / 100


LOSSES = {"emd": emd_loss, "emd2": squared_emd_loss}


def sample_curves(tile_pred: np.ndarray, meta: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    return aggregate(tile_pred, meta, how)


def curve_scores(curves: pd.DataFrame, labels: pd.DataFrame) -> dict:
    y_true = labels.set_index("sample_id").loc[curves.index, TARGET_COLS].to_numpy(float)
    fine, coarse = emd_parts(y_true, curves[TARGET_COLS].to_numpy())
    return {"emd": float((fine + coarse).mean()), "emd_fine_part": float(fine.mean()),
            "emd_coarse_part": float(coarse.mean()),
            "D50_log_err": float(d_value_errors(y_true, curves[TARGET_COLS].to_numpy())["D50_log_err"])}
