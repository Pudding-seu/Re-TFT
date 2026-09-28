"""Small, inspectable Re-TFT inference assembly for the revised paper.

Training, model selection and checkpoints are deliberately outside this
minimal public interface. Fit the Base first; copy its encoder into a fresh
residual module; optimize only the unfrozen residual parameters; fit the
affine coefficients on validation predictions only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from models.retft import CNN2TFTLSTMResidual, CNNFrontTFT
from utils.calibration import calib_affine


def frozen_residual_from_base(base: CNNFrontTFT, *, input_dim: int, calendar_dim: int = 8,
                              conv_out_each: int = 12, d_model: int = 64) -> CNN2TFTLSTMResidual:
    """Copy the trained Base CNN/LSTM weights and freeze the copied encoder."""
    residual = CNN2TFTLSTMResidual(
        d_enc_in=input_dim,
        d_known_in=calendar_dim,
        conv_out_each=conv_out_each,
        d_model=d_model,
        init_from_base_state_dict=base.state_dict(),
        freeze_backbone=True,
    )
    for module in (residual.convs, residual.enc_proj, residual.enc_lstm):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("Residual representation was not frozen")
    return residual


def fit_validation_affine(y_true: np.ndarray, y_base: np.ndarray,
                          y_residual: np.ndarray) -> dict[str, float]:
    """Fit the retained affine correction using validation targets only."""
    values = [np.asarray(x, dtype=float).reshape(-1) for x in (y_true, y_base, y_residual)]
    if len({len(x) for x in values}) != 1 or len(values[0]) < 2:
        raise ValueError("Equal-length validation vectors with >=2 rows are required")
    if not all(np.isfinite(x).all() for x in values):
        raise ValueError("Validation vectors must be finite")
    return calib_affine(*values, clip_a=(-3.0, 3.0))


@torch.no_grad()
def forecast_endpoint(base: CNNFrontTFT, residual: CNN2TFTLSTMResidual,
                      history: torch.Tensor, target_calendar: torch.Tensor,
                      affine: dict[str, float],
                      inverse_target: Callable[[torch.Tensor], torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Base and corrected scalar endpoints in original target units.

    The canonical Base emits a scaled target; its train-fitted scaler must be
    inverted before adding the residual and validation-fitted affine offset.
    """
    if history.ndim != 3 or history.shape[1] != 96:
        raise ValueError("history must be [batch, 96, encoder_features]")
    if target_calendar.ndim != 2 or target_calendar.shape != (history.shape[0], 8):
        raise ValueError("target_calendar must be [batch, 8]")
    if affine.get("mode") != "affine":
        raise ValueError("Validation-fitted affine calibration is required")
    base.eval()
    residual.eval()
    y_base = inverse_target(base(history, target_calendar))
    correction = residual(history, target_calendar)
    y_final = y_base + float(affine["a"]) * correction + float(affine["b"])
    return y_base, y_final
