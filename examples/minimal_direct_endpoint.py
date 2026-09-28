"""MINIMAL IMPLEMENTATION EXAMPLE: past 96 rows -> one H96 target.

Synthetic inputs only. Random weights and illustrative affine coefficients are
not a reproduction of any published metric or trained checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.pipeline import forecast_endpoint, frozen_residual_from_base
from models.retft import CNNFrontTFT
from utils.features import build_known_future_timefeats


def main() -> None:
    torch.manual_seed(7)
    timestamps = pd.date_range("2023-01-01", periods=192, freq="15min")
    origin_end_index = 95
    horizon_steps = 96
    target_index = origin_end_index + horizon_steps
    assert target_index == 191

    # The 40 encoder columns stand for historical load/weather and their
    # engineered lag/rolling/calendar features. No future weather is supplied.
    historical_features = np.zeros((96, 40), dtype=np.float32)
    historical_features[:, 0] = np.linspace(0.2, 0.8, 96)  # history-only load
    historical_features[:, 1:6] = 0.5                     # history-only weather
    history = torch.from_numpy(historical_features[None, :, :])
    target_calendar = torch.from_numpy(
        build_known_future_timefeats(timestamps[target_index])[None, :]
    )

    base = CNNFrontTFT(d_enc_in=40, d_known_in=8, conv_out_each=12)
    residual = frozen_residual_from_base(base, input_dim=40, conv_out_each=12)
    # In a trained run these coefficients must come from validation data only,
    # and inverse_target must use the train-fitted target scaler. The synthetic
    # demonstration has no scaling, hence the explicit identity transform.
    illustrative_affine = {"mode": "affine", "a": 1.0, "b": 0.0}
    y_base, y_final = forecast_endpoint(
        base, residual, history, target_calendar, illustrative_affine,
        inverse_target=lambda scaled: scaled,
    )
    assert y_base.shape == y_final.shape == (1,)
    print(f"history ends: {timestamps[origin_end_index]}")
    print(f"H96 scalar target: {timestamps[target_index]}")
    print(f"random-weight Base={y_base.item():.4f}; corrected={y_final.item():.4f}")


if __name__ == "__main__":
    main()
