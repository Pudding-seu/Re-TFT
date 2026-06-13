# -*- coding: utf-8 -*-
"""Point and interval forecasting metrics."""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def point_metrics(y_true, y_pred) -> dict:
    """
    Compute MSE, RMSE, MAE, MAPE, R² on de-normalised load values.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse  = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_pred - y_true) / np.maximum(1e-6, np.abs(y_true)))) * 100.0)
    r2   = float(r2_score(y_true, y_pred))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def interval_metrics(y_true, y_lower, y_upper,
                     target_coverage: float = 0.80,
                     y_train_base=None) -> dict:
    """
    Compute probabilistic interval evaluation metrics.

    PICP  -- Prediction Interval Coverage Probability  (target = 80%)
    ACE   -- Average Coverage Error  = PICP - target  (ideal = 0)
    MPIW  -- Mean Prediction Interval Width
    PINAW -- MPIW normalised by training range
    CWC   -- Coverage-Width Criterion (penalises under-coverage)
    """
    y_true = np.asarray(y_true).reshape(-1)
    lower  = np.minimum(np.asarray(y_lower).reshape(-1), np.asarray(y_upper).reshape(-1))
    upper  = np.maximum(np.asarray(y_lower).reshape(-1), np.asarray(y_upper).reshape(-1))

    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mu       = float(target_coverage)
    mpiw     = float(np.mean(upper - lower))

    # Standard Khosravi CWC (matches paper Eq. 19):
    # CWC = MPIW * (1 + 1[PICP < mu] * exp(-eta * (PICP - mu))), eta = 50.
    # No penalty for over-coverage: when PICP >= mu, CWC = MPIW.
    if coverage < mu:
        cwc = float(mpiw * (1.0 + np.exp(-50.0 * (coverage - mu))))
    else:
        cwc = float(mpiw)

    if y_train_base is not None:
        data_range = float(np.max(y_train_base) - np.min(y_train_base))
    else:
        data_range = float(np.max(y_true) - np.min(y_true))
    pinaw = float(mpiw / (data_range + 1e-8))

    return {
        "PICP(%)": coverage * 100.0,
        "ACE(%)":  (coverage - mu) * 100.0,
        "MPIW":    mpiw,
        "PINAW(%)": pinaw * 100.0,
        "CWC":     cwc,
    }
