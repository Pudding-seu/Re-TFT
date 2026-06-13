# -*- coding: utf-8 -*-
"""
Residual calibration and fusion utilities.

After the residual head predicts ê_res on the validation set, an affine
calibration  ŷ_final = ŷ_base + a·ê_res + b  is fitted via OLS to maximally
align the residual prediction with the true residual.  Three modes are
supported: scalar, affine (default), and per-hour.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Calibration fitters
# ─────────────────────────────────────────────────────────────────────────────
def calib_scalar(y_true, y_base, y_res, clip=(-3.0, 3.0)) -> dict:
    """Scalar: ŷ_final = ŷ_base + w·ê_res, fitted by OLS projection."""
    e = y_true - y_base
    num = float(np.dot(y_res, e))
    den = float(np.dot(y_res, y_res) + 1e-8)
    w = float(np.clip(num / den, *clip))
    return {"mode": "scalar", "w": w}


def calib_affine(y_true, y_base, y_res, clip_a=(-3.0, 3.0), clip_b=None) -> dict:
    """Affine: ŷ_final = ŷ_base + a·ê_res + b, fitted by OLS."""
    e = (y_true - y_base).reshape(-1, 1)
    X = np.column_stack([y_res.reshape(-1, 1), np.ones_like(e)])
    sol, _, _, _ = np.linalg.lstsq(X, e, rcond=None)
    a = float(np.clip(sol[0, 0], *clip_a))
    b = float(sol[1, 0])
    if clip_b is not None:
        b = float(np.clip(b, *clip_b))
    return {"mode": "affine", "a": a, "b": b}


def calib_per_hour(y_true, y_base, y_res, dates,
                   min_count: int = 24, fallback: str = "affine") -> dict:
    """Hour-of-day affine calibration (24 separate (a, b) pairs)."""
    hours = pd.to_datetime(pd.Series(dates)).dt.hour.values
    fb = calib_affine(y_true, y_base, y_res) if fallback == "affine" \
        else calib_scalar(y_true, y_base, y_res)
    params = []
    for h in range(24):
        idx = np.where(hours == h)[0]
        if len(idx) < min_count:
            params.append((fb.get("a", fb.get("w", 0.0)), fb.get("b", 0.0)))
        else:
            sub = calib_affine(y_true[idx], y_base[idx], y_res[idx])
            params.append((sub["a"], sub["b"]))
    return {"mode": "per_hour", "ab": params}


# ─────────────────────────────────────────────────────────────────────────────
# Apply calibrated fusion
# ─────────────────────────────────────────────────────────────────────────────
def apply_fusion(calib: dict, y_base, y_res, dates=None) -> np.ndarray:
    """Compute ŷ_final = ŷ_base + calibrated_correction(ê_res)."""
    mode = calib["mode"]
    if mode == "scalar":
        return y_base + calib["w"] * y_res
    elif mode == "affine":
        return y_base + calib["a"] * y_res + calib["b"]
    elif mode == "per_hour":
        assert dates is not None, "per_hour mode requires dates array"
        hours = pd.to_datetime(pd.Series(dates)).dt.hour.values
        y_out = np.empty_like(y_base)
        for i in range(len(y_out)):
            a, b = calib["ab"][int(hours[i])]
            y_out[i] = y_base[i] + a * y_res[i] + b
        return y_out
    raise ValueError(f"Unknown calibration mode: {mode}")
