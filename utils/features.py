# -*- coding: utf-8 -*-
"""
Feature engineering and dataset utilities.
Paper Section 2.3 — Data Preparation and Feature Engineering.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Cyclical time encoding  (Section 2.3.2)
# ─────────────────────────────────────────────────────────────────────────────
def _sincos(val, period):
    ang = 2 * np.pi * (val % period) / period
    return np.sin(ang), np.cos(ang)


def build_known_future_timefeats(ts, peak_hours=(8, 21)):
    """
    Build the 8-dim known-future calendar embedding for a single timestamp.

    Features: [sin_h, cos_h, sin_dow, cos_dow, sin_month, cos_month,
               is_weekend, is_peak_hour]
    """
    h_s, h_c = _sincos(ts.hour, 24)
    dw_s, dw_c = _sincos(ts.dayofweek, 7)
    m_s, m_c = _sincos(ts.month - 1, 12)
    is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0
    is_peak = 1.0 if peak_hours[0] <= ts.hour <= peak_hours[1] else 0.0
    return np.array([h_s, h_c, dw_s, dw_c, m_s, m_c, is_weekend, is_peak], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering  (Section 2.3.3)
# ─────────────────────────────────────────────────────────────────────────────
def apply_feature_engineering(df: pd.DataFrame, date_col: str):
    """
    Augment raw DataFrame with:
      - Lag features: target load and temperature at lags {1, 4, 96, 672}
      - Rolling mean/std: windows {4, 24, 96} for load and temperature
      - Cross features: temperature × sin/cos(hour)

    Scaler statistics must be fitted ONLY on the training partition to
    prevent look-ahead bias (data leakage).

    Returns
    -------
    df_fe : pd.DataFrame   feature-augmented DataFrame (NaN rows dropped)
    enc_feature_cols : list[str]  ordered list of encoder feature column names
    """
    df = df.copy()

    lag_plan = [
        ("load",        [1, 4, 96, 96 * 7]),
        ("temperature", [1, 4, 96, 96 * 7]),
        ("maxtem",      [1, 4, 96]),
        ("mintem",      [1, 4, 96]),
        ("humidity",    [1, 4, 96]),
        ("speed",       [1, 4, 96]),
    ]
    for col, ks in lag_plan:
        for k in ks:
            df[f"{col}_lag{k}"] = df[col].shift(k)

    roll_plan = [("load", [4, 24, 96]), ("temperature", [4, 24, 96])]
    for col, ws in roll_plan:
        for w in ws:
            df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=w).mean()
            df[f"{col}_roll{w}_std"]  = df[col].rolling(w, min_periods=w).std()

    hsin = df[date_col].dt.hour.apply(lambda h: _sincos(h, 24)[0])
    hcos = df[date_col].dt.hour.apply(lambda h: _sincos(h, 24)[1])
    df["temp_x_hsin"] = df["temperature"] * hsin
    df["temp_x_hcos"] = df["temperature"] * hcos

    df = df.dropna().reset_index(drop=True)
    enc_feature_cols = [c for c in df.columns if c != date_col]
    return df, enc_feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Sequence construction
# ─────────────────────────────────────────────────────────────────────────────
def create_sequences(features_scaled: np.ndarray, dates, seq_len: int,
                     tgt_idx: int, pred_step: int = 1):
    """
    Build sliding-window sequences for the encoder/decoder.

    Parameters
    ----------
    features_scaled : np.ndarray  [N, F]  MinMax-scaled feature matrix
    dates           : array-like  timestamp array aligned with features_scaled
    seq_len         : int         encoder lookback length (default 96 = 24h at 15-min)
    tgt_idx         : int         column index of the target variable (load)
    pred_step       : int         forecasting horizon in timestep units

    Returns
    -------
    enc_seqs    : [M, T, F]   encoder input sequences
    dec_known   : [M, 8]      known-future calendar features at prediction time
    labels      : [M]         target values (scaled)
    label_dates : [M]         ISO timestamp strings for each prediction step
    """
    enc_seqs, dec_known, labels, label_dates = [], [], [], []
    N = len(features_scaled)
    for i in range(N - seq_len - pred_step + 1):
        tgt_pos = i + seq_len + pred_step - 1
        enc = features_scaled[i: i + seq_len]
        y = features_scaled[tgt_pos, tgt_idx]
        ts = pd.to_datetime(dates[tgt_pos]) if dates is not None else None
        kf = build_known_future_timefeats(ts) if ts is not None else np.zeros(8, dtype=np.float32)
        label_dates.append(ts.strftime("%Y/%m/%d %H:%M") if ts is not None else str(tgt_pos))
        enc_seqs.append(enc)
        dec_known.append(kf)
        labels.append(y)
    return (
        np.array(enc_seqs,    dtype=np.float32),
        np.array(dec_known,   dtype=np.float32),
        np.array(labels,      dtype=np.float32),
        np.array(label_dates),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, enc_seqs, dec_known, labels):
        self.enc_seqs  = enc_seqs
        self.dec_known = dec_known
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.enc_seqs[idx], self.dec_known[idx], self.labels[idx]
