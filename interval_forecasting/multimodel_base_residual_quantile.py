# -*- coding: utf-8 -*-
"""
Multi-model Base + Residual Quantile framework for 15-min electricity load forecasting.

核心设计：
1) 所有模型共享同一数据切分、scaler、seq_len、pred_step、known_future_timefeats。
2) Base 模型使用 MSE 做点预测。
3) Residual 模型使用 Pinball Loss 预测残差 q10/q90。
4) 最终区间：
      lower = base_pred + residual_q10
      upper = base_pred + residual_q90
5) Ours = CNNFrontTFT。
6) Baselines = LSTM / GRU / TCN / Transformer / SimpleTFT / NBEATS /
               DLinear / PatchTST / iTransformer / TimesNet。

说明：
- PatchTST / iTransformer / TimesNet 为轻量纯 PyTorch 版本，目标是公平接入同一
  Base+ResidualQuantile 训练框架，而不是官方仓库逐行复刻。
- 如果你要在论文中声称“官方 PatchTST / TimesNet”，建议替换为官方实现；
  如果写作中表述为“PatchTST-style / TimesNet-style baseline”，本脚本即可直接使用。
"""

import os
import time
import json
import math
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 通用工具
# ============================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 需要完全复现时可打开；但可能降低速度
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def _fmt_dur(sec: float) -> str:
    if sec is None:
        return "n/a"
    m, s = divmod(float(sec), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h}h {m:02d}m {s:04.1f}s"


def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def json_safe(obj):
    """便于保存 CONFIG。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_pred - y_true) / np.maximum(1e-6, np.abs(y_true)))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def interval_metrics(y_true, y_lower, y_upper, target_coverage=0.80):
    """
    区间指标：
    - PICP: Prediction Interval Coverage Probability
    - MPIW: Mean Prediction Interval Width
    - CWC : Coverage Width-based Criterion
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_lower = np.asarray(y_lower).reshape(-1)
    y_upper = np.asarray(y_upper).reshape(-1)

    lower = np.minimum(y_lower, y_upper)
    upper = np.maximum(y_lower, y_upper)

    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    width = float(np.mean(upper - lower))

    mu = float(target_coverage)
    gamma = 2.0
    eta = 50.0

    if coverage < mu:
        penalty = 1.0 + gamma
    else:
        penalty = 1.0 + gamma * np.exp(-eta * (coverage - mu))

    cwc = float(width * penalty)
    cross_rate = float(np.mean(y_lower > y_upper))

    return {
        "PICP": coverage,
        "PICP(%)": coverage * 100.0,
        "MPIW": width,
        "CWC": cwc,
        "CrossRate(%)": cross_rate * 100.0,
    }


# ============================================================
# Pinball Loss
# ============================================================
class PinballLoss(nn.Module):
    def __init__(self, tau_lower=0.10, tau_upper=0.90, crossing_penalty=0.0):
        super().__init__()
        self.tau_lower = float(tau_lower)
        self.tau_upper = float(tau_upper)
        self.crossing_penalty = float(crossing_penalty)

    @staticmethod
    def _pinball(pred_q, target, tau):
        diff = target - pred_q
        return torch.max(tau * diff, (tau - 1.0) * diff)

    def forward(self, pred, target):
        """
        pred:   [batch, 2]，第0列 q10_residual，第1列 q90_residual
        target: [batch]，真实残差
        """
        if pred.dim() != 2 or pred.size(-1) != 2:
            raise ValueError(f"PinballLoss 需要 pred.shape=[batch, 2]，但得到 {tuple(pred.shape)}")

        target = target.view(-1)
        lower = pred[:, 0]
        upper = pred[:, 1]

        loss_lower = self._pinball(lower, target, self.tau_lower)
        loss_upper = self._pinball(upper, target, self.tau_upper)
        loss = 0.5 * (loss_lower.mean() + loss_upper.mean())

        if self.crossing_penalty > 0:
            loss = loss + self.crossing_penalty * F.relu(lower - upper).mean()

        return loss


# ============================================================
# 已知未来时间特征
# ============================================================
def _sincos(val, period):
    ang = 2 * np.pi * (val % period) / period
    return np.sin(ang), np.cos(ang)


def build_known_future_timefeats(ts, use_is_peak=True, peak_hours=(8, 21)):
    h_s, h_c = _sincos(ts.hour, 24)
    dw_s, dw_c = _sincos(ts.dayofweek, 7)
    m_s, m_c = _sincos(ts.month - 1, 12)

    is_weekend = 1.0 if ts.dayofweek >= 5 else 0.0
    is_peak = 1.0 if (peak_hours[0] <= ts.hour <= peak_hours[1]) else 0.0
    if not use_is_peak:
        is_peak = 0.0

    feats = [h_s, h_c, dw_s, dw_c, m_s, m_c, is_weekend, is_peak]
    return np.array(feats, dtype=np.float32)


# ============================================================
# Dataset
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, enc_seqs, dec_known, labels):
        self.enc_seqs = torch.as_tensor(enc_seqs, dtype=torch.float32)
        self.dec_known = torch.as_tensor(dec_known, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.enc_seqs[idx], self.dec_known[idx], self.labels[idx]


# ============================================================
# 特征工程
# ============================================================
def apply_feature_engineering(df, date_col, target_col="load"):
    df = df.copy()

    lag_plan = [
        ("load", [1, 4, 96, 96 * 7]),
        ("temperature", [1, 4, 96, 96 * 7]),
        ("maxtem", [1, 4, 96]),
        ("mintem", [1, 4, 96]),
        ("humidity", [1, 4, 96]),
        ("speed", [1, 4, 96]),
    ]

    for col, ks in lag_plan:
        if col not in df.columns:
            continue
        for k in ks:
            df[f"{col}_lag{k}"] = df[col].shift(k)

    roll_plan = [
        ("load", [4, 24, 96]),
        ("temperature", [4, 24, 96]),
    ]

    for col, ws in roll_plan:
        if col not in df.columns:
            continue
        for w in ws:
            df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=w).mean()
            df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=w).std()

    if "temperature" in df.columns:
        hsin = df[date_col].dt.hour.apply(lambda h: _sincos(h, 24)[0])
        hcos = df[date_col].dt.hour.apply(lambda h: _sincos(h, 24)[1])
        df["temp_x_hsin"] = df["temperature"] * hsin
        df["temp_x_hcos"] = df["temperature"] * hcos

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    enc_feature_cols = [c for c in df.columns if c != date_col]
    if target_col not in enc_feature_cols:
        raise ValueError(f"target_col={target_col} 不在特征列中，请检查数据列名。")

    return df, enc_feature_cols


# ============================================================
# 序列构造
# ============================================================
def create_sequences(features_scaled, dates, seq_len, tgt_idx, pred_step=1):
    enc_seqs, dec_known, labels, label_dates = [], [], [], []
    N = len(features_scaled)

    for i in range(N - seq_len - pred_step + 1):
        enc = features_scaled[i: i + seq_len]
        tgt_pos = i + seq_len + pred_step - 1
        y = features_scaled[tgt_pos, tgt_idx]

        ts = pd.to_datetime(dates[tgt_pos]) if dates is not None else None
        if ts is not None:
            kf = build_known_future_timefeats(ts)
            label_dates.append(ts.strftime("%Y/%m/%d %H:%M"))
        else:
            kf = np.zeros(8, dtype=np.float32)
            label_dates.append(str(tgt_pos))

        enc_seqs.append(enc)
        dec_known.append(kf)
        labels.append(y)

    return (
        np.array(enc_seqs, dtype=np.float32),
        np.array(dec_known, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(label_dates),
    )


# ============================================================
# 模型基础模块
# ============================================================
class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.GLU()
        self.skip = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.net(x)
        h = self.gate(torch.cat([h, h], dim=-1))
        s = self.skip(x)
        return self.norm(h + s)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ============================================================
# Ours: CNNFrontTFT
# ============================================================
class CNNFrontTFT(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        conv_out_each=32,
        conv_kernels=(3, 5, 7),
        conv_dilations=(1, 2, 4),
        d_model=64,
        n_heads=4,
        lstm_layers=1,
        dropout=0.2,
        act="relu",
        **kwargs,
    ):
        super().__init__()
        assert len(conv_kernels) == len(conv_dilations), "kernels 与 dilations 长度需一致"
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.feature_dim = d_model
        self.convs = nn.ModuleList()

        for k, d in zip(conv_kernels, conv_dilations):
            assert k % 2 == 1, "为保持长度不变，请使用奇数卷积核"
            pad = (k - 1) * d // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels=d_enc_in,
                    out_channels=conv_out_each,
                    kernel_size=k,
                    dilation=d,
                    padding=pad,
                )
            )

        self.act = nn.ReLU() if str(act).lower() == "relu" else nn.ELU()
        self.cnn_dropout = nn.Dropout(dropout)

        total_c = d_enc_in + len(self.convs) * conv_out_each
        self.enc_proj = nn.Linear(total_c, d_model)

        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dec_known_proj = nn.Linear(d_known_in, d_model)
        self.ctx_grn = GRN(input_dim=2 * d_model, hidden_dim=d_model, dropout=dropout)

        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fuse_grn = GRN(input_dim=2 * d_model, hidden_dim=d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x0 = enc_seq
        x_conv_in = x0.transpose(1, 2)
        conv_feats = [self.act(conv(x_conv_in)) for conv in self.convs]

        c = torch.cat(conv_feats, dim=1)
        c = self.cnn_dropout(c).transpose(1, 2)

        x = torch.cat([x0, c], dim=-1)
        x = F.elu(self.enc_proj(x))

        enc_out, (h_n, _) = self.enc_lstm(x)
        h_last = h_n[-1]

        kf = F.elu(self.dec_known_proj(dec_known))
        ctx = self.ctx_grn(torch.cat([h_last, kf], dim=-1))

        q = ctx.unsqueeze(1)
        attn_out, _ = self.mha(q, enc_out, enc_out)

        z = self.fuse_grn(torch.cat([ctx, attn_out.squeeze(1)], dim=-1))
        return z

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: LSTM / GRU
# ============================================================
class SimpleLSTM(nn.Module):
    def __init__(self, d_enc_in, d_known_in, d_model=64, lstm_layers=2, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = d_model
        self.encoder = nn.LSTM(
            d_enc_in,
            d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        _, (h_n, _) = self.encoder(enc_seq)
        h = h_n[-1]
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


class SimpleGRU(nn.Module):
    def __init__(self, d_enc_in, d_known_in, d_model=64, gru_layers=2, dropout=0.2, **kwargs):
        super().__init__()
        self.feature_dim = d_model
        self.encoder = nn.GRU(
            d_enc_in,
            d_model,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        _, h_n = self.encoder(enc_seq)
        h = h_n[-1]
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: TCN
# ============================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = F.relu(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = F.relu(y)
        y = self.drop(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.norm(F.relu(y + res))


class SimpleTCN(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        d_model=64,
        tcn_layers=4,
        kernel_size=3,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = d_model

        layers = []
        in_ch = d_enc_in
        for i in range(tcn_layers):
            layers.append(
                TemporalBlock(
                    in_ch=in_ch,
                    out_ch=d_model,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            in_ch = d_model

        self.tcn = nn.Sequential(*layers)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x = enc_seq.transpose(1, 2)
        h_seq = self.tcn(x)
        h = h_seq[:, :, -1]
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: Transformer Encoder
# ============================================================
class SimpleTransformer(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        d_model=64,
        n_heads=4,
        e_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        seq_len=96,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_dim = d_model
        self.input_proj = nn.Linear(d_enc_in, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(4096, seq_len + 16), dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x = self.input_proj(enc_seq)
        x = self.pos(x)
        h_seq = self.encoder(x)
        h = h_seq[:, -1, :]
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: Simple TFT without CNN front-end
# ============================================================
class SimpleTFT(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        d_model=64,
        n_heads=4,
        lstm_layers=1,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_dim = d_model
        self.enc_proj = nn.Linear(d_enc_in, d_model)
        self.enc_lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dec_known_proj = nn.Linear(d_known_in, d_model)
        self.ctx_grn = GRN(2 * d_model, d_model, dropout=dropout)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.fuse_grn = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x = F.elu(self.enc_proj(enc_seq))
        enc_out, (h_n, _) = self.enc_lstm(x)
        h_last = h_n[-1]
        kf = F.elu(self.dec_known_proj(dec_known))
        ctx = self.ctx_grn(torch.cat([h_last, kf], dim=-1))
        attn_out, _ = self.mha(ctx.unsqueeze(1), enc_out, enc_out)
        z = self.fuse_grn(torch.cat([ctx, attn_out.squeeze(1)], dim=-1))
        return z

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: N-BEATS style MLP
# ============================================================
class SimpleNBEATS(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        seq_len=96,
        d_model=64,
        n_blocks=3,
        hidden=256,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = d_model
        in_dim = seq_len * d_enc_in
        blocks = []
        cur = in_dim
        for _ in range(n_blocks):
            blocks.extend(
                [
                    nn.Linear(cur, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            cur = hidden
        self.mlp = nn.Sequential(*blocks)
        self.to_feat = nn.Linear(hidden, d_model)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x = enc_seq.reshape(enc_seq.size(0), -1)
        h = F.elu(self.to_feat(self.mlp(x)))
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: DLinear style
# ============================================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: [B, L, D]
        if self.kernel_size <= 1:
            return x
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_pad = x_pad.transpose(1, 2)
        y = self.avg(x_pad).transpose(1, 2)
        return y


class SimpleDLinear(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        seq_len=96,
        d_model=64,
        moving_avg=25,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = d_model
        self.seq_len = seq_len
        self.decomp = MovingAvg(moving_avg)
        self.linear_seasonal = nn.Linear(seq_len, d_model)
        self.linear_trend = nn.Linear(seq_len, d_model)
        self.var_proj = nn.Linear(d_enc_in, 1)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        trend = self.decomp(enc_seq)
        seasonal = enc_seq - trend

        # [B, L, D] -> [B, D, L] -> [B, D, d_model]
        s = self.linear_seasonal(seasonal.transpose(1, 2))
        t = self.linear_trend(trend.transpose(1, 2))
        h_var = s + t  # [B, D, d_model]

        # 变量维度加权聚合
        weights = torch.softmax(self.var_proj(enc_seq.mean(dim=1)).unsqueeze(-1), dim=1)  # [B,1,1]
        if h_var.size(1) > 1:
            h = h_var.mean(dim=1)
        else:
            h = h_var.squeeze(1)

        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: PatchTST style
# ============================================================
class SimplePatchTST(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        seq_len=96,
        d_model=64,
        n_heads=4,
        e_layers=2,
        patch_len=16,
        stride=8,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_dim = d_model
        self.seq_len = seq_len
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        self.patch_proj = nn.Linear(self.patch_len * d_enc_in, d_model)
        max_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(4096, max_patches + 16), dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def _make_patches(self, enc_seq):
        # enc_seq: [B, L, D]
        B, L, D = enc_seq.shape
        if L < self.patch_len:
            pad_len = self.patch_len - L
            pad = enc_seq[:, -1:, :].repeat(1, pad_len, 1)
            enc_seq = torch.cat([enc_seq, pad], dim=1)
            L = enc_seq.size(1)

        patches = enc_seq.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold 后: [B, P, D, patch_len]
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(B, patches.size(1), self.patch_len * D)
        return patches

    def get_feature(self, enc_seq, dec_known):
        patches = self._make_patches(enc_seq)
        x = self.patch_proj(patches)
        x = self.pos(x)
        h_seq = self.encoder(x)
        h = h_seq.mean(dim=1)
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: iTransformer style
# ============================================================
class SimpleiTransformer(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        seq_len=96,
        d_model=64,
        n_heads=4,
        e_layers=2,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_dim = d_model
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.var_embedding = nn.Parameter(torch.randn(1, d_enc_in, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        # [B, L, D] -> [B, D, L]，变量作为 token
        x = enc_seq.transpose(1, 2)
        x = self.value_embedding(x) + self.var_embedding[:, : x.size(1), :]
        h_tokens = self.encoder(x)
        h = h_tokens.mean(dim=1)
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# Baseline: TimesNet style
# ============================================================
class InceptionTimeBlock(nn.Module):
    def __init__(self, d_model, kernels=(3, 5, 7), dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.convs.append(nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad))
        self.proj = nn.Conv1d(d_model * len(kernels), d_model, kernel_size=1)
        self.norm = nn.BatchNorm1d(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D, L]
        ys = [F.gelu(conv(x)) for conv in self.convs]
        y = torch.cat(ys, dim=1)
        y = self.proj(y)
        y = self.drop(y)
        return self.norm(F.gelu(y + x))


class SimpleTimesNet(nn.Module):
    def __init__(
        self,
        d_enc_in,
        d_known_in,
        seq_len=96,
        d_model=64,
        e_layers=3,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = d_model
        self.input_proj = nn.Linear(d_enc_in, d_model)
        self.blocks = nn.ModuleList([InceptionTimeBlock(d_model, dropout=dropout) for _ in range(e_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.known_proj = nn.Linear(d_known_in, d_model)
        self.fuse = GRN(2 * d_model, d_model, dropout=dropout)
        self.fc_out = nn.Linear(d_model, 1)

    def get_feature(self, enc_seq, dec_known):
        x = self.input_proj(enc_seq).transpose(1, 2)  # [B, d_model, L]
        for block in self.blocks:
            x = block(x)
        h = self.pool(x).squeeze(-1)
        kf = F.elu(self.known_proj(dec_known))
        return self.fuse(torch.cat([h, kf], dim=-1))

    def forward(self, enc_seq, dec_known):
        z = self.get_feature(enc_seq, dec_known)
        return self.fc_out(z).squeeze(-1)


# ============================================================
# 通用 Residual Quantile Wrapper
# ============================================================
class ResidualQuantileWrapper(nn.Module):
    """
    通用残差分位数模型：
    - 内部复制一个 Base 模型作为 backbone；
    - 加载 Base 的 state_dict；
    - 使用 backbone.get_feature(enc_seq, dec_known) 抽取特征；
    - 新增 quantile_head 输出 [q10_residual, q90_residual]。
    """
    def __init__(
        self,
        backbone,
        base_state_dict=None,
        freeze_backbone=True,
        dropout=0.2,
        output_dim=2,
    ):
        super().__init__()
        self.backbone = backbone

        if base_state_dict is not None:
            missing, unexpected = self.backbone.load_state_dict(base_state_dict, strict=False)
            print(
                f"[ResidualWrapper] Base 权重加载完成。"
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        d = int(getattr(self.backbone, "feature_dim"))
        self.quantile_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d, output_dim),
        )

    def forward(self, enc_seq, dec_known):
        z = self.backbone.get_feature(enc_seq, dec_known)
        return self.quantile_head(z)


# ============================================================
# Model Factory
# ============================================================
class ModelFactory:
    SUPPORTED = [
        "CNNFrontTFT",
        "LSTM",
        "GRU",
        "TCN",
        "Transformer",
        "TFT",
        "NBEATS",
        "DLinear",
        "PatchTST",
        "iTransformer",
        "TimesNet",
    ]

    @staticmethod
    def create_base_model(model_name: str, d_enc_in: int, d_known_in: int, config: dict):
        name = str(model_name)
        d_model = int(config.get("baseline_d_model", config.get("tft_d_model", 64)))
        dropout = float(config.get("baseline_dropout", config.get("tft_dropout", 0.2)))
        seq_len = int(config.get("seq_length", 96))
        n_heads = int(config.get("baseline_n_heads", config.get("tft_n_heads", 4)))

        if name == "CNNFrontTFT":
            return CNNFrontTFT(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                conv_out_each=config.get("conv_out_each", 32),
                conv_kernels=tuple(config.get("conv_kernels", [3, 5, 7])),
                conv_dilations=tuple(config.get("conv_dilations", [1, 2, 4])),
                d_model=config.get("tft_d_model", d_model),
                n_heads=config.get("tft_n_heads", n_heads),
                lstm_layers=config.get("tft_lstm_layers", 1),
                dropout=config.get("tft_dropout", dropout),
                act=config.get("conv_act", "relu"),
            )

        if name == "LSTM":
            return SimpleLSTM(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                d_model=d_model,
                lstm_layers=config.get("lstm_layers", 2),
                dropout=dropout,
            )

        if name == "GRU":
            return SimpleGRU(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                d_model=d_model,
                gru_layers=config.get("gru_layers", 2),
                dropout=dropout,
            )

        if name == "TCN":
            return SimpleTCN(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                d_model=d_model,
                tcn_layers=config.get("tcn_layers", 4),
                kernel_size=config.get("tcn_kernel_size", 3),
                dropout=dropout,
            )

        if name == "Transformer":
            return SimpleTransformer(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=config.get("transformer_layers", 2),
                dim_feedforward=config.get("transformer_ff", 4 * d_model),
                dropout=dropout,
                seq_len=seq_len,
            )

        if name == "TFT":
            return SimpleTFT(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                d_model=d_model,
                n_heads=n_heads,
                lstm_layers=config.get("tft_lstm_layers", 1),
                dropout=dropout,
            )

        if name == "NBEATS":
            return SimpleNBEATS(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                seq_len=seq_len,
                d_model=d_model,
                n_blocks=config.get("nbeats_blocks", 3),
                hidden=config.get("nbeats_hidden", 256),
                dropout=dropout,
            )

        if name == "DLinear":
            return SimpleDLinear(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                seq_len=seq_len,
                d_model=d_model,
                moving_avg=config.get("dlinear_moving_avg", 25),
                dropout=dropout,
            )

        if name == "PatchTST":
            return SimplePatchTST(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=config.get("patchtst_layers", 2),
                patch_len=config.get("patch_len", 16),
                stride=config.get("patch_stride", 8),
                dropout=dropout,
            )

        if name == "iTransformer":
            return SimpleiTransformer(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                seq_len=seq_len,
                d_model=d_model,
                n_heads=n_heads,
                e_layers=config.get("itransformer_layers", 2),
                dropout=dropout,
            )

        if name == "TimesNet":
            return SimpleTimesNet(
                d_enc_in=d_enc_in,
                d_known_in=d_known_in,
                seq_len=seq_len,
                d_model=d_model,
                e_layers=config.get("timesnet_layers", 3),
                dropout=dropout,
            )

        raise ValueError(f"未知模型: {model_name}. 支持: {ModelFactory.SUPPORTED}")

    @staticmethod
    def create_residual_model(
        model_name: str,
        d_enc_in: int,
        d_known_in: int,
        config: dict,
        base_state_dict=None,
        freeze_backbone=True,
    ):
        backbone = ModelFactory.create_base_model(model_name, d_enc_in, d_known_in, config)
        return ResidualQuantileWrapper(
            backbone=backbone,
            base_state_dict=base_state_dict,
            freeze_backbone=freeze_backbone,
            dropout=config.get("residual_head_dropout", config.get("baseline_dropout", config.get("tft_dropout", 0.2))),
            output_dim=2,
        )


# ============================================================
# 可视化
# ============================================================
def plot_losses(train_losses, val_losses, save_path, title="训练/验证损失", ylabel="Loss"):
    plt.figure(figsize=(9, 5))
    plt.plot(train_losses, label="训练")
    plt.plot(val_losses, label="验证")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_predictions_all(actual, predicted, dates, title, save_path):
    actual = np.asarray(actual).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)

    plt.figure(figsize=(14, 6))
    x_numeric = np.arange(len(actual))

    plt.plot(x_numeric, actual, label="实际")
    plt.plot(x_numeric, predicted, label="预测")

    plt.xlabel("时间")
    plt.ylabel("负荷")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if dates is not None and len(dates) == len(actual) and len(actual) > 0:
        step = max(1, len(actual) // 60)
        plt.xticks(ticks=x_numeric[::step], labels=np.asarray(dates)[::step], rotation=45)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_interval_predictions(actual, base_pred, lower, upper, dates, title, save_path):
    actual = np.asarray(actual).reshape(-1)
    base_pred = np.asarray(base_pred).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)

    low = np.minimum(lower, upper)
    up = np.maximum(lower, upper)

    plt.figure(figsize=(14, 6))
    x_numeric = np.arange(len(actual))

    plt.plot(x_numeric, actual, label="实际")
    plt.plot(x_numeric, base_pred, label="Base点预测")
    plt.fill_between(x_numeric, low, up, alpha=0.25, label="80%预测区间(q10-q90)")

    if dates is not None and len(dates) == len(actual) and len(actual) > 0:
        step = max(1, len(actual) // 60)
        plt.xticks(ticks=x_numeric[::step], labels=np.asarray(dates)[::step], rotation=45)

    plt.xlabel("时间")
    plt.ylabel("负荷")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_parity_scatter(y_true, y_pred, save_path, title="测试集：真实值 vs 预测值（散点）"):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5, edgecolors="none")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="y = x")

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    plt.title(title)
    plt.xlabel("真实值")
    plt.ylabel("预测值")
    plt.grid(True, linestyle=":")
    plt.legend(loc="upper left")

    txt = f"R² = {r2:.4f}\nRMSE = {rmse:.2f}"
    plt.gca().text(
        0.02,
        0.98,
        txt,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()


# ============================================================
# 保存与训练
# ============================================================
def safe_torch_save(state_dict, path: Path):
    path = Path(path)
    os.makedirs(str(path.parent), exist_ok=True)
    try:
        torch.save(state_dict, str(path), _use_new_zipfile_serialization=False)
        return
    except Exception as e:
        print(f"[SAVE WARN] zip-serialize失败({e})，尝试文件句柄保存 → {path}")

    try:
        with open(str(path), "wb") as f:
            torch.save(state_dict, f, _use_new_zipfile_serialization=False)
    except Exception as e2:
        print(f"[SAVE ERROR] 句柄保存仍失败({e2})，最终回退为最朴素方式")
        torch.save(state_dict, str(path))


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_path,
    scheduler=None,
    early_stop_patience=10,
    early_stop_min_delta=0.0,
    grad_clip_norm=None,
    tag="",
):
    best = float("inf")
    wait = 0
    tr_hist = []
    va_hist = []
    non_block = device.type == "cuda"
    t0 = time.time()

    for ep in range(num_epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        model.train()
        tr = 0.0

        for xs, kf, ys in train_loader:
            xs = xs.to(device=device, dtype=torch.float32, non_blocking=non_block)
            kf = kf.to(device=device, dtype=torch.float32, non_blocking=non_block)
            ys = ys.to(device=device, dtype=torch.float32, non_blocking=non_block)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xs, kf)
            loss = criterion(pred, ys)
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip_norm)

            optimizer.step()
            tr += loss.item() * xs.size(0)

        tr /= max(1, len(train_loader.dataset))
        tr_hist.append(tr)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xs, kf, ys in val_loader:
                xs = xs.to(device=device, dtype=torch.float32, non_blocking=non_block)
                kf = kf.to(device=device, dtype=torch.float32, non_blocking=non_block)
                ys = ys.to(device=device, dtype=torch.float32, non_blocking=non_block)
                va_loss = criterion(model(xs, kf), ys)
                va += va_loss.item() * xs.size(0)

        va /= max(1, len(val_loader.dataset))
        va_hist.append(va)

        print(f"{tag}Epoch {ep + 1}/{num_epochs} | LR {cur_lr:.5g} | 训练 {tr:.6f} | 验证 {va:.6f}")

        if va < best - early_stop_min_delta:
            best = va
            wait = 0
            if save_path:
                safe_torch_save(model.state_dict(), Path(save_path))
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(f"{tag}早停：{wait} 个 epoch 未提升（best_val={best:.6f}）")
                break

        if scheduler is not None:
            scheduler.step()

    t1 = time.time()
    return tr_hist, va_hist, t1 - t0, best


@torch.no_grad()
def predict_model(model, loader, device):
    model.eval()
    preds = []
    trues = []
    non_block = device.type == "cuda"

    for xs, kf, ys in loader:
        xs = xs.to(device=device, dtype=torch.float32, non_blocking=non_block)
        kf = kf.to(device=device, dtype=torch.float32, non_blocking=non_block)
        ys = ys.to(device=device, dtype=torch.float32, non_blocking=non_block)

        yhat = model(xs, kf)
        preds.append(yhat.detach().cpu().numpy())
        trues.append(ys.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    if trues.ndim == 2 and trues.shape[1] == 1:
        trues = trues[:, 0]

    return preds, trues


def build_scheduler(optimizer, total_epochs, warmup_epochs=5, sched_type="cosine", min_lr=0.0):
    warmup_epochs = int(warmup_epochs)
    total_epochs = int(total_epochs)

    if warmup_epochs > 0:
        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda ep: min(1.0, (ep + 1) / max(1, warmup_epochs)),
        )
    else:
        warmup = None

    rest_epochs = max(1, total_epochs - max(0, warmup_epochs))

    if sched_type == "cosine":
        tail = CosineAnnealingLR(optimizer, T_max=rest_epochs, eta_min=min_lr)
    else:
        start_lr = optimizer.param_groups[0]["lr"]
        min_factor = (min_lr / start_lr) if start_lr > 0 else 0.0

        def lin_lambda(ep):
            if rest_epochs <= 1:
                return min_factor
            frac = ep / (rest_epochs - 1)
            return (1 - frac) + frac * min_factor

        tail = LambdaLR(optimizer, lr_lambda=lin_lambda)

    if warmup is None:
        return tail

    return SequentialLR(optimizer, schedulers=[warmup, tail], milestones=[warmup_epochs])


def lr_warmup_search(
    model_ctor,
    criterion_ctor,
    dloaders,
    device,
    pilot_epochs,
    grid_lrs,
    grid_warmups,
    grid_types,
    weight_decay,
    grad_clip_norm,
    early_stop_patience,
    early_stop_min_delta,
    tag="",
):
    train_loader, val_loader = dloaders
    best_cfg = None
    best_val = float("inf")
    best_hist = None

    print(
        f"{tag}== LR+Warmup联合调参: "
        f"lr={grid_lrs}, warmup={grid_warmups}, 调度={grid_types}, pilot_epochs={pilot_epochs} =="
    )

    for lr in grid_lrs:
        for w in grid_warmups:
            for t in grid_types:
                model = model_ctor().to(device)
                criterion = criterion_ctor()

                trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
                if len(trainable_params) == 0:
                    raise RuntimeError("模型没有可训练参数，请检查 freeze 设置。")

                optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

                scheduler = build_scheduler(
                    optimizer,
                    total_epochs=pilot_epochs,
                    warmup_epochs=w,
                    sched_type=t,
                    min_lr=0.0,
                )

                tr, va, _, best = train_model(
                    model,
                    train_loader,
                    val_loader,
                    criterion,
                    optimizer,
                    pilot_epochs,
                    device,
                    save_path=None,
                    scheduler=scheduler,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    grad_clip_norm=grad_clip_norm,
                    tag=tag,
                )

                print(f"{tag}候选(lr={lr}, warmup={w}, type={t}) → 最佳验证 {best:.6f}")

                if best < best_val:
                    best_val = best
                    best_cfg = (lr, w, t)
                    best_hist = (tr, va)

    print(f"{tag}选择 lr={best_cfg[0]}, warmup={best_cfg[1]}, 调度={best_cfg[2]} | 最优验证={best_val:.6f}")
    return best_cfg, best_val, best_hist


# ============================================================
# 统一数据准备
# ============================================================
def prepare_data(CONFIG):
    df = pd.read_csv(CONFIG["data_path"])

    date_col = CONFIG["date_col"]
    target_col = CONFIG["target_col"]

    if date_col not in df.columns:
        raise ValueError(f"CSV需含日期列 {date_col}")
    for c in CONFIG["feature_cols"]:
        if c not in df.columns:
            raise ValueError(f"CSV缺少 feature_cols 中的列: {c}")

    df[date_col] = pd.to_datetime(
        df[date_col],
        format=CONFIG.get("date_format", None),
        errors="coerce",
    )
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)

    base_cols = CONFIG["feature_cols"]
    df[base_cols] = (
        df[base_cols]
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    df_fe, enc_feature_cols = apply_feature_engineering(df, date_col, target_col=target_col)

    N = len(df_fe)
    train_end = int(N * CONFIG.get("train_ratio", 0.60))
    val_end = int(N * (CONFIG.get("train_ratio", 0.60) + CONFIG.get("val_ratio", 0.20)))
    test_start = val_end

    print(f"行级切分: train=[0,{train_end})  val=[{train_end},{val_end})  test=[{test_start},{N})")

    X_all = df_fe[enc_feature_cols].values.astype(np.float32)
    tgt_idx = enc_feature_cols.index(target_col)

    scaler = MinMaxScaler(feature_range=CONFIG["scaler_range"])
    scaler.fit(X_all[:train_end])
    X_scaled = scaler.transform(X_all).astype(np.float32)

    seq_len = int(CONFIG["seq_length"])
    pred_step = int(CONFIG.get("pred_step", 1))

    enc_seqs, dec_known, labels, label_dates = create_sequences(
        X_scaled,
        df_fe[date_col].values,
        seq_len,
        tgt_idx,
        pred_step=pred_step,
    )

    max_seq = len(labels)
    train_seq_end = max(0, train_end - seq_len - pred_step + 1)
    val_seq_end = max(0, val_end - seq_len - pred_step + 1)
    train_seq_end = min(train_seq_end, max_seq)
    val_seq_end = min(val_seq_end, max_seq)

    X_tr = enc_seqs[:train_seq_end]
    KF_tr = dec_known[:train_seq_end]
    y_tr = labels[:train_seq_end]

    X_val = enc_seqs[train_seq_end:val_seq_end]
    KF_val = dec_known[train_seq_end:val_seq_end]
    y_val = labels[train_seq_end:val_seq_end]

    X_te = enc_seqs[val_seq_end:]
    KF_te = dec_known[val_seq_end:]
    y_te = labels[val_seq_end:]

    d_tr = label_dates[:train_seq_end]
    d_val = label_dates[train_seq_end:val_seq_end]
    d_te = label_dates[val_seq_end:]

    if min(len(X_tr), len(X_val), len(X_te)) <= 0:
        raise ValueError(
            f"序列数量不足: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}。"
            "请检查数据长度、seq_length、pred_step。"
        )

    print(f"序列数量 (pred_step={pred_step}): train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    return {
        "df_fe": df_fe,
        "enc_feature_cols": enc_feature_cols,
        "X_all": X_all,
        "tgt_idx": tgt_idx,
        "scaler": scaler,
        "splits": {
            "train_rows": train_end,
            "val_rows": val_end - train_end,
            "test_rows": N - val_end,
            "seq_len": seq_len,
        },
        "train": (X_tr, KF_tr, y_tr, d_tr),
        "val": (X_val, KF_val, y_val, d_val),
        "test": (X_te, KF_te, y_te, d_te),
    }


# ============================================================
# 单模型单步长运行
# ============================================================
def run_single_model(CONFIG, model_name="CNNFrontTFT", prepared=None):
    set_seed(CONFIG["seed"])

    pred_step = int(CONFIG.get("pred_step", 1))
    print(f"\n>>> 模型 = {model_name} | pred_step = {pred_step}\n")

    tau_lower = float(CONFIG.get("interval_tau_lower", 0.10))
    tau_upper = float(CONFIG.get("interval_tau_upper", 0.90))
    target_coverage = float(CONFIG.get("interval_target_coverage", tau_upper - tau_lower))

    save_root = Path(CONFIG["results_dir"]).expanduser().resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    print(f"保存目录: {save_root}")

    with open(save_root / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2, default=json_safe)

    device = torch.device("cuda" if (CONFIG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")

    if prepared is None:
        prepared = prepare_data(CONFIG)

    X_all = prepared["X_all"]
    tgt_idx = prepared["tgt_idx"]
    scaler = prepared["scaler"]

    X_tr, KF_tr, y_tr, d_tr = prepared["train"]
    X_val, KF_val, y_val, d_val = prepared["val"]
    X_te, KF_te, y_te, d_te = prepared["test"]

    pin = device.type == "cuda"
    bs = int(CONFIG["batch_size"])

    train_loader = DataLoader(SeqDataset(X_tr, KF_tr, y_tr), batch_size=bs, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(SeqDataset(X_val, KF_val, y_val), batch_size=bs, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(SeqDataset(X_te, KF_te, y_te), batch_size=bs, shuffle=False, pin_memory=pin)

    train_loader_eval = DataLoader(SeqDataset(X_tr, KF_tr, y_tr), batch_size=bs, shuffle=False, pin_memory=pin)
    val_loader_eval = DataLoader(SeqDataset(X_val, KF_val, y_val), batch_size=bs, shuffle=False, pin_memory=pin)
    test_loader_eval = DataLoader(SeqDataset(X_te, KF_te, y_te), batch_size=bs, shuffle=False, pin_memory=pin)

    d_enc_in = X_tr.shape[2]
    d_known_in = KF_tr.shape[1]

    def inv_only_target(arr_scaled):
        arr_scaled = np.asarray(arr_scaled).reshape(-1)
        dummy = np.zeros((len(arr_scaled), X_all.shape[1]), dtype=np.float32)
        dummy[:, tgt_idx] = arr_scaled
        return scaler.inverse_transform(dummy)[:, tgt_idx]

    def base_model_ctor():
        return ModelFactory.create_base_model(model_name, d_enc_in, d_known_in, CONFIG)

    # ========================================================
    # Base LR pilot
    # ========================================================
    model_ckpt = save_root / f"{model_name}_base_best.pth"
    base_loss_png = save_root / f"{model_name}_base_loss.png"

    _t0_base_pilot = time.time()
    if CONFIG.get("force_base_sched", None) is None:
        pilot_cfg, _, _ = lr_warmup_search(
            model_ctor=base_model_ctor,
            criterion_ctor=lambda: nn.MSELoss(),
            dloaders=(train_loader, val_loader),
            device=device,
            pilot_epochs=CONFIG["pilot_epochs"],
            grid_lrs=CONFIG.get("lr_grid", [CONFIG["lr"]]),
            grid_warmups=CONFIG["warmup_grid"],
            grid_types=CONFIG["sched_type_grid"],
            weight_decay=CONFIG["weight_decay"],
            grad_clip_norm=CONFIG["grad_clip"],
            early_stop_patience=CONFIG["early_stop_patience"],
            early_stop_min_delta=CONFIG["early_stop_min_delta"],
            tag=f"[Pilot Base {model_name}] ",
        )
        best_lr, best_warmup, best_sched_type = pilot_cfg
    else:
        forced = CONFIG["force_base_sched"]
        if len(forced) == 2:
            best_lr = CONFIG["lr"]
            best_warmup = int(forced[0])
            best_sched_type = str(forced[1])
        else:
            best_lr = float(forced[0])
            best_warmup = int(forced[1])
            best_sched_type = str(forced[2])
        print(f"[Pilot Base {model_name}] 强制使用 lr={best_lr}, warmup={best_warmup}, sched={best_sched_type}")

    _base_pilot_sec = time.time() - _t0_base_pilot

    # ========================================================
    # Base 正式训练
    # ========================================================
    base_model = base_model_ctor().to(device)
    optimizer = optim.Adam(base_model.parameters(), lr=best_lr, weight_decay=CONFIG["weight_decay"])
    scheduler = build_scheduler(
        optimizer,
        total_epochs=CONFIG["epochs"],
        warmup_epochs=best_warmup,
        sched_type=best_sched_type,
        min_lr=CONFIG["min_lr"],
    )

    tr_hist, va_hist, base_train_sec, best_val = train_model(
        base_model,
        train_loader,
        val_loader,
        nn.MSELoss(),
        optimizer,
        CONFIG["epochs"],
        device,
        save_path=model_ckpt,
        scheduler=scheduler,
        early_stop_patience=CONFIG["early_stop_patience"],
        early_stop_min_delta=CONFIG["early_stop_min_delta"],
        grad_clip_norm=CONFIG["grad_clip"],
        tag=f"[Base {model_name}] ",
    )

    plot_losses(tr_hist, va_hist, base_loss_png, f"{model_name} Base 训练/验证损失 (step={pred_step})", ylabel="MSE")

    base_model.load_state_dict(torch.load(str(model_ckpt), map_location=device))

    # ========================================================
    # Base 预测与指标
    # ========================================================
    t0 = time.time()
    tr_pred_s, tr_true_s = predict_model(base_model, train_loader_eval, device)
    base_infer_train_sec = time.time() - t0

    t0 = time.time()
    va_pred_s, va_true_s = predict_model(base_model, val_loader_eval, device)
    base_infer_val_sec = time.time() - t0

    t0 = time.time()
    te_pred_s, te_true_s = predict_model(base_model, test_loader_eval, device)
    base_infer_test_sec = time.time() - t0

    y_tr_true = inv_only_target(tr_true_s)
    y_tr_pred = inv_only_target(tr_pred_s)
    y_val_true_b = inv_only_target(va_true_s)
    y_val_pred_b = inv_only_target(va_pred_s)
    y_te_true_b = inv_only_target(te_true_s)
    y_te_pred_b = inv_only_target(te_pred_s)

    met_base_train = _metrics(y_tr_true, y_tr_pred)
    met_base_val = _metrics(y_val_true_b, y_val_pred_b)
    met_base_test = _metrics(y_te_true_b, y_te_pred_b)

    print(f"\n=== 测试集指标 [Base {model_name}] ===")
    print(
        " ".join(
            [
                f"{k}={v:.6f}" if k != "MAPE" else f"{k}={v:.2f}%"
                for k, v in met_base_test.items()
            ]
        )
    )

    pd.DataFrame(
        {
            "时间": pd.Series(d_te).astype(str),
            "真实": y_te_true_b,
            "预测_base": y_te_pred_b,
        }
    ).to_csv(save_root / "test_predictions_base.csv", index=False, encoding="utf-8")

    plot_predictions_all(
        y_te_true_b,
        y_te_pred_b,
        dates=d_te,
        title=f"{model_name}: 测试集实际 vs Base点预测 (step={pred_step})",
        save_path=save_root / "test_pred_vs_true_base.png",
    )

    plot_parity_scatter(
        y_te_true_b,
        y_te_pred_b,
        save_path=save_root / "test_parity_scatter_base.png",
        title=f"{model_name} Base 点预测 (step={pred_step})：真实 vs 预测",
    )

    with open(save_root / "metrics_base.txt", "w", encoding="utf-8") as f:
        f.write(f"model={model_name}\n")
        f.write(f"pred_step={pred_step}\n")
        f.write(f"best lr={best_lr}, best warmup={best_warmup}, best sched={best_sched_type}\n")
        f.write(f"best_val_mse_scaled={best_val:.8f}\n")
        f.write("\n== Base 指标（反归一化后） ==\n")
        for split_name, met in [("Train", met_base_train), ("Val", met_base_val), ("Test", met_base_test)]:
            f.write(
                f"[{split_name}] "
                + ", ".join(
                    [
                        f"{k}={v:.6f}" if k != "MAPE" else f"{k}={v:.2f}%"
                        for k, v in met.items()
                    ]
                )
                + "\n"
            )
        f.write("\n== 用时 ==\n")
        f.write(f"Pilot: {_fmt_dur(_base_pilot_sec)}\n")
        f.write(f"Train: {_fmt_dur(base_train_sec)}\n")
        f.write(
            f"Infer(train/val/test): "
            f"{_fmt_dur(base_infer_train_sec)} / "
            f"{_fmt_dur(base_infer_val_sec)} / "
            f"{_fmt_dur(base_infer_test_sec)}\n"
        )

    # ========================================================
    # Residual 标签
    # ========================================================
    label_space = CONFIG.get("residual_label_space", "original").lower()

    if label_space == "original":
        res_tr_lbl = (y_tr_true - y_tr_pred).astype(np.float32)
        res_val_lbl = (y_val_true_b - y_val_pred_b).astype(np.float32)
        res_te_lbl = (y_te_true_b - y_te_pred_b).astype(np.float32)

        def to_original_residual(arr):
            return np.asarray(arr, dtype=np.float32)

    elif label_space == "scaled":
        res_tr_lbl = (tr_true_s - tr_pred_s).astype(np.float32)
        res_val_lbl = (va_true_s - va_pred_s).astype(np.float32)
        res_te_lbl = (te_true_s - te_pred_s).astype(np.float32)

        def to_original_residual(arr):
            factor = 1.0 / float(scaler.scale_[tgt_idx])
            return np.asarray(arr, dtype=np.float32) * factor

    else:
        raise ValueError("residual_label_space 仅支持 'original' 或 'scaled'。")

    res_train_loader = DataLoader(SeqDataset(X_tr, KF_tr, res_tr_lbl), batch_size=bs, shuffle=True, pin_memory=pin)
    res_val_loader = DataLoader(SeqDataset(X_val, KF_val, res_val_lbl), batch_size=bs, shuffle=False, pin_memory=pin)
    res_test_loader = DataLoader(SeqDataset(X_te, KF_te, res_te_lbl), batch_size=bs, shuffle=False, pin_memory=pin)

    base_sd_cpu = torch.load(str(model_ckpt), map_location="cpu")

    def residual_model_ctor():
        return ModelFactory.create_residual_model(
            model_name=model_name,
            d_enc_in=d_enc_in,
            d_known_in=d_known_in,
            config=CONFIG,
            base_state_dict=base_sd_cpu,
            freeze_backbone=CONFIG.get("res_backbone_freeze", True),
        )

    # ========================================================
    # Residual Quantile pilot
    # ========================================================
    res_ckpt = save_root / f"{model_name}_residual_quantile_best.pth"
    res_loss_png = save_root / f"{model_name}_residual_quantile_loss.png"

    _t0_res_pilot = time.time()
    if CONFIG.get("force_res_sched", None) is None:
        (res_lr, res_warmup, res_sched), _, _ = lr_warmup_search(
            model_ctor=residual_model_ctor,
            criterion_ctor=lambda: PinballLoss(
                tau_lower=tau_lower,
                tau_upper=tau_upper,
                crossing_penalty=CONFIG.get("crossing_penalty", 0.0),
            ),
            dloaders=(res_train_loader, res_val_loader),
            device=device,
            pilot_epochs=CONFIG.get("res_pilot_epochs", 10),
            grid_lrs=CONFIG.get("res_lr_grid", [CONFIG.get("res_lr", 5e-4)]),
            grid_warmups=CONFIG.get("res_warmup_grid", [0, 3, 5]),
            grid_types=CONFIG.get("res_sched_type_grid", ["cosine", "linear"]),
            weight_decay=CONFIG.get("res_weight_decay", 0.0),
            grad_clip_norm=CONFIG.get("res_grad_clip", 1.0),
            early_stop_patience=CONFIG["res_early_stop_patience"],
            early_stop_min_delta=CONFIG["res_early_stop_min_delta"],
            tag=f"[Pilot Residual {model_name}] ",
        )
    else:
        forced = CONFIG["force_res_sched"]
        if len(forced) == 2:
            res_lr = CONFIG.get("res_lr", 5e-4)
            res_warmup = int(forced[0])
            res_sched = str(forced[1])
        else:
            res_lr = float(forced[0])
            res_warmup = int(forced[1])
            res_sched = str(forced[2])
        print(f"[Pilot Residual {model_name}] 强制使用 lr={res_lr}, warmup={res_warmup}, sched={res_sched}")

    _res_pilot_sec = time.time() - _t0_res_pilot

    # ========================================================
    # Residual Quantile 正式训练
    # ========================================================
    residual_model = residual_model_ctor().to(device)
    res_criterion = PinballLoss(
        tau_lower=tau_lower,
        tau_upper=tau_upper,
        crossing_penalty=CONFIG.get("crossing_penalty", 0.0),
    )
    trainable_params = list(filter(lambda p: p.requires_grad, residual_model.parameters()))
    res_optimizer = optim.Adam(trainable_params, lr=res_lr, weight_decay=CONFIG.get("res_weight_decay", 0.0))
    res_scheduler = build_scheduler(
        res_optimizer,
        total_epochs=CONFIG.get("res_epochs", 30),
        warmup_epochs=res_warmup,
        sched_type=res_sched,
        min_lr=CONFIG.get("res_min_lr", 0.0),
    )

    tr_hist_r, va_hist_r, res_train_sec, best_res_val = train_model(
        residual_model,
        res_train_loader,
        res_val_loader,
        res_criterion,
        res_optimizer,
        CONFIG.get("res_epochs", 30),
        device,
        save_path=res_ckpt,
        scheduler=res_scheduler,
        early_stop_patience=CONFIG["res_early_stop_patience"],
        early_stop_min_delta=CONFIG["res_early_stop_min_delta"],
        grad_clip_norm=CONFIG.get("res_grad_clip", 1.0),
        tag=f"[Residual Quantile {model_name}] ",
    )

    plot_losses(
        tr_hist_r,
        va_hist_r,
        res_loss_png,
        f"{model_name} Residual Quantile 训练/验证损失 (step={pred_step})",
        ylabel="Pinball Loss",
    )

    # ========================================================
    # Residual Quantile 预测
    # ========================================================
    residual_model.load_state_dict(torch.load(str(res_ckpt), map_location=device))

    t0 = time.time()
    val_pred_res_q_s, _ = predict_model(residual_model, res_val_loader, device)
    res_infer_val_sec = time.time() - t0

    t0 = time.time()
    te_pred_res_q_s, _ = predict_model(residual_model, res_test_loader, device)
    res_infer_test_sec = time.time() - t0

    y_val_res_q = to_original_residual(val_pred_res_q_s)
    y_te_res_q = to_original_residual(te_pred_res_q_s)

    if y_val_res_q.ndim != 2 or y_val_res_q.shape[1] != 2:
        raise ValueError(f"残差分位数验证集输出形状异常：{y_val_res_q.shape}")
    if y_te_res_q.ndim != 2 or y_te_res_q.shape[1] != 2:
        raise ValueError(f"残差分位数测试集输出形状异常：{y_te_res_q.shape}")

    y_val_lower_raw = y_val_pred_b + y_val_res_q[:, 0]
    y_val_upper_raw = y_val_pred_b + y_val_res_q[:, 1]

    y_te_lower_raw = y_te_pred_b + y_te_res_q[:, 0]
    y_te_upper_raw = y_te_pred_b + y_te_res_q[:, 1]

    y_val_lower = np.minimum(y_val_lower_raw, y_val_upper_raw)
    y_val_upper = np.maximum(y_val_lower_raw, y_val_upper_raw)
    y_te_lower = np.minimum(y_te_lower_raw, y_te_upper_raw)
    y_te_upper = np.maximum(y_te_lower_raw, y_te_upper_raw)

    met_interval_val = interval_metrics(
        y_val_true_b,
        y_val_lower_raw,
        y_val_upper_raw,
        target_coverage=target_coverage,
    )
    met_interval_test = interval_metrics(
        y_te_true_b,
        y_te_lower_raw,
        y_te_upper_raw,
        target_coverage=target_coverage,
    )

    print(f"\n=== 区间指标 [Base {model_name} + Residual q10/q90] ===")
    print(f"tau_lower={tau_lower:.2f}, tau_upper={tau_upper:.2f}, target_coverage={target_coverage:.2f}")
    print(
        "[Val Interval]  "
        + ", ".join(
            [
                f"{k}={v:.6f}" if "%" not in k else f"{k}={v:.2f}%"
                for k, v in met_interval_val.items()
            ]
        )
    )
    print(
        "[Test Interval] "
        + ", ".join(
            [
                f"{k}={v:.6f}" if "%" not in k else f"{k}={v:.2f}%"
                for k, v in met_interval_test.items()
            ]
        )
    )

    plot_interval_predictions(
        actual=y_te_true_b,
        base_pred=y_te_pred_b,
        lower=y_te_lower,
        upper=y_te_upper,
        dates=d_te,
        title=f"{model_name}: Base点预测 + 残差分位数80%区间(q10-q90, step={pred_step})",
        save_path=save_root / "test_interval_base_plus_residual_q10_q90.png",
    )

    hit = ((y_te_true_b >= y_te_lower) & (y_te_true_b <= y_te_upper)).astype(int)

    pd.DataFrame(
        {
            "时间": pd.Series(d_te).astype(str),
            "真实": y_te_true_b,
            "预测_base": y_te_pred_b,
            "残差_q10": y_te_res_q[:, 0],
            "残差_q90": y_te_res_q[:, 1],
            "区间下界_q10": y_te_lower,
            "区间上界_q90": y_te_upper,
            "区间宽度": y_te_upper - y_te_lower,
            "是否命中80区间": hit,
        }
    ).to_csv(save_root / "test_predictions_interval_q10_q90.csv", index=False, encoding="utf-8")

    with open(save_root / "metrics_interval_q10_q90.txt", "w", encoding="utf-8") as f:
        f.write(f"model={model_name}\n")
        f.write(f"pred_step={pred_step}\n")
        f.write(f"tau_lower={tau_lower}, tau_upper={tau_upper}, target_coverage={target_coverage}\n")
        f.write(f"residual_label_space={label_space}\n")
        f.write(f"res_backbone_freeze={CONFIG.get('res_backbone_freeze', True)}\n")
        f.write(f"best residual lr={res_lr}, warmup={res_warmup}, sched={res_sched}\n")
        f.write(f"best_res_val_pinball={best_res_val:.8f}\n")

        f.write("\n== Base 点预测 Test 指标（反归一化） ==\n")
        f.write(
            ", ".join(
                [
                    f"{k}={v:.6f}" if k != "MAPE" else f"{k}={v:.2f}%"
                    for k, v in met_base_test.items()
                ]
            )
            + "\n"
        )

        f.write("\n== Val 区间指标 ==\n")
        f.write(
            ", ".join(
                [
                    f"{k}={v:.6f}" if "%" not in k else f"{k}={v:.2f}%"
                    for k, v in met_interval_val.items()
                ]
            )
            + "\n"
        )

        f.write("\n== Test 区间指标 ==\n")
        f.write(
            ", ".join(
                [
                    f"{k}={v:.6f}" if "%" not in k else f"{k}={v:.2f}%"
                    for k, v in met_interval_test.items()
                ]
            )
            + "\n"
        )

        f.write("\n== 用时 ==\n")
        f.write(f"Base Pilot: {_fmt_dur(_base_pilot_sec)}\n")
        f.write(f"Base Train: {_fmt_dur(base_train_sec)}\n")
        f.write(f"Base Infer(train/val/test): {_fmt_dur(base_infer_train_sec)} / {_fmt_dur(base_infer_val_sec)} / {_fmt_dur(base_infer_test_sec)}\n")
        f.write(f"Residual Pilot: {_fmt_dur(_res_pilot_sec)}\n")
        f.write(f"Residual Train: {_fmt_dur(res_train_sec)}\n")
        f.write(f"Residual Infer(val/test): {_fmt_dur(res_infer_val_sec)} / {_fmt_dur(res_infer_test_sec)}\n")

    # ========================================================
    # 单目录总览 CSV
    # ========================================================
    overview_rows = [
        {
            "model_name": model_name,
            "pred_step": pred_step,
            "model": "Base",
            "split": "train",
            **met_base_train,
            "PICP": np.nan,
            "PICP(%)": np.nan,
            "MPIW": np.nan,
            "CWC": np.nan,
            "CrossRate(%)": np.nan,
            "time_pilot": _fmt_dur(_base_pilot_sec),
            "time_train": _fmt_dur(base_train_sec),
            "time_infer": _fmt_dur(base_infer_train_sec),
            "best_lr": best_lr,
            "best_warmup": best_warmup,
            "best_sched": best_sched_type,
        },
        {
            "model_name": model_name,
            "pred_step": pred_step,
            "model": "Base",
            "split": "val",
            **met_base_val,
            "PICP": np.nan,
            "PICP(%)": np.nan,
            "MPIW": np.nan,
            "CWC": np.nan,
            "CrossRate(%)": np.nan,
            "time_pilot": _fmt_dur(_base_pilot_sec),
            "time_train": _fmt_dur(base_train_sec),
            "time_infer": _fmt_dur(base_infer_val_sec),
            "best_lr": best_lr,
            "best_warmup": best_warmup,
            "best_sched": best_sched_type,
        },
        {
            "model_name": model_name,
            "pred_step": pred_step,
            "model": "Base",
            "split": "test",
            **met_base_test,
            "PICP": np.nan,
            "PICP(%)": np.nan,
            "MPIW": np.nan,
            "CWC": np.nan,
            "CrossRate(%)": np.nan,
            "time_pilot": _fmt_dur(_base_pilot_sec),
            "time_train": _fmt_dur(base_train_sec),
            "time_infer": _fmt_dur(base_infer_test_sec),
            "best_lr": best_lr,
            "best_warmup": best_warmup,
            "best_sched": best_sched_type,
        },
        {
            "model_name": model_name,
            "pred_step": pred_step,
            "model": "Base+ResidualQuantile(q10-q90)",
            "split": "val_interval",
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            **met_interval_val,
            "time_pilot": _fmt_dur(_res_pilot_sec),
            "time_train": _fmt_dur(res_train_sec),
            "time_infer": _fmt_dur(res_infer_val_sec),
            "best_lr": res_lr,
            "best_warmup": res_warmup,
            "best_sched": res_sched,
        },
        {
            "model_name": model_name,
            "pred_step": pred_step,
            "model": "Base+ResidualQuantile(q10-q90)",
            "split": "test_interval",
            "MSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            **met_interval_test,
            "time_pilot": _fmt_dur(_res_pilot_sec),
            "time_train": _fmt_dur(res_train_sec),
            "time_infer": _fmt_dur(res_infer_test_sec),
            "best_lr": res_lr,
            "best_warmup": res_warmup,
            "best_sched": res_sched,
        },
    ]

    overview_df = pd.DataFrame(overview_rows)
    metric_cols = [
        "model_name",
        "pred_step",
        "model",
        "split",
        "MSE",
        "RMSE",
        "MAE",
        "MAPE",
        "R2",
        "PICP",
        "PICP(%)",
        "MPIW",
        "CWC",
        "CrossRate(%)",
        "time_pilot",
        "time_train",
        "time_infer",
        "best_lr",
        "best_warmup",
        "best_sched",
    ]
    overview_df = overview_df[metric_cols]
    overview_df.to_csv(save_root / "metrics_overview.csv", index=False, encoding="utf-8")

    print(f"指标总览 CSV：{save_root / 'metrics_overview.csv'}")
    return overview_df


# ============================================================
# 汇总表输出
# ============================================================
def write_summary_tables(df_all, out_prefix="Q105090_all_models"):
    df_all = df_all.copy()
    df_all.to_csv(f"{out_prefix}_metrics_overview.csv", index=False, encoding="utf-8")

    df_base_test = df_all[(df_all["model"] == "Base") & (df_all["split"] == "test")].copy()
    df_base_test = df_base_test[
        ["model_name", "pred_step", "MSE", "RMSE", "MAE", "MAPE", "R2", "time_train", "time_infer"]
    ].sort_values(["pred_step", "model_name"])
    df_base_test.to_csv(f"{out_prefix}_base_point_test.csv", index=False, encoding="utf-8")

    tex_base = df_base_test.to_latex(
        index=False,
        float_format="%.3f",
        caption="不同模型与预测步长下 Base 点预测模型在测试集上的性能对比",
        label="tab:all_models_multi_step_base_point_performance",
    )
    with open(f"{out_prefix}_base_point_test.tex", "w", encoding="utf-8") as f:
        f.write(tex_base)

    df_interval_test = df_all[
        (df_all["model"] == "Base+ResidualQuantile(q10-q90)") & (df_all["split"] == "test_interval")
    ].copy()
    df_interval_test = df_interval_test[
        ["model_name", "pred_step", "PICP", "PICP(%)", "MPIW", "CWC", "CrossRate(%)", "time_train", "time_infer"]
    ].sort_values(["pred_step", "model_name"])
    df_interval_test.to_csv(f"{out_prefix}_interval_test.csv", index=False, encoding="utf-8")

    tex_interval = df_interval_test.to_latex(
        index=False,
        float_format="%.3f",
        caption="不同模型与预测步长下 Base 点预测加残差分位数区间模型在测试集上的区间预测性能对比",
        label="tab:all_models_multi_step_interval_performance",
    )
    with open(f"{out_prefix}_interval_test.tex", "w", encoding="utf-8") as f:
        f.write(tex_interval)

    # 论文常用透视表
    for metric in ["RMSE", "MAE", "MAPE", "R2"]:
        pivot = df_base_test.pivot(index="model_name", columns="pred_step", values=metric)
        pivot.to_csv(f"{out_prefix}_pivot_base_{metric}.csv", encoding="utf-8")

    for metric in ["PICP(%)", "MPIW", "CWC"]:
        pivot = df_interval_test.pivot(index="model_name", columns="pred_step", values=metric)
        safe_metric = metric.replace("(%)", "pct")
        pivot.to_csv(f"{out_prefix}_pivot_interval_{safe_metric}.csv", encoding="utf-8")

    print(f"汇总表已输出，前缀: {out_prefix}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    FREEZE_BACKBONE_MODE = True

    BASE_CONFIG = {
        # 数据
        "data_path": "belgium.csv",
        "date_col": "date",
        "date_format": "%Y/%m/%d %H:%M",
        "feature_cols": ["humidity", "speed", "mintem", "temperature", "maxtem", "load"],
        "target_col": "load",

        # 行级切分
        "train_ratio": 0.60,
        "val_ratio": 0.20,

        # 窗口/缩放
        "seq_length": 96,
        "scaler_range": (0, 1),

        # 设备/随机
        "seed": 42,
        "use_gpu": True,

        # DataLoader
        "batch_size": 32,

        # Ours: CNNFrontTFT
        "conv_out_each": 12,
        "conv_kernels": [3, 5, 7],
        "conv_dilations": [1, 2, 4],
        "conv_act": "relu",

        "tft_d_model": 64,
        "tft_n_heads": 4,
        "tft_lstm_layers": 1,
        "tft_dropout": 0.2,

        # Baseline 公共超参
        "baseline_d_model": 64,
        "baseline_n_heads": 4,
        "baseline_dropout": 0.2,

        # 可按需调 baseline 细节
        "lstm_layers": 2,
        "gru_layers": 2,
        "tcn_layers": 4,
        "tcn_kernel_size": 3,
        "transformer_layers": 2,
        "transformer_ff": 256,
        "nbeats_blocks": 3,
        "nbeats_hidden": 256,
        "dlinear_moving_avg": 25,
        "patch_len": 16,
        "patch_stride": 8,
        "patchtst_layers": 2,
        "itransformer_layers": 2,
        "timesnet_layers": 3,

        # Base 优化
        "lr": 1e-3,
        "lr_grid": [1e-4, 5e-4, 1e-3],
        "min_lr": 0.0,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "epochs": 50,

        # Base warmup pilot
        "pilot_epochs": 12,
        "warmup_grid": [0, 3, 5, 10],
        "sched_type_grid": ["cosine", "linear"],

        # Base 训练控制
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.0,

        # 可选：强制 Base 调度，跳过 pilot
        # 格式：[lr, warmup, "sched_type"] 或 [warmup, "sched_type"]
        "force_base_sched": None,

        # 输出基目录，run 时会覆盖为各模型各步长目录
        "results_dir": "./Results-Q105090",

        # 区间预测设置：q10/q90 -> 80%
        "interval_tau_lower": 0.10,
        "interval_tau_upper": 0.90,
        "interval_target_coverage": 0.80,

        # 残差标签空间
        # original：e = y_true_original - y_base_original
        # scaled：   e = y_true_scaled - y_base_scaled
        "residual_label_space": "original",

        # Residual 模型
        "res_backbone_freeze": FREEZE_BACKBONE_MODE,
        "residual_head_dropout": 0.2,
        "crossing_penalty": 0.0,

        "res_lr": 1e-3 if FREEZE_BACKBONE_MODE else 5e-4,
        "res_lr_grid": [5e-4, 1e-3, 3e-3, 5e-3]
        if FREEZE_BACKBONE_MODE
        else [1e-5, 5e-5, 1e-4, 5e-4],

        "res_min_lr": 0.0,
        "res_weight_decay": 0.0,
        "res_grad_clip": 1.0,
        "res_epochs": 30,

        "res_early_stop_patience": 8,
        "res_early_stop_min_delta": 0.0,

        "res_pilot_epochs": 10,
        "res_warmup_grid": [0, 2, 5] if FREEZE_BACKBONE_MODE else [0, 3, 5, 10],
        "res_sched_type_grid": ["cosine"] if FREEZE_BACKBONE_MODE else ["cosine", "linear"],

        # 可选：强制 Residual 调度，跳过 pilot
        "force_res_sched": None,
    }

    # 建议调试顺序：
    # 1) 先只跑 ["CNNFrontTFT"] 和 pred_steps=[1]
    # 2) 确认无误后再增加模型和步长


    models_to_test = ["CNNFrontTFT", "LSTM", "GRU", "TCN", "Transformer"]
    pred_steps = [1, 4, 24, 96]

    # Quick mode: skip pilot, fixed params, fewer epochs
    QUICK = True
    if QUICK:
        BASE_CONFIG["force_base_sched"] = [1e-3, 5, "cosine"]
        BASE_CONFIG["epochs"] = 20
        BASE_CONFIG["force_res_sched"] = [1e-3, 3, "cosine"]
        BASE_CONFIG["res_epochs"] = 15

    all_dfs = []

    for h in pred_steps:
        # 同一步长内所有模型共享同一份数据准备结果，保证切分/scaler/序列完全一致
        base_cfg_for_data = deepcopy(BASE_CONFIG)
        base_cfg_for_data["pred_step"] = h
        print(f"\n================ 准备数据 | pred_step={h} ================\n")
        prepared = prepare_data(base_cfg_for_data)

        for model_name in models_to_test:
            CONFIG = deepcopy(BASE_CONFIG)
            CONFIG["pred_step"] = h
            CONFIG["results_dir"] = f"./Result-{model_name}-Q105090-step{h}"

            print(f"\n=============== 运行 {model_name} | 预测步长 = {h} ===============")

            try:
                df_metrics = run_single_model(CONFIG, model_name=model_name, prepared=prepared)
                all_dfs.append(df_metrics)
            except Exception as e:
                print(f"[ERROR] {model_name} | pred_step={h} 运行失败：{repr(e)}")
                # 不中断全部实验，继续下一个模型
                continue

    if len(all_dfs) > 0:
        df_all = pd.concat(all_dfs, ignore_index=True)
        write_summary_tables(df_all, out_prefix="Q105090_all_models_comparison")
    else:
        print("没有任何模型成功运行，无法生成汇总表。")
