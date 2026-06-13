# -*- coding: utf-8 -*-
"""
Re-TFT Training Script
======================
Reproduces the Belgium dataset benchmark results reported in:

  "Bridging Prediction Accuracy and Operational Value: A Residual-Enhanced
   Transformer for Demand Response and Battery Storage Scheduling"

Usage
-----
  # Single forecasting horizon (e.g. step = 1):
  python train.py --data data/belgium.csv --steps 1

  # Reproduce all four horizons (1 / 4 / 24 / 96):
  python train.py --data data/belgium.csv --steps 1 4 24 96

  # Skip the LR pilot search (use paper-reported hyperparameters):
  python train.py --data data/belgium.csv --steps 1 4 24 96 --force_lr 0.001

Output
------
  results/step<N>/
      cnntft_best.pth          base model checkpoint
      residual_best.pth        residual head checkpoint
      test_predictions.csv     [time, y_true, y_base, y_final] on test set
      metrics_summary.txt      point metrics for Base and Re-TFT
      loss_base.png / loss_res.png
"""

import os
import sys
import time
import argparse
import warnings
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ── Local imports ──────────────────────────────────────────────────────────
from models.retft import CNNFrontTFT, CNN2TFTLSTMResidual
from utils.features import apply_feature_engineering, create_sequences, SeqDataset
from utils.metrics import point_metrics
from utils.training import (
    train_model, predict_model, build_scheduler, lr_warmup_search, safe_torch_save,
)
from utils.calibration import calib_affine, calib_scalar, calib_per_hour, apply_fusion


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION
# All values below reproduce the paper's primary Belgium benchmark.
# Override any value via command-line arguments or by editing this dict.
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # ── Data ──────────────────────────────────────────────────────────────
    "data_path":    "data/belgium.csv",
    "date_col":     "date",
    "date_format":  "%Y/%m/%d %H:%M",
    "feature_cols": ["humidity", "speed", "mintem", "temperature", "maxtem", "load"],
    "target_col":   "load",

    # ── Sequence / normalisation ───────────────────────────────────────────
    "seq_length":   96,   # 24 h at 15-min resolution
    "scaler_range": (0, 1),

    # ── Device / reproducibility ──────────────────────────────────────────
    "seed":    42,
    "use_gpu": True,

    # ── DataLoader ────────────────────────────────────────────────────────
    "batch_size": 32,

    # ── CNN front-end ──────────────────────────────────────────────────────
    "conv_out_each":  12,
    "conv_kernels":   [3, 5, 7],
    "conv_dilations": [1, 2, 4],
    "conv_act":       "relu",

    # ── TFT back-end (Base Head) ──────────────────────────────────────────
    "tft_d_model":     64,
    "tft_n_heads":     4,
    "tft_lstm_layers": 1,
    "tft_dropout":     0.2,

    # ── Base training ─────────────────────────────────────────────────────
    "epochs":       50,
    "lr":           1e-3,
    "lr_grid":      [1e-4, 5e-4, 1e-3],
    "min_lr":       0.0,
    "weight_decay": 0.0,
    "grad_clip":    1.0,

    # ── Base pilot search ─────────────────────────────────────────────────
    "pilot_epochs":      12,
    "warmup_grid":       [0, 3, 5, 10],
    "sched_type_grid":   ["cosine", "linear"],
    "early_stop_patience":    10,
    "early_stop_min_delta":   0.0,

    # ─────────────────────────────────────────────────────────────────────
    # RESIDUAL HEAD CONFIGURATION
    # ─────────────────────────────────────────────────────────────────────
    "res_backbone_freeze": True,    # ← freeze CNN+LSTM during residual training
    "residual_label_space": "original",
    "res_fusion_mode": "affine",
    "res_weight_clip": (-3.0, 3.0),
    "per_hour_min_count": 24,

    # ── Residual training ─────────────────────────────────────────────────
    "res_epochs":       30,
    "res_lr":           1e-3,
    "res_lr_grid":      [5e-4, 1e-3, 3e-3, 5e-3],
    "res_min_lr":       0.0,
    "res_weight_decay": 0.0,
    "res_grad_clip":    1.0,
    "res_use_huber":    True,
    "res_pilot_epochs": 10,
    "res_warmup_grid":  [0, 2, 5],
    "res_sched_type_grid": ["cosine"],
    "res_early_stop_patience":  8,
    "res_early_stop_min_delta": 0.0,

    # ─────────────────────────────────────────────────────────────────────
    # GUARD MECHANISM  (paper Eq. 9  —  Section 2.6.2)
    # ─────────────────────────────────────────────────────────────────────
    # The Guard is a heuristic deployment safeguard, NOT a core learning
    # contribution.  It activates (or suppresses) the residual correction
    # based on two validation-set criteria:
    #
    #   I_guard = 1   if  ρ(ê_res_val, e_val) > τ_corr
    #                AND  ΔRMSE_val > δ_gain         (residual improves RMSE)
    #           = 0   otherwise  →  fall back to ŷ_base
    #
    # BENCHMARK SETTING  (reproduces all paper tables):
    #   Both thresholds are set to 0.  With these values the conditions
    #   reduce to "any correlation / any positive RMSE gain", which means
    #   the Guard NEVER triggers and I_guard ≡ 1 throughout.
    #   This is IDENTICAL to bypassing the Guard entirely, exactly as
    #   stated in Section 2.6.2:
    #       "For the primary benchmark experiments … this heuristic gate
    #        was entirely bypassed (I_guard ≡ 1)."
    #
    # DEPLOYMENT OVERRIDE:
    #   To activate selective gating (e.g. for out-of-distribution datasets),
    #   raise the thresholds, e.g.:
    #       "res_guard_min_corr": 0.10,  # require ρ > 0.10
    #       "res_guard_min_gain": 0.005, # require ≥ 0.5 % RMSE improvement
    # ─────────────────────────────────────────────────────────────────────
    "res_guard_require_pos_corr": False,  # if True: require ρ > 0 (strict)
    "res_guard_min_corr":         0.0,   # τ_corr  — set > 0 for selective activation
    "res_guard_min_gain":         0.0,   # δ_gain  — set > 0 for selective activation

    # ── Output ────────────────────────────────────────────────────────────
    "results_dir": "results",
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: inverse-normalise target column only
# ═══════════════════════════════════════════════════════════════════════════
def _make_inv(scaler, X_all, tgt_idx):
    def inv(arr):
        arr = np.asarray(arr).reshape(-1)
        dummy = np.zeros((len(arr), X_all.shape[1]), dtype=np.float32)
        dummy[:, tgt_idx] = arr
        return scaler.inverse_transform(dummy)[:, tgt_idx]
    return inv


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: pretty-print duration
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(sec):
    if sec is None: return "n/a"
    m, s = divmod(float(sec), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h}h {m:02d}m {s:04.1f}s"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def run(cfg: dict):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    pred_step = int(cfg.get("pred_step", 1))
    print(f"\n{'='*60}")
    print(f"  Re-TFT  |  pred_step = {pred_step} timestep(s)  ({pred_step * 15} min ahead)")
    print(f"{'='*60}\n")

    out_dir = Path(cfg["results_dir"]) / f"step{pred_step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if cfg["use_gpu"] and torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Load and pre-process data ──────────────────────────────────────────
    df = pd.read_csv(cfg["data_path"])
    df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]],
                                          format=cfg["date_format"], errors="coerce")
    df = df.dropna(subset=[cfg["date_col"]]).sort_values(cfg["date_col"]).reset_index(drop=True)

    base_cols = cfg["feature_cols"]
    df[base_cols] = (df[base_cols]
                     .replace([np.inf, -np.inf], np.nan)
                     .interpolate(limit_direction="both")
                     .ffill().bfill())

    df_fe, enc_cols = apply_feature_engineering(df, cfg["date_col"])
    N = len(df_fe)

    # ── 60/20/20 temporal split ───────────────────────────────────────────
    train_end = int(N * 0.60)
    val_end   = int(N * 0.80)
    print(f"Rows: total={N}  train=[0,{train_end})  val=[{train_end},{val_end})  test=[{val_end},{N})")

    # ── Fit scaler on TRAIN partition only (no look-ahead) ────────────────
    X_all = df_fe[enc_cols].values.astype(np.float32)
    tgt_idx = enc_cols.index(cfg["target_col"])
    scaler = MinMaxScaler(feature_range=cfg["scaler_range"])
    scaler.fit(X_all[:train_end])
    X_scaled = scaler.transform(X_all).astype(np.float32)
    inv = _make_inv(scaler, X_all, tgt_idx)

    # ── Build sequences ───────────────────────────────────────────────────
    seq_len = cfg["seq_length"]
    enc_seqs, dec_known, labels, label_dates = create_sequences(
        X_scaled, df_fe[cfg["date_col"]].values, seq_len, tgt_idx, pred_step)
    M = len(labels)
    tr_end = min(max(0, train_end - seq_len - pred_step + 1), M)
    va_end = min(max(0, val_end   - seq_len - pred_step + 1), M)

    X_tr,  KF_tr,  y_tr,  d_tr  = enc_seqs[:tr_end],  dec_known[:tr_end],  labels[:tr_end],  label_dates[:tr_end]
    X_val, KF_val, y_val, d_val  = enc_seqs[tr_end:va_end], dec_known[tr_end:va_end], labels[tr_end:va_end], label_dates[tr_end:va_end]
    X_te,  KF_te,  y_te,  d_te   = enc_seqs[va_end:],  dec_known[va_end:],  labels[va_end:],  label_dates[va_end:]
    print(f"Sequences: train={len(X_tr)}  val={len(X_val)}  test={len(X_te)}")

    pin = device.type == "cuda"
    bs  = cfg["batch_size"]
    train_loader = DataLoader(SeqDataset(X_tr,  KF_tr,  y_tr),  bs, shuffle=True,  pin_memory=pin)
    val_loader   = DataLoader(SeqDataset(X_val, KF_val, y_val), bs, shuffle=False, pin_memory=pin)
    test_loader  = DataLoader(SeqDataset(X_te,  KF_te,  y_te),  bs, shuffle=False, pin_memory=pin)

    d_enc_in  = X_tr.shape[2]
    d_known   = KF_tr.shape[1]

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Base Model (CNN-TFT)
    # ══════════════════════════════════════════════════════════════════════
    def base_ctor():
        return CNNFrontTFT(
            d_enc_in=d_enc_in, d_known_in=d_known,
            conv_out_each=cfg["conv_out_each"],
            conv_kernels=tuple(cfg["conv_kernels"]),
            conv_dilations=tuple(cfg["conv_dilations"]),
            d_model=cfg["tft_d_model"], n_heads=cfg["tft_n_heads"],
            lstm_layers=cfg["tft_lstm_layers"], dropout=cfg["tft_dropout"],
            act=cfg["conv_act"],
        )

    # Optional pilot search for best (lr, warmup, scheduler)
    if cfg.get("force_lr") is None:
        (best_lr, best_wu, best_st), _, _ = lr_warmup_search(
            model_ctor=base_ctor,
            criterion_ctor=lambda: nn.MSELoss(),
            dloaders=(train_loader, val_loader),
            device=device,
            pilot_epochs=cfg["pilot_epochs"],
            grid_lrs=cfg.get("lr_grid", [cfg["lr"]]),
            grid_warmups=cfg["warmup_grid"],
            grid_types=cfg["sched_type_grid"],
            weight_decay=cfg["weight_decay"],
            grad_clip_norm=cfg["grad_clip"],
            early_stop_patience=cfg["early_stop_patience"],
            early_stop_min_delta=cfg["early_stop_min_delta"],
            tag="[Base Pilot] ",
        )
    else:
        best_lr = float(cfg["force_lr"])
        best_wu = cfg["warmup_grid"][0]
        best_st = cfg["sched_type_grid"][0]
        print(f"[Base] Forced lr={best_lr}, warmup={best_wu}, sched={best_st}")

    base_model = base_ctor().to(device)
    base_ckpt  = out_dir / "cnntft_best.pth"
    opt = optim.Adam(base_model.parameters(), lr=best_lr, weight_decay=cfg["weight_decay"])
    sch = build_scheduler(opt, cfg["epochs"], best_wu, best_st, cfg["min_lr"])
    tr_h, va_h, base_sec, _ = train_model(
        base_model, train_loader, val_loader,
        nn.MSELoss(), opt, cfg["epochs"], device, base_ckpt,
        scheduler=sch,
        early_stop_patience=cfg["early_stop_patience"],
        early_stop_min_delta=cfg["early_stop_min_delta"],
        grad_clip_norm=cfg["grad_clip"],
        tag="[Base] ",
    )
    # Save loss curve
    plt.figure(figsize=(9, 4))
    plt.plot(tr_h, label="train"); plt.plot(va_h, label="val")
    plt.title(f"Base loss (step={pred_step})"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(str(out_dir / "loss_base.png"), dpi=150); plt.close()

    base_model.load_state_dict(torch.load(str(base_ckpt), map_location=device))

    # Predictions in scaled domain → inverse-normalise
    val_pred_s,  val_true_s  = predict_model(base_model, DataLoader(SeqDataset(X_val, KF_val, y_val), bs, pin_memory=pin), device)
    test_pred_s, test_true_s = predict_model(base_model, DataLoader(SeqDataset(X_te,  KF_te,  y_te),  bs, pin_memory=pin), device)

    y_val_base  = inv(val_pred_s);  y_val_true  = inv(val_true_s)
    y_te_base   = inv(test_pred_s); y_te_true   = inv(test_true_s)

    met_base = point_metrics(y_te_true, y_te_base)
    print(f"\n[Base Test] RMSE={met_base['RMSE']:.2f}  MAE={met_base['MAE']:.2f}  "
          f"MAPE={met_base['MAPE']:.2f}%  R²={met_base['R2']:.4f}  "
          f"time={_fmt(base_sec)}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Residual Correction Head
    # Targets: e_t = y_t − ŷ_{base,t}   (paper Section 2.6)
    # ══════════════════════════════════════════════════════════════════════
    res_tr_lbl  = (y_val_true[:0] * 0)   # placeholder; computed below
    # Residual labels in original-domain units
    tr_pred_s, tr_true_s = predict_model(base_model, DataLoader(SeqDataset(X_tr, KF_tr, y_tr), bs, pin_memory=pin), device)
    y_tr_true = inv(tr_true_s); y_tr_base = inv(tr_pred_s)
    res_tr  = (y_tr_true  - y_tr_base).astype(np.float32)
    res_val = (y_val_true - y_val_base).astype(np.float32)
    res_te  = (y_te_true  - y_te_base ).astype(np.float32)

    res_train_loader = DataLoader(SeqDataset(X_tr,  KF_tr,  res_tr),  bs, shuffle=True,  pin_memory=pin)
    res_val_loader   = DataLoader(SeqDataset(X_val, KF_val, res_val), bs, shuffle=False, pin_memory=pin)
    res_test_loader  = DataLoader(SeqDataset(X_te,  KF_te,  res_te),  bs, shuffle=False, pin_memory=pin)

    base_sd = torch.load(str(base_ckpt), map_location="cpu")

    def res_ctor():
        return CNN2TFTLSTMResidual(
            d_enc_in=d_enc_in, d_known_in=d_known,
            conv_out_each=cfg["conv_out_each"],
            conv_kernels=tuple(cfg["conv_kernels"]),
            conv_dilations=tuple(cfg["conv_dilations"]),
            d_model=cfg.get("res_cnnl_d_model", cfg["tft_d_model"]),
            lstm_layers=cfg["tft_lstm_layers"],
            dropout=cfg.get("res_cnnl_dropout", 0.2),
            act=cfg["conv_act"],
            init_from_base_state_dict=base_sd,
            freeze_backbone=cfg["res_backbone_freeze"],
        )

    crit_res = nn.SmoothL1Loss(beta=1.0) if cfg["res_use_huber"] else nn.MSELoss()

    (res_lr, res_wu, res_st), _, _ = lr_warmup_search(
        model_ctor=res_ctor,
        criterion_ctor=lambda: (nn.SmoothL1Loss(beta=1.0) if cfg["res_use_huber"] else nn.MSELoss()),
        dloaders=(res_train_loader, res_val_loader),
        device=device,
        pilot_epochs=cfg["res_pilot_epochs"],
        grid_lrs=cfg.get("res_lr_grid", [cfg["res_lr"]]),
        grid_warmups=cfg["res_warmup_grid"],
        grid_types=cfg["res_sched_type_grid"],
        weight_decay=cfg["res_weight_decay"],
        grad_clip_norm=cfg["res_grad_clip"],
        early_stop_patience=cfg["res_early_stop_patience"],
        early_stop_min_delta=cfg["res_early_stop_min_delta"],
        tag="[Res Pilot] ",
    )

    res_model = res_ctor().to(device)
    res_ckpt  = out_dir / "residual_best.pth"
    opt_r = optim.Adam(res_model.parameters(), lr=res_lr, weight_decay=cfg["res_weight_decay"])
    sch_r = build_scheduler(opt_r, cfg["res_epochs"], res_wu, res_st, cfg["res_min_lr"])
    tr_hr, va_hr, res_sec, _ = train_model(
        res_model, res_train_loader, res_val_loader,
        crit_res, opt_r, cfg["res_epochs"], device, res_ckpt,
        scheduler=sch_r,
        early_stop_patience=cfg["res_early_stop_patience"],
        early_stop_min_delta=cfg["res_early_stop_min_delta"],
        grad_clip_norm=cfg["res_grad_clip"],
        tag="[Residual] ",
    )
    plt.figure(figsize=(9, 4))
    plt.plot(tr_hr, label="train"); plt.plot(va_hr, label="val")
    plt.title(f"Residual loss (step={pred_step})"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(str(out_dir / "loss_res.png"), dpi=150); plt.close()

    res_model.load_state_dict(torch.load(str(res_ckpt), map_location=device))
    val_res_s, _  = predict_model(res_model, res_val_loader,  device)
    te_res_s,  _  = predict_model(res_model, res_test_loader, device)

    # Residual predictions are in scaled domain; convert to original units
    factor = 1.0 / float(scaler.scale_[tgt_idx])
    y_val_res = val_res_s.astype(np.float32) * factor
    y_te_res  = te_res_s .astype(np.float32) * factor

    # ── Affine calibration on validation set ─────────────────────────────
    fusion_mode = cfg.get("res_fusion_mode", "affine")
    if fusion_mode == "scalar":
        calib = calib_scalar(y_val_true, y_val_base, y_val_res,
                              clip=cfg.get("res_weight_clip", (-3.0, 3.0)))
    elif fusion_mode == "per_hour":
        calib = calib_per_hour(y_val_true, y_val_base, y_val_res,
                                dates=d_val,
                                min_count=cfg.get("per_hour_min_count", 24))
    else:
        calib = calib_affine(y_val_true, y_val_base, y_val_res,
                              clip_a=cfg.get("res_weight_clip", (-3.0, 3.0)))

    y_val_final = apply_fusion(calib, y_val_base, y_val_res, dates=d_val)
    rmse_val_base  = float(np.sqrt(np.mean((y_val_true - y_val_base ) ** 2)))
    rmse_val_final = float(np.sqrt(np.mean((y_val_true - y_val_final) ** 2)))
    val_gain_pct   = 100.0 * (rmse_val_base - rmse_val_final) / max(1e-9, rmse_val_base)

    # Pearson correlation between residual prediction and true residual (val)
    corr_val = float(np.corrcoef(
        (y_val_true - y_val_base).reshape(-1),
        y_val_res.reshape(-1)
    )[0, 1])

    # ══════════════════════════════════════════════════════════════════════
    # GUARD MECHANISM EVALUATION  (paper Eq. 9)
    # ══════════════════════════════════════════════════════════════════════
    #
    # Decision rule (two conditions must BOTH hold):
    #   (1) Correlation gate:
    #       if res_guard_require_pos_corr = True  → require ρ > 0
    #       else                                  → require |ρ| ≥ τ_corr
    #   (2) RMSE-gain gate:
    #       require RMSE_val_final < RMSE_val_base × (1 − δ_gain)
    #
    # BENCHMARK SETTING (τ_corr = 0, δ_gain = 0):
    #   Condition (2) passes whenever RMSE_val_final ≤ RMSE_val_base, i.e.
    #   any non-negative validation improvement.  Because the calibrated
    #   fusion is guaranteed to be at least as good as the base on the val
    #   set (OLS fit), this condition is always satisfied.
    #   Condition (1) with τ_corr = 0 is also trivially satisfied for any
    #   non-zero correlation.
    #   → I_guard ≡ 1  (Guard never triggers)  ← reproduces paper tables.
    #
    req_pos_corr = cfg["res_guard_require_pos_corr"]
    tau_corr     = float(cfg["res_guard_min_corr"])   # τ_corr
    delta_gain   = float(cfg["res_guard_min_gain"])   # δ_gain

    bad_corr = (corr_val <= 0.0) if req_pos_corr else (abs(corr_val) < tau_corr)
    no_gain  = (rmse_val_final >= rmse_val_base * (1.0 - delta_gain))

    if bad_corr or no_gain:
        # Guard triggered: fall back to base forecast
        print(f"[Guard] TRIGGERED  corr={corr_val:.4f}  val_gain={val_gain_pct:+.2f}%"
              f"  → using base forecast (I_guard = 0)")
        y_pred_final = y_te_base.copy()
        guard_active = False
    else:
        # Guard passed: apply residual correction
        print(f"[Guard] PASSED  corr={corr_val:.4f}  val_gain={val_gain_pct:+.2f}%"
              f"  → residual correction applied (I_guard = 1)")
        y_pred_final = apply_fusion(calib, y_te_base, y_te_res, dates=d_te)
        guard_active = True

    # ── Final test metrics ────────────────────────────────────────────────
    met_final = point_metrics(y_te_true, y_pred_final)
    delta_rmse = 100.0 * (met_base["RMSE"] - met_final["RMSE"]) / max(1e-9, met_base["RMSE"])

    print(f"\n[Re-TFT Test] RMSE={met_final['RMSE']:.2f}  MAE={met_final['MAE']:.2f}  "
          f"MAPE={met_final['MAPE']:.2f}%  R²={met_final['R2']:.4f}  "
          f"ΔRMSE={delta_rmse:+.2f}%")

    # ── Save outputs ──────────────────────────────────────────────────────
    pd.DataFrame({
        "time":    pd.Series(d_te).astype(str),
        "y_true":  y_te_true,
        "y_base":  y_te_base,
        "y_final": y_pred_final,
    }).to_csv(str(out_dir / "test_predictions.csv"), index=False, encoding="utf-8")

    with open(str(out_dir / "metrics_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"pred_step = {pred_step}\n")
        f.write(f"guard_triggered = {not guard_active}  (I_guard={'1' if guard_active else '0'})\n")
        f.write(f"corr_val = {corr_val:.6f}  val_gain = {val_gain_pct:+.3f}%\n")
        f.write(f"tau_corr = {tau_corr}  delta_gain = {delta_gain}  "
                f"(both=0 ↔ I_guard≡1, reproduces paper)\n\n")
        f.write("=== Base (CNN-TFT) ===\n")
        for k, v in met_base.items():
            f.write(f"  {k}: {v:.4f}{'%' if k == 'MAPE' else ''}\n")
        f.write(f"  Training time: {_fmt(base_sec)}\n\n")
        f.write("=== Re-TFT (Base + Residual) ===\n")
        for k, v in met_final.items():
            f.write(f"  {k}: {v:.4f}{'%' if k == 'MAPE' else ''}\n")
        f.write(f"  ΔRMSE: {delta_rmse:+.2f}%\n")
        f.write(f"  Residual training time: {_fmt(res_sec)}\n")

    print(f"\nResults saved to: {out_dir}/")
    return met_base, met_final


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-TFT training script")
    parser.add_argument("--data",     default=DEFAULT_CONFIG["data_path"],
                        help="Path to belgium.csv (or other 15-min load CSV)")
    parser.add_argument("--steps",    nargs="+", type=int, default=[1, 4, 24, 96],
                        help="Forecasting horizons to train (default: 1 4 24 96)")
    parser.add_argument("--force_lr", type=float, default=None,
                        help="Skip pilot search, use this fixed LR")
    parser.add_argument("--no_gpu",   action="store_true", help="Disable GPU")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out",      default=DEFAULT_CONFIG["results_dir"])
    args = parser.parse_args()

    summary_rows = []
    for step in args.steps:
        cfg = deepcopy(DEFAULT_CONFIG)
        cfg["data_path"]   = args.data
        cfg["pred_step"]   = step
        cfg["seed"]        = args.seed
        cfg["use_gpu"]     = not args.no_gpu
        cfg["results_dir"] = args.out
        if args.force_lr is not None:
            cfg["force_lr"] = args.force_lr

        met_b, met_f = run(cfg)
        summary_rows.append({"step": step, "model": "Base",   **met_b})
        summary_rows.append({"step": step, "model": "Re-TFT", **met_f})

    df_sum = pd.DataFrame(summary_rows)
    out_p  = Path(args.out)
    df_sum.to_csv(str(out_p / "all_steps_summary.csv"), index=False, encoding="utf-8")
    print(f"\n{'='*60}")
    print("All horizons complete. Summary:")
    print(df_sum[["step", "model", "RMSE", "MAE", "MAPE", "R2"]].to_string(index=False))
    print(f"{'='*60}")
    print(f"Full summary saved to: {out_p / 'all_steps_summary.csv'}")
