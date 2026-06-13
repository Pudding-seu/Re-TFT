# -*- coding: utf-8 -*-
"""Training loop, inference, scheduler, and hyperparameter search utilities."""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR


# ─────────────────────────────────────────────────────────────────────────────
# Safe checkpoint saving
# ─────────────────────────────────────────────────────────────────────────────
def safe_torch_save(state_dict: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(state_dict, str(path), _use_new_zipfile_serialization=False)
        return
    except Exception as e:
        print(f"[SAVE WARN] zip-serialize failed ({e}), retrying with file handle …")
    try:
        with open(str(path), "wb") as f:
            torch.save(state_dict, f, _use_new_zipfile_serialization=False)
    except Exception as e2:
        print(f"[SAVE ERROR] file-handle save also failed ({e2}), falling back to plain save")
        torch.save(state_dict, str(path))


# ─────────────────────────────────────────────────────────────────────────────
# Learning-rate scheduler builder
# ─────────────────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, total_epochs: int, warmup_epochs: int = 5,
                    sched_type: str = "cosine", min_lr: float = 0.0):
    """Linear warmup followed by cosine or linear decay."""
    warmup = (
        LambdaLR(optimizer, lr_lambda=lambda ep: min(1.0, (ep + 1) / max(1, warmup_epochs)))
        if warmup_epochs > 0 else None
    )
    rest = max(1, total_epochs - max(0, warmup_epochs))
    if sched_type == "cosine":
        tail = CosineAnnealingLR(optimizer, T_max=rest, eta_min=min_lr)
    else:
        start_lr = optimizer.param_groups[0]["lr"]
        min_f = (min_lr / start_lr) if start_lr > 0 else 0.0
        tail = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / max(rest - 1, 1)) + ep / max(rest - 1, 1) * min_f)
    if warmup is None:
        return tail
    return SequentialLR(optimizer, schedulers=[warmup, tail], milestones=[warmup_epochs])


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs: int, device, save_path,
                scheduler=None, early_stop_patience: int = 10,
                early_stop_min_delta: float = 0.0,
                grad_clip_norm=None, tag: str = ""):
    """
    Standard training loop with early stopping and best-checkpoint saving.

    Returns
    -------
    tr_hist   : list[float]  per-epoch training loss
    va_hist   : list[float]  per-epoch validation loss
    elapsed   : float        wall-clock seconds
    best_val  : float        best validation loss achieved
    """
    best = float("inf")
    wait = 0
    tr_hist, va_hist = [], []
    non_block = device.type == "cuda"
    t0 = time.time()

    for ep in range(num_epochs):
        cur_lr = optimizer.param_groups[0]["lr"]
        model.train()
        tr = 0.0
        for xs, kf, ys in train_loader:
            xs = xs.to(device, dtype=torch.float32, non_blocking=non_block)
            kf = kf.to(device, dtype=torch.float32, non_blocking=non_block)
            ys = ys.to(device, dtype=torch.float32, non_blocking=non_block)
            optimizer.zero_grad()
            loss = criterion(model(xs, kf), ys)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            tr += loss.item() * xs.size(0)
        tr /= len(train_loader.dataset)
        tr_hist.append(tr)

        model.eval()
        va = 0.0
        with torch.no_grad():
            for xs, kf, ys in val_loader:
                xs = xs.to(device, dtype=torch.float32, non_blocking=non_block)
                kf = kf.to(device, dtype=torch.float32, non_blocking=non_block)
                ys = ys.to(device, dtype=torch.float32, non_blocking=non_block)
                va += criterion(model(xs, kf), ys).item() * xs.size(0)
        va /= len(val_loader.dataset)
        va_hist.append(va)
        print(f"{tag}Epoch {ep+1}/{num_epochs} | LR {cur_lr:.5g} | Train {tr:.6f} | Val {va:.6f}")

        if va < best - early_stop_min_delta:
            best = va
            wait = 0
            if save_path:
                safe_torch_save(model.state_dict(), Path(save_path))
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(f"{tag}Early stop at epoch {ep+1} (best_val={best:.6f})")
                break

        if scheduler is not None:
            scheduler.step()

    return tr_hist, va_hist, time.time() - t0, best


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_model(model, loader, device):
    """Run model in eval mode; return numpy arrays (preds, trues)."""
    model.eval()
    preds, trues = [], []
    nb = device.type == "cuda"
    for xs, kf, ys in loader:
        xs = xs.to(device, dtype=torch.float32, non_blocking=nb)
        kf = kf.to(device, dtype=torch.float32, non_blocking=nb)
        preds.extend(model(xs, kf).cpu().numpy().flatten())
        trues.extend(ys.numpy().flatten())
    return np.array(preds), np.array(trues)


# ─────────────────────────────────────────────────────────────────────────────
# Joint LR + warmup + scheduler-type grid search  (pilot search)
# ─────────────────────────────────────────────────────────────────────────────
def lr_warmup_search(model_ctor, criterion_ctor, dloaders, device,
                     pilot_epochs, grid_lrs, grid_warmups, grid_types,
                     weight_decay, grad_clip_norm,
                     early_stop_patience, early_stop_min_delta, tag=""):
    """
    Quick pilot search over (lr × warmup × sched_type) to select the best
    hyperparameter configuration before full training.

    Returns (best_lr, best_warmup, best_sched_type), best_val_loss, best_hist.
    """
    train_loader, val_loader = dloaders
    best_cfg = None
    best_val = float("inf")
    best_hist = None

    print(f"{tag}== Pilot search: lr={grid_lrs}, warmup={grid_warmups}, "
          f"sched={grid_types}, pilot_epochs={pilot_epochs} ==")

    for lr in grid_lrs:
        for w in grid_warmups:
            for t in grid_types:
                m = model_ctor().to(device)
                crit = criterion_ctor()
                opt = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
                sch = build_scheduler(opt, total_epochs=pilot_epochs,
                                      warmup_epochs=w, sched_type=t, min_lr=0.0)
                _, va, _, bv = train_model(
                    m, train_loader, val_loader, crit, opt,
                    pilot_epochs, device, save_path=None,
                    scheduler=sch,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    grad_clip_norm=grad_clip_norm, tag=tag,
                )
                print(f"{tag}  (lr={lr}, warmup={w}, sched={t}) → best_val={bv:.6f}")
                if bv < best_val:
                    best_val = bv
                    best_cfg = (lr, w, t)
                    best_hist = ([], va)

    print(f"{tag}Selected: lr={best_cfg[0]}, warmup={best_cfg[1]}, "
          f"sched={best_cfg[2]} | best_val={best_val:.6f}")
    return best_cfg, best_val, best_hist
