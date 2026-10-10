"""Train the repository's two Re-TFT stages on a small synthetic CPU dataset.

This method demonstration does not reproduce manuscript experiments or metrics.
It reads no research data and writes no checkpoints or prediction files.
Run from a repository containing this example with:
    python examples/minimal_two_stage_workflow.py
For review before installation, put the real repository on PYTHONPATH instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.pipeline import (
    fit_validation_affine,
    forecast_endpoint,
    frozen_residual_from_base,
)
from models.retft import CNNFrontTFT
from utils.features import SeqDataset, apply_feature_engineering, create_sequences
from utils.training import predict_model, train_model

ENCODER_PREFIXES = ("convs.", "enc_proj.", "enc_lstm.")


def synthetic_frame() -> pd.DataFrame:
    """Same-time synthetic weather and load in arbitrary synthetic load units."""
    rng = np.random.default_rng(7)
    step = np.arange(1408)
    daily = 2 * np.pi * step / 96
    weekly = 2 * np.pi * step / 672
    temperature = 15 + 3 * np.sin(daily - 0.4) + 0.5 * np.cos(weekly)
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=len(step), freq="15min"),
        "humidity": 55 + 12 * np.cos(daily),
        "speed": 3 + 0.4 * np.sin(daily + 0.8),
        # These bounds describe the same timestamp, not an entire day's extrema.
        "mintem": temperature - 0.5,
        "temperature": temperature,
        "maxtem": temperature + 0.5,
        "load": 10 + 1.8 * np.sin(daily) + 0.6 * np.cos(weekly)
                + 0.08 * (temperature - 15) + rng.normal(0, 0.08, len(step)),
    })


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def loader(x, calendar, labels, *, shuffle=False) -> DataLoader:
    return DataLoader(SeqDataset(x, calendar, labels), batch_size=32, shuffle=shuffle)


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    sequence_length, horizon = 96, 96
    epochs = 3  # Fixed in advance; no search or synthetic-test selection.

    frame, columns = apply_feature_engineering(synthetic_frame(), "date")
    assert len(columns) == 40
    train_end, val_end = int(len(frame) * 0.60), int(len(frame) * 0.80)
    target_column = columns.index("load")
    raw_features = frame[columns].to_numpy(dtype=np.float32)
    scaler = MinMaxScaler().fit(raw_features[:train_end])
    scaled = scaler.transform(raw_features).astype(np.float32)
    x, calendar, labels, target_dates = create_sequences(
        scaled, frame["date"].to_numpy(), sequence_length, target_column, pred_step=horizon
    )
    target_indices = np.arange(sequence_length + horizon - 1, len(frame))
    masks = (
        target_indices < train_end,
        (target_indices >= train_end) & (target_indices < val_end),
        target_indices >= val_end,
    )
    train_idx, val_idx, test_idx = (target_indices[mask] for mask in masks)
    assert train_idx[-1] < val_idx[0] <= val_idx[-1] < test_idx[0]
    assert len(set(train_idx) & set(val_idx)) == len(set(val_idx) & set(test_idx)) == 0
    assert all(len(indices) > 1 for indices in (train_idx, val_idx, test_idx))
    # Split by target timestamp. Later partitions may use already observed history;
    # none supplies future weather, and test targets never enter a fit or selection.
    (x_train, k_train, y_train), (x_val, k_val, y_val), (x_test, k_test, _) = (
        (x[mask], calendar[mask], labels[mask]) for mask in masks
    )
    assert x_train.shape[1:] == (96, 40) and k_train.shape[1] == 8

    target_scale = float(scaler.scale_[target_column])
    target_offset = float(scaler.min_[target_column])

    def inverse_target(values):
        # Works on either numpy arrays or torch tensors; no residual rescaling.
        return (values - target_offset) / target_scale

    print("SYNTHETIC CPU METHOD DEMO; no manuscript metrics or reproduction claim")
    print(f"train/validation/test endpoints: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print("history [96,40]; target calendar [8]; H96 -> one scalar endpoint")

    # Stage 1: learn a Base forecaster in the train-normalized target domain.
    base = CNNFrontTFT(d_enc_in=40, d_known_in=8, conv_out_each=12, d_model=64).to(device)
    base_initial = snapshot(base)
    train_model(
        base, loader(x_train, k_train, y_train, shuffle=True), loader(x_val, k_val, y_val),
        nn.MSELoss(), torch.optim.Adam(base.parameters(), lr=1e-3), epochs, device,
        save_path=None, grad_clip_norm=1.0, early_stop_patience=epochs, tag="Base: ",
    )
    trained_base = snapshot(base)
    assert any(not torch.equal(base_initial[key], value) for key, value in trained_base.items())

    # Complete Base training before constructing signed residual supervision.
    base_train_scaled, _ = predict_model(base, loader(x_train, k_train, y_train), device)
    base_val_scaled, _ = predict_model(base, loader(x_val, k_val, y_val), device)
    base_train = inverse_target(base_train_scaled).astype(np.float32)
    base_val = inverse_target(base_val_scaled).astype(np.float32)
    truth_train = inverse_target(y_train).astype(np.float32)
    truth_val = inverse_target(y_val).astype(np.float32)
    np.testing.assert_allclose(truth_train, frame["load"].to_numpy()[train_idx], atol=2e-6)
    residual_train = truth_train - base_train
    residual_val = truth_val - base_val

    # Stage 2: the public factory copies only pretrained CNN/projection/LSTM.
    # Calendar projection, GRN and scalar residual output remain independent.
    residual = frozen_residual_from_base(base, input_dim=40, conv_out_each=12, d_model=64)
    copied = snapshot(residual)
    base_parameters = dict(base.named_parameters())
    encoder_names = [name for name, _ in residual.named_parameters() if name.startswith(ENCODER_PREFIXES)]
    for name, parameter in residual.named_parameters():
        assert all(parameter.data_ptr() != other.data_ptr() for other in base_parameters.values())
        if name.startswith(ENCODER_PREFIXES):
            assert torch.equal(parameter, base_parameters[name]) and not parameter.requires_grad
        else:
            assert parameter.requires_grad
    assert residual.fuse_grn is not base.fuse_grn
    head_parameters = [parameter for parameter in residual.parameters() if parameter.requires_grad]
    train_history, val_history, _, _ = train_model(
        residual, loader(x_train, k_train, residual_train, shuffle=True),
        loader(x_val, k_val, residual_val), nn.SmoothL1Loss(beta=1.0),
        torch.optim.Adam(head_parameters, lr=1e-3), epochs, device,
        save_path=None, grad_clip_norm=1.0, early_stop_patience=epochs, tag="Residual: ",
    )
    after_residual = residual.state_dict()
    assert all(torch.equal(copied[name], after_residual[name]) for name in encoder_names)
    assert all(parameter.grad is None for name, parameter in residual.named_parameters()
               if name in encoder_names)
    assert any(not torch.equal(copied[name], after_residual[name]) for name in after_residual
               if not name.startswith(ENCODER_PREFIXES))
    assert all(torch.equal(value, base.state_dict()[key]) for key, value in trained_base.items())
    assert np.isfinite(train_history + val_history).all()
    print(f"copied/frozen encoder tensors: {len(encoder_names)}; independent residual head updated")

    # Calibration: only validation truth and predictions reach this fitter.
    validation_residual, _ = predict_model(residual, loader(x_val, k_val, residual_val), device)
    affine = fit_validation_affine(truth_val, base_val, validation_residual)
    assert -3.0 <= affine["a"] <= 3.0
    fixed_affine = dict(affine)
    print(f"validation-only affine: a={affine['a']:.6f}, b={affine['b']:.6f}")

    # Test inference begins only after both stages and calibration are complete.
    history = torch.from_numpy(x_test)
    known_calendar = torch.from_numpy(k_test)
    base_test, retft_test = forecast_endpoint(
        base, residual, history, known_calendar, affine, inverse_target=inverse_target
    )
    with torch.no_grad():
        raw_residual_test = residual(history, known_calendar)
    assert base_test.shape == retft_test.shape == (len(test_idx),)
    assert torch.isfinite(base_test).all() and torch.isfinite(retft_test).all()
    torch.testing.assert_close(retft_test, base_test + affine["a"] * raw_residual_test + affine["b"])
    assert affine == fixed_affine
    print("test target | Base | Re-TFT (arbitrary synthetic load units)")
    for row, date in enumerate(target_dates[masks[2]][:3]):
        print(f"{date} | {base_test[row].item():.6f} | {retft_test[row].item():.6f}")
    print("PASS: sequential training, signed original-unit residuals, copied frozen encoder,")
    print("independent head, validation-only calibration, fixed scalar test assembly")
    print("The short run demonstrates the method; forecast improvement is not guaranteed.")


if __name__ == "__main__":
    main()
