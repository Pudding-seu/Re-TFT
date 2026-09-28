"""Fast synthetic tests for the minimal public implementation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from bess.peak_shaving import Battery, optimize_forecast_schedule, realized_peak_reduction
from models.pipeline import fit_validation_affine, forecast_endpoint, frozen_residual_from_base
from models.retft import CNNFrontTFT


class DirectEndpointTests(unittest.TestCase):
    def test_scalar_output_and_frozen_copy(self):
        torch.manual_seed(2)
        base = CNNFrontTFT(d_enc_in=40, d_known_in=8, conv_out_each=12)
        residual = frozen_residual_from_base(base, input_dim=40, conv_out_each=12)
        for module in (residual.convs, residual.enc_proj, residual.enc_lstm):
            self.assertTrue(all(not p.requires_grad for p in module.parameters()))
        x = torch.zeros((2, 96, 40))
        calendar = torch.zeros((2, 8))
        base_y, final_y = forecast_endpoint(
            base, residual, x, calendar, {"mode": "affine", "a": 1.0, "b": 0.0},
            inverse_target=lambda value: value,
        )
        self.assertEqual(tuple(base_y.shape), (2,))
        self.assertEqual(tuple(final_y.shape), (2,))
        with self.assertRaises(ValueError):
            forecast_endpoint(base, residual, torch.zeros((2, 95, 40)), calendar,
                              {"mode": "affine", "a": 1.0, "b": 0.0}, lambda value: value)

    def test_affine_is_fitted_on_passed_validation_vectors(self):
        truth = np.array([2.0, 4.0, 6.0])
        base = np.array([1.0, 1.0, 1.0])
        residual = np.array([0.0, 1.0, 2.0])
        params = fit_validation_affine(truth, base, residual)
        self.assertAlmostEqual(params["a"], 2.0)
        self.assertAlmostEqual(params["b"], 1.0)


class BatteryTests(unittest.TestCase):
    def test_fixed_schedule_scored_only_after_optimization(self):
        t = np.arange(96)
        forecast = 2400 + 350 * np.exp(-((t - 70) / 9) ** 2)
        battery = Battery()
        schedule = optimize_forecast_schedule(forecast, battery)
        self.assertEqual(schedule.charge_kw.shape, (96,))
        self.assertEqual(schedule.discharge_kw.shape, (96,))
        self.assertAlmostEqual(schedule.energy_kwh[0], 2000.0)
        self.assertAlmostEqual(schedule.energy_kwh[-1], 2000.0, places=4)
        self.assertGreaterEqual(schedule.energy_kwh.min(), battery.minimum_kwh - 1e-4)
        self.assertLessEqual(schedule.energy_kwh.max(), battery.maximum_kwh + 1e-4)
        # One fixed schedule is evaluated against two distinct truths.
        a = realized_peak_reduction(forecast, schedule)
        b = realized_peak_reduction(forecast + 10 * np.sin(t), schedule)
        self.assertTrue(np.isfinite(a) and np.isfinite(b))
        self.assertGreater(abs(a - b), 1e-4)
        with self.assertRaises(ValueError):
            optimize_forecast_schedule(np.ones(95), battery)


if __name__ == "__main__":
    unittest.main()
