"""Paper-aligned two-stage daily BESS LP and realized-truth evaluation.

The schedule is optimized from the forecast alone. Ground Truth enters only
after both LP stages have finished and the charge/discharge vectors are fixed.
This module deliberately omits private forecast archives and study statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    power_kw: float = 1000.0
    capacity_kwh: float = 4000.0
    minimum_kwh: float = 200.0   # 5% SOC
    maximum_kwh: float = 3800.0  # 95% SOC
    initial_kwh: float = 2000.0  # 50% SOC
    terminal_kwh: float = 2000.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    step_hours: float = 0.25
    stage2_peak_tolerance_kw: float = 1e-6


@dataclass(frozen=True)
class Schedule:
    charge_kw: np.ndarray
    discharge_kw: np.ndarray
    energy_kwh: np.ndarray
    forecast_peak_after_kw: float


def _profile(values: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (96,) or not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{label} must contain 96 finite nonnegative 15-min values")
    return arr


def _constraints(forecast: np.ndarray, battery: Battery, peak_cap: float | None = None):
    n = len(forecast)
    n_vars = 1 + 2 * n  # peak, charge[0:96], discharge[0:96]
    ch = np.arange(1, n + 1)
    dis = np.arange(n + 1, 2 * n + 1)
    rows: list[np.ndarray] = []
    limits: list[float] = []
    for t in range(n):
        row = np.zeros(n_vars)
        row[0], row[ch[t]], row[dis[t]] = -1.0, 1.0, -1.0
        rows.append(row)
        limits.append(-float(forecast[t]))
    for end in range(n):
        upper = np.zeros(n_vars)
        upper[ch[: end + 1]] = battery.charge_efficiency * battery.step_hours
        upper[dis[: end + 1]] = -battery.step_hours / battery.discharge_efficiency
        rows.extend((upper, -upper))
        limits.extend((battery.maximum_kwh - battery.initial_kwh,
                       battery.initial_kwh - battery.minimum_kwh))
    if peak_cap is not None:
        row = np.zeros(n_vars)
        row[0] = 1.0
        rows.append(row)
        limits.append(peak_cap)
    equality = np.zeros((1, n_vars))
    equality[0, ch] = battery.charge_efficiency * battery.step_hours
    equality[0, dis] = -battery.step_hours / battery.discharge_efficiency
    target = np.array([battery.terminal_kwh - battery.initial_kwh])
    bounds = [(0.0, None)] + [(0.0, battery.power_kw)] * (2 * n)
    return np.asarray(rows), np.asarray(limits), equality, target, bounds, ch, dis


def optimize_forecast_schedule(forecast_kw: np.ndarray,
                               battery: Battery = Battery()) -> Schedule:
    """Minimize predicted peak, then throughput within the first-stage peak."""
    forecast = _profile(forecast_kw, "forecast_kw")
    args = _constraints(forecast, battery)
    objective = np.zeros(1 + 2 * len(forecast))
    objective[0] = 1.0
    first = linprog(objective, A_ub=args[0], b_ub=args[1], A_eq=args[2],
                    b_eq=args[3], bounds=args[4], method="highs")
    if not first.success:
        raise RuntimeError(f"Stage-1 LP failed: {first.message}")
    cap = float(first.x[0] + battery.stage2_peak_tolerance_kw)
    args = _constraints(forecast, battery, peak_cap=cap)
    throughput = np.zeros_like(objective)
    throughput[args[5]] = battery.step_hours
    throughput[args[6]] = battery.step_hours
    second = linprog(throughput, A_ub=args[0], b_ub=args[1], A_eq=args[2],
                     b_eq=args[3], bounds=args[4], method="highs")
    if not second.success:
        raise RuntimeError(f"Stage-2 LP failed: {second.message}")
    charge = np.asarray(second.x[args[5]])
    discharge = np.asarray(second.x[args[6]])
    energy = np.r_[battery.initial_kwh,
                   battery.initial_kwh + np.cumsum(
                       battery.step_hours * (battery.charge_efficiency * charge
                                             - discharge / battery.discharge_efficiency))]
    if abs(energy[-1] - battery.terminal_kwh) > 1e-5:
        raise RuntimeError("Terminal SOC constraint failed")
    return Schedule(charge_kw=charge, discharge_kw=discharge, energy_kwh=energy,
                    forecast_peak_after_kw=float(np.max(forecast + charge - discharge)))


def realized_peak_reduction(true_load_kw: np.ndarray, schedule: Schedule) -> float:
    """Score the fixed schedule on truth; negative reductions remain negative."""
    truth = _profile(true_load_kw, "true_load_kw")
    if schedule.charge_kw.shape != (96,) or schedule.discharge_kw.shape != (96,):
        raise ValueError("Schedule must have 96 fixed charge/discharge decisions")
    realized_grid = truth + schedule.charge_kw - schedule.discharge_kw
    return float(np.max(truth) - np.max(realized_grid))
