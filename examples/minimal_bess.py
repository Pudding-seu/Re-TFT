"""MINIMAL IMPLEMENTATION EXAMPLE: forecast schedule, then truth evaluation.

Synthetic 96-point profiles only; no private test predictions are included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bess.peak_shaving import Battery, optimize_forecast_schedule, realized_peak_reduction


def main() -> None:
    quarter_hours = np.arange(96)
    forecast_kw = 2300.0 + 520.0 * np.exp(-((quarter_hours - 72) / 9.0) ** 2)
    truth_kw = forecast_kw + 45.0 * np.sin(quarter_hours / 8.0)
    battery = Battery()
    schedule = optimize_forecast_schedule(forecast_kw, battery)
    reduction_kw = realized_peak_reduction(truth_kw, schedule)
    print(f"Forecast-optimized daily peak: {schedule.forecast_peak_after_kw:.2f} kW")
    print(f"Realized peak reduction on truth: {reduction_kw:.2f} kW")
    print("This is synthetic and does not reproduce the 71-day paper result.")


if __name__ == "__main__":
    main()
