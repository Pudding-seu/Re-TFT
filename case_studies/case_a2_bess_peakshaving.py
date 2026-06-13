# -*- coding: utf-8 -*-
"""
Case A-II: Industrial BESS Peak Shaving via Linear Programming
==============================================================
Paper Section 4.2 (Case A-II)

Scenario
--------
A large industrial facility (1000 kW / 4000 kWh BESS) minimises monthly peak
grid demand to reduce demand charges.  Under-forecasting the peak causes
under-discharge and directly limits achievable savings.

Optimisation (Eq. 15–16 in paper)
-----------------------------------
    min J = P_peak

    subject to:
      P_load(t) + P_ch(t) − P_dis(t) ≤ P_peak           (peak constraint)
      SOC(t+1) = SOC(t) − P_dis(t)·Δt/E_cap + P_ch(t)·Δt/E_cap
      0 ≤ P_ch(t)  ≤ P_rated                              (charge limit)
      0 ≤ P_dis(t) ≤ P_rated                              (discharge limit)
      SOC_min ≤ SOC(t) ≤ SOC_max
      P_load(t) is the *predicted* load (from base or Re-TFT)

Usage
-----
    python case_studies/case_a2_bess_peakshaving.py \
        --pred_csv results/step1/test_predictions.csv \
        --demand_rate 40       # CNY / kW / month
        --duration_days 30

Requirements
------------
    pip install scipy
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from scipy.optimize import linprog
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not found. Install with: pip install scipy")


# ─────────────────────────────────────────────────────────────────────────────
# BESS parameters (paper Table 7)
# ─────────────────────────────────────────────────────────────────────────────
BESS = {
    "P_rated":  1000.0,   # kW
    "E_cap":    4000.0,   # kWh
    "eta":      1.0,      # round-trip efficiency (paper uses 95%, simplified to 1 here)
    "SOC_min":  0.05,     # 5%
    "SOC_max":  0.95,     # 95%
    "SOC_init": 0.50,
}
DT = 0.25   # 15-min interval [h]


def lp_peak_shaving(load_forecast: np.ndarray, bess: dict = BESS):
    """
    Solve a single-pass LP for optimal BESS dispatch given a load forecast.

    Returns: peak_kW (float), P_ch array, P_dis array, SOC array.
    Returns (None, ...) if scipy not available or infeasible.
    """
    if not HAS_SCIPY:
        return None, None, None, None

    T = len(load_forecast)
    P  = bess["P_rated"]
    E  = bess["E_cap"]
    s0 = bess["SOC_init"]
    smin, smax = bess["SOC_min"], bess["SOC_max"]

    # Decision variables: [P_peak, P_ch_0..T-1, P_dis_0..T-1]
    # Total: 1 + 2T
    n = 1 + 2 * T
    i_peak = 0
    i_ch   = np.arange(1, T + 1)
    i_dis  = np.arange(T + 1, 2 * T + 1)

    # Objective: min P_peak
    c = np.zeros(n); c[i_peak] = 1.0

    # ── Inequality constraints (Ax ≤ b) ──────────────────────────────────
    A_ub_rows, b_ub_rows = [], []

    # (1) Peak constraint: load(t) + P_ch(t) - P_dis(t) ≤ P_peak → rearranged
    for t in range(T):
        row = np.zeros(n)
        row[i_peak]    = -1.0
        row[i_ch[t]]   =  1.0
        row[i_dis[t]]  = -1.0
        A_ub_rows.append(row)
        b_ub_rows.append(-load_forecast[t])

    # (2) SOC bounds via cumulative energy balance
    soc = [s0]
    for t in range(T):
        # SOC(t+1) = SOC(t) - dis·Δt/E + ch·Δt/E
        row_max = np.zeros(n)
        row_min = np.zeros(n)
        for tt in range(t + 1):
            row_max[i_ch[tt]]  = -DT / E
            row_max[i_dis[tt]] =  DT / E
            row_min[i_ch[tt]]  =  DT / E
            row_min[i_dis[tt]] = -DT / E
        A_ub_rows.append(row_max); b_ub_rows.append(smax - s0)
        A_ub_rows.append(row_min); b_ub_rows.append(s0 - smin)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # ── Variable bounds ───────────────────────────────────────────────────
    bounds = [(0, None)]           # P_peak ≥ 0
    bounds += [(0, P)] * T         # P_ch
    bounds += [(0, P)] * T         # P_dis

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return None, None, None, None

    peak_kw = float(res.x[i_peak])
    p_ch    = res.x[i_ch]
    p_dis   = res.x[i_dis]
    soc_arr = [s0]
    s = s0
    for t in range(T):
        s += (p_ch[t] - p_dis[t]) * DT / E
        soc_arr.append(s)

    return peak_kw, p_ch, p_dis, np.array(soc_arr)


def run_case(pred_csv: str, demand_rate_cny: float = 40.0, n_days: int = 30):
    """
    Compare no-BESS, base-strategy, and Re-TFT-strategy peak shaving.

    demand_rate_cny: CNY per kW per month (Chinese demand charge tariff).
    """
    df = pd.read_csv(pred_csv)
    steps_per_day = 96   # 15-min × 96 = 24h
    N = min(len(df), n_days * steps_per_day)
    df = df.iloc[:N].reset_index(drop=True)

    y_true  = df["y_true"].values
    y_base  = df["y_base"].values
    y_final = df["y_final"].values

    peak_true = float(np.max(y_true))
    peak_base_strat, _, _, _ = lp_peak_shaving(y_base)
    peak_ours_strat, _, _, _ = lp_peak_shaving(y_final)

    if peak_base_strat is None:
        print("[ERROR] LP solver unavailable. Install scipy.")
        return

    # Monthly demand charge = peak × rate
    charge_no_bess  = peak_true        * demand_rate_cny
    charge_base     = peak_base_strat  * demand_rate_cny
    charge_ours     = peak_ours_strat  * demand_rate_cny

    saving_base = charge_no_bess - charge_base
    saving_ours = charge_no_bess - charge_ours
    improvement = (saving_ours - saving_base) / max(1e-9, saving_base) * 100.0

    print("\n=== Case A-II: BESS Peak Shaving Results ===")
    print(f"  Demand rate: {demand_rate_cny:.1f} CNY/kW/month")
    print(f"  Evaluation window: {n_days} days ({N} timesteps)\n")
    print(f"  {'Scenario':<20}  {'Peak (kW)':>10}  {'Charge (CNY)':>14}  {'Saving (CNY)':>14}")
    print(f"  {'-'*60}")
    print(f"  {'No BESS':<20}  {peak_true:>10.1f}  {charge_no_bess:>14.1f}  {'0':>14}")
    print(f"  {'Base strategy':<20}  {peak_base_strat:>10.1f}  {charge_base:>14.1f}  {saving_base:>14.1f}")
    print(f"  {'Re-TFT strategy':<20}  {peak_ours_strat:>10.1f}  {charge_ours:>14.1f}  {saving_ours:>14.1f}")
    print(f"\n  Improvement in savings (Re-TFT vs Base): {improvement:+.1f}%")

    return {
        "peak_no_bess_kW":    peak_true,
        "peak_base_strat_kW": peak_base_strat,
        "peak_ours_strat_kW": peak_ours_strat,
        "demand_charge_no_bess_CNY":   charge_no_bess,
        "demand_charge_base_CNY":      charge_base,
        "demand_charge_ours_CNY":      charge_ours,
        "saving_base_CNY":  saving_base,
        "saving_ours_CNY":  saving_ours,
        "improvement_pct":  improvement,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv",     required=True)
    parser.add_argument("--demand_rate",  type=float, default=40.0,
                        help="Demand charge rate [CNY/kW/month]  (default 40)")
    parser.add_argument("--duration_days", type=int,  default=30)
    args = parser.parse_args()
    run_case(args.pred_csv, args.demand_rate, args.duration_days)
