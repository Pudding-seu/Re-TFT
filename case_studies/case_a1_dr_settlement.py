# -*- coding: utf-8 -*-
"""
Case A-I: DR Settlement — Enhancing Resource Visibility
========================================================
Paper Section 4.1 (Case A-I)

Problem
-------
In demand response (DR) markets the baseline forecast determines the measurable
flexible portion of load.  When a standard MSE-optimised model exhibits systematic
peak underestimation, the inferred flexibility becomes artificially small, making
real flexibility "invisible" to operators.

Flexibility estimator (Eq. 14 in paper)
----------------------------------------
    P_adj(t) = η · max(0, ŷ(t) − P_base)
    E_adj    = Σ_t P_adj(t) · Δt

    where:
      ŷ(t)   : predicted load at time t  [kW]
      P_base : rigid base load = 5th-percentile of historical load  [kW]
      η      : flexibility factor (fraction of controllable load)
      Δt     : time-step duration = 0.25 h  (15-min resolution)

Evaluation metric
-----------------
    Error Recovery Rate = (E_missed_base − E_missed_ours) / E_missed_base × 100%

    where E_missed_m = Σ_t |P_adj_true(t) − P_adj_m(t)| · Δt

Usage
-----
    python case_studies/case_a1_dr_settlement.py \
        --pred_csv results/step1/test_predictions.csv \
        --eta 0.4 \
        --pbase_pct 5

    # Sensitivity analysis over η and P_base:
    python case_studies/case_a1_dr_settlement.py \
        --pred_csv results/step1/test_predictions.csv \
        --sensitivity
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DT = 0.25  # 15-min timestep in hours


def adjustable_potential(y_hat: np.ndarray, p_base: float, eta: float) -> np.ndarray:
    """Compute adjustable potential P_adj(t) for each timestep."""
    return eta * np.maximum(0.0, y_hat - p_base)


def energy_adj(p_adj: np.ndarray) -> float:
    """Total adjustable energy [MWh] over the evaluation window."""
    return float(np.sum(p_adj) * DT / 1e3)   # kW × h → MWh


def missed_potential(p_adj_true: np.ndarray, p_adj_pred: np.ndarray) -> float:
    """Cumulative |error| in adjustable potential estimation [MWh]."""
    return float(np.sum(np.abs(p_adj_true - p_adj_pred)) * DT / 1e3)


def evaluate(df: pd.DataFrame, eta: float, pbase_pct: int):
    """
    Evaluate DR settlement accuracy for both Base and Re-TFT models.

    Parameters
    ----------
    df : DataFrame with columns [y_true, y_base, y_final]
    eta       : flexibility factor ∈ (0, 1]
    pbase_pct : percentile of y_true used as P_base  (paper uses 5th)
    """
    y_true  = df["y_true"].values
    y_base  = df["y_base"].values
    y_final = df["y_final"].values

    p_base = float(np.percentile(y_true, pbase_pct))

    p_adj_true  = adjustable_potential(y_true,  p_base, eta)
    p_adj_base  = adjustable_potential(y_base,  p_base, eta)
    p_adj_ours  = adjustable_potential(y_final, p_base, eta)

    e_true = energy_adj(p_adj_true)
    e_base = energy_adj(p_adj_base)
    e_ours = energy_adj(p_adj_ours)

    miss_base = missed_potential(p_adj_true, p_adj_base)
    miss_ours = missed_potential(p_adj_true, p_adj_ours)

    recovery_rate = (miss_base - miss_ours) / max(1e-9, miss_base) * 100.0
    missed_ratio_base = miss_base / max(1e-9, energy_adj(p_adj_true)) * 100.0
    missed_ratio_ours = miss_ours / max(1e-9, energy_adj(p_adj_true)) * 100.0

    return {
        "eta":              eta,
        "P_base_pct":       pbase_pct,
        "P_base_kW":        p_base,
        "E_true_MWh":       e_true,
        "E_base_MWh":       e_base,
        "E_ours_MWh":       e_ours,
        "Miss_base_MWh":    miss_base,
        "Miss_ours_MWh":    miss_ours,
        "Missed_ratio_base(%)": missed_ratio_base,
        "Missed_ratio_ours(%)": missed_ratio_ours,
        "Error_Recovery_Rate(%)": recovery_rate,
    }


def sensitivity_analysis(df: pd.DataFrame):
    """
    Reproduce Table 16–17 (paper Section 4.3.1):
    Grid search over η ∈ {0.2, 0.4, 0.6, 0.8} and P_base ∈ {1,5,10,15}th pct.
    """
    rows = []
    for eta in [0.2, 0.4, 0.6, 0.8]:
        for pct in [1, 5, 10, 15]:
            r = evaluate(df, eta=eta, pbase_pct=pct)
            rows.append(r)
    results = pd.DataFrame(rows)
    print("\n=== Sensitivity: Error Recovery Rate (%) ===")
    pivot = results.pivot_table(
        values="Error_Recovery_Rate(%)", index="P_base_pct", columns="eta"
    )
    print(pivot.to_string())
    print("\n(Note: Recovery Rate is invariant to η, as η cancels in the ratio.)")
    print("      Recovery Rate increases as P_base threshold focuses on fat-tail peaks.")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv",   required=True,
                        help="test_predictions.csv from train.py")
    parser.add_argument("--eta",        type=float, default=0.4,
                        help="Flexibility factor (default 0.4)")
    parser.add_argument("--pbase_pct",  type=int,   default=5,
                        help="Percentile for P_base (default 5)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run full η × P_base sensitivity grid")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.pred_csv)
    assert {"y_true", "y_base", "y_final"}.issubset(df.columns), \
        "CSV must have columns: y_true, y_base, y_final"

    if args.sensitivity:
        results = sensitivity_analysis(df)
        out = args.out or str(Path(args.pred_csv).parent / "case_a1_sensitivity.csv")
        results.to_csv(out, index=False)
        print(f"\nSaved: {out}")
    else:
        r = evaluate(df, eta=args.eta, pbase_pct=args.pbase_pct)
        print("\n=== Case A-I: DR Settlement Results ===")
        for k, v in r.items():
            print(f"  {k:35s}: {v:.4f}" if isinstance(v, float) else f"  {k:35s}: {v}")
        out = args.out or str(Path(args.pred_csv).parent / "case_a1_results.csv")
        pd.DataFrame([r]).to_csv(out, index=False)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
