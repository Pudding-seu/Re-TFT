# Case B — 3×3 Penalty-Rate Sensitivity Sweep

**Single source of truth:** manuscript Table "Quantitative Performance Comparison" (Case B).

**Fixed physical outcomes (penalty-independent):**

| Quantity | Deterministic MPC | Risk-aware MPC |
|---|---|---|
| Voltage violations (V < 0.95 p.u.) | **15 events** | **0 events** |
| Contract breach (3.0 MW limit) | 0.15 MW (peak 3.15 MW) | none (peak 2.60 MW) |
| Operational cost | \$5,953 | \$5,867 (−1.4%) |

**Cost model:** `cost_det(D, V) = 5,953 + 15·V + 0.15263·D`; `cost_risk = 5,867` (zero violations in all scenarios).
Reference cell (D = \$100k/MW, V = \$5k/event) reproduces the manuscript exactly: \$96,216 → \$5,867 = **93.90%**.

## Reduction in Total Generalized Cost (%)

| Demand penalty \ Voltage penalty | \$500/event | \$5,000/event | \$20,000/event |
|---|---:|---:|---:|
| **\$10,000 / MW**  | 60.83 | 92.89 | 98.09 |
| **\$100,000 / MW** | 79.57 | **93.90** ★ | 98.17 |
| **\$500,000 / MW** | 93.46 | 96.27 | 98.47 |

★ Manuscript reference scenario.

## Full Cost Table

| D ($/MW) | V ($/event) | Cost Det. ($) | Cost Risk ($) | Reduction (%) |
|---:|---:|---:|---:|---:|
| 10,000 | 500 | 14,979.30 | 5,867.00 | 60.83 |
| 10,000 | 5,000 | 82,479.30 | 5,867.00 | 92.89 |
| 10,000 | 20,000 | 307,479.30 | 5,867.00 | 98.09 |
| 100,000 | 500 | 28,716.00 | 5,867.00 | 79.57 |
| 100,000 | 5,000 | 96,216.00 | 5,867.00 | 93.90 |
| 100,000 | 20,000 | 321,216.00 | 5,867.00 | 98.17 |
| 500,000 | 500 | 89,768.00 | 5,867.00 | 93.46 |
| 500,000 | 5,000 | 157,268.00 | 5,867.00 | 96.27 |
| 500,000 | 20,000 | 382,268.00 | 5,867.00 | 98.47 |

## Findings

1. **Penalty-invariant violation elimination.** Penalty rates appear only in the cost objective, never in the constraint set (voltage bounds, thermal limits, contract cap, BESS envelope) or the dynamic reserve rule. The dispatch trajectories — and therefore the violation counts (15 → 0 voltage events; 0.15 MW breach → none) — are identical across all nine scenarios.

2. **Monotone economic viability with a 60.8% floor.** The reduction is monotone in both penalty dimensions. At the most lenient corner, the advantage decomposes into \$9,026 of avoided penalties and \$86 of operational savings — penalties dominate by construction in a constraint-compliance scenario, but the sign of the advantage and the negligible cost of reserve conservatism (−1.4% operational cost, i.e., no premium paid for safety) hold everywhere.

3. **Reference scenario is interior, not a corner.** 93.90% lies strictly between the sweep minimum (60.83%) and maximum (98.47%).
