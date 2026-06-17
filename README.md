# Re-TFT: Residual-Enhanced Temporal Fusion Transformer

Reproducible code for the paper  
**"Bridging Prediction Accuracy and Operational Value: A Residual-Enhanced
Transformer for Demand Response and Battery Storage Scheduling"**  
submitted to *Applied Energy*.

---

## Quick-start (reproduce all paper results in three commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place Belgium dataset  (see data/README.md for download instructions)
#    Expected path: data/belgium.csv

# 3. Point forecasting — train the base model and Re-TFT (Tables 4-5; feeds Case A)
python train.py --data data/belgium.csv --steps 1 4 24 96

# Point results land in: results/step{1,4,24,96}/
#   test_predictions.csv   — time / y_true / y_base / y_final   (point forecasts)
#   metrics_summary.txt    — RMSE, MAE, MAPE, R^2               (point metrics)

# 4. Interval (P10/P90) forecasting — single-stage direct quantile (Tables 7-8)
python interval_forecasting/fair_compare.py
#   -> results/interval_fair/fair_comparison.csv   (PICP / MPIW / CWC per model & horizon)
```

---

## Repository layout

```
Re-TFT/
├── train.py                        # Point forecasting: base model + Re-TFT
├── requirements.txt
├── data/
│   └── README.md                   # Dataset download instructions
├── models/
│   ├── __init__.py
│   └── retft.py                    # GRN · CNNFrontTFT · CNN2TFTLSTMResidual
├── utils/
│   ├── features.py                 # Feature engineering, sequence builder
│   ├── metrics.py                  # Point + interval metrics
│   ├── training.py                 # Training loop, LR search, scheduler
│   └── calibration.py             # Affine calibration (OLS)
├── case_studies/
│   ├── case_a1_dr_settlement.py    # Case A-I: DR settlement flexibility
│   ├── case_a2_bess_peakshaving.py # Case A-II: LP-based BESS peak shaving
│   └── case_b_risk_mpc.py          # Case B: Risk-aware rolling horizon MPC
├── interval_forecasting/           # P10/P90 interval forecasting (Tables 7-8)
│   ├── fair_compare.py             # Entry point: single-stage direct quantile
│   └── multimodel_base_residual_quantile.py  # Architectures + pinball engine
└── results/                        # Auto-created by the scripts
```

---

## Step-by-step reproduction

### Single-step forecast (step = 1, Table 3 row 1)

```bash
python train.py --data data/belgium.csv --steps 1
```

### Multi-step forecast (steps 4 / 24 / 96, Table 5)

```bash
python train.py --data data/belgium.csv --steps 4 24 96
```

### Ablation study (Table 6 — frozen vs unfrozen backbone)

```bash
python train.py --data data/belgium.csv --steps 1 96 --ablation
```

---

## Case study reproduction

All case studies consume the `test_predictions.csv` produced by `train.py`.

### Case A-I — DR Settlement Flexibility (Section 4.1)

```bash
# Default run (η=0.4, P_base=5th percentile)
python case_studies/case_a1_dr_settlement.py \
    --pred_csv results/step1/test_predictions.csv

# Full sensitivity grid over η × P_base  (Table 16–17)
python case_studies/case_a1_dr_settlement.py \
    --pred_csv results/step1/test_predictions.csv \
    --sensitivity
```

### Case A-II — BESS Peak Shaving (Section 4.2)

```bash
python case_studies/case_a2_bess_peakshaving.py \
    --pred_csv results/step1/test_predictions.csv \
    --demand_rate 40 \
    --duration_days 30
```

### Case B — Risk-Aware Rolling Horizon MPC (Section 4.3)

```bash
# Default run (reproduces Table 10 numbers)
python case_studies/case_b_risk_mpc.py \
    --pred_csv results/step1/test_predictions.csv

# Penalty sensitivity sweep  (addresses Reviewer #2 concern)
python case_studies/case_b_risk_mpc.py \
    --pred_csv results/step1/test_predictions.csv \
    --sensitivity
```

---

## Guard mechanism — clarification for reviewers

**Paper Eq. 9** defines the Guard gate:

```
I_guard = 1  if  ρ(ê_res_val, e_val) > τ_corr  AND  ΔRMSE > δ_gain
          0  otherwise
```

**Benchmark configuration** (`train.py` default): `τ_corr = 0`, `δ_gain = 0`.  
With these values the conditions reduce to:

- "Pearson correlation > 0" → always true for any positively-correlated residual
- "RMSE improvement > 0" → true whenever the residual head reduces error at all

In practice **both conditions are met throughout training**, so `I_guard ≡ 1`
uniformly — the Guard **never triggers** and the residual correction is always
applied. This is **identical to bypassing the Guard entirely**, which is exactly
what Section 2.6.2 states:

> *"For the primary benchmark experiments … this heuristic gate was entirely
> bypassed (I_guard ≡ 1)."*

The Guard code is retained in `train.py` for completeness and is activated
automatically when non-zero thresholds are passed via `--tau_corr` / `--delta_gain`.

**There is no `torch.clamp(min=0)` Guard substitute anywhere in this codebase.**
The actual implementation is a Pearson-correlation + ΔRMSE conditional branch
(see `train.py`, Guard mechanism section).

---

## Datasets & baselines

- **Datasets.** Three load-forecasting datasets are used: the Belgian national
  grid (Elia, primary benchmark), the Spain national grid (Kaggle), and the
  London microgrid (Smart Meters in London). None of the raw/processed CSVs are
  redistributed here; **`data/README.md`** documents how to obtain and assemble
  each one, including the expected column schema and, for Belgium, the SHA-256
  checksum of the exact file used in the paper. The London weather features come
  from archived Dark Sky records (the API was retired in 2023); see the
  reproducibility note in `data/README.md` for an Open-Meteo substitute.

- **Baselines.** The third-party baselines (LSTM, GRU, TCN, Transformer, NBEATS,
  TFT, PatchTST, iTransformer, TimesNet, FEDformer, DLinear) were evaluated with
  the external [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
  under the **same evaluation protocol** as Re-TFT: chronological 6:2:2
  train/val/test split, identical look-back window, and identical forecast
  horizons (steps 1 / 4 / 24 / 96). Re-TFT itself is trained and evaluated by the
  `train.py` script in this repository.

- **Point vs. interval baselines.** The *point*-forecasting baseline numbers
  (Tables 4, 9, 10) come from the external Time-Series-Library / Darts runs
  described above. The *interval*-forecasting comparison (Tables 7-8) is produced
  by the self-contained harness in `interval_forecasting/`, which re-implements
  every architecture with an **identical single-stage direct-quantile head**
  (q10/q50/q90) trained under a multi-quantile pinball loss, so that all models
  are compared under the same probabilistic protocol and CWC definition. In that
  harness the proposed model corresponds to the fully frozen variant
  (`Re-TFT (FullFreeze)`); `CNNFrontTFT` without freezing is the CNN-TFT base.
  Interval forecasting is reported for the Belgium and London settings, the
  scenarios most directly tied to tail-risk and the downstream operational
  decisions; the smoother, highly aggregated Spain grid is used for
  point-forecast transfer only.

## Environment

| Requirement | Tested version |
|-------------|---------------|
| Python      | 3.10          |
| PyTorch     | 2.1.0 (CUDA 11.8) |
| NumPy       | 1.24.3        |
| pandas      | 2.0.3         |
| scikit-learn| 1.3.0         |
| scipy       | 1.11.3        |

GPU is strongly recommended; a single training run takes ~15 min on an
NVIDIA RTX 3090 (24 GB VRAM) for the full 6-year Belgium dataset.

---

## Citation

```bibtex
@article{retft2026,
  title   = {Bridging Prediction Accuracy and Operational Value:
             A Residual-Enhanced Transformer for Demand Response
             and Battery Storage Scheduling},
  author  = {Qi, Yu and Xu, Qingshan and Yang, Yongbiao and Huang, Yixuan},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Under review},
}
```
