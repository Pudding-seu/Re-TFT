# Bridging Forecast Accuracy and Operational Value: Residual-Enhanced Load Forecasting for Battery Peak Shaving

## Core methodological contribution

Re-TFT combines a pretrained multi-scale CNN–LSTM forecaster with an independently trained residual correction stage. The second stage reuses a frozen copy of the pretrained encoder, followed by validation-calibrated additive refinement.

1. **Base prediction:** train a multi-scale dilated CNN–LSTM forecaster with a TFT-style GRN and attention head to predict one direct scalar endpoint from 96 historical rows and a target-time calendar vector. The paper configuration uses 12 output channels per convolution branch.
2. **Residual refinement:** form signed residual targets as truth minus the completed Base prediction in original load units. Copy the pretrained CNN, linear projection and LSTM into a new residual branch and freeze those encoder parameters. Train its independent calendar projection, GRN and scalar output head; Base and Residual GRN parameters are not shared.
3. **Validation calibration:** fit an affine residual adjustment on validation data only, clip its slope to `[-3, 3]`, and hold its coefficients fixed during test prediction. In original units, the final scalar forecast is `Base + a * raw_residual + b`.

This is a two-stage forecasting method with pretrained representation reuse and independent residual learning. It uses a TFT-style head rather than the full original TFT architecture. Freezing is a design choice; the reported comparisons do not isolate its causal effect.

## Paper

This repository accompanies the revised EPSR manuscript, “Bridging Forecast Accuracy and Operational Value: Residual-Enhanced Load Forecasting for Battery Peak Shaving.” The manuscript evaluates direct-endpoint load forecasting and forecast-driven battery peak shaving.

## Overview and scope

This public repository provides the core Re-TFT model implementation and **minimal paper-aligned examples**. It is **not** the complete archive of training searches, baseline runs, checkpoints, test predictions, evidence freezes, or internal provenance. Running the examples does not reproduce the numerical tables in the manuscript.

## Re-TFT architecture

`models/retft.py` implements a multi-scale dilated CNN and LSTM historical encoder, a TFT-style Base Head with attention and GRNs, and a residual branch that copies and freezes the pretrained CNN/LSTM representation. `models/pipeline.py` exposes the frozen-copy step, validation-only affine calibration, and scalar Base-plus-residual endpoint. Freezing is a design choice; the manuscript does not establish that freezing alone causes the observed improvement. No guard/fallback result supports the revised paper.

## Direct-endpoint task

For each horizon `h ∈ {1, 4, 24, 96}`, the input is the **past 96 rows** and the output is **one scalar target** `y(t+h)`, not a 96-value forecast trajectory. Historic load and weather are encoder inputs. The target-time input is calendar-only: target-time load and meteorological values are not supplied to the head. This input boundary alone does not establish the time availability of every derived historical covariate. Belgium is sampled every 15 minutes; H96 is a fixed 24-hour lead. Spain and London are processed hourly/nominally hourly and have dataset-specific timing caveats in the data documentation.

## Data sources

Read [data/README.md](data/README.md) and [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). The manuscript's Spain and London sources have been identified, but exact byte-for-byte acquisition of the private processed benchmarks is not claimed. The London processed series is a fixed-scale household-profile scenario and contains a three-hour timestamp jump caused by missing archived weather rows. Its retained `mintem` and `maxtem` are actual full-day temperature summaries backfilled to hourly rows; for origins within that day, these retrospective covariates can include subsequent observations. The London benchmark must be interpreted with that information boundary.

No dataset, checkpoint, prediction archive, or evidence-freeze package is included here.

## Minimal forecasting example

Install compatible Python dependencies from `requirements.txt`, then run:

```bash
python examples/minimal_direct_endpoint.py
```

This **MINIMAL IMPLEMENTATION EXAMPLE** uses synthetic 40-channel history, an eight-value target-calendar vector, randomly initialized Base and residual modules, and an illustrative affine adjustment. It demonstrates the H96 scalar shape and freeze path; it does not report a paper metric or train canonical weights.

## Minimal two-stage workflow example

```bash
python examples/minimal_two_stage_workflow.py
```

This CPU-only synthetic example uses the repository's actual networks and computation interfaces. It trains a Base, constructs original-unit residual targets, copies and freezes the trained encoder, trains an independent residual head, fits an affine adjustment on a separate validation partition, and prints Base and Re-TFT predictions on a separate synthetic test partition. The short training budget demonstrates the method and does not guarantee improvement. It does not read research data, publish checkpoints or prediction archives, or reproduce manuscript tables.

## Minimal BESS example

Battery peak shaving is a downstream operational-value evaluation, not a neural-network module or decision-aware training. In the paper's evaluation, the 96 daily entries are assembled from target-aligned scalar H96 forecasts issued at 96 distinct origins; they are not a 96-step trajectory issued at one origin.

```bash
python examples/minimal_bess.py
```

`bess/peak_shaving.py` optimizes a 96-point daily forecast using the manuscript's 1,000-kW/4,000-kWh battery, 5–95% SOC bounds, 50% initial/terminal SOC, 0.95 efficiencies, 15-minute steps, and two-stage HiGHS LP. The schedule is fixed before scoring it on Ground Truth. The example uses synthetic profiles, **not** the paper's private 71-day H96 forecast archive. Belgium-to-facility scaling in the paper is a shape-preserving scenario assumption (1 kW/MW), not a physical conversion or measured facility demand.

## Cross-dataset evaluation scope

The manuscript also reports Spain and London deterministic direct-endpoint benchmarks. Source/provenance information is documented to the extent supported by retained experiment records; this repository does not supply those processed datasets, baseline training trees, or a full cross-dataset reproduction workflow.

## Legacy code

[legacy/](legacy/README.md) records the publication boundary for earlier Applied Energy-era scripts. Legacy scripts from earlier research versions are not included in the current EPSR-supported snapshot. Some remain accessible through the repository's public Git history, but they are not part of the workflow or experimental evidence supporting the present manuscript.

## Citation

Cite the revised EPSR manuscript by its final bibliographic record when available. Until publication, use the repository-oriented metadata in [CITATION.cff](CITATION.cff); no DOI or acceptance claim is made here.

## License and data

Author-owned source code is released under the [MIT License](LICENSE). Third-party libraries retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Third-party datasets retain the terms of their original providers, are not covered by the repository software license, and are not redistributed here; see [data/README.md](data/README.md).
