# Bridging Forecast Accuracy and Operational Value: Residual-Enhanced Load Forecasting for Battery Peak Shaving

## Paper

This repository accompanies the revised EPSR manuscript, “Bridging Forecast Accuracy and Operational Value: Residual-Enhanced Load Forecasting for Battery Peak Shaving.” The manuscript evaluates direct-endpoint load forecasting and forecast-driven battery peak shaving.

## Overview and scope

This public repository provides the core Re-TFT model implementation and **minimal paper-aligned examples**. It is **not** the complete archive of training searches, baseline runs, checkpoints, test predictions, evidence freezes, or internal provenance. Running the examples does not reproduce the numerical tables in the manuscript.

## Re-TFT architecture

`models/retft.py` implements a multi-scale dilated CNN and LSTM historical encoder, a TFT-style Base Head with attention and GRNs, and a residual branch that copies and freezes the pretrained CNN/LSTM representation. `models/pipeline.py` exposes the frozen-copy step, validation-only affine calibration, and scalar Base-plus-residual endpoint. Freezing is a design choice; the manuscript does not establish that freezing alone causes the observed improvement. No guard/fallback result supports the revised paper.

## Direct-endpoint task

For each horizon `h ∈ {1, 4, 24, 96}`, the input is the **past 96 rows** and the output is **one scalar target** `y(t+h)`, not a 96-value forecast trajectory. Historic load and weather are encoder inputs. At the target timestamp, the model uses calendar information only: **no future load or meteorological observation** enters the forecast. Belgium is sampled every 15 minutes; H96 is a fixed 24-hour lead. Spain and London are processed hourly/nominally hourly and have dataset-specific timing caveats in the data documentation.

## Data sources

Read [data/README.md](data/README.md) and [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). The manuscript's Spain and London sources have been identified, but exact byte-for-byte acquisition of the private processed benchmarks is not claimed. The London processed series is a fixed-scale household-profile scenario and contains a three-hour timestamp jump caused by missing archived weather rows.

No dataset, checkpoint, prediction archive, or evidence-freeze package is included here.

## Minimal forecasting example

Install compatible Python dependencies from `requirements.txt`, then run:

```bash
python examples/minimal_direct_endpoint.py
```

This **MINIMAL IMPLEMENTATION EXAMPLE** uses synthetic 40-channel history, an eight-value target-calendar vector, randomly initialized Base and residual modules, and an illustrative affine adjustment. It demonstrates the H96 scalar shape and freeze path; it does not report a paper metric or train canonical weights.

## Minimal BESS example

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
