# Third-party notices

The repository's MIT License applies only to author-owned source code. The
projects below are runtime dependencies; their source code is not copied,
adapted, vendored or redistributed in this repository. Each dependency remains
subject to its own license.

| Project | URL | Role | License | Inclusion status |
|---|---|---|---|---|
| PyTorch | https://github.com/pytorch/pytorch | Neural-network layers and tensor operations | BSD-style; consult the upstream distribution's complete license files | Imported dependency only |
| NumPy | https://github.com/numpy/numpy | Numerical arrays and calculations | BSD-3-Clause for the main project; distributions may contain separately licensed components | Imported dependency only |
| pandas | https://github.com/pandas-dev/pandas | Timestamp and tabular utilities used by the minimal example and feature helpers | BSD-3-Clause for the main project; consult bundled-license files | Imported dependency only |
| SciPy | https://github.com/scipy/scipy | Linear programming for the BESS example | BSD-3-Clause for the main project; consult bundled-license files | Imported dependency only |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | Forecast metric helpers | BSD-3-Clause | Imported dependency only |

The public implementation uses ideas described in the Temporal Fusion
Transformer literature, but it does not redistribute source code from PyTorch
Forecasting, Darts, THUML Time-Series-Library/Autoformer, or an N-BEATS
implementation. Those projects are not runtime dependencies of this minimal
repository.

Third-party datasets are not software dependencies and are not covered by the
repository license. See `data/README.md` and `docs/DATA_PROVENANCE.md` for the
dataset-specific provider terms and redistribution boundaries.
