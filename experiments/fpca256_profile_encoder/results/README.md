# Frozen experiment results

Tracked, readable footprint for the completed FPCA256 experiment.

Runtime lineage is pinned to `pyFAI==2026.5.0` with the exact `integrate1d`
default `("bbox", "csr", "cython")`; the source is
`pyfai_integrate1d_default`. Independently regenerated H5 inputs produced exact
dataframe and metric parity with the pinned artifacts.

- `common/`: matched npt100/npt256 cohort used for direct comparisons.
- `full_npt256/`: complete accepted npt256 cohort.
- `aggregate_summary.csv`: fold mean/SD, repeat-averaged cross-fitted, and train-on-all metrics.
- `fold_metrics.csv`: metrics and threshold for every outer test fold.
- `fold_manifest.csv`: patient-safe train/test assignment for every split.
- `paired_fold_deltas.csv`: split-wise FPCA minus baseline differences.
- `paired_delta_summary.csv`: descriptive paired-delta summaries; quantiles are not confidence intervals.
- `pca_explained_variance.csv`: fold-local and train-on-all component variance.
- `pca_basis_components.csv`: train-on-all mean profile and basis loadings.
- `roc_comparison.png` and `fpca_component_convergence.png`: descriptive visual summaries.

Large executable models, fold PCA objects, and per-case predictions remain generated outputs under `outputs/` and are intentionally excluded from Git.
