# Results

## Decision

Keep `npt=100` with `bbox / csr / cython` in the product pipeline.

Increasing radial resolution did not improve patient-safe discrimination. It
increased in-sample separation and reached perfect train-on-all classification
at `npt=512`, while held-out ROC AUC remained approximately `0.64–0.66`. This is
overfitting, not improved generalization.

## Controlled common cohort

All variants used the same `803` measurements, `164` patients, `164`
target-breast cases, split IDs, and random seed.

| Variant | q step, nm^-1 | ROC AUC | Sensitivity | Specificity | Train-all ROC | Train-all specificity |
|---|---:|---:|---:|---:|---:|---:|
| `100/bbox` | 0.210 | **0.679 ± 0.076** | **0.826 ± 0.111** | 0.400 ± 0.120 | 0.876 | 0.564 |
| `150/bbox` | 0.140 | 0.641 ± 0.077 | 0.748 ± 0.121 | 0.463 ± 0.115 | 0.920 | 0.574 |
| `200/bbox` | 0.105 | 0.633 ± 0.083 | 0.657 ± 0.136 | 0.533 ± 0.106 | 0.955 | 0.713 |
| `250/bbox` | 0.084 | 0.617 ± 0.080 | 0.613 ± 0.133 | 0.557 ± 0.117 | 0.977 | 0.809 |
| `256/bbox` | 0.082 | 0.636 ± 0.079 | 0.585 ± 0.134 | 0.614 ± 0.116 | 0.980 | 0.915 |
| `512/bbox` | 0.041 | 0.636 ± 0.086 | 0.471 ± 0.133 | 0.718 ± 0.100 | 1.000 | 1.000 |
| `512/no` | 0.041 | 0.657 ± 0.080 | 0.437 ± 0.148 | **0.764 ± 0.095** | 1.000 | 1.000 |
| `512/full` | 0.041 | 0.642 ± 0.086 | 0.465 ± 0.134 | 0.733 ± 0.097 | 1.000 | 1.000 |

The apparent specificity increase is paired with a large sensitivity loss.
None of the higher-resolution variants preserved the intended high-sensitivity
behavior on held-out patients.

## End-to-end cohorts

This analysis includes npt-dependent QC retention.

| Variant | Measurements | Target cases | ROC AUC | Sensitivity | Specificity | Train-all ROC | Train-all specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100/bbox` | 893 | 175 | **0.645 ± 0.069** | **0.818 ± 0.099** | 0.376 ± 0.133 | 0.865 | 0.495 |
| `150/bbox` | 886 | 173 | 0.631 ± 0.081 | 0.695 ± 0.121 | 0.485 ± 0.124 | 0.926 | 0.670 |
| `200/bbox` | 883 | 172 | 0.623 ± 0.084 | 0.683 ± 0.128 | 0.513 ± 0.118 | 0.949 | 0.701 |
| `250/bbox` | 877 | 171 | 0.593 ± 0.094 | 0.593 ± 0.146 | 0.524 ± 0.138 | 0.973 | 0.804 |
| `256/bbox` | 876 | 171 | 0.627 ± 0.085 | 0.590 ± 0.133 | 0.603 ± 0.122 | 0.978 | 0.825 |
| `512/bbox` | 844 | 168 | 0.649 ± 0.093 | 0.483 ± 0.136 | 0.733 ± 0.114 | 1.000 | 1.000 |
| `512/no` | 805 | 164 | 0.658 ± 0.079 | 0.440 ± 0.147 | **0.769 ± 0.096** | 1.000 | 1.000 |
| `512/full` | 838 | 168 | 0.649 ± 0.095 | 0.468 ± 0.137 | 0.750 ± 0.109 | 1.000 | 1.000 |

The `100/bbox` row reproduces the frozen product evaluation and train-on-all
metrics.

## Interpretation

- `npt` does not need to divide `512` or `768`.
- `250` and `256` provide no evidence of a power-of-two or divisibility benefit.
- Higher `npt` lowers calculated SNR because fewer detector counts contribute
  to each radial bin. Therefore changing `npt` also changes QC retention.
- On the fixed common cohort, higher `npt` still does not improve ROC AUC.
- `bbox` and `full` give similar held-out ROC at `npt=512`.
- `no` splitting increases specificity but loses most of the required
  sensitivity.
- Perfect `512` train-on-all metrics alongside modest held-out ROC are direct
  evidence that the larger LR1 profile can memorize this small cohort.

These are research-draft results, not independent clinical validation.
