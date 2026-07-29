# Results

## Decision

Keep `npt=100` with `bbox / csr / cython` in the product pipeline. This
experiment does not change the frozen product configuration.

Increasing radial resolution did not improve patient-safe discrimination. It
increased in-sample separation and reached perfect train-on-all classification
at `npt=512`, while held-out ROC AUC remained approximately `0.64–0.66`. This is
overfitting, not improved generalization.

The low-resolution sweep identified `npt=60 / bbox` as a research candidate,
not a promoted product setting. Relative to `npt=100 / bbox`, its paired mean
held-out ROC AUC difference was `+0.019` and its sensitivity difference was
`+0.042`; specificity decreased by `-0.043`. ROC AUC was higher in `67/100`
paired folds. These results were obtained while comparing 15 preprocessing
variants on one small cohort, so they are insufficient to select a new product
configuration. Confirmation on new data and a separately specified comparison
are required.

At the fixed product resolution (`npt=100`), changing only the pyFAI pixel
splitting rule did not change held-out performance materially. `no`, `bbox`,
and `full` gave ROC AUC `0.681`, `0.679`, and `0.676`, respectively, on the
same held-out cases. `bbox` remains the frozen product method: it retains the
existing validated preprocessing definition without a measurable performance
disadvantage.

## Controlled common cohort

All variants used the same `802` measurements, `164` patients, `164`
target-breast cases, split IDs, and random seed.

| Variant | q step, nm^-1 | ROC AUC | Sensitivity | Specificity | Train-all ROC | Train-all specificity |
|---|---:|---:|---:|---:|---:|---:|
| `50/bbox` | 0.420 | 0.678 ± 0.079 | **0.868 ± 0.106** | 0.308 ± 0.118 | 0.813 | 0.436 |
| `60/bbox` | 0.350 | **0.698 ± 0.079** | **0.868 ± 0.101** | 0.357 ± 0.127 | 0.831 | 0.426 |
| `70/bbox` | 0.300 | 0.686 ± 0.076 | 0.851 ± 0.102 | 0.381 ± 0.123 | 0.843 | 0.521 |
| `80/bbox` | 0.263 | 0.675 ± 0.081 | 0.840 ± 0.113 | 0.359 ± 0.117 | 0.852 | 0.532 |
| `90/bbox` | 0.233 | 0.661 ± 0.079 | 0.808 ± 0.120 | 0.395 ± 0.120 | 0.862 | 0.511 |
| `100/bbox` | 0.210 | **0.679 ± 0.076** | **0.826 ± 0.111** | 0.400 ± 0.120 | 0.876 | 0.564 |
| `100/no` | 0.210 | **0.681 ± 0.076** | 0.823 ± 0.117 | 0.395 ± 0.125 | 0.877 | 0.574 |
| `100/full` | 0.210 | 0.676 ± 0.077 | 0.826 ± 0.114 | 0.398 ± 0.126 | 0.877 | 0.553 |
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
| `50/bbox` | 897 | 175 | 0.655 ± 0.073 | **0.850 ± 0.103** | 0.314 ± 0.099 | 0.791 | 0.384 |
| `60/bbox` | 896 | 175 | 0.647 ± 0.075 | **0.844 ± 0.095** | 0.295 ± 0.100 | 0.795 | 0.273 |
| `70/bbox` | 895 | 175 | **0.667 ± 0.072** | 0.819 ± 0.100 | 0.400 ± 0.116 | 0.834 | 0.384 |
| `80/bbox` | 896 | 175 | 0.659 ± 0.071 | 0.818 ± 0.102 | 0.378 ± 0.115 | 0.850 | 0.374 |
| `90/bbox` | 893 | 175 | 0.636 ± 0.075 | 0.803 ± 0.107 | 0.379 ± 0.112 | 0.848 | 0.364 |
| `100/bbox` | 893 | 175 | **0.645 ± 0.069** | **0.818 ± 0.099** | 0.376 ± 0.133 | 0.865 | 0.495 |
| `100/no` | 891 | 175 | **0.646 ± 0.070** | 0.816 ± 0.106 | 0.380 ± 0.136 | 0.868 | 0.505 |
| `100/full` | 893 | 175 | 0.640 ± 0.070 | 0.815 ± 0.105 | 0.367 ± 0.134 | 0.871 | 0.374 |
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
- `npt=60 / bbox` had the highest mean ROC AUC in the controlled sweep, but
  it traded specificity for sensitivity and was selected after examining many
  variants. Treat it as a research candidate, not evidence for a product
  change.
- `250` and `256` provide no evidence of a power-of-two or divisibility benefit.
- Higher `npt` lowers calculated SNR because fewer detector counts contribute
  to each radial bin. Therefore changing `npt` also changes QC retention.
- On the fixed common cohort, higher `npt` still does not improve ROC AUC.
- At `npt=100`, `no`, `bbox`, and `full` splitting are indistinguishable at
  this cohort size; no performance rationale supports changing the product
  `bbox` method.
- `bbox` and `full` give similar held-out ROC at `npt=512`.
- `no` splitting increases specificity but loses most of the required
  sensitivity.
- Perfect `512` train-on-all metrics alongside modest held-out ROC are direct
  evidence that the larger LR1 profile can memorize this small cohort.

These are research-draft results, not independent clinical validation.
