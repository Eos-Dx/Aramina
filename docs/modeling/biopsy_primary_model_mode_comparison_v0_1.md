# Biopsy Primary Model Mode Comparison v0.1

Status: research draft. Not clinical validation.

Dataset:

```text
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
```

Training target:

```text
inferred target breast = biopsied breast
```

Models:

```text
M0: target-breast profile evidence only
M1: M0 + cosine symmetry context
M2: M1 + age and age_available
```

`M2` is an age audit/comparison branch, not the current primary v0.1-beta
candidate.

Metrics below use the threshold selected on training data for target
sensitivity 0.95. Therefore test sensitivity is not forced to equal 0.95.

| mode | model | ROC AUC | sensitivity | specificity | PPV | NPV |
|---|---|---:|---:|---:|---:|---:|
| 70/30 x50 | M0 | 0.574 +/- 0.061 | 0.738 +/- 0.122 | 0.359 +/- 0.141 | 0.501 | 0.619 |
| 70/30 x50 | M1 | 0.554 +/- 0.062 | 0.710 +/- 0.114 | 0.354 +/- 0.138 | 0.490 | 0.584 |
| 70/30 x50 | M2 | 0.615 +/- 0.066 | 0.743 +/- 0.124 | 0.395 +/- 0.123 | 0.516 | 0.654 |
| stratified 5-fold | M0 | 0.614 +/- 0.086 | 0.751 +/- 0.154 | 0.366 +/- 0.078 | 0.505 | 0.654 |
| stratified 5-fold | M1 | 0.599 +/- 0.099 | 0.728 +/- 0.150 | 0.407 +/- 0.055 | 0.513 | 0.653 |
| stratified 5-fold | M2 | 0.660 +/- 0.092 | 0.811 +/- 0.107 | 0.386 +/- 0.094 | 0.537 | 0.711 |
| LOOVM | M0 | 0.593 | 0.833 | 0.292 | 0.507 | 0.667 |
| LOOVM | M1 | 0.576 | 0.738 | 0.292 | 0.477 | 0.560 |
| LOOVM | M2 | 0.639 | 0.821 | 0.427 | 0.556 | 0.732 |
| train-all | M0 | 0.851 | 0.952 | 0.490 | 0.620 | 0.922 |
| train-all | M1 | 0.858 | 0.952 | 0.562 | 0.656 | 0.931 |
| train-all | M2 | 0.871 | 0.952 | 0.552 | 0.650 | 0.930 |

Interpretation:

```text
70/30 and k-fold are the most useful generalization checks.
LOOVM is useful for small-data sensitivity but is high variance by design.
train-all is an optimistic in-sample ceiling only.
M1 does not currently improve M0 in honest split modes.
M2 is consistently stronger, but the gain is partly age-driven.
```

Machine-readable table:

```text
docs/modeling/results/biopsy_primary_model_mode_comparison_v0_1.csv
```
