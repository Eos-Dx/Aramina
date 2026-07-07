# M1Q T100 Regularization Experiment v0.1

Status: research draft.

Purpose: tune L2 regularization for the current Aramis M1Q candidate using the
T100 biopsy-patient model-input DataFrame. Regularization is selected on
repeated patient-safe stratified 5-fold validation, not on train-all metrics.

Dataset:

```text
examples/outputs/model_selection_m1q_v0_1/preprocessing/aramis_t100_biopsy_patients_model_input.joblib
```

Selection rule:

```text
primary mode: repeated patient-safe stratified 5-fold x20
target sensitivity: 0.95
penalty: L2 LogisticRegression
C grid: [0.03, 0.1, 0.3, 1.0]
selected C: 0.1
rule: highest K-fold ROC AUC, then smaller C if ROC AUC differs by less than 0.005
```

## K-fold Regularization Grid

| mode | logreg_c | splits | roc_auc_mean | roc_auc_std | sensitivity_target_mean | sensitivity_target_std | specificity_target_mean | specificity_target_std | balanced_accuracy_target_mean | ppv_target_mean | npv_target_mean | threshold_target_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kfold_5x20 | 0.100 | 100 | 0.618 | 0.079 | 0.766 | 0.126 | 0.409 | 0.105 | 0.587 | 0.522 | 0.689 | 0.384 |
| kfold_5x20 | 0.300 | 100 | 0.616 | 0.077 | 0.705 | 0.124 | 0.462 | 0.110 | 0.583 | 0.525 | 0.656 | 0.365 |
| kfold_5x20 | 0.030 | 100 | 0.605 | 0.083 | 0.803 | 0.111 | 0.308 | 0.099 | 0.556 | 0.495 | 0.668 | 0.409 |
| kfold_5x20 | 1.000 | 100 | 0.604 | 0.079 | 0.636 | 0.127 | 0.506 | 0.125 | 0.571 | 0.523 | 0.625 | 0.367 |

## Selected-C Validation Modes

| mode | logreg_c | splits | roc_auc_mean | roc_auc_std | sensitivity_target_mean | sensitivity_target_std | specificity_target_mean | specificity_target_std | balanced_accuracy_target_mean | ppv_target_mean | npv_target_mean | threshold_target_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kfold_5x20 | 0.100 | 100 | 0.618 | 0.079 | 0.766 | 0.126 | 0.409 | 0.105 | 0.587 | 0.522 | 0.689 | 0.384 |
| patient_80_20_x50 | 0.100 | 50 | 0.613 | 0.084 | 0.748 | 0.119 | 0.407 | 0.106 | 0.577 | 0.513 | 0.674 | 0.384 |
| loovm | 0.100 | 1 | 0.622 | 0.000 | 0.840 | 0.000 | 0.326 | 0.000 | 0.583 | 0.512 | 0.707 | 0.328 |
| train_all | 0.100 | 1 | 0.881 | 0.000 | 0.960 | 0.000 | 0.494 | 0.000 | 0.727 | 0.615 | 0.936 | 0.328 |

## Interpretation

The K-fold grid is used to choose regularization. The final train-all artifact is
an optimistic fitted model candidate, not validation evidence. Thresholds are
selected on training folds for the 0.95 sensitivity target and then applied to
held-out patients for split-based modes.
