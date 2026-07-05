# M1Q Threshold And Validation-Mode Comparison v0.1

Status: research draft. Not clinical validation.

Cohort rule: biopsy-patient cohort; contralateral rows kept for symmetry; NORMAL mapped to BENIGN; EXCLUDE dropped.

Model: M1Q only.

Metric note: target threshold is selected on train data to target sensitivity
0.95, then applied to held-out patients. Therefore honest test sensitivity is
not forced to be 0.95 and can be lower.

| threshold | mode | patients | cancer_patients | benign_patients | roc_auc_mean | roc_auc_std | sensitivity_target_mean | sensitivity_target_std | specificity_target_mean | specificity_target_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T100 | loovm | 164 | 75 | 89 | 0.605 | 0.000 | 0.747 | 0.000 | 0.438 | 0.000 |
| T100 | patient_80_20_x50 | 164 | 75 | 89 | 0.607 | 0.078 | 0.629 | 0.126 | 0.501 | 0.107 |
| T100 | stratified_5fold | 164 | 75 | 89 | 0.577 | 0.074 | 0.613 | 0.078 | 0.472 | 0.082 |
| T100 | train_all | 164 | 75 | 89 | 0.920 | 0.000 | 0.960 | 0.000 | 0.618 | 0.000 |
| T130 | loovm | 180 | 84 | 96 | 0.588 | 0.000 | 0.690 | 0.000 | 0.438 | 0.000 |
| T130 | patient_80_20_x50 | 180 | 84 | 96 | 0.589 | 0.075 | 0.702 | 0.122 | 0.423 | 0.102 |
| T130 | stratified_5fold | 180 | 84 | 96 | 0.622 | 0.093 | 0.716 | 0.144 | 0.417 | 0.091 |
| T130 | train_all | 180 | 84 | 96 | 0.901 | 0.000 | 0.952 | 0.000 | 0.667 | 0.000 |
| T70 | loovm | 149 | 68 | 81 | 0.626 | 0.000 | 0.735 | 0.000 | 0.481 | 0.000 |
| T70 | patient_80_20_x50 | 149 | 68 | 81 | 0.627 | 0.078 | 0.689 | 0.145 | 0.507 | 0.104 |
| T70 | stratified_5fold | 149 | 68 | 81 | 0.633 | 0.074 | 0.695 | 0.109 | 0.544 | 0.078 |
| T70 | train_all | 149 | 68 | 81 | 0.940 | 0.000 | 0.956 | 0.000 | 0.728 | 0.000 |

Machine-readable table: `docs/modeling/results/m1q_threshold_mode_comparison_v0_1.csv`

## Current Reading

For honest repeated patient-safe `80/20 x50`, T70 gives the best M1Q ROC AUC
and specificity in this run:

```text
T70:  ROC AUC 0.627 +/- 0.078, sensitivity 0.689 +/- 0.145, specificity 0.507 +/- 0.104
T100: ROC AUC 0.607 +/- 0.078, sensitivity 0.629 +/- 0.126, specificity 0.501 +/- 0.107
T130: ROC AUC 0.589 +/- 0.075, sensitivity 0.702 +/- 0.122, specificity 0.423 +/- 0.102
```

For stratified 5-fold, T70 is also best in this run:

```text
T70:  ROC AUC 0.633 +/- 0.074, specificity 0.544 +/- 0.078
T130: ROC AUC 0.622 +/- 0.093, specificity 0.417 +/- 0.091
T100: ROC AUC 0.577 +/- 0.074, specificity 0.472 +/- 0.082
```

Train-all results are optimistic discovery ceilings and should not be used as
validation evidence. They are kept only to show the upper bound when the model
sees all patients during fitting.
