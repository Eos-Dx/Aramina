# T130 Paired Held-out Comparison

Three T100-trained procedures were scored on the same locked T130 case manifest: 22 target-breast cases from 17 patient-disjoint patients (11 CANCER, 11 BENIGN).

| procedure | roc_auc | pr_auc | sensitivity | specificity | balanced_accuracy | ppv | npv | log_loss | brier_score | true_positives | true_negatives | false_positives | false_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_current_product | 0.6281 | 0.5989 | 0.7273 | 0.5455 | 0.6364 | 0.6154 | 0.6667 | 0.9059 | 0.2778 | 8 | 6 | 5 | 3 |
| recalibrated_joint_same_fitted_lr1 | 0.5950 | 0.5931 | 0.8182 | 0.4545 | 0.6364 | 0.6000 | 0.7143 | 0.7932 | 0.2704 | 9 | 5 | 6 | 2 |
| recalibrated_joint_lr1_oof | 0.5950 | 0.5931 | 0.8182 | 0.2727 | 0.5455 | 0.5294 | 0.6000 | 0.7844 | 0.2679 | 9 | 3 | 8 | 2 |

## Interpretation

- Relative to the frozen product, same-data LR2 training changes one CANCER FN to TP but also one BENIGN TN to FP: sensitivity is +9.09 percentage points, specificity is -9.09 points, and ROC AUC is -0.0331.
- The OOF-trained LR2 has the same T130 ROC AUC and sensitivity as same-data LR2, but two fewer true negatives at its locked lower threshold (0.18564 versus 0.25747).
- There is no clear overall winner on this subset. The frozen product retains higher ROC AUC and specificity; both recalibrated procedures have lower Brier score and log loss, while same-data LR2 shifts the thresholded error trade-off toward sensitivity.
- Research-only same-source check. It is not independent external validation.
- T130 uses looser calibration QC than T100 and selects MRI-or-biopsy cases; MRI denotes that MRI was performed, not an MRI outcome.
- Five patients contribute two target breasts, so 22 cases are not 22 independent patients.
- Each CANCER or BENIGN case changes sensitivity or specificity by 1/11 = 9.09 percentage points. `metrics.csv` reports descriptive case-level Wilson 95% intervals; they do not account for paired breasts within five patients.
- The deliberately balanced 11/11 class composition is not clinical prevalence. PR AUC, PPV, and NPV are descriptive for this subset and must not be projected to clinical workflow.
- The frozen product comparator is scored directly from its immutable model artifact. The current T100 input joblib SHA is recorded separately because it differs from the SHA stored in that artifact.
