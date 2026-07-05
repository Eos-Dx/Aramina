# Patient Model Structures v0.2

Status: research draft. Not clinical validation.

## Goal

Aramis is a breast XRD decision-support prototype for a radiologist-facing
biopsy-risk question. In the current product concept, the clinician supplies
the suspicious target breast. The model should estimate `p_cancer` for that
target breast, then use same-patient breast symmetry as supporting context.

This experiment compares the current patient-level aggregation design before
the target-breast product route is finalized.

## Current Decisions

The B structure is dropped for now. It used three logistic regressors:

```text
profile -> LR1 p_cancer
symmetry features -> LR p_cancer
profile p_cancer + symmetry p_cancer (+ age) -> final LR
```

It did not add useful performance over the simpler A structure and made the
interpretation less clear. The active structure is:

```text
profile -> LR1 p_cancer
LR1 p_cancer + raw cosine symmetry features (+ optional age) -> final LR
```

Age is kept only as an explicit audit branch. It improves performance strongly,
but it can also dominate the biological XRD signal. Therefore every model table
must show `AGE_ONLY` next to profile/symmetry models.

## Datasets

Both datasets are built from the same broad monochromaticity-filtered pool:

```text
wide source:
  examples/outputs/threshold_grid_patient_cohorts/wide_pools/aramis_wide_t130.joblib

wide rows / patients / specimens:
  1222 / 219 / 430
```

Label mapping:

```text
CANCER, ATYPICAL, PRE_CANCEROUS -> CANCER
BENIGN, NORMAL -> BENIGN
NA and unlabeled rows are not used for model labels
```

`all_patients` uses all labelled patients:

```text
patients: 178
cancer / non-cancer patients: 72 / 106
```

`biopsy_patients` includes patients with at least one biopsy row. LR1 trains
only on biopsy-confirmed specimen measurements, while symmetry features use the
full patient context, including contralateral breast rows:

```text
final patients: 153
cancer / non-cancer patients: 72 / 81
LR1 biopsy rows / patients / specimens: 546 / 180 / 193
LR1 biopsy labels: BENIGN=298, CANCER=248
```

## Protocols

`honest_70_30_train_threshold` is the main validation-style check. Patients are
split 70/30 for 50 repeats. Measurements from the same patient never appear in
both train and test. The decision threshold is selected on train scores and
then applied to test scores.

`oracle_70_30_target95` is exploratory only. It selects the threshold on test
labels to inspect the best possible specificity at about 95% sensitivity. It is
not a validation estimate.

`patient_loocv_pooled` holds out one patient at a time and pools predictions
before selecting the reporting threshold.

`train_all_discovery` trains and evaluates on the same patients. It is an
optimistic discovery upper bound, not validation.

## Results

`R` is ROC AUC, `S` is sensitivity, `Sp` is specificity.

These tables are historical model-development results. After the 2026-07-04
switch to LR1 logit-average aggregation, rerun the model grid before treating
the numbers below as current.

### Honest 70/30 Train Threshold

| dataset | model | R | S | Sp |
| --- | --- | --- | --- | --- |
| all_patients | AGE_ONLY | 0.698 +/- 0.051 | 0.953 +/- 0.051 | 0.251 +/- 0.062 |
| all_patients | M0 profile LR1 | 0.574 +/- 0.071 | 0.780 +/- 0.101 | 0.325 +/- 0.102 |
| all_patients | M1 profile + symmetry | 0.602 +/- 0.067 | 0.724 +/- 0.118 | 0.399 +/- 0.115 |
| all_patients | M2 profile + symmetry + age | 0.672 +/- 0.063 | 0.782 +/- 0.100 | 0.424 +/- 0.115 |
| biopsy_patients | AGE_ONLY | 0.734 +/- 0.057 | 0.953 +/- 0.050 | 0.304 +/- 0.077 |
| biopsy_patients | M0 profile LR1 | 0.563 +/- 0.068 | 0.708 +/- 0.114 | 0.382 +/- 0.115 |
| biopsy_patients | M1 profile + symmetry | 0.578 +/- 0.063 | 0.659 +/- 0.124 | 0.440 +/- 0.116 |
| biopsy_patients | M2 profile + symmetry + age | 0.639 +/- 0.067 | 0.731 +/- 0.106 | 0.445 +/- 0.115 |

### Oracle 70/30 Target 95

| dataset | model | R | S | Sp |
| --- | --- | --- | --- | --- |
| all_patients | AGE_ONLY | 0.698 +/- 0.051 | 0.961 +/- 0.016 | 0.280 +/- 0.093 |
| all_patients | M0 profile LR1 | 0.574 +/- 0.071 | 0.955 +/- 0.000 | 0.141 +/- 0.084 |
| all_patients | M1 profile + symmetry | 0.602 +/- 0.067 | 0.955 +/- 0.000 | 0.172 +/- 0.077 |
| all_patients | M2 profile + symmetry + age | 0.672 +/- 0.063 | 0.955 +/- 0.000 | 0.181 +/- 0.096 |
| biopsy_patients | AGE_ONLY | 0.734 +/- 0.057 | 0.963 +/- 0.017 | 0.329 +/- 0.112 |
| biopsy_patients | M0 profile LR1 | 0.563 +/- 0.068 | 0.955 +/- 0.000 | 0.111 +/- 0.070 |
| biopsy_patients | M1 profile + symmetry | 0.578 +/- 0.063 | 0.955 +/- 0.000 | 0.154 +/- 0.088 |
| biopsy_patients | M2 profile + symmetry + age | 0.639 +/- 0.067 | 0.955 +/- 0.000 | 0.196 +/- 0.104 |

### Patient LOOCV Pooled

| dataset | model | R | S | Sp |
| --- | --- | --- | --- | --- |
| all_patients | AGE_ONLY | 0.686 | 0.958 | 0.245 |
| all_patients | M0 profile LR1 | 0.563 | 0.958 | 0.123 |
| all_patients | M1 profile + symmetry | 0.598 | 0.958 | 0.113 |
| all_patients | M2 profile + symmetry + age | 0.683 | 0.958 | 0.132 |
| biopsy_patients | AGE_ONLY | 0.721 | 0.958 | 0.309 |
| biopsy_patients | M0 profile LR1 | 0.564 | 0.958 | 0.049 |
| biopsy_patients | M1 profile + symmetry | 0.580 | 0.958 | 0.123 |
| biopsy_patients | M2 profile + symmetry + age | 0.661 | 0.958 | 0.148 |

### Train-All Discovery

| dataset | model | R | S | Sp |
| --- | --- | --- | --- | --- |
| all_patients | AGE_ONLY | 0.698 | 0.958 | 0.245 |
| all_patients | M0 profile LR1 | 0.749 | 0.958 | 0.368 |
| all_patients | M1 profile + symmetry | 0.821 | 0.958 | 0.434 |
| all_patients | M2 profile + symmetry + age | 0.870 | 0.958 | 0.509 |
| biopsy_patients | AGE_ONLY | 0.733 | 0.958 | 0.309 |
| biopsy_patients | M0 profile LR1 | 0.824 | 0.958 | 0.457 |
| biopsy_patients | M1 profile + symmetry | 0.871 | 0.958 | 0.444 |
| biopsy_patients | M2 profile + symmetry + age | 0.895 | 0.958 | 0.457 |

## Critical Interpretation

The biggest current problem is age. Age-only gives ROC AUC 0.698 on
`all_patients` and 0.734 on `biopsy_patients` in the honest 70/30 protocol.
That means age is a strong baseline and possible confounder. XRD profile and
symmetry must be judged against age-only, not against chance alone.

The profile-only model is weak in honest patient-safe validation:

```text
all_patients M0: ROC AUC 0.574 +/- 0.071
biopsy_patients M0: ROC AUC 0.563 +/- 0.068
```

Symmetry helps modestly but does not yet create a strong standalone model:

```text
all_patients M0 -> M1: 0.574 -> 0.602
biopsy_patients M0 -> M1: 0.563 -> 0.578
```

M2 improves specificity in the honest 70/30 setting, but much of that may come
from age:

```text
all_patients M2 specificity: 0.424 +/- 0.115
biopsy_patients M2 specificity: 0.445 +/- 0.115
```

The strong train-all values are useful only as a discovery ceiling. They should
not be presented as expected prospective performance.

The oracle target-95 table is useful for understanding possible high-sensitivity
operating points, but it uses test labels to set the threshold. It must never be
used as the main validation result.

## Metric Policy For Small Patient-Level Data

All model-development metrics must be counted at patient level. Measurement rows
from the same patient must never be split between train and test.

The primary metrics are:

```text
ROC AUC
specificity at fixed sensitivity target, reported as Sp@S>=0.95
```

ROC AUC answers the ranking question:

```text
Does the model tend to assign higher p_cancer to cancer patients than to
benign patients?
```

Sp@S>=0.95 answers the product question:

```text
If Aramis is operated in a safety-first mode with sensitivity near 95%, how
many benign patients can avoid being flagged?
```

For this small dataset, sensitivity and specificity should not be interpreted
from one arbitrary threshold or one arbitrary split. The preferred reporting
view is:

```text
patient-safe repeated 70/30 splits, mean +/- std
patient-level LOOCV pooled predictions
train-all discovery ceiling, clearly marked as optimistic
```

The repeated split threshold policy must be explicit:

```text
honest threshold:
  choose threshold on train scores
  apply the same threshold to test scores
  use as the main validation-style estimate

oracle threshold:
  choose threshold on test scores to force S>=0.95
  use only as an exploratory operating-point view
  do not use as validation
```

Secondary metrics should be added when the train function is formalized:

```text
PR AUC
Brier score
calibration slope
calibration intercept
TP / FN / TN / FP counts at the selected threshold
bootstrap confidence intervals by patient
```

Brier score and calibration are needed because Aramis reports `p_cancer`, not
only a rank. A model can rank patients reasonably by ROC AUC but still produce
poorly calibrated probabilities.

Model comparisons should be paired by patient split:

```text
M1 - M0 ROC AUC
M2 - M1 ROC AUC
M1 - M0 Sp@S>=0.95
M2 - M1 Sp@S>=0.95
```

If the paired confidence interval crosses zero, the result should be described
as a trend, not a confirmed improvement.

## Current Working Conclusion

For the next modeling iteration, use the A structure only:

```text
M0: profile LR1 patient logit-averaged p_cancer
M1: profile LR1 patient logit-averaged p_cancer + raw cosine symmetry features
M2: profile LR1 patient logit-averaged p_cancer + raw cosine symmetry features + age
```

Keep both datasets:

```text
all_patients
biopsy_patients
```

Report all results against `AGE_ONLY`. The next product implementation should
move from current patient-level aggregation to the intended clinical route:
clinician-specified target breast profile plus same-patient contralateral
symmetry context.

## Artifacts

```text
examples/aramis_biopsy_patient_model_structures_experiment.py
examples/outputs/patient_model_structures_v0_2/model_structure_summary.csv
examples/outputs/patient_model_structures_v0_2/model_structure_predictions.csv
examples/outputs/patient_model_structures_v0_2/comparison_pivot.csv
examples/outputs/patient_model_structures_v0_2/comparison_pivot.md
examples/outputs/patient_model_structures_v0_2/all_patients_feature_table.csv
examples/outputs/patient_model_structures_v0_2/biopsy_patients_feature_table.csv
```
