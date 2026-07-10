# M2Q Age Candidate Update v0.1

Status: superseded research record. The current optional-symmetry Core4
candidate is documented in `m2q_core4_optional_symmetry_candidate_v0_1_9.md`.

This note records the switch from M1Q to M2Q as the current Aramis primary
candidate model. M2Q keeps the M1Q structure and adds `age` and
`age_available`.

## Rationale

Age is a clinically meaningful breast-cancer risk prior: older women have a
higher baseline cancer risk. The model should therefore be allowed to use age
when it is available, while keeping age explicitly visible in documentation,
the feature schema, and the internal report.

This is not a claim that age replaces XRD information. Age is used together
with target-breast profile score, target/contralateral SK symmetry features,
and reliability counters. Because age can dominate small cohorts, M2Q must be
reported as a decision-support research model and reviewed against future
validation data.

## Fixed Settings

```text
dataset: T100 biopsy-patient model-input DataFrame
patients: 164
CANCER patients: 75
BENIGN patients: 89
regularization: L2 LogisticRegression, C=0.1
target sensitivity: 0.95
```

## M1Q vs M2Q

| mode | M1Q ROC AUC | M2Q ROC AUC | M1Q sensitivity | M2Q sensitivity | M1Q specificity | M2Q specificity |
|---|---:|---:|---:|---:|---:|---:|
| stratified 5-fold x20 | 0.618 +/- 0.079 | 0.673 +/- 0.074 | 0.766 +/- 0.126 | 0.803 +/- 0.112 | 0.409 +/- 0.105 | 0.419 +/- 0.111 |
| patient-safe 80/20 x50 | 0.613 +/- 0.084 | 0.662 +/- 0.082 | 0.748 +/- 0.119 | 0.781 +/- 0.119 | 0.407 +/- 0.106 | 0.430 +/- 0.115 |
| LOOVM | 0.622 | 0.677 | 0.840 | 0.840 | 0.326 | 0.326 |
| train-all | 0.881 | 0.889 | 0.960 | 0.960 | 0.494 | 0.517 |

The clearest fitted-cohort operating-point change is specificity:

```text
M1Q train-all specificity: 0.494
M2Q train-all specificity: 0.517
absolute gain: +0.022, approximately 2-3 percentage points
```

The patient-safe split views show a larger ROC AUC gain, especially in
stratified 5-fold and 80/20 modes. These are still small-cohort estimates and
must not be presented as clinical validation.

## Product Decision

Current product candidate:

```text
model_id: aramis_m2q_t100_train_all_c0p1
selected_model: M2Q
threshold_target: 0.302291
```

M1Q remains the age-free comparison model. M2Q becomes the primary candidate
because age is clinically meaningful and improves the current model comparison
without changing preprocessing, symmetry features, reliability features, or
regularization.
