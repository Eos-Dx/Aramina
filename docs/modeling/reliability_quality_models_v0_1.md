# Reliability Quality Models v0.1

Clinical framing: research-draft decision support only; requires radiologist
review. `p_cancer` is the risk score. Reliability describes result confidence
and must not be treated as a direct rule that lowers risk.

## Question

The current profile-only signal is useful but weak in patient-safe validation.
The practical product question is whether measurement reliability can help the
model reduce false positives without hiding the risk score itself.

The intended product separation is:

```text
p_cancer = risk
reliability = confidence in this result
```

This is why reliability is tested both as a model feature and as a report field.
If the report says `p_cancer = 0.62` and `reliability = low`, the risk remains
high, but the result must be read as less confident and may justify repeat scan
or closer radiologist review.

## Dataset

Input preprocessing artifact:

```text
examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
```

Training dataset summary from the generated model artifact:

| field | value |
|---|---:|
| measurement rows | 968 |
| patients | 180 |
| specimens/breasts | 342 |
| LR1 rows | 546 |
| LR1 patients | 180 |
| final patients | 180 |
| final CANCER patients | 84 |
| final BENIGN patients | 96 |

Reliability distribution in the patient-level feature table:

| label | high | medium | low |
|---|---:|---:|---:|
| BENIGN | 67 | 17 | 12 |
| CANCER | 64 | 14 | 6 |

Important: these counts belong to this exact preprocessing artifact. Other
historical Aramis tables may use older cohort snapshots and must not be mixed
silently.

## Model Definitions

All models use the same patient-safe split assignments within a given validation
mode. All rows for one patient are kept either in train or in test.

```text
M0:
  LR1 profile LogisticRegression
  target-breast measurement p_cancer values
  logit-average aggregation to patient p_cancer

M0Q:
  M0 patient p_cancer
  plus reliability/quality counters

M1:
  M0 patient p_cancer
  plus SK target/contralateral symmetry block

M1Q:
  M1
  plus reliability/quality counters

M2:
  M1
  plus age and age_available

M2Q:
  M1Q
  plus age and age_available
```

SK means Slava Kubitskyi-style target/contralateral symmetry features. Age
models are audit/comparison branches because age can encode clinical prior
probability and may dominate XRD signal.

## Reliability Features

The Q models use these numerical features:

```text
profile_p_cancer_n_measurements
target_measurements
contralateral_measurements
min_measurements_per_breast
target_measurements_ok
contralateral_measurements_ok
paired_measurements_ok
```

The artifact also stores report/audit fields:

```text
result_reliability
result_reliability_reason
```

Reliability levels are assigned as:

```text
high:
  paired breast symmetry is available
  and at least 3 valid measurements exist per breast

medium:
  paired breast symmetry is available
  but at least one breast has fewer than 3 valid measurements

low:
  paired breast symmetry is unavailable
```

These fields are meant to express confidence in the model result. They are not
a clinical diagnosis and are not a rule to reduce `p_cancer`.

## Validation Modes

Four modes were run:

```text
70/30 x50:
  patient-level StratifiedShuffleSplit
  50 repeated train/test splits
  30% patient test set each split
  reported as mean +/- std

stratified 5-fold:
  patient-level StratifiedKFold
  shuffled 5-fold split
  reported as mean +/- std

LOOVM:
  leave-one-patient-out style run
  pooled left-out predictions
  no std because one pooled result is reported

train-all:
  train and evaluate on all patients
  discovery ceiling only
  not validation
```

For split-based modes, LR1 and the second-layer model are fit only on the train
patients. Test patients are scored without fitting on their rows.

## Threshold Logic

For each split and each model, the operating threshold is selected on training
scores to target sensitivity `0.95`. Sensitivity and specificity are then
measured on test scores. Therefore realized test sensitivity can be below
`0.95`, especially in small patient-safe splits.

Metrics:

```text
ROC AUC:
  ranking quality independent of selected threshold

sensitivity:
  test sensitivity at the training-selected target-sensitivity threshold

specificity:
  test specificity at the same threshold
```

False negatives are safety-critical for this product, so specificity is
interpreted only together with sensitivity.

## How Results Were Generated

Main training config:

```text
config/training/aramis_biopsy_patients_m0_m1_m2_v0_1.yaml
```

The config was run with model set:

```text
M0, M0Q, M1, M1Q, M2, M2Q
```

and with these evaluation modes:

```text
repeated_stratified_shuffle
stratified_kfold
loovm
all_on_all
```

The machine-readable output table is:

```text
docs/modeling/results/biopsy_reliability_quality_model_mode_comparison_v0_1.csv
```

Temporary generated model artifacts for this comparison are under:

```text
examples/outputs/training/reliability_modes_tmp/
```

## Results

| mode | model | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|---:|
| 70/30 x50 | M0 | 0.574 +/- 0.061 | 0.738 +/- 0.122 | 0.359 +/- 0.141 |
| 70/30 x50 | M0Q | 0.574 +/- 0.065 | 0.714 +/- 0.116 | 0.383 +/- 0.137 |
| 70/30 x50 | M1 | 0.582 +/- 0.059 | 0.658 +/- 0.103 | 0.455 +/- 0.112 |
| 70/30 x50 | M1Q | 0.576 +/- 0.063 | 0.644 +/- 0.116 | 0.471 +/- 0.120 |
| 70/30 x50 | M2 | 0.623 +/- 0.059 | 0.702 +/- 0.108 | 0.470 +/- 0.111 |
| 70/30 x50 | M2Q | 0.616 +/- 0.061 | 0.692 +/- 0.124 | 0.484 +/- 0.119 |
| stratified 5-fold | M0 | 0.614 +/- 0.086 | 0.751 +/- 0.154 | 0.366 +/- 0.078 |
| stratified 5-fold | M0Q | 0.613 +/- 0.084 | 0.751 +/- 0.130 | 0.376 +/- 0.094 |
| stratified 5-fold | M1 | 0.630 +/- 0.094 | 0.693 +/- 0.144 | 0.470 +/- 0.104 |
| stratified 5-fold | M1Q | 0.622 +/- 0.093 | 0.716 +/- 0.144 | 0.417 +/- 0.091 |
| stratified 5-fold | M2 | 0.675 +/- 0.084 | 0.764 +/- 0.110 | 0.449 +/- 0.135 |
| stratified 5-fold | M2Q | 0.679 +/- 0.095 | 0.776 +/- 0.140 | 0.397 +/- 0.059 |
| LOOVM | M0 | 0.593 | 0.833 | 0.292 |
| LOOVM | M0Q | 0.583 | 0.821 | 0.323 |
| LOOVM | M1 | 0.599 | 0.750 | 0.385 |
| LOOVM | M1Q | 0.588 | 0.690 | 0.438 |
| LOOVM | M2 | 0.648 | 0.810 | 0.344 |
| LOOVM | M2Q | 0.633 | 0.798 | 0.375 |
| train-all | M0 | 0.851 | 0.952 | 0.490 |
| train-all | M0Q | 0.862 | 0.952 | 0.510 |
| train-all | M1 | 0.895 | 0.952 | 0.573 |
| train-all | M1Q | 0.901 | 0.952 | 0.667 |
| train-all | M2 | 0.904 | 0.964 | 0.542 |
| train-all | M2Q | 0.909 | 0.952 | 0.552 |

## Interpretation

### Honest 70/30 x50

This is the most useful current check for comparing candidate product behavior.

Reliability improves specificity for every base family:

```text
M0 -> M0Q: 0.359 -> 0.383
M1 -> M1Q: 0.455 -> 0.471
M2 -> M2Q: 0.470 -> 0.484
```

However, reliability does not improve ROC AUC:

```text
M0 -> M0Q: 0.574 -> 0.574
M1 -> M1Q: 0.582 -> 0.576
M2 -> M2Q: 0.623 -> 0.616
```

Interpretation: reliability helps the operating threshold reduce false
positives, but it does not improve global ranking in this run.

Best 70/30 ROC AUC:

```text
M2: 0.623 +/- 0.059
```

Best 70/30 specificity:

```text
M2Q: 0.484 +/- 0.119
```

Best non-age 70/30 specificity:

```text
M1Q: 0.471 +/- 0.120
```

### Stratified 5-fold

In 5-fold, reliability helps M0 only:

```text
M0 -> M0Q: specificity 0.366 -> 0.376
```

For M1 and M2, Q features decrease specificity:

```text
M1 -> M1Q: 0.470 -> 0.417
M2 -> M2Q: 0.449 -> 0.397
```

This means reliability is not universally beneficial. It depends on split mode
and probably on how many low/medium reliability patients land in each fold.

### LOOVM

LOOVM shows the same specificity pattern as 70/30:

```text
M0 -> M0Q: 0.292 -> 0.323
M1 -> M1Q: 0.385 -> 0.438
M2 -> M2Q: 0.344 -> 0.375
```

But sensitivity falls for Q models. This is acceptable only as an exploratory
observation, not as a product decision.

### Train-all

Train-all improves with Q features, especially:

```text
M1 -> M1Q specificity: 0.573 -> 0.667
```

This is a discovery ceiling, not validation. It shows the features can fit the
cohort, but does not estimate generalization.

## Current Decision

For product discussion:

```text
Primary non-age candidate:
  M1 or M1Q

Best 70/30 non-age specificity:
  M1Q

Best 70/30 ROC AUC:
  M2

Best 70/30 specificity overall:
  M2Q
```

Recommended interpretation now:

```text
Keep M1 as clean profile+symmetry baseline.
Keep M1Q as reliability-aware non-age branch.
Keep M2/M2Q as age-audit branches, not primary product decision yet.
Report reliability separately from p_cancer in every future predict report.
```

## Limitations

This experiment is not clinical validation. Main weaknesses:

```text
small patient cohort
threshold selected on small train splits
test sensitivity below target 0.95 in honest modes
age may encode clinical prior probability
reliability effects are split-dependent
no external validation cohort
```

The next development step should compare M1 and M1Q in a prediction-style report
prototype where `p_cancer`, `risk_level`, `reliability`, and
`result_reliability_reason` are shown together.

## Report Behavior

The future predict/report layer should return risk and reliability separately:

```yaml
p_cancer: 0.62
risk_level: high
reliability: low
reason: only 1 valid target-breast measurement
review: requires radiologist review
```

