# Historical Aramis Candidate Artifact v0.1

Status: superseded research record.

This document records the earlier unrestricted-SK candidate. It is retained as
research history only. The current paired-breast Core4 candidate is documented
in `m2q_core4_paired_candidate_v0_1_8.md`.

## Current Candidate

```text
preprocessing: T100 biopsy-patient model-input DataFrame
model family: M2Q
LR penalty: L2
LR regularization C: 0.1
target sensitivity: >= 0.95 on fitted development cohort
threshold_target: 0.302291
```

Current fitted artifact:

```text
examples/prediction_models/aramis_m2q_t100_train_all_c0p1.joblib
```

Model id:

```text
aramis_m2q_t100_train_all_c0p1
```

This id is stored as `training.name` in the model artifact and must be supplied
as `model.model_id` in prediction YAML.

The experiment artifact used to select this model is preserved at:

```text
examples/outputs/model_selection_m1q_regularization_v0_1/aramis_m1q_t100_selected_c_train_all.joblib
```

## Why T100

T100 is the current development preprocessing default. It is a middle-ground
AgBH monochromaticity threshold:

```text
T70: stricter, slightly better in some M1Q checks, but loses more patients
T100: selected development compromise
T130: keeps more patients, but current M1Q checks show weaker ROC/specificity
```

T100 keeps enough biopsy-patient cases for patient-safe model selection while
excluding more questionable calibration days than T130.

## Why M2Q

M2Q uses:

```text
LR1 profile p_cancer
SK target-vs-contralateral symmetry features
measurement reliability counters
age + age_available
```

It is preferred over M1Q because age is a clinically meaningful risk prior for
breast cancer: older women have higher baseline risk. In the current
model-selection comparison, adding age improved the candidate by roughly 3
percentage points while preserving the same T100 preprocessing, SK symmetry
features, reliability counters, and L2 LogisticRegression regularization
(`C=0.1`).

Age remains explicitly documented because it can act as a clinical prior and
may dominate the XRD signal in a small development cohort. The model output
must therefore be described as decision support, not autonomous diagnosis.

## Regularization Selection

Regularization was selected before fitting the final train-all model. The
initial selection experiment used repeated patient-safe stratified 5-fold
validation on the T100 candidate family:

```text
validation mode: repeated stratified patient K-fold
folds: 5
repeats: 20
patient leakage protection: split by patientId
C grid: [0.03, 0.1, 0.3, 1.0]
selection rule: highest K-fold ROC AUC, then smaller C if ROC differs by < 0.005
selected C: 0.1
```

The final M2Q candidate keeps this same `C=0.1` regularization so that the
model-family change isolates the addition of age rather than changing
regularization at the same time.

Original K-fold grid:

| C | ROC AUC | sensitivity | specificity |
|---:|---:|---:|---:|
| 0.1 | 0.618 +/- 0.079 | 0.766 +/- 0.126 | 0.409 +/- 0.105 |
| 0.3 | 0.616 +/- 0.077 | 0.705 +/- 0.124 | 0.462 +/- 0.110 |
| 0.03 | 0.605 +/- 0.083 | 0.803 +/- 0.111 | 0.308 +/- 0.099 |
| 1.0 | 0.604 +/- 0.079 | 0.636 +/- 0.127 | 0.506 +/- 0.125 |

## Final Train-All Fit

After model family and regularization were selected, the candidate model was
trained once on all available development patients:

```text
patients: 164
CANCER patients: 75
BENIGN patients: 89
```

The train-all fit gives the current candidate artifact and fitted-cohort
operating threshold. The threshold is selected to reach the high-sensitivity
operating point on the full development cohort.

Train-all fitted-cohort metrics:

| metric | value |
|---|---:|
| ROC AUC | 0.889 |
| threshold_target | 0.302291 |
| sensitivity | 0.960 |
| specificity | 0.517 |
| TP | 72 |
| FN | 3 |
| TN | 46 |
| FP | 43 |
| PPV | 0.626 |
| NPV | 0.939 |

## Conservative Validation Views

The same selected settings were also evaluated with patient-safe validation:

| mode | ROC AUC | sensitivity | specificity |
|---|---:|---:|---:|
| repeated stratified 5-fold x20 | 0.673 +/- 0.074 | 0.803 +/- 0.112 | 0.419 +/- 0.111 |
| patient-safe 80/20 x50 | 0.662 +/- 0.082 | 0.781 +/- 0.119 | 0.430 +/- 0.115 |
| LOOVM | 0.677 | 0.840 | 0.326 |
| train-all | 0.889 | 0.960 | 0.517 |

## Interpretation

The final artifact is not an average of K-fold models. K-fold is used to select
the model family and regularization and to estimate expected generalization.
The delivered candidate artifact is one train-all model fitted on the full
development cohort with the selected hyperparameters.

Train-all metrics are optimistic fitted-cohort metrics. Patient-safe K-fold,
80/20 splits, and LOOVM are more conservative estimates. The model can be
described as a high-sensitivity candidate model locked for the next validation
cohort, not as a clinically proven diagnostic model.

## Predict Contract

Prediction uses the model artifact and the prediction preprocessing config
stored inside it. The incoming H5 does not define the target side. The suspicious
breast side must be supplied explicitly in predict YAML:

```yaml
patient:
  patient_id: PATIENT_ID
  target_side: Left
model:
  model_id: aramis_m2q_t100_train_all_c0p1
  selected_model: M2Q
```

The per-patient report must keep risk and reliability separate:

```text
p_cancer = model risk score
reliability = data sufficiency / measurement confidence
suggested_class = thresholded decision-support class
```
