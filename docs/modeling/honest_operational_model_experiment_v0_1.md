# Honest Operational Model Experiment v0.1

Status: historical research draft, development experiment. Not a released model claim.

This document records the earlier paired/fallback architecture and its results.
It is historical comparison evidence; the current gated target-case M2Q record
is `m2q_gated_target_case_model_v0_1.md`.

## Purpose

This experiment changes model evaluation to answer the operational question
without encoding missing contralateral data as a diagnostic number.

```text
paired route:
  LR1 target profile score
  + SK Core4 target/contralateral symmetry
  + age and age_available

fallback_no_symmetry route:
  LR1 target profile score
  + age and age_available
```

`symmetry_available` selects the route. It is not a model feature. Measurement
counts are reliability fields only and do not enter either LogisticRegression.

## Dataset

```text
preprocessing: T100 biopsy-patient model input
measurement rows: 893
patients: 164
CANCER patients: 75
BENIGN patients: 89
paired patients: 150
patients without contralateral data: 14
```

The biopsied historical suspicious breast supplies the target label. The
contralateral breast supplies symmetry context only.

## Honest Validation Design

Outer evaluation:

```text
repeated stratified patient-safe 5-fold
5 repeats
25 outer test folds
no patient can occur in outer train and outer test together
```

Inside every outer-train fold:

```text
4-fold patient-safe inner CV
select LR1 C from [0.1, 0.3]
select LR2 C from [0.1, 0.3]
selection metric: inner out-of-fold operational ROC AUC
derive paired and fallback thresholds from inner out-of-fold predictions
target sensitivity: 0.95
```

The selected models and thresholds are then applied unchanged to the outer
test fold. Outer test labels do not select regularization, features, or
thresholds.

The final train-all artifact repeats the same inner-CV selection on the full
training cohort, refits the selected models on all patients, and stores the
inner out-of-fold thresholds. Train-all fitted-cohort metrics are diagnostics,
not validation evidence.

## Evaluation Views

```text
operational:
  paired patients use paired route
  unpaired patients use fallback route

paired:
  paired route evaluated only on patients with both breasts

fallback_no_symmetry:
  fallback route evaluated on every test patient with symmetry intentionally omitted
```

Evaluating fallback on every test patient avoids claiming performance from the
14 naturally unpaired patients alone.

## Results

Patient-level pooled predictions average each patient's outer-test predictions
across repeats. Confidence intervals use 500 patient-level bootstrap samples.

| model | view | patients | split ROC mean +/- SD | pooled ROC (95% CI) | sensitivity mean +/- SD | specificity mean +/- SD | Brier mean |
|---|---|---:|---:|---:|---:|---:|---:|
| M0 profile only | operational | 164 | 0.624 +/- 0.064 | 0.633 (0.547-0.712) | 0.955 +/- 0.062 | 0.102 +/- 0.065 | 0.239 |
| A0 age only | operational | 164 | 0.716 +/- 0.074 | 0.715 (0.640-0.787) | 0.941 +/- 0.093 | 0.305 +/- 0.099 | 0.217 |
| M1 profile + routed symmetry | operational | 164 | 0.637 +/- 0.056 | 0.657 (0.571-0.733) | 0.965 +/- 0.047 | 0.117 +/- 0.074 | 0.244 |
| M1 profile + symmetry | paired | 150 | 0.611 +/- 0.059 | 0.630 (0.544-0.713) | 0.962 +/- 0.052 | 0.115 +/- 0.076 | 0.254 |
| M1 profile fallback | fallback_no_symmetry | 164 | 0.624 +/- 0.064 | 0.634 (0.543-0.719) | 0.939 +/- 0.065 | 0.136 +/- 0.084 | 0.253 |
| M2Q profile + routed symmetry + age | operational | 164 | 0.704 +/- 0.040 | 0.720 (0.634-0.801) | 0.949 +/- 0.060 | 0.153 +/- 0.081 | 0.222 |
| M2Q paired | paired | 150 | 0.676 +/- 0.039 | 0.692 (0.602-0.785) | 0.945 +/- 0.066 | 0.139 +/- 0.081 | 0.233 |
| M2Q profile + age fallback | fallback_no_symmetry | 164 | 0.685 +/- 0.049 | 0.693 (0.620-0.775) | 0.933 +/- 0.073 | 0.173 +/- 0.080 | 0.231 |

## Hyperparameter Stability

Full-cohort inner CV selected:

```text
LR1 C: 0.1
LR2 C: 0.3
inner OOF operational ROC AUC: 0.716
paired threshold_target: 0.105872
fallback threshold_target: 0.075409
```

Across 25 outer folds:

| LR1 C | LR2 C | selected folds |
|---:|---:|---:|
| 0.1 | 0.1 | 12 |
| 0.1 | 0.3 | 8 |
| 0.3 | 0.1 | 4 |
| 0.3 | 0.3 | 1 |

The preference for `LR1 C=0.1` is reasonably consistent. LR2 selection is less
stable. The final `C=0.3` choice is therefore recorded as a data-dependent
selection, not as a universally established constant.

## Interpretation

- The profile-only model contains signal above random ranking, but confidence
  intervals remain wide.
- Age alone is a strong predictor in this cohort. M2Q cannot be interpreted as
  evidence that XRD and symmetry alone produce its full ROC AUC.
- Routed symmetry adds some ranking information to profile plus age: pooled ROC
  is 0.720 operational versus 0.693 for the fallback view.
- At a train-derived high-sensitivity operating point, specificity remains low
  and unstable. This is the principal current limitation.
- Measurement counts remain important for report reliability but are excluded
  from diagnostic prediction to avoid acquisition-protocol leakage.
- Train-all fitted-cohort values must not replace the patient-safe outer-CV
  estimates above.

## Artifact

```text
config:
  config/training/aramis_t100_honest_nested_experiment_v0_1.yaml

generated model artifact:
  examples/outputs/training/aramis_t100_honest_nested_operational_experiment.joblib
```

The generated output directory is experimental and is not the released model
registry.
