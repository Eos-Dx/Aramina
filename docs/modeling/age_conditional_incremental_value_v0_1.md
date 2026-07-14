# Age-Conditional Incremental Value Analysis v0.1

Status: historical research draft. This is an evidence record, not a released-model claim.

The routed architecture described below is historical. The current gated
target-case M2Q record is `m2q_gated_target_case_model_v0_1.md`.

## Question

Age is an expected clinical prior in the biopsy-referred cohort. The relevant
question is therefore not whether XRD should replace age, but whether profile
and symmetry data improve the patient-level probability after age is known.

The comparison uses the same patient-safe outer-test predictions for every
patient:

```text
A0: age and age_available only
M1: target profile plus routed SK Core4 symmetry; no age
M2Q: age plus target profile plus routed SK Core4 symmetry
```

`M1` and `M2Q` use `paired` when a contralateral breast exists and
`fallback_no_symmetry` otherwise. Measurement counts are reliability fields and
do not enter risk prediction.

## Source Predictions

```text
training artifact:
  examples/outputs/training/aramis_t100_honest_nested_operational_experiment.joblib

outer evaluation:
  repeated patient-safe stratified 5-fold, 5 repeats

patients:
  164 (75 CANCER, 89 BENIGN)
```

Each patient has five outer-test predictions. Scores and train-derived margins
are averaged per patient before paired comparisons. Bootstrap resampling always
resamples the same patients jointly for both models.

## Incremental Value Relative to Age

| comparison | metric difference | estimate | paired 95% bootstrap CI | interpretation |
|---|---|---:|---:|---|
| M1 - A0 | ROC AUC | -0.058 | -0.161 to 0.046 | XRD/symmetry alone does not improve age ranking |
| M1 - A0 | Brier score | +0.021 | -0.012 to 0.055 | probability quality is not improved |
| M1 - A0 | log loss | +0.051 | -0.026 to 0.130 | probability quality is not improved |
| M2Q - A0 | ROC AUC | +0.005 | -0.075 to 0.084 | no demonstrated incremental ranking gain |
| M2Q - A0 | Brier score | -0.000 | -0.027 to 0.028 | no demonstrated incremental probability gain |
| M2Q - A0 | log loss | +0.003 | -0.060 to 0.068 | no demonstrated incremental probability gain |

At this sample size, M2Q may help some patients and harm others, but the net
incremental effect over age is indistinguishable from zero.

## Individual Decision Changes

At each model's train-derived high-sensitivity threshold, M2Q changed the
patient-level decision relative to A0 for 23 of 164 patients:

| M2Q change | BENIGN | CANCER | total |
|---|---:|---:|---:|
| escalates age-only BENIGN to CANCER | 15 | 3 | 18 |
| downgrades age-only CANCER to BENIGN | 3 | 2 | 5 |
| unchanged | 71 | 70 | 141 |

Relative to the known historical label:

| outcome | BENIGN | CANCER | total |
|---|---:|---:|---:|
| M2Q corrects A0 | 3 | 3 | 6 |
| M2Q harms A0 | 15 | 2 | 17 |
| same correctness | 71 | 70 | 141 |

This operating-point analysis is threshold-dependent. It should not be
interpreted as a prospective clinical benefit claim.

## Age Bands

Predefined age bands show that any apparent XRD contribution is not uniform.
Small bands have wide uncertainty.

| age band | patients | CANCER | BENIGN | A0 ROC | M1 ROC | M2Q ROC |
|---|---:|---:|---:|---:|---:|---:|
| <45 | 59 | 13 | 46 | 0.635 | 0.579 | 0.604 |
| 45-54 | 47 | 27 | 20 | 0.582 | 0.648 | 0.656 |
| 55-64 | 41 | 23 | 18 | 0.662 | 0.621 | 0.628 |
| 65+ | 17 | 12 | 5 | 0.600 | 0.750 | 0.733 |

The 45-54 and 65+ bands suggest potential refinement, but the evidence is too
small and heterogeneous to select an age-specific product rule.

## Limitation of This First Paired Analysis

Outer test patients are identical across A0, M1, and M2Q, so there is no outer
test leakage. However, the current inner regularization grid is selected for
M2Q and then reused for the comparison models. That is conservative for the
M2Q comparison but not a final model-family selection procedure. The next fair
experiment must perform model-specific inner tuning for A0, M1, and M2Q while
keeping the same outer patient splits.

## Generated Review Artifacts

These intentionally live outside the Git repository because they contain
patient-level review data:

```text
/Users/sad/dev/aramis_artifacts/aramis_t100_age_conditional_patient_predictions.csv
/Users/sad/dev/aramis_artifacts/aramis_t100_age_conditional_incremental_metrics.csv
/Users/sad/dev/aramis_artifacts/aramis_t100_age_conditional_age_bands.csv
/Users/sad/dev/aramis_artifacts/aramis_t100_age_conditional_changed_patients.csv
```
