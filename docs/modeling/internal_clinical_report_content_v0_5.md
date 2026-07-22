# Aramis Internal Clinical Report Content v0.5

Status: research-draft internal audit contract. It is not for autonomous diagnosis.

The report is produced once per request. It contains one formal decision-support
result for the caller-specified target breast and an internal full-model score
for the contralateral breast when it remains available after preprocessing. The
contralateral score uses the same profile-and-age model with the optional SK
symmetry refinement neutralized. The external report contains only the target
decision-support result.

```yaml
output_type: aramis_internal_clinical_report
report_version: "0.5"
reference_doc: ./docs/modeling/internal_clinical_report_content_v0_5.md
report_id: GENERATED_UNIQUE_ID
created_at: "2026-07-16T14:11:00+02:00"
analysis_author: REQUESTING_ANALYST
prediction_comment: "optional free-text request comment"
model:
  id: MODEL_ARTIFACT_ID
  name: aramis_target_breast_risk
  version: 0.2.10-beta
  artifact_sha256: SHA256
model_metrics:
  dataset: train_on_all_target_breast_cases
  validation: not_performed
  sensitivity: 0.96053
  specificity: 0.46465
decision_threshold:
  threshold_id: target_sensitivity_0.95
  threshold: 0.24451
  applies_to: [target.final_prediction, contralateral.final_prediction]
scan_metadata:
  patient_id: PATIENT_ID
  target_side: left
  patient_age: 53.0
  patient_age_available: true
  session_id: H5_SESSION_ID
  scan_date_time: "2026-07-16T14:11:00+02:00"
  operator_id: EOSCAN_OPERATOR
  hardware_version: human-1 v1.2.0
  eoscan_version: unknown
  experimental_protocol_version: unknown
  mammography_suspicious_field: unknown
  mammography_conclusion: unknown
  measurement_summary: {}
breast_predictions:
  target: {}
  contralateral: {}
```

`analysis_author` is the person who requested the analysis. `operator_id` is the EoScan operator read from H5 metadata. Optional scan metadata not present in the container is written as `unknown`. Numerical output is rounded to five decimal places; every binary output is `true` or `false`.

`decision_threshold` is one shared threshold policy for both available breast
scores:

```yaml
threshold_id: target_sensitivity_0.95
threshold: 0.24451
applies_to: [target.final_prediction, contralateral.final_prediction]
```

The available target block has this shape:

```yaml
available: true
side: left
azimuthal_integration_target_profile:
  available: true
  p_cancer: 0.63482
  per_measurement_p_cancer: [0.62203, 0.64908, 0.63332]
final_prediction:
  p_cancer: 0.59489
  suggested_class: CANCER
  level: TRA 3
  score_percentiles:
    reference_score: final_prediction.p_cancer
    reference_population: train_on_all_target_breast_cases
    all_training_target_cases: 0.74000
    benign_training_target_cases: 0.93000
    cancer_training_target_cases: 0.32000
reliability:
  level: high
  reason: at least 3 valid measurements per breast
symmetry:
  available: true
model_execution:
  scoring_path: azimuthal_integration_age_with_symmetry
```

`azimuthal_integration_target_profile.p_cancer` is the first-layer LR1 score.
The final score combines the profile, optional gated SK symmetry refinement, and
age when available. The decision uses only `target.final_prediction`: `CANCER`
when `p_cancer >= decision_threshold.threshold`, otherwise `BENIGN`.

`final_prediction.level` is the product-facing ordinal Tissue Risk Assessment.
The model calculates its unreported internal index as the all-training-target-
case percentile of the final score. `TRA 1` is below the 20th percentile,
`TRA 2` is 20th–below 50th, `TRA 3` is 50th–below 80th, `TRA 4` is 80th–below
90th, and `TRA 5` is at or above the 90th percentile. TRA is a fixed rank in
the frozen reference cohort, not an individual cancer probability, diagnosis,
or clinical calibration. It does not replace the shared threshold-based
`suggested_class` or clinician review.

Target `score_percentiles` are empirical percentiles of final `p_cancer` against
the frozen train-on-all target-breast final-score distributions: all accepted
training cases, BENIGN cases, and CANCER cases. A value of `0.90286` means that
90.286% of the relevant frozen training scores were lower than the reported
score. These values are descriptive evidence only: they are not probability,
diagnosis, population risk, calibration, or a decision rule. They are never
recalculated from incoming scans.

The available contralateral block contains the LR1 profile evidence and the
full final-model score with SK symmetry terms neutralized. Its `suggested_class` is
calculated with the same shared threshold as the target breast. The target
remains the caller-supplied primary decision-support result:

```yaml
available: true
side: right
azimuthal_integration_contralateral_profile:
  available: true
  p_cancer: 0.51234
  per_measurement_p_cancer: [0.50120, 0.51492, 0.52087]
final_prediction:
  p_cancer: 0.48611
  suggested_class: CANCER
  level: TRA 3
  score_percentiles:
    reference_score: final_prediction.p_cancer
    reference_population: train_on_all_target_breast_cases
    all_training_target_cases: 0.51000
    benign_training_target_cases: 0.67000
    cancer_training_target_cases: 0.25000
reliability:
  level: low
  reason: SK symmetry refinement is intentionally unavailable for contralateral scoring
symmetry:
  available: false
model_execution:
  scoring_path: azimuthal_integration_age
```

Both target and contralateral `final_prediction.p_cancer` values use the same
frozen final-score distribution from historical target-breast cases. This
comparison is descriptive evidence only and does not make the contralateral
score equivalent to a symmetry-supported target result. Contralateral
reliability is always `low`: its final score intentionally does not use SK
symmetry refinement.

If the contralateral breast is absent after preprocessing, its block is:

```yaml
available: false
side: unknown
azimuthal_integration_contralateral_profile:
  available: false
  p_cancer: unknown
  per_measurement_p_cancer: []
final_prediction:
  p_cancer: unknown
  suggested_class: unknown
  level: unknown
  score_percentiles:
    reference_score: unknown
    reference_population: unknown
    all_training_target_cases: unknown
    benign_training_target_cases: unknown
    cancer_training_target_cases: unknown
reliability:
  level: unknown
  reason: unknown
reason: contralateral breast is unavailable after preprocessing
```

When the target breast has no usable contralateral breast, the target block
still has a valid final prediction. Its `model_execution.scoring_path` is
`azimuthal_integration_age`: the same LR2 is used, but the optional
SK symmetry terms are neutral and do not affect the score. This is not a second
model and it must not be interpreted as a symmetry-supported result.

`model_metrics` contains sensitivity and specificity calculated for the frozen
final model on its accepted train-on-all target cases. `dataset` identifies
these cases as `train_on_all_target_breast_cases`; `validation: not_performed`
makes explicit that these are not independent validation estimates. Full
evaluation records remain in the model artifact and its adjacent evaluation
files.
`prediction_comment` is copied unchanged from the caller's predict YAML.

The report deliberately excludes estimator objects, model weights, raw SK feature values, feature contributions, duplicate prediction-config fields, filesystem paths, and training configuration. Those remain in the model joblib and its `model_description.yaml`.
