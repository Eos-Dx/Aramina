# Aramis Internal Clinical Report Content

Status: research-draft internal audit contract. It is not for autonomous diagnosis.

The report is produced once per request and contains two consistently shaped blocks: the caller-specified target breast and the contralateral breast when it remains available after preprocessing. The external report contains only the target decision-support result.

```yaml
output_type: aramis_internal_clinical_report
report_version: "0.1"
reference_doc: ./docs/modeling/internal_clinical_report_content_v0_1.md
report_id: GENERATED_UNIQUE_ID
created_at: "2026-07-16T14:11:00+02:00"
analysis_author: REQUESTING_ANALYST
prediction_comment: "optional free-text request comment"
model:
  id: MODEL_ARTIFACT_ID
  name: aramis_m2q_t100
  version: 0.2.7-beta
  artifact_sha256: SHA256
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

Each available breast block has this shape:

```yaml
available: true
side: left
azimuthal_integration_target_profile:
  available: true
  p_cancer: 0.63482
  per_measurement_p_cancer: [0.62203, 0.64908, 0.63332]
final_prediction:
  p_cancer: 0.59489
  decision_threshold_id: target_sensitivity_0.95
  decision_threshold: 0.32787
  suggested_class: CANCER
  score_percentiles:
    all_training_patients: 0.74000
    benign_training_patients: 0.93000
    cancer_training_patients: 0.32000
  reliability:
    level: high
    reason: at least 3 valid measurements per breast
symmetry:
  available: true
  status: applied
model_execution:
  scoring_path: profile_age_with_symmetry
  symmetry_refinement: applied
```

`azimuthal_integration_target_profile.p_cancer` is the first-layer LR1 score. The final score combines the profile, optional gated SK symmetry refinement, and age when available. The decision uses only `final_prediction`: `CANCER` when `p_cancer >= decision_threshold`, otherwise `BENIGN`.

`score_percentiles` are empirical percentiles of final `p_cancer` against frozen train-on-all target-breast score distributions: all accepted training cases, BENIGN cases, and CANCER cases. A value of `0.90286` means that 90.286% of the relevant frozen training scores were lower than the reported score. These values are descriptive evidence only: they are not probability, diagnosis, population risk, calibration, or a decision rule. They are never recalculated from incoming scans.

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
  decision_threshold_id: unknown
  decision_threshold: unknown
  suggested_class: unknown
  score_percentiles:
    all_training_patients: unknown
    benign_training_patients: unknown
    cancer_training_patients: unknown
  reliability:
    level: unknown
    reason: unknown
symmetry:
  available: false
  status: not_available
reason: contralateral breast is unavailable after preprocessing
```

When the target breast has no usable contralateral breast, the target block
still has a valid final prediction. Its `model_execution.scoring_path` is
`profile_age_with_neutral_symmetry_gate`: the same LR2 is used, but the optional
SK symmetry terms are neutral and do not affect the score. This is not a second
model and it must not be interpreted as a symmetry-supported result.

The report deliberately excludes estimator objects, model weights, raw SK feature values, feature contributions, duplicate prediction-config fields, filesystem paths, and training configuration. Those remain in the model joblib and its `model_description.yaml`.
