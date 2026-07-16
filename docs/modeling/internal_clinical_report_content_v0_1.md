# Aramis Internal Clinical Report Content

Status: research-draft internal audit contract. It is not for autonomous diagnosis.

The report is produced once per request and contains two consistently shaped blocks: the caller-specified target breast and the contralateral breast when it remains available after preprocessing. The external report contains only the target decision-support result.

```yaml
output_type: aramis_internal_clinical_report
report_version: "0.1"
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
  operator_id: EOSCAN_OPERATOR
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
profile_only:
  p_cancer: 0.63482
final_prediction:
  p_cancer: 0.59489
  decision_threshold_id: target_sensitivity_0.95
  decision_threshold: 0.32787
  suggested_class: CANCER
training_cohort_quantile: 0.74000
benign_cohort_quantile: 0.93000
cancer_cohort_quantile: 0.32000
features:
  symmetry:
    available: true
    status: applied
  reliability:
    level: high
    reason: at least 3 valid measurements per breast
model_execution:
  scoring_path: profile_age_with_symmetry
  symmetry_refinement: applied
```

`profile_only.p_cancer` is the first-layer LR1 score. The final score is M2Q: profile, optional gated SK symmetry refinement, and age when available. The decision uses only `final_prediction`: `CANCER` when `p_cancer >= decision_threshold`, otherwise `BENIGN`.

The three quantiles are empirical percentiles of final `p_cancer` against frozen train-on-all target-breast score distributions: all accepted training cases, BENIGN cases, and CANCER cases. They are internal descriptive evidence, not calibrated risk or a decision rule.

If the contralateral breast is absent after preprocessing, its block is:

```yaml
available: false
side: unknown
profile_only:
  p_cancer: unknown
final_prediction:
  p_cancer: unknown
  decision_threshold_id: unknown
  decision_threshold: unknown
  suggested_class: unknown
training_cohort_quantile: unknown
benign_cohort_quantile: unknown
cancer_cohort_quantile: unknown
reason: contralateral breast is unavailable after preprocessing
```

When the target breast has no usable contralateral breast, the target block
still has a valid final prediction. Its `model_execution.scoring_path` is
`profile_age_with_neutral_symmetry_gate`: the same LR2 is used, but the optional
SK symmetry terms are neutral and do not affect the score. This is not a second
model and it must not be interpreted as a symmetry-supported result.

The report deliberately excludes estimator objects, model weights, raw SK feature values, feature contributions, duplicate prediction-config fields, filesystem paths, and training configuration. Those remain in the model joblib and its `model_description.yaml`.
