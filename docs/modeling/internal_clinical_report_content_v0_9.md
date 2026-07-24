# Aramis Internal Clinical Report Content v0.9

Status: research-draft internal audit contract. It is not for autonomous diagnosis.

One report contains a formal target-breast decision-support result and, when
available after preprocessing, an internal contralateral score. The caller
selects the target side. The target alone receives the threshold-derived risk
level and `biopsy_required`; contralateral evidence never creates a second
biopsy action.

```yaml
output_type: aramis_internal_clinical_report
report_version: "0.9"
reference_doc: ./docs/modeling/internal_clinical_report_content_v0_9.md
report_id: GENERATED_UNIQUE_ID
created_at: "2026-07-24T14:11:00+02:00"
analysis_author: REQUESTING_ANALYST
prediction_comment: "optional free-text request comment"
model:
  id: MODEL_ARTIFACT_ID
  name: aramis_target_breast_risk
  version: MODEL_VERSION
  artifact_sha256: SHA256
model_metrics:
  dataset: train_on_all_target_breast_cases
  validation: not_performed
  sensitivity: 0.96053
  specificity: 0.49495
decision_threshold:
  threshold_id: target_sensitivity_0.95
  threshold: 0.24666
  applies_to: [target.final_prediction]
scan_metadata:
  patient_id: PATIENT_ID
  target_side: left
  patient_age: 53.0
  patient_age_available: true
  session_id: H5_SESSION_ID
  scan_date_time: "2026-07-24T14:11:00+02:00"
  operator_id: EOSCAN_OPERATOR
  hardware_version: human-1 v1.2.0
  eoscan_version: unknown
  experimental_protocol_version: unknown
  mammography_suspicious_field: unknown
  mammography_conclusion: unknown
  measurement_summary:
    target_valid_measurements: 3
    contralateral_available: true
    contralateral_valid_measurements: 3
breast_predictions:
  target: {}
  contralateral: {}
```

All numerical values are rounded to five decimal places. All binary values are
`true` or `false`. Optional H5 metadata that is missing or conflicting after
QC is `unknown`.

## Target Breast

```yaml
available: true
side: left
azimuthal_integration_profile:
  p_cancer: 0.63482
  per_measurement_p_cancer: [0.62203, 0.64908, 0.63332]
final_prediction:
  p_cancer: 0.59489
  reference_class: BENIGN
  target_class: CANCER
  target_class_risk_level: high
  biopsy_required: true
  level: TRA 4
  score_percentiles:
    reference_population: train_on_all_target-breast_cases
    all: 0.74000
    reference_class: 0.93000
    target_class: 0.32000
reliability:
  level: high
  reason: at least 2 valid measurements per breast; symmetry refinement applied
symmetry:
  available: true
model_execution:
  scoring_path: azimuthal_integration_age_with_symmetry
```

`azimuthal_integration_profile.p_cancer` is the first-layer LR1 score. The
final `p_cancer` combines profile evidence, age when available, and optional
gated SK symmetry refinement. `target_class_risk_level` and
`biopsy_required` are both derived only from the frozen decision threshold:

```text
p_cancer < decision_threshold  -> target_class_risk_level: low, biopsy_required: false
p_cancer >= decision_threshold -> target_class_risk_level: high, biopsy_required: true
```

This is research-draft decision support, not an autonomous diagnosis.

`score_percentiles` locate final `p_cancer` in the frozen train-on-all
target-breast score distributions. `all`, `reference_class`, and
`target_class` are descriptive ranks, not probabilities, calibration, or a
decision rule. They are never recalculated from an incoming scan.

`level` is the internal TRA score tier. It is not the decision rule and is not
a probability. Its threshold-centred calibration is defined in
`docs/contracts/tissue_risk_assessment_v0_2.md`.

## Contralateral Breast

```yaml
available: true
side: right
azimuthal_integration_profile:
  p_cancer: 0.51234
  per_measurement_p_cancer: [0.50120, 0.51492, 0.52087]
final_prediction:
  p_cancer: 0.48611
  reference_class: BENIGN
  target_class: CANCER
  level: TRA 4
  score_percentiles:
    reference_population: train_on_all_target-breast_cases
    all: 0.51000
    reference_class: 0.67000
    target_class: 0.25000
reliability:
  level: low
  reason: SK symmetry refinement is intentionally unavailable for contralateral scoring
symmetry:
  available: false
model_execution:
  scoring_path: azimuthal_integration_age
```

The contralateral score uses the same frozen model with SK symmetry inputs
neutralized. It provides internal evidence only. It has no
`target_class_risk_level`, no `biopsy_required`, and no independent decision
threshold application.

If no usable contralateral breast remains after preprocessing:

```yaml
available: false
side: unknown
azimuthal_integration_profile:
  p_cancer: unknown
  per_measurement_p_cancer: []
final_prediction:
  p_cancer: unknown
  reference_class: BENIGN
  target_class: CANCER
  level: unknown
  score_percentiles:
    reference_population: unknown
    all: unknown
    reference_class: unknown
    target_class: unknown
reliability:
  level: unknown
  reason: unknown
reason: contralateral breast is unavailable after preprocessing
```

`model_metrics` are train-on-all target-case metrics for the selected frozen
artifact. `validation: not_performed` makes clear that they are not independent
validation estimates. Patient-safe evaluation remains in the model artifact and
adjacent evaluation files.
