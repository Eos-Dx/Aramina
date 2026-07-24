# Model Training Results Contract v0.1

Status: research draft. This contract describes artifacts written by
`python -m aramis train` and `python -m aramis preprocess-train` when
`run.train_on_all: true`.

## Output Set

```text
model.joblib
model_description.yaml
evaluation.yaml                         # when run.evaluation: true
evaluation_metrics.csv                  # when run.evaluation: true
evaluation_predictions.csv              # when run.evaluation: true
```

`model.joblib` is executable. YAML files are human-readable internal records.
CSV is used only for row-oriented data: one row per evaluation fold and one
row per held-out target-breast prediction. YAML numbers are rounded to five
decimal places; IDs, SHA256 values, labels, and paths remain strings.

## model.joblib

Required top-level sections:

```yaml
kind: aramis_training_artifact
version: "0.3"
model_identity: {name, version, model_author, clinical_stage, intended_use}
models: {aramis_target_breast_risk: <fitted LR1 and final sklearn estimators>}
feature_schema: {final_model: ...}
model_performance: ...
final_fit_training_metrics: ...
evaluation: ...
training_config_yaml: <resolved training YAML>
historical_preprocessing_yaml: <resolved historical preprocessing YAML>
prediction_preprocessing_yaml: <resolved operational preprocessing YAML>
prediction_contract_yaml: <embedded prediction contract YAML>
reproducibility: <H5 checksum, source-code provenance, runtime versions>
```

`model_performance` is the held-out validation record: patient-safe method,
fold count, repeat count, seed, target sensitivity, and ROC AUC, sensitivity,
and specificity as fold mean and standard deviation.

The final model entry also contains `tissue_risk_assessment`, a frozen
`aramis_tra_v0_2` policy. It is derived automatically from the final threshold
and the patient-safe OOF predictions produced during this training run. It
records the OOF calibration population, decision-stability counts, logit-margin
boundaries, and threshold-dependent probability boundaries. TRA does not alter
`p_cancer` or the target-side `biopsy_required` action. Full contract:
`docs/contracts/tissue_risk_assessment_v0_2.md`.

`final_fit_training_metrics` describes the frozen model fitted on all accepted
target-breast cases:

```yaml
evaluation_status: in_sample_not_independent
target_cases: <integer>
cancer_target_cases: <integer>
benign_target_cases: <integer>
decision_threshold: <float>
roc_auc: <float>
pr_auc: <float>
sensitivity: <float>
specificity: <float>
balanced_accuracy: <float>
ppv: <float>
npv: <float>
brier_score: <float>
log_loss: <float>
calibration_intercept: <float>
calibration_slope: <float>
true_positives: <integer>
true_negatives: <integer>
false_negatives: <integer>
false_positives: <integer>
```

These are in-sample values: they describe the fitted artifact, not expected
performance on new patients.

## model_description.yaml

Internal human-readable projection of the final model. Required fields:

```yaml
output_type: aramis_model_description
version: "0.1"
model:
  id: <model name + version + first 12 SHA256 characters>
  name: aramis_target_breast_risk
  version: <version>
  artifact_sha256: <full SHA256>
model_summary:
  architecture:
    stage_1: target_xrd_profile_logistic_regression
    stage_2: age_and_optional_symmetry_refinement
    symmetry_behavior: neutralized_unless_2_valid_measurements_per_breast_and_finite_core4_features
  lr1_profile_model: <fitted pipeline summary>
  final_model: <fitted gated logistic-regression summary>
  symmetry_feature_contract: aramis_sk_symmetry_v0_2
model_joblib: model.joblib
model_performance: <held-out validation record>
final_fit_training_metrics: <in-sample final-fit record>
decision_thresholds: <Youden and target-sensitivity thresholds>
feature_schema: {final_model: <feature schema>}
dataset_summary: <accepted-cohort counts>
evaluation_artifacts: {summary, metrics, predictions}
reproducibility: <H5 checksum, config checksums, code and runtime provenance>
clinical_stage: research draft
```

Logistic-regression `classes` must be `BENIGN` and `CANCER`, never `0` and
`1`.

## evaluation.yaml

Complete aggregated patient-safe validation footprint:

```yaml
output_type: aramis_evaluation_artifact
version: "0.1"
created_at: <ISO-8601 timestamp with second precision>
model: <same immutable identity as model_description.yaml>
threshold_selection: train_fold_target_sensitivity
target_sensitivity: 0.95
training_config_sha256: <SHA256 of the frozen training_config.yaml text>
decision_threshold:
  id: target_sensitivity_0_95
  value: <final train-on-all threshold>
dataset_summary: <accepted-cohort counts>
metric_summary:
- evaluation_mode: stratified_kfold
  evidence_status: patient_safe_validation
  splits: 100
  <mean/std, pooled and 95% CI metrics>
files:
  metrics: evaluation_metrics.csv
  predictions: evaluation_predictions.csv
```

`evidence_status: patient_safe_validation` means all target cases from one
patient are exclusively in either train or held-out data in every fold.
`metric_summary` includes ROC AUC, PR AUC, thresholded classification metrics,
probability-quality metrics, calibration, patient-bootstrap 95% confidence
intervals, and mean TP/TN/FP/FN.

The final train-on-all threshold is retained for traceability. Held-out fold
sensitivity and specificity use a threshold selected on that fold's training
data; they are not recomputed with the final train-on-all threshold.

## CSV Artifacts

`evaluation_metrics.csv`: one row per fold, including fold ID, train/test
counts, ROC AUC, PR AUC, thresholded metrics, calibration, confusion counts,
and train-fold thresholds.

`evaluation_predictions.csv`: one row per held-out target-breast case per fold,
including target-case ID, patient ID, true label, predicted probability,
threshold, predicted class, and fold ID.

Artifact-file references are sibling relative names. The full model directory
is therefore portable as one unit; model ID and SHA256 remain immutable
identifiers.

## Promotion

`python -m aramis promote --run-folder <completed-run-folder>` copies a reviewed
final-fit run to `models/<immutable_model_id>/`. Promotion does not retrain or
modify the source run, refuses an existing destination, and requires the
evaluation YAML plus its fold CSVs. It verifies that model name/version,
artifact SHA256, feature schema, and evaluation model reference match the
executable joblib before copying the complete portable model directory.
