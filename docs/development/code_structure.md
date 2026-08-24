# Aramina Code Structure

## Purpose

This document defines the internal source-code boundaries for the Aramina
research-draft decision-support product. It does not change the clinical
model, preprocessing, labels, threshold, or validation protocol.

## Module Boundaries

| Module | Responsibility |
|---|---|
| `config_paths.py` | Apply one documented relative-path policy to public YAML. |
| `pipelines.py` | Build and run YAML-declared preprocessing. |
| `workflows.py` | Run the combined `preprocess-train` workflow. |
| `patient_features.py` | Build historical target cases, LR1 evidence, age and reliability fields. |
| `symmetry_features.py` | Calculate target-versus-contralateral SK symmetry features. |
| `target_breast_model.py` | Define the profile and gated final estimators. |
| `m2q_model.py` | Compatibility imports for released joblib artifacts only. |
| `training_model.py` | Fit LR1 and the final model on accepted target cases. |
| `training_evaluation.py` | Patient-safe repeated stratified folds and evaluation metrics. |
| `model_metrics.py` | Calculate shared discrimination, threshold, calibration, and confusion metrics. |
| `training_artifacts.py` | Build training lineage, reproducibility, evaluation and final-model payloads. |
| `mlflow_artifacts.py` | Build required product dataset, split, metric, prediction and model tracking files. |
| `mlflow_tracking.py` | Manage one bounded fail-closed MLflow run without changing model behavior. |
| `model_description.py` | Write model descriptions, YAML and immutable artifact identifiers. |
| `runtime_identity.py` | Shared SHA256, file-stem, package-version and Git identity helpers. |
| `training.py` | Public training API and sklearn-compatible orchestration only. |
| `prediction_contract.py` | Validate prediction YAML, H5 v0.3 and output paths. |
| `prediction_scoring.py` | Score target and contralateral breasts using the frozen artifact. |
| `prediction_reports.py` | Construct internal and external reports. |
| `prediction.py` | Public prediction API and orchestration only. |
| `prediction_api.py` | Immutable-model HTTP adapter for multipart H5 requests. |

## Dependency Direction

```text
CLI / workflows
  -> preprocessing | training | prediction
  -> feature, estimator, evaluation, artifact, contract, report modules
```

`prediction.py` must not import `training.py`. Shared target-case calculations
belong in `patient_features.py`. Reporting must not fit models. Evaluation must
not write model artifacts.

## Size and Refactor Rules

- Product source files under `src/aramina/` should remain at or below 700 lines.
- A module has one responsibility. Do not mix feature calculation, evaluation,
  artifact serialisation and report construction.
- Preserve public APIs: `run_preprocessing_from_config`,
  `run_training_from_config`, `run_preprocess_train_from_config`, and
  `run_prediction_from_config`.
- Preserve import compatibility for custom classes stored in released joblibs.
  `aramina.m2q_model` remains a compatibility module; new code uses
  `aramina.target_breast_model`.
- Structural changes must not silently alter product behaviour.

## Required Verification

After structural refactoring:

```bash
ruff check .
pytest -q
```

Released model artifacts and their one-patient H5 examples must retain stable
target and contralateral `p_cancer` values. An intentional preprocessing,
feature, or model change requires a new evaluated artifact and updated regression
expectations before release.
