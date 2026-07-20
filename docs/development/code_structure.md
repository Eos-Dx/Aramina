# Aramis Code Structure

## Purpose

This document defines the internal source-code boundaries for the Aramis
research-draft decision-support product. It does not change the clinical
model, preprocessing, labels, threshold, or validation protocol.

## Module Boundaries

| Module | Responsibility |
|---|---|
| `pipelines.py` | Build and run YAML-declared preprocessing. |
| `workflows.py` | Run the combined `preprocess-train` workflow. |
| `patient_features.py` | Build historical target cases, LR1 evidence, age and reliability fields. |
| `symmetry_features.py` | Calculate target-versus-contralateral SK symmetry features. |
| `m2q_model.py` | Define LR1 and the gated final logistic estimator. |
| `training_model.py` | Fit LR1 and the final model on accepted target cases. |
| `training_evaluation.py` | Patient-safe repeated stratified folds and evaluation metrics. |
| `training_artifacts.py` | Build training lineage, reproducibility, evaluation and final-model payloads. |
| `model_description.py` | Write model descriptions, YAML and immutable artifact identifiers. |
| `training.py` | Public training API and sklearn-compatible orchestration only. |
| `prediction_contract.py` | Validate prediction YAML, H5 v0.3 and output paths. |
| `prediction_scoring.py` | Score target and contralateral breasts using the frozen artifact. |
| `prediction_reports.py` | Construct internal and external reports. |
| `prediction.py` | Public prediction API and orchestration only. |

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

- Product source files under `src/aramis/` should remain at or below 700 lines.
- A module has one responsibility. Do not mix feature calculation, evaluation,
  artifact serialisation and report construction.
- Preserve public APIs: `run_preprocessing_from_config`,
  `run_training_from_config`, `run_preprocess_train_from_config`, and
  `run_prediction_from_config`.
- Preserve the import path of any custom class stored in a released joblib.
  `GatedSymmetryLogistic` remains in `aramis.m2q_model` for this reason.
- Structural changes must not silently alter product behaviour.

## Required Verification

After structural refactoring:

```bash
ruff check .
pytest -q
```

The frozen final model and all three one-patient H5 examples must retain their
stable target and contralateral `p_cancer` values. The regression test in
`tests/test_prediction.py` enforces this requirement.
