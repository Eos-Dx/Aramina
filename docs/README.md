# Aramina Documentation

Use this page as the documentation entry point. Each product fact has one
canonical owner; other documents link to it instead of repeating it.

## Product

| Question | Canonical document |
|---|---|
| What architecture does current source train? | [Current FPCA30 model](modeling/aramina_fpca30_target_case_model_v0_3.md) |
| Why use a profile basis, and why 30 FPCA coefficients? | [FPCA30 basis decision](modeling/fpca30_component_selection_v0_1.md) |
| What executable model artifacts are preserved? | [Model artifacts](../models/README.md) |
| What was retrained with XRD v0.1.9-beta? | [Candidate record](modeling/aramina_t100_target_case_model_v0_2.md) |
| What data enter training? | [Current model DataFrame](modeling/current_model_dataframe_v0_1.md) |
| How are profiles preprocessed? | [Data preprocessing](data_preprocessing.md) |
| How is one patient scored? | [Prediction pipeline](modeling/prediction_pipeline_v0_1.md) |
| What do symmetry fields mean? | [Symmetry features](modeling/sk_symmetry_features_v0_1.md) |
| How is TRA derived? | [TRA decision record](modeling/tra_decision_record_v0_2.md) |
| What are the product limits? | [Development rules](product_development_rules.md) |

## Contracts

| Input or output | Contract |
|---|---|
| Preprocessing YAML | [Preprocessing config](contracts/preprocessing_config_v0_2.md) |
| Training YAML | [Training config](contracts/training_config_v0_1.md) |
| Combined preprocessing and training YAML | [Preprocess-train config](contracts/preprocess_train_config_v0_1.md) |
| Prediction YAML | [Prediction config](contracts/prediction_config_v0_1.md) |
| Training outputs | [Model training results](contracts/model_training_results_v0_1.md) |
| Internal prediction report | [Internal report](modeling/internal_clinical_report_content_v0_9.md) |
| TRA policy | [TRA contract](contracts/tissue_risk_assessment_v0_2.md) |
| Acquisition protocol | [Aramina acquisition protocol](contracts/aramina_acquisition_protocol_v0_1.md) |

Runnable input YAMLs live under [`config/`](../config/README.md). Filled output
examples live under [`contracts/`](../contracts/README.md). Frozen model files
under [`models/`](../models/README.md) are immutable run outputs, not a second
documentation source.

## Interfaces

- [Product CLI and Python API](product_api.md)
- [Install and environment](eosproduct_environment.md)
- [Data release policy](../DATA_RELEASE.md)

## Evidence And Development

- [AgBH exclusion rationale](agbh_quality_exclusions.md)
- [Controlled Human-1 metadata](meta/README.md)
- [Code structure](development/code_structure.md)
- [Future development](future_development_steps.md)
- [Patient collection and independent validation plan](development/Aramina_Patient_Collection_and_Independent_Validation_Plan.docx)

## Documentation Rules

- `config/` defines runnable inputs.
- `docs/contracts/` defines fields and validation rules.
- `contracts/` contains filled examples only.
- `models/<model_id>/` records one immutable training result.
- `docs/modeling/` explains architecture, evidence, and limitations.
- `docs/meta/` preserves controlled metadata and decision evidence.
- Historical experiments stay on experiment branches.
