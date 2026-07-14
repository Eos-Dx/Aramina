# Aramis Documentation Audit v0.1

Reviewed for the gated M2Q target-case model on 2026-07-13.

| file | role after review | action |
|---|---|---|
| `README.md` | project entry point | current model and documentation-map links updated |
| `examples/README.md` | examples index | retained; direct training config marked as development fixture pending config packaging |
| `docs/data_preprocessing.md` | preprocessing contract | retained; no model-architecture change required |
| `docs/product_api.md` | developer API contract | target-case rule retained; model-artifact naming deferred to config stage |
| `docs/product_development_rules.md` | development controls | retained; applies to current architecture |
| `docs/mlflow.md` | future tracking contract | retained; implementation work remains separate |
| `docs/agbh_quality_exclusions.md` | calibration-quality rationale | retained; independent from model architecture |
| `docs/eosproduct_environment.md` | environment reference | retained; dependency/config refresh deferred to config stage |
| `docs/meta/README.md` | metadata provenance index | retained; target-case training rule added |
| `docs/modeling/README.md` | modeling entry point | current record moved to gated M2Q document |
| `docs/modeling/m2q_gated_target_case_model_v0_1.md` | current model record | added with architecture and current evidence |
| `docs/modeling/current_model_pipeline_and_risks_v0_1.md` | product interpretation and risks | updated to current gated architecture and evidence |
| `docs/modeling/current_model_dataframe_v0_1.md` | cohort and level definitions | updated for target-breast cases and bilateral rule |
| `docs/modeling/training_pipeline_classes_v0_1.md` | training implementation contract | updated for one LR2 and patient-safe target cases |
| `docs/modeling/prediction_pipeline_v0_1.md` | prediction contract | retained; current model-artifact reference deferred to config stage |
| `docs/modeling/sk_symmetry_features_v0_1.md` | SK mathematical definitions | retained; Core4 remains current block |
| `docs/modeling/internal_clinical_report_content_v0_1.md` | internal-report reference | retained; final target-side result and LR1-only contralateral output remain valid |
| `docs/modeling/Aramis_model_status_and_validation_limitations_v0_1.docx` | generated Word draft | retained outside the Markdown source-of-truth set |
| `docs/meta/*.json`, `*.csv`, `*.py` | controlled metadata and audit evidence | reviewed as data/source artifacts; metadata index remains `docs/meta/README.md` |

Historical experiment records are retained in
`experiment/aramis-model-selection-v0.1`, not in this product-development
branch.
