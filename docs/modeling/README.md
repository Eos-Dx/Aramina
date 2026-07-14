# Aramis Modeling Notes

Status: research draft.

Product-clean documents:

```text
current_model_pipeline_and_risks_v0_1.md
current_model_dataframe_v0_1.md
final_candidate_model_artifact_v0_1.md
training_pipeline_classes_v0_1.md
prediction_pipeline_v0_1.md
reliability_quality_models_v0_1.md
patient_model_structures_v0_2.md
modeling_results_interpretation_v0_1.md
t100_peak_delta_symmetry_experiment_v0_1.md
t100_m2q_symmetry_feature_selection_v0_1.md
t100_m2q_pairwise_cosine_experiment_v0_1.md
t100_m0q_m1q_m2q_controlled_comparison_v0_1.md
```

Additional archived decision records retained on this branch:

```text
age_conditional_incremental_value_v0_1.md
honest_operational_model_experiment_v0_1.md
m2q_age_candidate_update_v0_1.md
m2q_core4_optional_symmetry_candidate_v0_1_9.md
m2q_core4_paired_candidate_v0_1_8.md
```

Archived experiment documents may still mention removed result CSVs. Their full
source tables and exploratory scripts are preserved on:

```text
experiment/aramis-v0.1-research-state
```

Current product direction:

```text
preprocess: YAML-governed XRD preprocessing
train: YAML-governed patient-level model training
predict: model joblib supplies prediction preprocessing YAML, incoming H5 supplies patient scans
primary candidate under experiment: M2Q-style profile risk plus same-patient symmetry plus reliability fields plus age
```

All results remain research-draft decision-support evidence only. They do not
represent clinical validation, FDA clearance, or autonomous diagnosis.
