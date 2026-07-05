# Aramis Training YAML Layout

Runnable training configs define:

```text
training.branch
io.input_dataframe_joblib_path
io.output_model_joblib_path
model profile/label/group/specimen columns
model.extra_feature_columns, for scalar features appended to radial profiles
evaluation patient-safe split settings
```

Current first route:

```text
aramis_v0_1_beta_primary_train.yaml
aramis_one_to_many_logistic_v0_1.yaml
aramis_biopsy_patients_m0_m1_m2_v0_1.yaml
model_grid_v0_1/
```

`aramis_v0_1_beta_primary_train.yaml` is the current primary train command for
v0.1-beta. It uses biopsy-patient data and trains only M0/M1.

`aramis_biopsy_patients_m0_m1_m2_v0_1.yaml` is the biopsy-cohort comparison
route. It uses the same endpoint logic but also includes M2 as age
audit/comparison.

In the biopsy cohort, the biopsied breast is treated as the training target:
it is the clinically suspicious breast and has the BENIGN/CANCER endpoint.

`aramis_all_patients_m0_m1_m2_v0_1.yaml` is exploratory only. It exists to test
label-policy and metadata sensitivity, not to produce the primary product
model.

It loads a preprocessing artifact joblib, trains a one-to-many
LogisticRegression research-draft model, evaluates repeated patient-safe splits,
then writes a model artifact joblib.

Current one-to-many route uses normalized `radial_profile_data` as the primary
predictor. `sample_thickness_mm` remains available for audit/control checks but
is not used as a primary cancer predictor.

The patient route loads one preprocessing artifact joblib and writes one model
artifact joblib containing sklearn-style model entries:

```text
M0: LR1 profile p_cancer logit-averaged to patient level
M0Q: M0 + reliability/quality counters -> LogisticRegression
M1: M0 score + same-patient target/contralateral SK symmetry block -> LogisticRegression
M1Q: M1 + reliability/quality counters -> LogisticRegression
M2: M1 + age and age_available -> LogisticRegression
M2Q: M1Q + age and age_available -> LogisticRegression
```

SK means Slava Kubitskyi-style symmetry features. Current cosine symmetry
features remain in the patient feature table for audit/comparison, but the
primary M1/M2 feature schema uses the SK block.

Reliability is separate from risk. `p_cancer` is the decision-support risk
score. Reliability fields describe whether the score is based on enough valid
target/contralateral measurements. They should be reported as result confidence,
not used as a rule to directly lower `p_cancer`.

LR1 aggregation is:

```text
measurement p_cancer -> logit(p_cancer) -> mean logit -> sigmoid(mean logit)
```

The training artifact keeps both `profile_p_cancer_logit_average` for model use
and `profile_p_cancer_probability_mean` for audit.

The training YAML text and SHA256 are stored inside the model artifact. Optional
JSON and YAML summaries are written to paths under `io.output_json_path` and
`io.output_yaml_path`.

`model_grid_v0_1/` contains one runnable YAML per model and validation mode:

```text
NN_<model>_<validation_mode>_<dataset>_model_v0_1.yaml
```

The current grid is:

```text
primary dataset: biopsy_patients
exploratory dataset: all_patients
models: M0, M0Q, M1, M1Q, M2, M2Q
validation modes: all_on_all, loovm, stratified_kfold
```

`all_on_all` is an optimistic sanity check. `loovm` is leave-one-patient-out
with pooled left-out predictions. `stratified_kfold` uses patient-level
StratifiedKFold.
