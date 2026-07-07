# Aramis Development Cleanup Plan v0.1

Status: research draft.

Goal: turn the experiment branch state into a product-clean development branch
centered on one candidate model.

## Fixed Candidate

```text
preprocessing: T100 biopsy-patient model-input DataFrame
model_id: aramis_m1q_t100_train_all_c0p1
selected_model: M1Q
regularization: L2 LogisticRegression, C=0.1
threshold_target: 0.327873
prediction target side: required in predict YAML
```

## Done

- [x] Merge model-selection experiment into `aramis-development-v0.1`.
- [x] Keep experiment evidence documents for T70/T100/T130 and C-grid.
- [x] Fix prediction contract: `patient.target_side` comes from predict YAML.
- [x] Fix prediction contract: `model.model_id` is required and checked against
  `training.name` in the model artifact.
- [x] Update primary training YAML to produce the M1Q/T100/C=0.1 train-all
  candidate artifact.
- [x] Point prediction YAML templates to the M1Q/T100/C=0.1 model artifact.

## Next Cleanup Items

- [ ] Regenerate the primary preprocessing artifact with the product T100 YAML
  if the local output is stale.
- [x] Run the primary train YAML and confirm the resulting artifact has:
  `training.name = aramis_m1q_t100_train_all_c0p1`.
- [x] Run one prediction smoke test against the generated primary artifact.
- [ ] Move exploratory scripts/configs under clearly marked experiment folders
  or keep them only as model-selection references.
- [ ] Keep the main README focused on three commands:
  `aramis preprocess`, `aramis train`, `aramis predict`.
- [ ] Add final report schema after predict JSON/YAML fields are locked.
- [ ] Add MLflow after the product-clean preprocess/train/predict route is
  stable.

## Non-Product Evidence

The following remain evidence, not product execution defaults:

```text
config/training/model_selection_m1q_v0_1/
examples/aramis_m1q_threshold_mode_experiment_v0_1.py
examples/aramis_m1q_regularization_experiment_v0_1.py
docs/modeling/m1q_threshold_mode_comparison_v0_1.md
docs/modeling/m1q_regularization_experiment_v0_1.md
```
