# Aramis Model Grid v0.1

One YAML trains one research-draft decision-support model variant.

Naming:

```text
NN_<model>_<validation_mode>_<dataset>_model_v0_1.yaml
```

Models:

```text
M0: profile LogisticRegression logit-averaged to patient p_cancer
M1: M0 plus same-patient SK symmetry block
M2: M1 plus age and age_available
```

Current v0.1-beta primary candidate is M1. M2 is kept as an age
audit/comparison branch.

SK means Slava Kubitskyi-style symmetry features. The older current-cosine
symmetry block is retained only for audit/comparison.

Validation modes:

```text
all_on_all: train and score on the same patient table; optimistic sanity check only
loovm: leave-one-patient-out validation mode; pooled left-out predictions
stratified_kfold: patient-level StratifiedKFold, n_splits=5
```

Datasets:

```text
biopsy_patients: examples/outputs/model_input/aramis_biopsy_patients_model_input_v0_1.joblib
  primary v0.1-beta training dataset
  inferred target breast = biopsied breast

all_patients: examples/outputs/model_input/aramis_all_patients_model_input_v0_1.joblib
  exploratory only
  label-policy / metadata sensitivity check
```

Run example:

```bash
cd /Users/sad/dev/Aramis
conda activate eosproduct
python -m aramis train --config config/training/model_grid_v0_1/13_m1_all_on_all_biopsy_patients_model_v0_1.yaml
```
