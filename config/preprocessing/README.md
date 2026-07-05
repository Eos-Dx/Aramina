# Aramis Preprocessing YAML

Status: research draft.

Runnable root YAMLs extend shared fragments from this folder. Aramis passes the
resolved YAML to `xrd_preprocessing`, which builds the sklearn-style transformer
pipeline and writes a preprocessing artifact joblib.

Product-clean model-input configs:

```text
aramis_all_patients_model_input_v0_1.yaml
aramis_biopsy_patients_model_input_v0_1.yaml
aramis_prediction_patient_model_input_v0_1.yaml
```

Current primary training data uses:

```text
aramis_biopsy_patients_model_input_v0_1.yaml
```

It keeps patients with at least one biopsy row, includes contralateral breast
rows for same-patient symmetry features, maps NORMAL to BENIGN, applies AgBH
T130 monochromaticity exclusions, and outputs only model/audit columns.

Prediction uses:

```text
aramis_prediction_patient_model_input_v0_1.yaml
```

This config is stored inside trained model joblibs. `aramis predict` loads it
from the model artifact, injects the incoming patient H5 path and output path,
then runs the same preprocessing lineage needed by the model.

Legacy notebook/dataframe examples kept for inspection:

```text
aramis_one_to_one_max_v0_1.yaml
aramis_one_to_one_min_v0_1.yaml
aramis_one_to_one_biopsy_max_v0_1.yaml
aramis_one_to_one_biopsy_min_v0_1.yaml
aramis_one_to_many_max_v0_1.yaml
aramis_one_to_many_min_v0_1.yaml
aramis_one_to_many_biopsy_max_v0_1.yaml
aramis_one_to_many_biopsy_min_v0_1.yaml
```

Shared fragments:

```text
shared/aramis_policy_v0_1.yaml
shared/aramis_pipeline_v0_1.yaml
outputs/max_output_v0_1.yaml
outputs/min_output_v0_1.yaml
outputs/model_input_output_v0_1.yaml
outputs/prediction_model_input_output_v0_1.yaml
branches/*.yaml
exclusions/agbh_quality_exclusions_v0_1.yaml
exclusions/agbh_quality_exclusions_t130_v0_1.yaml
```

Experimental threshold grids and old FDA-like cohorts are archived on branch:

```text
experiment/aramis-v0.1-research-state
```
