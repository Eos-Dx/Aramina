# Preprocess-train YAML

`preprocess-train` runs the approved preprocessing YAML once, saves its traceable DataFrame artifact, passes the DataFrame directly to training, and writes all run artifacts under one output folder.

```yaml
contract: aramis_preprocess_train_config_v0_1
preprocess_train:
  name: aramis_m2q_t100_preprocess_train
  created_by: Sergey Denisov
  output_folder: ./examples/outputs/preprocess_train
preprocessing_config_path: ./config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
training_config_path: ./config/training/aramis_m2q_t100_primary_train_v0_1.yaml
```

All relative paths resolve from the Aramis project root. `created_by` is the person initiating the combined run. The referenced preprocessing and training configs remain separate immutable snapshots in the resulting model artifact.

```bash
cd /path/to/Aramis
python -m aramis preprocess-train \
  --config config/preprocess_train/aramis_biopsy_patients_primary_preprocess_train_v0_1.yaml
```

Output:

```text
<unique preprocess-train folder>/preprocessing/dataframe.joblib
<unique preprocess-train folder>/preprocessing/cohort_summary.json
<unique preprocess-train folder>/training/<unique training folder>/...
```
