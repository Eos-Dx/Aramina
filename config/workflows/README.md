# Preprocess-train workflow

The workflow runs preprocessing once, saves its DataFrame artifact, and passes
the in-memory DataFrame directly into training.

```yaml
contract: aramis_preprocess_train_workflow_v0_1
workflow:
  name: aramis_m2q_t100_preprocess_train
  created_by: Sergey Denisov
  created_at: "2026-07-14"
  output_folder: ../../examples/outputs/workflows
preprocessing_config_path: ../preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
training_config_path: ../training/aramis_m2q_t100_primary_train_v0_1.yaml
```

Run:

```bash
python -m aramis preprocess-train \
  --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
```

Output:

```text
<unique workflow folder>/preprocessing/dataframe.joblib
<unique workflow folder>/training/<unique training folder>/...
```

Unknown fields fail immediately. Paths are relative to the YAML that declares
them.
