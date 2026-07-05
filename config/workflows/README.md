# Aramis Workflow YAML

Workflow YAML runs a complete research-draft product route:

```text
preprocess YAML -> preprocessing joblib -> train YAML -> model joblib
```

The workflow file stores only references to the two sub-YAML files. The
preprocessing YAML remains the source of truth for H5-to-DataFrame construction.
The training YAML remains the source of truth for model training and evaluation.

Default mode is `memory`:

```text
preprocess builds DataFrame
preprocess saves preprocessing joblib footprint
the same in-memory DataFrame is passed directly to train
train writes model joblib
```

This avoids reloading the preprocessing joblib during one combined run while
still preserving the dataset artifact. Use `mode: artifact` when the training
step should explicitly reload the saved preprocessing joblib.

Run current product-clean workflow:

```bash
python -m aramis run --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml
```

Required fields:

```text
workflow.name
workflow.mode
workflow.run_preprocessing
workflow.run_training
workflow.validate_io_match
preprocessing.config_path
training.config_path
```

When `validate_io_match` is true, Aramis checks that:

```text
preprocessing.io.output_joblib_path == training.io.input_dataframe_joblib_path
```

This prevents silently training on an old or different preprocessing artifact.
