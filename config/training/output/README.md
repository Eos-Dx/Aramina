# Training Output Examples

Canonical contract: `docs/contracts/model_training_results_v0_1.md`.

This directory contains small, non-clinical examples of the files written by
`aramis train` and `aramis preprocess-train`. They demonstrate the schema only.
Real model records are stored beside their `model.joblib` under `models/`.

```text
model_description_example.yaml
evaluation_example.yaml
evaluation_metrics_example.csv
evaluation_predictions_example.csv
```

CSV is used only for one-row-per-fold and one-row-per-held-out-case data.
