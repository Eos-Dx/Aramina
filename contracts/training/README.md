# Training Output Examples

Canonical contract: `docs/contracts/model_training_results_v0_1.md`.

This directory contains small, non-clinical examples of the files written by
`aramis train` and `aramis preprocess-train`. They demonstrate the schema only.
Real model records are stored beside their `model.joblib` under `models/`.

```text
examples/model_description.yaml
examples/evaluation.yaml
examples/fold_metrics.csv
examples/fold_predictions.csv
```

CSV is used only for one-row-per-fold and one-row-per-held-out-case data.
