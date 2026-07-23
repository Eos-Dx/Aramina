# Tissue Risk Assessment Contract v0.1

Status: research draft. TRA is a frozen ordinal model-score index for Aramis
decision support. It is not an individual cancer probability, diagnosis,
clinical-risk calibration, or autonomous clinical decision.

## Reference Scale

For every final model, training stores the sorted final `p_cancer` values for
all accepted train-on-all target-breast cases. For an incoming breast score:

```text
TRA index = percentage of frozen reference scores less than or equal to p_cancer
```

The index is used internally to assign a TRA level. It is never recalculated
from incoming scans and is not emitted in the internal or external report.

| TRA level | Index range |
| --- | --- |
| `TRA 1` | below 20 |
| `TRA 2` | 20 to below 50 |
| `TRA 3` | 50 to below 80 |
| `TRA 4` | 80 to below 90 |
| `TRA 5` | 90 to 100 |

`TRA 5` therefore means that the final model score is at or above the 90th
percentile of the model's frozen target-breast reference cohort. It does not
mean a 90% cancer probability.

## Report Fields

Internal report stores the derived TRA level under both available breast
results:

```yaml
breast_predictions:
  target:
    final_prediction:
      level: TRA 3
  contralateral:
    final_prediction:
      level: TRA 3
```

External reports do not expose TRA. They expose `risk_probability`,
`decision_threshold`, and `target_class_risk_level`: `high` when the probability meets or
exceeds the threshold, otherwise `low`. `reliability` remains a separate
measurement-quality statement. TRA does not change either field.
