# TRA Decision Record v0.1

Status: research draft.

## Decision

Aramis reports a Tissue Risk Assessment (`TRA`) level for every available final
breast prediction. The level is an ordinal representation of the final model
score relative to the frozen target-breast reference cohort stored in the
selected model artifact.

```text
TRA index = percentage of frozen final target-case scores <= incoming final p_cancer
```

The internally calculated index ranges from `0` to `100` and assigns the
reported level. A higher TRA level indicates that the model score is high
relative to historical target-breast cases and therefore gives stronger model
support for considering biopsy. The index itself is not emitted in either
report. TRA does not replace radiologist review and is not an individual cancer
probability or diagnosis.

| TRA level | TRA index | Meaning |
| --- | --- | --- |
| `TRA 1` | below 20 | lowest model-score range |
| `TRA 2` | 20 to below 50 | lower model-score range |
| `TRA 3` | 50 to below 80 | intermediate model-score range |
| `TRA 4` | 80 to below 90 | high model-score range |
| `TRA 5` | 90 to 100 | highest model-score range |

The threshold-derived `suggested_class` remains the formal decision-support
class. `reliability` remains independent and describes input-data sufficiency.
TRA does not alter the fixed threshold or reduce a score for low reliability.

## Frozen Artifact Content

The model joblib contains:

```yaml
tissue_risk_assessment:
  contract: aramis_tra_v0_1
  reference_score: final_prediction.p_cancer
  reference_population: train_on_all_target-breast_cases
  levels:
    - {level: TRA 1, minimum_percentile: 0, maximum_percentile: 20}
    - {level: TRA 2, minimum_percentile: 20, maximum_percentile: 50}
    - {level: TRA 3, minimum_percentile: 50, maximum_percentile: 80}
    - {level: TRA 4, minimum_percentile: 80, maximum_percentile: 90}
    - {level: TRA 5, minimum_percentile: 90, maximum_percentile: 100}
```

The frozen reference distribution is the final-fit distribution for the 175
accepted historical target-breast cases in the T100 cohort. Incoming scans do
not modify it.

## Interpretation Limit

The current cohort is small. Patient-safe repeated stratified 5-fold evaluation
demonstrated discrimination but does not establish a stable cancer prevalence
for each TRA level. TRA must therefore be presented as a model-score tier, not
as a calibrated clinical risk band. A later independent validation cohort can
replace this ordinal scale with a clinically calibrated representation without
changing the underlying prediction-record structure.
