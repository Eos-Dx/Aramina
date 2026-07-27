# Codex Rules For Aramina Product Development

These instructions control Codex behavior in this repository.

Aramina and related EOS products are medical decision-support software.

## Regulatory Posture

- Treat all model work as FDA-aligned clinical decision support / SaMD research.
- Do not present any result as FDA-cleared, clinically validated, or released unless explicit evidence exists in the repository.
- Use research/draft language by default.
- Never describe the model as replacing a radiologist.
- Never describe the output as autonomous diagnosis.
- Prefer: `decision support`, `risk score`, `p_cancer`, `suggested class`, `requires clinical review`.

## Product Intent

Aramina:

```text
Target population: women with BI-RADS 3 or BI-RADS 4 findings
Clinical question: does this patient likely need biopsy?
Clinical user: radiologist / qualified breast-imaging clinician
Output: p_cancer and BENIGN/CANCER decision-support class
```

Bremen:

```text
Clinical question: should patient continue to MRI?
Clinical user: radiologist / qualified breast-imaging clinician
Output: MRI-continuation decision support
```

Do not mix Aramina and Bremen target populations, endpoints, filters, or labels.

## Required Pipeline Discipline

Preserve the product lineage:

```text
H5 container
-> H5SessionSelectorTransformer / product H5 filters
-> H5MeasurementSetAuditTransformer(optional metadata-only audit)
-> H5ToDataFrameTransformer / h5_to_df
-> ProductStatusGroupFilter / PatientFilter
-> FaultyPixelDetector
-> AzimuthalIntegration(error_model="poisson")
-> SNRTransformer(snr_method="poisson")
-> SNRFilter(min_snr_db=18.0)
-> QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
-> product-specific model
```

Do not silently change:

```text
H5 reader
product filter
faulty-pixel rule
PONI / integration settings
SNR method
SNR threshold
q normalization window
label mapping
feature schema
model threshold
train/test split logic
```

If changing any of these, update docs, tests, MLflow logging, and change-control notes.

## FDA-Aligned Development Behavior

Every modeling task must preserve:

```text
intended use
target population
clinical user
label definition
input data contract
preprocessing version
model version
decision threshold
limitations
validation status
```

Always check for:

```text
patient leakage
specimen leakage
measurement-level leakage
single-class training data
missing labels
ambiguous labels
silent row drops
unstable feature schema
untracked preprocessing changes
```

When uncertain about clinical target, label mapping, endpoint, threshold, or release claim, ask the user before editing.

## Traceability

MLflow is not part of the current product baseline. Traceability is stored in
preprocessing, evaluation, and model artifacts. Add MLflow only through a
separate controlled change after the product workflow is stable.

## Model Rules

Start with the simplest useful baseline.

Preferred first baselines:

```text
LogisticRegression
LightGBM
```

Always report:

```text
sensitivity
specificity
ROC AUC
balanced accuracy
PPV
NPV
confusion matrix
threshold
confidence intervals when available
```

For Aramina, false negatives are safety-critical.

For Bremen, track both false negatives and MRI-workflow burden.

## Release Language

Use:

```text
research draft
prototype
model-development run
not for autonomous diagnosis
requires radiologist review
not clinically validated unless explicit validation exists
```

Do not use:

```text
diagnoses cancer
rules out cancer
FDA-ready
clinically proven
approved
safe for deployment
replacement for biopsy/MRI/radiologist
```

## Code Change Rules

- Prefer repo-local reusable modules over notebook-only logic.
- Keep notebooks thin.
- Put reusable training, MLflow, filtering, and evaluation code in `src/aramina`.
- Add tests for every reusable function.
- Run `ruff check .` and `pytest -q` after code edits.
- For marimo notebooks, also run `python -m marimo check <notebook.py>`.
