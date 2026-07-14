# Aramis Product Development Rules

Status: research draft.

These rules describe Aramis development controls. They are not a regulatory
clearance claim.

## Allowed Product Language

Use:

```text
decision support
p_cancer
suggested class
risk score
requires radiologist review
not for autonomous diagnosis
research draft
```

Do not use:

```text
diagnoses cancer
rules out cancer
biopsy replacement
radiologist replacement
clinically proven
approved
FDA-cleared
autonomous diagnosis
```

## Product Intent

```text
product: Aramis
target population: women with BI-RADS 3 / BI-RADS 4 suspicious findings
clinical question: does the clinically suspicious breast likely need biopsy?
clinical user: radiologist / qualified breast-imaging clinician
output: p_cancer, suggested BENIGN/CANCER class, reliability metadata
```

## Required Traceability

Every product dataset and model artifact must be traceable to:

```text
input H5 path and SHA256
preprocessing YAML text and SHA256
training YAML text and SHA256
prediction preprocessing YAML text and SHA256
Aramis version / git SHA
XRD-preprocessing version / tag
selected measurement rows
dropped measurement rows and reasons when retained by the preprocessing artifact
feature schema
label mapping
model threshold
```

## Preprocessing Lock

Do not silently change:

```text
raw data source
H5 filters
AgBH quality exclusions
PONI q-range requirement
sample thickness requirement
calibrant thickness requirement
faulty-pixel rule
azimuthal integration settings
SNR method or threshold
normalization window
profile gate
output columns
label mapping
```

Any change requires:

```text
reason
code/config diff
updated docs
updated tests
new model artifact or explicit statement that model artifact is unchanged
```

## Label Rules

Current product grouping is specimenId-level:

```text
BENIGN + NORMAL -> BENIGN
CANCER + ATYPICAL + PRE_CANCEROUS -> CANCER
NA -> excluded
```

Original `specimen_status` must be retained when available. Product labels must
be written to separate product columns.

## Split Rules

Model validation must be patient-safe:

```text
same patientId cannot appear in both train and test
measurement-level random split is forbidden
specimen-level split is insufficient when patient leakage is possible
```

Current product evaluation policy:

```text
use repeated patient-safe stratified 5-fold x20 with random_seed=42
  to evaluate the frozen M2Q recipe
use train-all only after evaluation to fit the final packaged artifact
derive the deployment threshold from train-all scores at sensitivity >=0.95
state clearly that train-all is fitted-cohort evidence, not validation
```

## Prediction Rules

Prediction must require:

```text
one patient per H5
H5 schema_version / format matching predict YAML
patient.patient_id matching H5 patientId
patient.target_side supplied by predict YAML
model identity derived from the selected model artifact SHA256
prediction_preprocessing_config stored inside model artifact
```

Prediction must output:

```text
p_cancer
suggested_class
threshold_key
threshold
risk_level
reliability
reliability_reason
provenance checksums
decision-support limitations
```

Risk and reliability must remain separate.

## Stop Conditions

Do not report a model as usable when:

```text
single-class data
unknown label mapping
patient leakage
missing preprocessing config
missing training config
missing prediction preprocessing config
unstable feature schema
unsupported H5 container schema
unknown target_side
missing model threshold
```
