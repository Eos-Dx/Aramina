# Product Development Rules

These rules are for Aramis and future EOS medical-model products.

They are development controls, not a regulatory clearance claim.

## Regulatory Posture

Treat every model as clinical decision support / SaMD until proven otherwise.

The product must help a qualified clinician decide next action.

The product must not make an autonomous diagnosis.

The product must show enough evidence, uncertainty, and limitations for a
radiologist to understand when to rely on it.

Primary FDA references:

```text
Clinical Decision Support Software guidance
Good Machine Learning Practice for Medical Device Development
Predetermined Change Control Plan guidance for AI-enabled device software
Transparency for Machine Learning-Enabled Medical Devices
Software as a Medical Device guidance
```

## Product Intent

### Aramis

Clinical question:

```text
For women with BI-RADS 3 or BI-RADS 4 findings, does the patient likely need biopsy?
```

Draft output:

```text
p_cancer
suggested class: BENIGN or CANCER
```

Clinical user:

```text
radiologist or qualified breast-imaging clinician
```

Clinical use:

```text
decision support for biopsy decision
not final diagnosis
not replacement for radiologist judgment
```

Target population:

```text
women with BI-RADS 3 or BI-RADS 4
```

### Bremen

Clinical question:

```text
Should patient continue to MRI?
```

Clinical use:

```text
decision support for MRI continuation workflow
not final diagnosis
not replacement for radiologist judgment
```

## Product Separation

The same H5 container may feed different products.

Each product must define its own:

```text
intended use
target population
inclusion criteria
exclusion criteria
label definition
clinical endpoint
model dataset
decision threshold
performance requirements
```

Never mix Aramis and Bremen filters silently.

Every MLflow run must include:

```text
product_name
product_version
intended_use_id
input_h5_id
input_h5_checksum
product_filter_rules
selected_measurement_ids
dropped_measurements_with_reasons
preprocessing_config
model_dataset_fingerprint
```

## Aramis Human-1 Versioning

Product versioning is tracked in:

```text
docs/meta/aramis_product_versioning.json
```

Conservative K-alpha-only rule:

```text
include data_batch: 3, 4, 5, 7
exclude data_batch: 1, 2, 6
review required: null
```

Batch 7 is K-alpha and product-usable according to the canonical JSON.

AGBH reference thickness:

```text
before 2026-04-22: 40 mm
from 2026-04-22: 10 mm
preferred H5/DataFrame field: calibrant_thickness_mm
```

If the H5 container lacks these fields, add them before product dataset build:

```text
calibrant_thickness_mm
kbeta_absent
xray_spectrum
product_batch_usable
product_batch_id
human1_data_batch
product_protocol_version
```

## Standard Data Pipeline

All products start from the same controlled preprocessing contract:

```text
h5_to_df
PatientFilter
FaultyPixelDetector
AzimuthalIntegration(error_model="poisson")
SNRTransformer(snr_method="poisson")
SNRFilter(min_snr_db=20.0)
QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")
product-specific analysis
```

Any deviation from this pipeline requires:

```text
reason
code diff
MLflow run comparison
clinical review
version bump
```

## Dataset Rules

Each row used for training must be traceable to:

```text
patient_id
specimen_id
measurement_id
raw H5 container
raw detector source
preprocessing version
label source
label timestamp or source file version
```

Forbidden:

```text
patient leakage between train/test
measurement-level split when patient/specimen leakage is possible
training on unlabeled or ambiguous labels without explicit rule
silent relabeling
silent row dropping
```

Required:

```text
patient-safe split
locked label mapping
locked feature schema
stored train/test split manifest
stored selected and dropped measurement IDs
```

## Label Rules

Aramis current label grouping is defined at `specimenId` / breast-side level.
Full branch-specific preprocessing rules are in:

```text
docs/data_preprocessing.md
```

Current grouping:

```text
BENIGN -> BENIGN
CANCER/PRE_CANCEROUS/ATYPICAL -> CANCER
NORMAL -> NORMAL
NA -> exclude
```

If clinical team changes this mapping, create a new label mapping version.

Do not overwrite old labels.

## Model Rules

First model can be simple.

Allowed draft models:

```text
LogisticRegression
LightGBM
```

Required outputs:

```text
p_cancer
class threshold
predicted class
model version
preprocessing version
```

Later required outputs:

```text
confidence interval
calibration plot
out-of-distribution warning
reason for abstention
```

## Clinical Performance Rules

Report at minimum:

```text
sensitivity
specificity
ROC AUC
balanced accuracy
PPV
NPV
confusion matrix
threshold
confidence intervals
```

For Aramis, sensitivity and false-negative analysis are safety-critical.

For Bremen, false-negative and false-positive workflow burden must both be
tracked.

## MLflow Rules

One MLflow run equals one complete product dataset build plus model training.

Log preprocessing and modeling together.

Required MLflow artifacts:

```text
preprocessing_config.json
product_filter_rules.json
selected_measurement_ids.csv
dropped_measurements.csv
preprocessed_dataset.parquet or .csv
feature_schema.json
label_mapping.json
train_test_split.csv
model.joblib
metrics.json
predictions.csv
```

Required MLflow tags:

```text
product = Aramis or Bremen
intended_use_id
pipeline_version
preprocessing_git_sha
model_git_sha
data_contract = eos_h5_v0.3
clinical_stage = research / locked_validation / released
```

## Change Control

Any change below creates a new controlled run:

```text
H5 container version
h5_to_df behavior
patient/product filter
faulty-pixel rules
PONI/integration settings
SNR method or threshold
q-range normalization
label mapping
feature schema
model type
decision threshold
```

For future adaptive or frequently updated models, define a Predetermined Change
Control Plan before release.

## Human Review

Before using any model outside research:

```text
radiologist review
clinical risk review
failure case review
bias/subgroup review
software verification
data lineage audit
locked validation run
```

The UI must show:

```text
prediction
probability
threshold
model version
data quality flags
SNR status
limitations
not-for-autonomous-diagnosis statement
```

## Stop Conditions

Do not train or report a model as usable when:

```text
single class only
patient leakage detected
unknown label mapping
missing raw-data traceability
missing preprocessing config
missing train/test manifest
unstable feature schema
unreviewed clinical target
```

## Release Rule

Research notebook results are not product release.

Product release requires:

```text
locked code
locked preprocessing
locked labels
locked validation dataset
locked threshold
traceable MLflow run
clinical sign-off
documented intended use
documented limitations
```
