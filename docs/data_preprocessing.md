# Aramis Data Preprocessing

Status: research draft.

This document defines the current Aramis data-preprocessing contract for model
development. Aramis output remains decision support: `p_cancer` and suggested
BENIGN/CANCER class, requiring radiologist / qualified clinician review.

## Identifier Levels

All filtering and transformations must state the data level.

```text
measurementId level:
  one detector measurement / one XRD frame-derived profile

specimenId level:
  one breast side / specimen
  all valid measurements from the same specimen inherit the same product label

patientId level:
  one patient
  contains left and right breast-side specimen groups when both are available
```

Do not mix these levels silently. Counts and filters must report whether they
operate on `measurementId`, `specimenId`, or `patientId`.

## Shared H5-Level Product Filters

These filters run before GFRM decode and before `h5_to_df` materializes the
model DataFrame.

```text
1. measurementId/session level:
   exclude sessions whose AgBH calibration failed quality QC
   primary key: linked_agbh_session_uid
   fallback only when primary key column is absent: started_at date

2. measurementId/session level:
   keep sessions whose PONI geometry can provide the required q range

3. measurementId/session level:
   require sample thickness metadata
   no thickness means azimuthal integration with thickness correction is invalid

4. measurementId/session level:
   require calibrant_thickness_mm in H5 metadata
   current AgBH safety range: 10..40 mm

5. measurementId/session level:
   keep only canonical measurement positions P1, P2, P3 before frame loading

6. measurementId/session level:
   read detector data from the source declared in the branch preprocessing YAML
   allowed source values for Aramis v0.1: gfrm
```

`calibrant_thickness_mm` is calibrant-generic. Current AgBH data use 10 or 40
mm. Later calibrants, for example LaB6, may use different values but must keep
an explicit field and documented safety range.

Production policy:

```text
Aramis v0.1 product preprocessing uses gfrm only
raw_file/artifacts/gfrm are the only product H5 blob candidates
npy is allowed only in synthetic tests, not in product YAML
```

Runtime quality exclusions live in the branch preprocessing YAML:

```text
filters.quality_exclusions.primary_key.excluded_values
filters.quality_exclusions.fallback_date.excluded_dates
```

The reason and session-linking policy are documented in:

```text
Aramis/docs/agbh_quality_exclusions.md
```

Date fallback is compatibility-only. If `linked_agbh_session_uid` exists in H5,
preprocessing excludes by session ID, not by calendar date.

Thickness correction is required for azimuthal integration:

```text
filters.thickness.sample.column:
  H5/sample attribute for specimen thickness filtering
  current value: sample_thickness_mm

filters.thickness.calibrant.column:
  H5/session attribute for calibrant/reference thickness filtering
  current value: calibrant_thickness_mm

correction:
  AzimuthalIntegration receives the same two YAML column names
  missing or invalid thickness means the measurement cannot be used
```

The YAML thickness contract is shared by H5 filtering and integration. If the
filter column and integration column differ, preprocessing must fail instead of
silently integrating with another thickness attribute.

## Shared YAML-Declared XRD Pipeline

After H5 filtering and `h5_to_df`, all Aramis branches use the same ordered
XRD-preprocessing transformer route. Cohort-specific steps are enabled or
disabled from YAML.

```text
1. measurementId level:
   ProductColumnBuilder
   create auditable product_status_group / product_diagnosis columns

2. measurementId level:
   q-range, position, sample-thickness, and calibrant-thickness filters

3. specimenId / patientId level:
   optional biopsy filters
   one-to-many biopsy: row-level biopsy == true
   one-to-one biopsy: patient-level GroupValueFilter, any biopsy == true

4. specimenId / patientId level:
   ProductStatusGroupFilter
   optional PairedGroupFilter for one-to-one

5. measurementId level:
   FaultyPixelDetector
   output: one `faulty_pixel_mask` column

6. measurementId level:
   ConstantQRangeTransformer

7. measurementId level:
   AzimuthalIntegration(error_model="poisson")
   use YAML thickness_correction columns

8. measurementId level:
   SNRTransformer(snr_method="poisson")

9. measurementId level:
   SNRFilter(min_snr_db=18.0 in current notebooks)

10. specimenId / patientId level:
    PatientSpecimenValidityFilter

11. measurementId level:
    QRangeValueNormalizer(q_min=6.7, q_max=7.1, statistic="median")

12. measurementId level:
    RadialProfileValueFilter

13. output schema:
    KeepColumnsTransformer(metadata.output_columns + branch_settings.output_columns)
```

The historical canonical threshold was 20 dB. Current exploratory Aramis
notebooks use 18 dB. Any final product change must be versioned and logged in
MLflow with the selected measurement IDs and dropped measurement IDs.

## Pipeline Entrypoints

Current draft code composes reusable `xrd_preprocessing` transformers. Aramis
does not own preprocessing transformer implementations.

```text
XRD-preprocessing/src/xrd_preprocessing/config.py
  load_preprocessing_config(...)

XRD-preprocessing/src/xrd_preprocessing/pipeline.py
  build_pipeline_from_config(...)
  transformer registry
  YAML $ref / $concat / enabled handling

XRD-preprocessing/src/xrd_preprocessing/configs/preprocessing_branch_config_template.yaml
  reusable branch-specific preprocessing YAML template/contract

XRD-preprocessing H5ToDataFrameTransformer
  H5 path -> decoded measurement DataFrame
  applies h5_filters before GFRM frame loading
  materializes only selected SAMPLE/SAMPLE rows

Aramis/config/preprocessing/shared/aramis_pipeline_v0_1.yaml
  single ordered Aramis preprocessing route

Aramis/config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml
Aramis/config/preprocessing/aramis_one_to_one_max_v0_1.yaml
Aramis/config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
Aramis/config/preprocessing/aramis_one_to_many_max_v0_1.yaml
  concrete Aramis project preprocessing config
  separate branch configs because one-to-one, one-to-many, standard, and
  biopsy-only cohorts use different rules

src/aramis/pipelines.py
  AramisOneToOnePreprocessingPipeline(...).fit_transform(h5_path)
  AramisOneToManyPreprocessingPipeline(...).fit_transform(h5_path)
  run_one_to_one_preprocessing_pipeline(...)
  run_one_to_many_preprocessing_pipeline(...)
```

The pipeline classes follow the sklearn transformer contract:

```text
input object X:
  H5 container path

output object:
  final preprocessing DataFrame

fit(X):
  no-op, returns self

transform(X):
  reads the H5, applies configured preprocessing, returns DataFrame

fit_transform(X):
  one-call DataFrame build
```

Each pipeline returns the final DataFrame and can write a preprocessing artifact
`.joblib` file. The artifact stores the DataFrame together with the resolved
YAML config, original YAML text, config SHA256, H5 SHA256, Aramis version/git
SHA, and branch. This is preprocessing output, not a trained classifier.

## Product Command Shape

The intended product code is split into three command/config stages:

```text
python -m aramis preprocess --config /path/to/preprocess.yaml
python -m aramis train --config /path/to/training.yaml
python -m aramis run --config /path/to/workflow.yaml
python -m aramis predict --config /path/to/predict.yaml
```

Current work covers preprocessing, training, prediction, and a combined
preprocess+train workflow. The preprocessing config is self-contained: it
defines input H5 path, output DataFrame/joblib path, raw-data source, branch
rules, quality exclusions, thickness correction, SNR, normalization, and payload
retention. The training config defines the input preprocessing joblib, model
family, selected model, validation mode, and output artifacts. The workflow
config references one preprocessing YAML and one training YAML. The prediction
config defines the one-patient H5 path, trained model joblib, patient id,
target-side source, and report output paths. The trained model joblib stores the
prediction preprocessing YAML.

For preprocessing, input and output paths are not command-line data parameters.
They live in YAML:

```text
io.input_h5_path
io.output_joblib_path
```

The command receives only the config path:

```text
python -m aramis preprocess --config Aramis/config/preprocessing/<branch>.yaml
```

Execution is YAML-driven:

```text
aramis.__main__.main
-> run_preprocessing_from_config(config_path)
-> load_preprocessing_config(config_path)
-> build_pipeline_from_config(config)
-> sklearn Pipeline declared in pipeline.steps
-> KeepColumnsTransformer(metadata.output_columns + branch_settings.output_columns)
-> preprocessing artifact joblib at io.output_joblib_path
```

Downstream code should use `xrd_preprocessing.load_preprocessing_dataframe`
when it needs only the DataFrame. The full joblib object keeps the preprocessing
lineage needed for audit and reproducibility.

The combined workflow can pass the freshly built DataFrame directly into
training while still writing the preprocessing joblib footprint:

```text
python -m aramis run --config /path/to/workflow.yaml
-> run preprocessing YAML
-> write preprocessing artifact joblib
-> pass in-memory DataFrame to training pipeline, when workflow.mode=memory
-> write model artifact joblib / JSON / YAML
```

Use `workflow.mode=artifact` when training should explicitly reload the saved
preprocessing joblib.

Aramis should not hardcode preprocessing steps. It owns concrete branch YAMLs
and product paths. XRD-preprocessing owns transformer classes, the transformer
registry, YAML `$ref` resolution, and pipeline construction.

Prediction v0.1 input:

```text
one-patient H5
trained Aramis model joblib
prediction preprocessing YAML stored in model joblib
one patient
clinician-supplied target_side from predict YAML
model_id and selected_model from predict YAML
two breast-side specimen groups when available
machine-readable JSON/YAML output for report generation
```

Prediction must not infer `target_side` from diagnosis labels or biopsy
metadata. In a clinical workflow the suspicious breast side is supplied by the
clinician-facing predict YAML.

Synthetic tests use one known H5 container with both `raw/data` and
`processed/data` 2D arrays:

```text
tests/synthetic_aramis_h5.py
```

Expected fixture behavior:

```text
P1: BENIGN-CANCER pair
  one-to-one: keep
  one-to-many: keep both specimens

P2: single BENIGN breast
  one-to-one: drop
  one-to-many: keep

P3: NORMAL-ATYPICAL pair
  one-to-one: keep as CANCER-NORMAL
  one-to-many: keep ATYPICAL as CANCER, drop NORMAL

P4: BENIGN-CANCER pair but one side lacks sample thickness
  one-to-one: drop patient after thickness/sample-pair validity
  one-to-many: keep valid CANCER side only

P5: CANCER with calibrant_thickness_mm=50
  both branches: drop by calibrant thickness safety range
```

The tests verify exact final DataFrame columns, expected patients/specimens,
label grouping, thickness-correction metadata, dropped heavy detector payloads,
and artifact-joblib roundtrip equality.

## Label Grouping

Label grouping is defined at `specimenId` / breast-side level.

```text
specimenId level: BENIGN -> BENIGN
specimenId level: CANCER -> CANCER
specimenId level: ATYPICAL -> CANCER
specimenId level: PRE_CANCEROUS -> CANCER
specimenId level: NORMAL -> NORMAL
specimenId level: NA -> exclude
```

The broad CANCER group is the current product grouping:

```text
CANCER + ATYPICAL + PRE_CANCEROUS -> CANCER
```

This grouping is applied at `specimenId` / breast-side level before the branch
datasets are finalized.

Preserve the original `specimen_status`. Write product labels to a separate
column so label mapping remains auditable.

MAX preprocessing outputs should preserve available non-heavy scalar H5
metadata. MIN outputs keep only declared model-essential columns. Biopsy
metadata is especially important because it tracks whether the diagnosis context
is biopsy-associated:

```text
biopsy / sample_biopsy:
  whether biopsy was taken for the patient/specimen context

sample_biopsy_type:
  Pre-biopsy / Post-biopsy when present

sample_status:
  Non-cancer / Cancer / Prior cancer context when present
```

## One-To-Many BENIGN/CANCER Datasets

Purpose:

```text
specimenId-level BENIGN vs CANCER classifier
compare the suspicious / target breast side against breast sides from other patients
```

Clinical level:

```text
patient has a suspicious side after mammography or other breast imaging
target side is measured at three nearby positions around the suspicious region
one-to-many training uses specimenId-level BENIGN/CANCER labels for such breast sides
contralateral-side rows may exist in the H5 container but are not required for
one-to-many validity
```

Two one-to-many preprocessing YAMLs are currently defined:

```text
standard:
  Aramis/config/preprocessing/aramis_one_to_many_max_v0_1.yaml
  no biopsy requirement

biopsy-only:
  Aramis/config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
  H5 measurementId/session level: require biopsy == true before GFRM loading
  DataFrame measurementId level: require biopsy == true as a safety check
```

The biopsy-only rule follows the Clinical_trials FDA model notebook convention:
use only rows where the biopsy flag is true. In the Aramis H5 container this is
the scalar metadata field `biopsy`.

Preprocessing steps:

```text
1. H5 measurementId/session level:
   apply shared AgBH-date, q-range, sample-thickness, and calibrant-thickness filters

2. H5 specimenId level:
   keep BENIGN, CANCER, ATYPICAL, PRE_CANCEROUS
   exclude NORMAL and NA for this binary dataset

3. h5_to_df measurementId level:
   materialize only selected SAMPLE/SAMPLE measurement rows from GFRM

4. DataFrame specimenId level:
   group labels:
     BENIGN -> BENIGN
     CANCER/ATYPICAL/PRE_CANCEROUS -> CANCER

5. measurementId level:
   run shared XRD preprocessing

6. specimenId level:
   apply post-SNR validity
   current rule: at least 1 valid measurement per specimen

7. patientId level:
   retain patientId only for leakage control and split logic
   do not require both breasts for one-to-many
```

Output unit:

```text
model rows are measurement-level/profile rows
labels are specimenId-level breast-side labels
splits must be patient-safe
```

## One-To-One Dataset

Purpose:

```text
patientId-level paired-breast symmetry feature generation
compare target and contralateral breast profiles within the same patient
```

Clinical level:

```text
target side:
  side with suspicious finding after mammography / breast imaging
  first product build uses biopsy-only patients
  target-side biopsy label is BENIGN or CANCER

contralateral side:
  patient-internal comparison side
  not assumed to be perfectly healthy
  may carry NORMAL, BENIGN, CANCER, ATYPICAL, or PRE_CANCEROUS metadata
```

Two one-to-one preprocessing YAMLs are currently defined:

```text
standard:
  Aramis/config/preprocessing/aramis_one_to_one_max_v0_1.yaml
  no biopsy requirement

biopsy-only:
  Aramis/config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml
  DataFrame patientId level: keep patient if any row has biopsy == true
  keep both breast sides after patient selection
```

Preprocessing steps:

```text
1. H5 measurementId/session level:
   apply shared AgBH-date, q-range, sample-thickness, and calibrant-thickness filters

2. H5 specimenId level:
   keep BENIGN, CANCER, NORMAL, ATYPICAL, PRE_CANCEROUS
   exclude NA before frame loading

3. h5_to_df measurementId level:
   materialize selected SAMPLE/SAMPLE measurement rows from GFRM

4. DataFrame patientId level, biopsy branch only:
   keep patients with at least one biopsy == true row
   do not drop contralateral non-biopsy rows before pair validation

5. DataFrame specimenId level:
   group labels:
     BENIGN -> BENIGN
     CANCER/ATYPICAL/PRE_CANCEROUS -> CANCER
     NORMAL -> NORMAL

6. DataFrame patientId level:
   keep first ML pair types as unordered grouped pairs:
     BENIGN-CANCER
     BENIGN-NORMAL
     CANCER-NORMAL

   BENIGN-CANCER includes both target-BENIGN/contralateral-CANCER and
   target-CANCER/contralateral-BENIGN orientation. Target/contralateral
   orientation is preserved as metadata for audit and side-specific reporting,
   but the first one-to-one feature generator uses symmetric distance features.

   BENIGN-NORMAL, BENIGN-CANCER, and CANCER-NORMAL distances are expected to be
   nonzero. The preprocessing contract does not assume zero distance for any
   valid pair.

7. DataFrame patientId level:
   exclude first ML pair types:
     BENIGN-BENIGN
     NORMAL-NORMAL
     CANCER-CANCER

8. measurementId level:
   run shared XRD preprocessing

9. patientId level after SNR:
   require paired breast availability
   current rule:
     min_measurements_per_specimen = 1
     min_specimens_per_patient = 2
```

Output unit:

```text
symmetry features are derived from paired breast profiles
pair validity is patientId-level
breast labels are specimenId-level
raw profiles remain measurementId-level
target/contralateral orientation remains metadata-level for reporting
symmetry_available records whether a valid feature was computed
```

If both breast sides are clinically suspicious, the first Aramis version does
not treat this as one coupled bilateral decision. The product should create
side-specific decision-support reports for each breast, so the clinical user can
review whether left, right, or both sides need biopsy / further work-up.

The first symmetry metric follows the Ulster mammary-gland symmetry pattern:

```text
between_mean:
  all pairwise target-vs-contralateral profile distances

within_left_mean / within_right_mean:
  pairwise replicate distances inside each breast side

within_mean:
  mean(within_left_mean, within_right_mean)

asymmetry_score:
  between_mean - within_mean
```

If this cannot be computed, do not set the feature to 0. Keep the one-to-many
side-specific output available, mark `symmetry_available = false`, and exclude
the row from first fusion-model training unless an explicit fallback model is
versioned.

## Fusion Dataset

Purpose:

```text
combine one-to-many evidence and one-to-one symmetry evidence
produce final p_cancer decision-support score
```

Fusion input concept:

```text
patientId level:
  target breast one-to-many score
  one-to-one symmetry coefficient / asymmetry risk
  symmetry_available flag
  quality summaries
  age/BMI candidate covariates when available before biopsy decision

output:
  patient-level p_cancer
  BENIGN/CANCER decision-support class
```

Preprocessing contract:

```text
1. patientId level:
   join one-to-many breast-side outputs and one-to-one patient-pair outputs

2. specimenId level:
   preserve side-specific one-to-many scores and product labels
   keep target-side score and target/contralateral orientation metadata

3. patientId level:
   preserve one-to-one symmetry coefficient / asymmetry risk
   encode missing symmetry explicitly:
     symmetry_available = 0
     symmetry_distance_value = 0
     symmetry_distance_x_available = 0

4. patientId level:
   add candidate quality features:
     snr_db_mean
     n_valid_target_measurements
     n_valid_contralateral_measurements
     target_profile_replicate_distance

5. patientId level:
   add candidate clinical covariates only if available before biopsy decision:
     age
     bmi
     age_available
     bmi_available

6. patientId level:
   train fusion model with patient-safe splits only
```

The fusion model must not re-split measurement rows independently. Patient,
specimen, and measurement lineage must remain traceable to the original H5
container.

Biopsy-derived metadata, biopsy type, and post-biopsy status are allowed for
filtering, label confidence, and audit. They must not be used as prediction
features in the decision-support model.

## Required Audit Artifacts

Every dataset build should produce MLflow artifacts:

```text
preprocessing_config.json
product_filter_rules.json
selected_measurement_ids.csv
dropped_measurements.csv
preprocessed_dataset.parquet or .csv
feature_schema.json
label_mapping.json
train_test_split.csv
```

Every stage counter must say which level was counted:

```text
measurementId count
specimenId count
patientId count
diagnosis/status count at specimenId level
```
