# Aramis Preprocessing YAML

Status: research draft.

Aramis preprocessing is YAML-governed. Aramis loads a concrete YAML, resolves
its shared fragments through `xrd_preprocessing.load_preprocessing_config(...)`,
builds the sklearn-style transformer route with
`xrd_preprocessing.build_pipeline_from_config(...)`, and writes a DataFrame
joblib artifact.

## Runnable Product Configs

```text
aramis_all_patients_model_input_v0_1.yaml
aramis_biopsy_patients_model_input_v0_1.yaml
aramis_prediction_patient_model_input_v0_1.yaml
```

`aramis_biopsy_patients_model_input_v0_1.yaml` is the current primary
development training dataset config. It keeps patients with at least one biopsy
row, keeps contralateral rows for symmetry features, maps NORMAL to BENIGN,
applies AgBH T100 monochromaticity exclusions, and outputs model/audit columns.

`aramis_all_patients_model_input_v0_1.yaml` is the broader comparison dataset.
It keeps all labelled patients, maps NORMAL to BENIGN, applies the same T100
quality exclusions, and writes the same model-input schema.

`aramis_prediction_patient_model_input_v0_1.yaml` is stored inside trained model
joblibs and is used by `aramis predict`. It has no historical date, diagnosis,
biopsy, or AgBH cohort filters. The predict YAML supplies the incoming one-
patient H5 path, target side, analysis author, and one output folder. Model
identity, prediction preprocessing, report contract, and decision threshold are
read only from the selected model joblib.

## Shared Fragments

```text
shared/aramis_policy_v0_1.yaml
shared/aramis_pipeline_v0_1.yaml
branches/one_to_many_all_patients_normal_as_benign_v0_1.yaml
branches/one_to_many_biopsy_patients_normal_as_benign_v0_1.yaml
branches/prediction_patient_v0_1.yaml
outputs/model_input_output_v0_1.yaml
outputs/prediction_model_input_output_v0_1.yaml
exclusions/agbh_quality_exclusions_t100_v0_1.yaml
```

T100 is the current development default. It is a middle-ground
monochromaticity threshold: stricter than T130, less data-hungry than T70, and
keeps enough biopsy-patient cases for patient-safe model selection.

## Canonical Product Order

The shared product pipeline expresses this canonical order explicitly:

```text
H5PoniGeometryCalculatorTransformer
-> H5SessionSelectorTransformer
-> H5ToDataFrameTransformer
-> ProductColumnBuilder and product filters
-> FaultyPixelDetector
-> AzimuthalIntegration
-> SNRTransformer and SNRFilter
-> PatientSpecimenValidityFilter
-> QRangeValueNormalizer
-> RadialProfileValueFilter
-> KeepColumnsTransformer
```

`H5PoniGeometryCalculatorTransformer` calculates `poni_q_max_nm_inv` from the
PONI geometry with pyFAI before H5 data frames are decoded. The following
session selector can therefore reject sessions that do not cover the required
q range without reading GFRM payloads. This is the current Aramis product
order, not a global restriction on every XRD-preprocessing YAML: another
workflow may declare another explicit pipeline order when its data contract
requires one.

## Commands

```bash
python -m aramis preprocess --config config/preprocessing/aramis_all_patients_model_input_v0_1.yaml
python -m aramis preprocess --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
```

Prediction preprocessing is normally not run directly. It is embedded in the
trained model joblib and invoked by:

```bash
python -m aramis predict --config config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

## Output Artifacts

Preprocessing joblibs contain:

```text
dataframe
resolved preprocessing YAML
original YAML text
preprocessing_config_sha256
input_h5_sha256
Aramis version / git SHA
branch metadata
```

These are data-preparation artifacts, not trained classifiers.
