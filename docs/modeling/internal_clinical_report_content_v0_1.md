# Aramis Internal Clinical Report Content

Status: research draft.

Source: `/Users/sad/Downloads/Aramis_internal_clinical_report_content.docx`.

Purpose: development reference for the internal clinical report schema. This
document distinguishes the implemented v0.1 output from future report fields.
It is not a public clinical report and is not for autonomous diagnosis.

## XRD Scan Information

The implemented v0.1 report emits the following examination fields:

- `patient_id`
- `target_side` and `contralateral_side`, lowercase
- target and contralateral valid-measurement counts
- patient age and an explicit `patient_age_available` boolean

Operator, scan-time, hardware, protocol, and mammography metadata are future
fields. They require a stable source in the prediction H5 input contract.

## Features

### Azimuthal Integration

The implemented v0.1 report emits, separately for target and contralateral
breasts:

- intensity distribution statistics:
  - minimum q
  - maximum q
  - average q
  - median q
  - first quartile q
  - third quartile q
- q mode: q value corresponding to the intensity peak
- highest intensity
- target-side risk probability of malignancy based on azimuthal integration only
- contralateral-side risk probability of malignancy based on azimuthal integration only

The healthy-reference distance has a structured `not_implemented` placeholder
until a fixed healthy reference profile is included in the model artifact.

Implementation note: current product direction is to calculate contralateral
breast prediction for the internal report only, using the first-layer profile
model only. It must not replace the target-side final decision-support output.

### Symmetry

The implemented v0.1 report emits available SK target-vs-contralateral features.
It also keeps a structured placeholder for a standalone symmetry-only score,
because no such model is fixed in the prediction artifact.

Future symmetry report fields may include:

- Wasserstein distance between target and contralateral side
- target-side risk probability of malignancy based on symmetry approach only
- contralateral-side risk probability of malignancy based on symmetry approach only

Implementation note: the current gated M2Q model uses SK symmetry as an
optional refinement. Detailed feature definitions are maintained in
`sk_symmetry_features_v0_1.md`.

### Age

The v0.1 report emits age once in `xrd_scan_information`. It does not emit an
age-only risk score because an age-only model is not part of the prediction
artifact.

## Final Prediction

The implemented v0.1 final prediction contains `p_cancer`, a decision-threshold
identifier and value, suggested class, and symmetry availability. There is no
paired/fallback model route. Model coefficients and component details remain in
training output, not in this prediction report.

Future internal-report candidates:

- ML-classifier version
- preprocessing version
- JP-index formula
- target-side risk probability of malignancy, JP-index, with optional confidence interval
- feature weights in JP-index:
  - azimuthal integration
  - symmetry
  - age
- contralateral-side risk probability
- classification threshold
- TRA level: low/high
- result provided to the client
- internal TRA level
- probability that biopsy will confirm the result of Aramis XRD-analysis, based on risk probability

Internal TRA note: a project-specific internal TRA scale may be based on
training-data quantiles. For example, 86% can mean that 86% of patients in the
reference set have a lower JP-index than the current patient.

## Current Development Consequences

- Prediction report output must include enough fields for both YAML and JSON
  internal report generation.
- Target breast remains the primary product prediction.
- Contralateral breast can be scored in parallel for internal review, but only
  with the first-layer profile model unless explicitly changed.
- Internal report exposes target and contralateral first-layer profile
  probabilities as XRD evidence, but it does not serialize estimator objects,
  feature weights, model schemas, or raw feature rows.
- Full model description, regularization, feature schema, training configuration,
  thresholds, scaler/imputer state, and LR coefficients live in the
  ML-classifier training output YAML under `model_registry`; the joblib remains
  the executable model artifact.
- Internal reports round every numerical value to five decimal places and use
  `true`/`false` for every binary field.
- Report fields preserve decision-support language: p_cancer, suggested class,
  reliability, and research-only status.
