# Aramis Internal Clinical Report Content

Status: research draft.

Source: `/Users/sad/Downloads/Aramis_internal_clinical_report_content.docx`.

Purpose: development reference for the internal clinical report schema. This
document defines report fields that Aramis prediction YAML/JSON outputs should
support. It is not a public clinical report and is not for autonomous diagnosis.

## XRD Scan Information

These fields describe the examination, patient context, and acquisition setup:

- patient id
- patient age
- operator id
- date and time of examination
- instrument version
- EoScan version
- experimental protocol version
- mammography target breast part
- mammography conclusion for target breast, for example benign/cancer/other
- target breast side: left or right
- number of measurements for target breast side
- number of measurements for control / contralateral breast side

## Features

### Azimuthal Integration

Report fields requested from the integrated radial profiles:

- intensity distribution statistics:
  - minimum q
  - maximum q
  - average q
  - median q
  - first quartile q
  - third quartile q
- q mode: q value corresponding to the intensity peak
- highest intensity
- Wasserstein distance to average healthy profile from training data
- target-side risk probability of malignancy based on azimuthal integration only
- contralateral-side risk probability of malignancy based on azimuthal integration only

Implementation note: current product direction is to calculate contralateral
breast prediction for the internal report only, using the first-layer profile
model only. It must not replace the target-side final decision-support output.

### Symmetry

Report fields requested from target-vs-contralateral comparison:

- Wasserstein distance between target and contralateral side
- target-side risk probability of malignancy based on symmetry approach only
- contralateral-side risk probability of malignancy based on symmetry approach only

Implementation note: the current candidate model uses SK symmetry features as
model features. Detailed feature definitions are maintained in
`sk_symmetry_features_v0_1.md`.

### Age

Report fields requested from age-related model components:

- prior risk probability of malignancy for target side
- target-side risk probability of malignancy based only on age feature
- prior risk probability of malignancy for contralateral side

## Final Prediction

Fields requested for final internal prediction:

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
- Intermediate model outputs should be exportable, including first-layer profile
  model summaries and the final candidate model summary.
- Internal report should expose final and intermediate probabilities separately:
  `final_prediction.p_cancer` for the final M2Q output,
  `intermediate_models.lr1_profile_model.profile_p_cancer` for target-breast
  LR1 profile-only probability, and
  `feature_row.profile_p_cancer_logit_average` only as model-audit input.
- Report fields must preserve decision-support language: p_cancer, suggested
  class, reliability, and requires radiologist review.
