# Aramis Modeling Results Interpretation v0.1

Status: historical research-draft snapshot.

This file records the earlier one-to-many ablation run. The current v0.1-beta
training route is documented in `current_model_pipeline_and_risks_v0_1.md` and
uses the target/contralateral patient-level M0/M1/M2 pipeline. Do not treat the
numbers in this file as the current primary training result table.

This document summarizes the current Aramis model-development results for the
BENIGN vs CANCER decision-support task. These results are not clinical
validation and must not be interpreted as autonomous diagnosis. The output
concept remains `p_cancer` and suggested class, requiring radiologist /
qualified clinician review.

## Reproducibility Scope

The numbers below were regenerated from the current Aramis joblib artifacts:

```text
one-to-many standard:
  examples/outputs/real_h5_yaml_validation/aramis_one_to_many_min_v0_1.joblib

one-to-many biopsy-only:
  examples/outputs/real_h5_yaml_validation/aramis_one_to_many_biopsy_min_v0_1.joblib

one-to-one biopsy-only:
  examples/outputs/real_h5_yaml_validation/aramis_one_to_one_biopsy_min_v0_1.joblib
```

Common evaluation settings:

```text
model family: LogisticRegression
outer split: repeated patient-safe 70/30 split
n_splits: 20
test_size: 0.30
random_state: 42
inner OOF splits: 5
target sensitivity threshold: 0.95
specimen aggregation: mean
```

Patient-safe means the same `patientId` is never present in both train and test
inside one split. The reported values are mean +/- standard deviation across 20
patient-safe splits. They are not the best fold and not a selected final model.

Machine-readable result tables are stored in:

```text
docs/modeling/results/one_to_many_feature_ablation_summary.csv
docs/modeling/results/fusion_ablation_summary.csv
docs/modeling/results/one_to_many_dataset_summary.csv
docs/modeling/results/fusion_feature_coverage.csv
docs/modeling/results/modeling_result_manifest.json
```

## Dataset Reliability

The biopsy-only dataset performs better than the broader standard dataset when
using the same one-to-many profile-only model.

```text
standard profile-only:
  rows: 777
  patients: 193
  specimens: 275
  ROC AUC: 0.515 +/- 0.058
  PR AUC: 0.325

biopsy-only profile-only:
  rows: 493
  patients: 163
  specimens: 174
  ROC AUC: 0.580 +/- 0.076
  PR AUC: 0.522
```

Interpretation:

```text
biopsy-only labels are more clinically reliable
standard labels are broader and noisier
biopsy-only should be the main training/evaluation cohort for the first
research-draft Aramis model
```

The improvement is modest but consistent with the product assumption: biopsy
confirmation gives a cleaner endpoint than mixed clinical metadata. The cost is
smaller sample size.

## One-To-Many Feature Ablation

The one-to-many model was tested in three variants:

```text
profile_only:
  radial_profile_data only

thickness_only:
  sample_thickness_mm only

profile_plus_thickness:
  radial_profile_data plus sample_thickness_mm
```

Results:

| dataset | feature route | ROC AUC | PR AUC | target sensitivity | target specificity |
|---|---:|---:|---:|---:|---:|
| standard | profile_only | 0.515 +/- 0.058 | 0.325 | 0.984 | 0.024 |
| standard | thickness_only | 0.500 +/- 0.040 | 0.307 | 0.998 | 0.021 |
| standard | profile_plus_thickness | 0.507 +/- 0.054 | 0.319 | 0.980 | 0.025 |
| biopsy_only | profile_only | 0.580 +/- 0.076 | 0.522 | 0.956 | 0.070 |
| biopsy_only | thickness_only | 0.589 +/- 0.051 | 0.504 | 0.982 | 0.067 |
| biopsy_only | profile_plus_thickness | 0.575 +/- 0.076 | 0.518 | 0.956 | 0.069 |

Interpretation:

```text
standard cohort:
  sample thickness alone is random-like
  adding sample thickness does not improve profile-only performance

biopsy-only cohort:
  sample thickness alone shows weak label association
  adding sample thickness does not improve profile-only performance
```

The important conclusion is not that sample thickness is a useful predictor.
The useful conclusion is that it does not improve the profile model. The weak
biopsy-only thickness-only signal is a warning sign for possible cohort,
protocol, or batch confounding. It should be tracked as a control variable and
quality/provenance feature, not accepted as a biological cancer signal.

Current recommendation:

```text
do not use sample_thickness_mm as a primary predictor
keep sample_thickness_mm in joblib metadata
continue reporting thickness-only control performance
investigate whether sample thickness is correlated with biopsy cohort, batch,
date, or measurement protocol
```

## Fusion Model Results

The fusion experiment uses the biopsy-only one-to-many target dataset and the
biopsy-only paired one-to-one dataset.

Input coverage:

```text
one-to-many biopsy rows: 493
one-to-many biopsy patients: 163
one-to-many biopsy specimens: 174

one-to-one biopsy rows: 631
one-to-one biopsy patients: 110
one-to-one biopsy specimens: 220

fusion target specimens: 174
symmetry available: 115
symmetry unavailable: 59
age available: 174
BMI available: 0
sample thickness available: 174
```

Patient-level paired-context audit:

```text
one-to-many biopsy patients: 163
one-to-one biopsy paired patients: 110
one-to-many patients with paired one-to-one context: 110
one-to-many patients without paired one-to-one context: 53
one-to-one patients with fewer than two breast sides: 0
```

The 53 patients without paired context are not patients with partially valid
one-to-one output. In the current `one_to_one_biopsy_min` DataFrame, every
included patient has two breast sides. These 53 patients are present in the
one-to-many biopsy target dataset but absent from the paired one-to-one biopsy
dataset after preprocessing and branch filtering.

Target specimens without symmetry:

```text
total target specimens without symmetry: 59
BENIGN Left: 29
BENIGN Right: 22
CANCER Left: 3
CANCER Right: 5
```

Detailed audit tables:

```text
docs/modeling/results/paired_context_summary.csv
docs/modeling/results/paired_context_patient_audit.csv
docs/modeling/results/paired_context_missing_target_specimens.csv
docs/modeling/results/fusion_target_specimen_symmetry_audit.csv
```

Feature sets:

```text
M0:
  one-to-many p_cancer only

M1:
  one-to-many p_cancer plus symmetry features

M2:
  M1 plus quality features

M3:
  M2 plus age/BMI features
```

Results:

| model | ROC AUC | PR AUC | target sensitivity | target specificity |
|---|---:|---:|---:|---:|
| M0 one-to-many only | 0.527 +/- 0.107 | 0.485 | 0.948 | 0.066 |
| M1 one-to-many + symmetry | 0.706 +/- 0.066 | 0.610 | 0.922 | 0.344 |
| M2 + quality | 0.675 +/- 0.074 | 0.582 | 0.909 | 0.363 |
| M3 + age/BMI | 0.750 +/- 0.068 | 0.648 | 0.908 | 0.436 |

Interpretation:

```text
M1 improves strongly over M0
M2 does not improve M1 in the current feature definition
M3 is numerically strongest
M3 improvement is not automatically acceptable, because age and feature
availability can carry clinical/protocol confounding
```

The current data support the idea that paired-breast context can improve the
one-to-many profile model. However, the symmetry-related improvement must be
interpreted carefully because paired one-to-one context after preprocessing is
not present for every one-to-many target patient in the current dataset.

## Age, BMI, Availability, And Thickness Controls

Additional control models:

| control model | ROC AUC | PR AUC | interpretation |
|---|---:|---:|---|
| A0 age only | 0.703 +/- 0.072 | 0.615 | age alone predicts label |
| A1 BMI only | 0.500 +/- 0.000 | 0.425 | BMI unavailable / non-informative here |
| A2 availability only | 0.703 +/- 0.053 | 0.565 | missingness/availability predicts label |
| F0 symmetry availability only | 0.702 +/- 0.055 | 0.561 | symmetry availability itself predicts label |
| F2 replicate availability only | 0.505 +/- 0.033 | 0.432 | replicate availability alone is random-like |
| T0 thickness only | 0.586 +/- 0.048 | 0.509 | weak thickness/protocol signal |

Age result:

```text
age-only ROC AUC is 0.703
M3 plus age/BMI ROC AUC is 0.750
M3a plus age and no BMI is also 0.750
M3b plus BMI and no age falls back to 0.675
```

Interpretation:

```text
the M3 gain is driven by age, not BMI
age is clinically plausible but dangerous as a shortcut
age can be useful only if explicitly accepted as a clinical covariate and
validated as part of intended-use modeling
```

BMI result:

```text
BMI availability is 0 in this current feature table
BMI-only is random
BMI does not currently add usable information
```

Availability result:

```text
availability-only ROC AUC is about 0.703
symmetry-availability-only ROC AUC is about 0.702
```

This is a major caution. It means the model can partially predict label from
whether the paired one-to-one context survived preprocessing and branch
filtering, not only from the actual spectral shape or measured biological
asymmetry. This can happen when valid paired context is correlated with
measurement completeness, clinical workflow, dates, biopsy status, or product
filter behavior.

Thickness result:

```text
standard thickness-only: ROC AUC 0.500
biopsy-only thickness-only: ROC AUC 0.589
fusion T0 thickness-only: ROC AUC 0.586
profile plus thickness does not improve profile-only performance
```

Sample thickness therefore should not be used as a direct predictive feature
for the current product model. It should remain required for physical
preprocessing and thickness-corrected azimuthal integration, and it should
remain available for audit. The weak biopsy-only signal should be investigated
as possible protocol or cohort confounding.

## Current Working Conclusions

1. Biopsy-only is the preferred development cohort for the first research-draft
   Aramis model because labels are more reliable and profile-only performance
   is better than in the standard cohort.

2. One-to-many profile-only performance is still weak. It should be treated as
   a baseline, not a final model.

3. Adding paired-breast information improves performance in the current
   experiment, but the gain is entangled with whether paired one-to-one context
   is present after preprocessing. We need to separate real asymmetry signal
   from paired-context availability and branch-filter effects.

4. Age alone predicts cancer status in this dataset. This is clinically
   plausible but potentially dangerous. Age may be included only as an explicit
   clinical covariate after deliberate decision, documentation, and validation.

5. BMI is currently not useful because it is unavailable in the current feature
   table.

6. Sample thickness does not improve the profile model. Thickness-only has a
   weak signal in biopsy-only data, so thickness must be treated as a
   preprocessing/audit/control variable, not as a primary predictor.

7. Specificity remains low when thresholds are selected for very high target
   sensitivity. This is expected in the current safety-oriented research draft,
   but must be optimized later against the intended clinical workflow.

## Next Decisions

Before selecting a final training route, resolve these questions:

```text
Should age be allowed as a clinical covariate?
Should paired-context availability flags be allowed, or should they be excluded
to avoid preprocessing/branch-filter shortcuts?
Can symmetry be recomputed only on patients with complete paired measurements
to remove paired-context availability confounding?
Is sample thickness correlated with batch, date, biopsy cohort, or diagnosis?
Should the first product model use M1-like symmetry without age, or M3-like
clinical covariates with explicit confounding controls?
```

Recommended next experiment:

```text
restrict to patients/specimens where paired one-to-one context is available
rerun M0/M1/M2/M3 without availability flags
rerun age-only and thickness-only controls on the same restricted cohort
compare against the current unrestricted biopsy-only results
```
