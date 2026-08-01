# Target versus Contralateral Profile Variability

## Question

The biopsy protocol samples three nearby points in the suspicious target
region. The contralateral breast is sampled at three more widely separated
positions. This experiment asks whether the within-breast variability of the
normalized XRD profiles differs between these two sampling schemes and whether
the paired difference depends on the target-breast diagnosis.

This is a data-variability experiment. It does not train or select a cancer
classifier.

## Frozen input

Use the prepared T100 product-model input artifact:

```text
examples/outputs/model_input/aramina_biopsy_patients_model_input_v0_1.joblib
```

The artifact contains 893 retained measurements from 164 patients. Every
profile has 100 points on one common q-grid and is already normalized by the
frozen preprocessing pipeline. No additional smoothing or normalization is
applied.

## Cohorts

Primary analysis:

- unilateral-biopsy patients;
- biopsied target breast and non-biopsied contralateral breast;
- three unique positions on each breast;
- target diagnosis `BENIGN` or `CANCER`.

The current artifact provides 115 primary cases: 62 BENIGN and 53 CANCER.

Sensitivity analyses:

- at least two unique positions on each breast: 130 cases, 70 BENIGN and 60
  CANCER;
- bilateral-biopsy cases reported separately, never treated as independent
  target/contralateral pairs.

## Primary metric

For two normalized profiles from the same breast:

```text
d2(a, b) = mean_q((profile_a(q) - profile_b(q))^2)
```

Within-breast variability is the mean `d2` over all position pairs. For each
patient:

```text
R = log((V_target + epsilon) / (V_contralateral + epsilon))
```

The primary effect is `exp(mean(R))`, the geometric target/contralateral
variability ratio. Values above one indicate larger target variability; values
below one indicate larger contralateral variability.

The primary distance uses the complete product profile grid
`2.105–22.895 nm^-1`. The restricted `7–15`, `15–23`, and `7–23 nm^-1`
ranges are secondary robustness checks.

Primary inference uses a paired test of `R` against zero with a 10,000-sample
patient-level bootstrap confidence interval. Wilcoxon and sign tests are
reported as robustness checks. The CANCER-versus-BENIGN contrast is evaluated
on the same patient-level log ratio with diagnosis-label permutation.

Secondary metrics include RMS, cosine and Wasserstein distances and separate
q-range results for `7–15` and `15–23 nm^-1`.

## Critical limitation

The joblib stores P1/P2/P3 labels but no physical coordinates or distances.
Target/contralateral role is therefore confounded with sampling geometry. The
result must be described as **protocol-conditioned variability**, not intrinsic
biological heterogeneity.

A prospective protocol capable of separating geometry from biology must sample
both nearby and widely separated triplets in both breasts and record physical
coordinates or normalized breast-profile positions.

## Run from CLI

```bash
PYTHONPATH=src python -m \
  experiments.profile_variability.run_profile_variability \
  --input-joblib examples/outputs/model_input/aramina_biopsy_patients_model_input_v0_1.joblib \
  --output-dir experiments/profile_variability/outputs/primary_3x3 \
  --min-measurements 3
```

Run the `>=2` sensitivity analysis by changing `--min-measurements 2` and the
output folder.

## Run the marimo notebook

```bash
export ARAMINA_PROFILE_JOBLIB=/path/to/aramina_biopsy_patients_model_input_v0_1.joblib
PYTHONPATH=src python -m marimo edit \
  experiments/profile_variability/profile_variability_notebook.py
```

The environment variable is optional when the artifact already exists at the
default project-relative path shown above. The notebook reports a clear input
warning rather than executing if the artifact is absent.

Patient-level output is written to `per_case_variability_local.csv` and is
gitignored. Aggregate summaries and figures contain no patient identifiers.

## All-patient analysis without historical K-beta exclusion

`config_all_patients_no_kbeta.yaml` is a research-only preprocessing route. It
retains historical AgBH/K-beta sessions and does not require biopsy metadata.
It still applies the technical pipeline: detector geometry, thickness filters,
faulty-pixel handling, azimuthal integration, SNR filtering, normalization and
profile gating. It is not a product-model input or a clinical validation set.

The resulting patients are split after preprocessing:

- `BIOPSY_BENIGN` and `BIOPSY_CANCER`: unilateral-biopsy patients, compared as
  target / contralateral;
- `NO_BIOPSY`: no biopsy on either breast, compared only as left / right;
- `BILATERAL_BIOPSY`: both breasts biopsied, retained separately;
- `BIOPSY_UNRESOLVED`: a biopsy is present but the historical label is neither
  BENIGN nor CANCER; retained without relabelling.

For `NO_BIOPSY`, left is merely the numerator of a fixed left/right ratio. It
does not mean target and has no cancer label. Biopsy and no-biopsy cohorts are
reported as separate descriptive analyses; they are not compared to each other.
The same oriented log-ratio scale is used for both and may be displayed in one
figure: target/contralateral for biopsy cases and left/right for no-biopsy
cases.

Run the full preprocessing once:

```bash
PYTHONPATH=src python -m \
  experiments.profile_variability.run_all_patient_variability \
  --input-h5 /path/to/combined_archive.h5 \
  --preprocessing-config experiments/profile_variability/config_all_patients_no_kbeta.yaml \
  --output-joblib experiments/profile_variability/local_data/aramina_all_patients_no_kbeta_profiles.joblib \
  --output-dir experiments/profile_variability/outputs/all_patients_no_kbeta \
  --min-measurements 3
```

Repeat only the descriptive analysis from the saved local artifact:

```bash
PYTHONPATH=src python -m \
  experiments.profile_variability.run_all_patient_variability \
  --input-joblib experiments/profile_variability/local_data/aramina_all_patients_no_kbeta_profiles.joblib \
  --output-dir experiments/profile_variability/outputs/all_patients_no_kbeta \
  --min-measurements 3
```

The all-patient notebook is `all_patient_variability_notebook.py`. It expects
`ARAMINA_ALL_PATIENT_PROFILE_JOBLIB` or the local artifact path above.
