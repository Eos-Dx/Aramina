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
PYTHONPATH=src python -m marimo edit \
  experiments/profile_variability/profile_variability_notebook.py
```

Patient-level output is written to `per_case_variability_local.csv` and is
gitignored. Aggregate summaries and figures contain no patient identifiers.
