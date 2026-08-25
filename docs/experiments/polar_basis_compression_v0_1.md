# Polar-Basis Compression Experiment v0.1

## Question

The experiment tests whether a fixed 256 q x 36 chi polar cake can be reduced
to 15, 30, or 50 interpretable coefficients without losing target-breast
BENIGN/CANCER decision-support performance. Reconstruction is a technical
control. The primary endpoint remains the same target-breast endpoint used by
the product architecture.

This is a research-only comparison. It does not change product `p_cancer`, the
frozen decision threshold, report contracts, or the immutable 0.2.14-beta
model artifact. It is not an independent clinical validation and is not for
autonomous diagnosis.

## Controlled Data And Lineage

The experiment uses only the DVC-tracked archive and frozen model lineage:

```text
data/combined_archive.h5
SHA256 d2d61e83850b282c3d2479ea436deed821c4488b96983252d294f3d56ee3f1f9

models/aramina_target_breast_risk_0_2_14-beta_98526329f40d/model.joblib
```

Missing full-cohort cakes may be generated from this archive. No new
measurements are added. Cake artifacts are deduplicated by deterministic
measurement key and cached independently of learned representations.

## Polar Representation

Each cake is normalized once. The normalization factor is the median of its
count-weighted angular mean over 6.7-7.1 nm^-1. Individual chi sectors are not
normalized separately.

Integration uses one explicit detector-independent axis contract: 256 q-bin
centres over 2.0-23.0 nm^-1 and 36 chi-bin centres over -180 to 180 degrees.
The contract is fingerprinted in the cache manifest. Every reused or newly
generated cake is validated against it. pyFAI bin centres within 1e-4 of the
contract are represented on the canonical grid; larger differences fail. Thus,
row-level integration ranges cannot silently change the feature geometry.

The detector provides fewer than nine independent angular sectors above
approximately 12.8 nm^-1 for some accepted measurements. A simultaneous fit of
`m=0..4` would therefore be underdetermined over that region. The v0.1 harmonic
model uses the prespecified common range 2.1-12.7 nm^-1 while preserving the
full 256-bin cake in cache. This geometry-derived restriction is applied before
labels or folds are evaluated and is recorded in `q_chi_axes.npz`. Missing
angular sectors are never replaced by physical zeros.

At each q value, weighted least squares estimates

\[
I(q,\chi)=a_0(q)+\sum_{m=1}^{4}
[a_m(q)\cos(m\chi)+b_m(q)\sin(m\chi)].
\]

The candidate signal contains

\[
a_0(q),\qquad
A_2(q)=\sqrt{a_2(q)^2+b_2(q)^2},\qquad
A_4(q)=\sqrt{a_4(q)^2+b_4(q)^2}.
\]

`A2` and `A4` are invariant to rotation of an otherwise unchanged cake. This
avoids treating an uncontrolled detector orientation as biological evidence.
Odd modes `m=1` and `m=3` are reduced to QC energies. They are evaluated for
age, thickness, calibration-session, and acquisition-date confounding but are
excluded from cancer prediction.

## Matched Encoders

Three encoder families are compared at exact output budgets 15, 30, and 50:

1. `fourier_bspline`: fixed radial B-spline bases on the shared physical q grid;
2. `fourier_bessel`: a dimension-matched exploratory control using low-order
   Bessel functions after mapping the measured q interval to [0, 1];
3. `fourier_fpca`: PCA of the flattened `a0/A2/A4` harmonic vector.

B-spline and Bessel bases are recreated inside each training fold even though
their q-grid construction is deterministic. FPCA mean and components are fit
to one mean harmonic tensor per training target case, so patients with more
measurements do not dominate the covariance. Individual measurements are then
transformed for LR1. The coefficient scaler and LR1 are also fit inside the
same training fold.

Fourier-Bessel functions impose a global oscillatory basis and an artificial
boundary convention on a truncated annular q interval. They are therefore a
matched-dimensional control, not the preferred physical interpretation.

The uncompressed 100-bin radial product architecture is retrained on the exact
same outer folds as `raw100`. It uses the same LR1, LR2, regularization, and
training-fold threshold policy. `polar_to_raw100_comparison.csv` reports the
metric difference for every compressed variant. This matched baseline is
required to interpret whether compression preserved decision-support signal.

## Patient-Safe Product Architecture

Every encoder is evaluated through the same sequence:

```text
target-breast polar cake
-> fold-local polar encoder
-> fold-local LR1 probability per measurement
-> target-side mean in logit space
-> age + age_available + neutral-gated SK Core4 symmetry LR2
-> target-breast p_cancer
-> threshold selected from training patients only
```

The fold manifest is generated once and reused by all nine variants. All
target cases from one patient remain in the same partition. No basis, scaler,
LR stage, or threshold sees an outer-test patient during fitting.

The default experimental threshold policy reproduces the product-development
procedure: the target-sensitivity threshold is selected from fitted training
scores within each outer fold. This does not replace or rewrite the immutable
product threshold. Because the training score used for threshold selection is
not inner-OOF, thresholded metrics may remain optimistic; this limitation is
explicitly recorded.

## Outputs

Each MLflow run logs DVC, model, Aramina, and XRD-preprocessing lineage plus:

- effective experiment and preprocessing YAML;
- shared q/chi axes and polar cake manifest;
- one `basis.joblib` with fold-specific bases;
- basis metadata and fingerprints;
- one patient/case fold manifest;
- out-of-fold coefficient table and predictions;
- matched raw-100 fold metrics, predictions, and metric-difference table;
- fold and aggregate classification metrics;
- harmonic reconstruction and radial-profile preservation errors;
- compressed reconstruction examples;
- confounder availability and analysis.

Classification output includes sensitivity, specificity, ROC AUC, balanced
accuracy, PPV, NPV, confusion-matrix counts, and the threshold used in each
fold. Reconstruction relative RMSE is reported separately and cannot establish
clinical utility.

Unavailable confounders are reported as unavailable with a reason. They are
not silently imputed. Session prediction records test labels absent from the
training fold.

## Commands

Pilot:

```bash
aramina experiment-polar-basis-compression \
  --config config/experiments/config_polar_basis_compression_pilot_v0_1.yaml \
  --verbose
```

Full 5-fold x20 comparison:

```bash
aramina experiment-polar-basis-compression \
  --config config/experiments/config_polar_basis_compression_v0_1.yaml \
  --verbose
```

## Interpretation Limits

- All variants use one retrospective training archive.
- Comparing nine variants on the same folds makes these folds development
  evidence, not an untouched blind test.
- The polar variants use the common harmonic range 2.1-12.7 nm^-1, whereas the
  raw-100 baseline retains its frozen product q range. A performance difference
  therefore combines representation compression with loss of the high-q region;
  it cannot be attributed to coefficient count alone.
- Patient-safe splitting does not by itself remove calibration-session,
  acquisition-date, thickness, age, hardware, or operator confounding.
- Only age, thickness, session, and date are analyzed because they are the
  confounders requested by the experiment contract. Missing metadata cannot be
  reconstructed from the present archive.
- Polar pixel splitting creates covariance between neighboring q/chi bins;
  diagonal pyFAI uncertainty is not used as an exact covariance model here.
- A final representation requires confirmation on an independent cohort and a
  separately frozen acquisition protocol.
