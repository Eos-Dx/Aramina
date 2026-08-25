# Measurement Uncertainty And Polar-Cake Experiment v0.1

Status: research draft experiment design. This document does not change the
Aramina product model, preprocessing configuration, model artifact, report
contract, or clinical interpretation.

Aramina remains a research-draft decision-support prototype for women with
BI-RADS 3 or BI-RADS 4 findings. Any result described here is not a clinical
validation result and is not for autonomous diagnosis.

## Purpose

The experiment has two related aims:

1. quantify how detector and preprocessing uncertainty changes a fixed
   patient's `p_cancer`;
2. test whether a fixed polar-cake representation can preserve useful XRD
   information while replacing a large, weakly interpretable pixel vector with
   a compact, physically constrained vector.

The product baseline remains the reference. The experimental work must be
compared on the same patient-safe folds and must not silently replace the
current radial-profile model.

## Two Different Uncertainties

### Measurement-induced prediction interval

The measurement interval answers:

> If the same physical sample were measured again under the estimated detector,
> calibration, thickness, and preprocessing variability, how much could the
> fixed model's `p_cancer` change?

The model coefficients, model version, patient label, and decision policy stay
fixed. Only the measurement realization is varied. The result is therefore a
conditional uncertainty interval for one scored case.

This is not:

- a Wilson confidence interval for sensitivity or specificity;
- a confidence interval for the population prevalence;
- a model-training or coefficient uncertainty interval;
- an independent validation estimate;
- a clinical probability statement that replaces clinician review.

The deterministic product value remains unchanged. The experiment may add a
separate measurement-stability record, for example:

```yaml
final_prediction:
  p_cancer: 0.300
  decision_threshold: 0.24666
  measurement_uncertainty:
    method: detector_level_monte_carlo_v0_1
    draws: 1000
    interval_level: 0.95
    p_cancer_low: 0.210
    p_cancer_high: 0.410
    probability_above_threshold: 0.78
    threshold_crossing: true
```

The numeric interval above is illustrative only. It is not a result from the
current model.

`p_cancer_low` and `p_cancer_high` are the empirical 2.5th and 97.5th
percentiles of the Monte Carlo scores. `probability_above_threshold` is the
fraction of draws at or above the frozen decision threshold. A threshold
crossing means that the decision-support class is sensitive to the simulated
measurement variation. It must not be presented as an autonomous diagnosis.

### Wilson or model uncertainty

Wilson intervals describe a binomial metric estimated from labelled cases. For
sensitivity, the relevant denominator is the number of reference CANCER cases;
for specificity, it is the number of reference BENIGN cases. Wilson intervals
describe uncertainty in an estimated population metric. They do not describe
how one patient's detector noise changes `p_cancer`.

The two outputs must remain separate:

```text
patient score:       deterministic p_cancer + measurement-induced interval
cohort performance:   sensitivity/specificity + Wilson interval
```

## Monte Carlo Design

### Detector-level reference method

The reference implementation should start from the original detector frame.
For Monte Carlo draw (b):

1. sample detector counting noise using a Poisson model when the input is in
   calibrated photon/count units:

   \[
   N_{ij}^{(b)} \sim \operatorname{Poisson}(\max(N_{ij},0)).
   \]

2. sample calibrated nuisance parameters only when their covariance is
   available. Candidate parameters are beam centre, detector distance,
   detector rotations, wavelength, and sample thickness;
3. apply the same faulty-pixel mask, geometry, pixel-splitting policy, and
   azimuthal integration as the reference pipeline;
4. apply the same q-range normalization and feature transformation;
5. aggregate the same measurements for each breast and calculate the same
   symmetry features;
6. score every draw with the frozen model and collect `p_cancer`.

The dependency structure matters. Counting noise may be frame-specific, while
calibration error is shared by measurements from one acquisition session.
Thickness uncertainty is shared at the appropriate sample or breast level.
The three measurements from one breast must be re-aggregated in every draw.

The model is not retrained inside a draw. Sampling model coefficients would
mix measurement uncertainty with model or training uncertainty and would answer
a different question.

### Profile- or cake-level approximation

A faster approximation can start from an integrated profile or polar cake:

\[
\mathbf{x}^{(b)} = \boldsymbol{\mu} + L\mathbf{z},
\qquad LL^\mathsf{T}=\Sigma,
\qquad \mathbf{z}\sim\mathcal{N}(0,I).
\]

If only per-bin `sigma` is available, the first approximation uses a diagonal
\(\Sigma\). This is incomplete because pixel splitting and normalization can
create covariance between neighbouring bins. The approximation must therefore
be calibrated against detector-level Monte Carlo on a representative subset.

For a weighted radial bin:

\[
I_k = \frac{\sum_p w_{kp}X_p}{\sum_p w_{kp}},
\]

and, under independent detector-pixel noise:

\[
\operatorname{Var}(I_k)=
\frac{\sum_p w_{kp}^2\operatorname{Var}(X_p)}
     {(\sum_p w_{kp})^2}.
\]

With pixel splitting, the more complete expression is:

\[
\operatorname{Cov}(I_k,I_l)=
\sum_p A_{kp}A_{lp}\operatorname{Var}(X_p).
\]

Normalization makes analytic propagation less convenient. If
\(z_i=I_i/M\), where \(M\) is the median in the q-window 6.7-7.1 nm^-1, all
normalized bins depend on the same random (M). Monte Carlo naturally
propagates this shared dependency.

For a fixed FPCA basis:

\[
\mathbf{c}=\Phi^\mathsf{T}(\mathbf{z}-\boldsymbol{\mu}),
\qquad
\operatorname{Cov}(\mathbf{c})=\Phi^\mathsf{T}\Sigma_z\Phi.
\]

The frozen classifier then produces:

\[
p_{\mathrm{cancer}}^{(b)}=
\sigma(\beta_0+\boldsymbol{\beta}^{\mathsf{T}}\mathbf{c}^{(b)}).
\]

The final interval is calculated from the empirical distribution of these
scores, not from a linear approximation after the logistic transformation.

### Draw count and convergence

The first implementation should compare 250, 500, and 1,000 draws. The
planned operational setting is 1,000 profile/cake-level draws per case after
the detector-level reference has been checked. A convergence check should
record changes in the lower quantile, upper quantile, and
`probability_above_threshold`. A provisional engineering target is a change
below 0.005 for each score summary between the final two draw counts; this is a
design criterion, not an established clinical threshold.

## Polar Cake Contract

The polar representation is a fixed physical-coordinate matrix:

\[
I(q,\chi),
\]

where (q) is the radial scattering coordinate and \(\chi\) is the
azimuthal coordinate. `pyFAI.integrate2d` is the intended integration route.

The experiment must persist, together:

```text
cake_intensity[q, chi]
q_axis[q]
chi_axis[chi]
cake_sigma[q, chi] or cake_variance[q, chi]
missing_or_masked_bins[q, chi]
integration_geometry
q_range
chi_convention
pixel_splitting_policy
normalization_window
```

The same q and chi axes must be used for every case. Candidate experimental
settings are 256 q bins and 36 or 72 azimuthal sectors. These are not product
settings and are not fixed by the 512x768 detector dimensions. `pyFAI` maps
detector pixels to physical q/chi bins; the number of bins does not need to
divide the detector dimensions.

The existing product normalization window, 6.7-7.1 nm^-1, remains a controlled
reference. A broader experimental q range may be evaluated, but q limits,
normalization, geometry, masking, and chi orientation must be frozen before
patient-safe comparison.

The angular mean of the cake must reproduce the current radial profile within a
predefined numerical tolerance. This is the first parity test before any model
comparison.

## Physically Constrained Harmonic Vector

The full cake must not be flattened directly into a logistic-regression input.
With 164 patients, thousands of unconstrained cake pixels would create a high
overfitting risk and would be difficult to interpret.

The proposed future representation is a low-order angular expansion:

\[
I(q,\chi)=a_0(q)+\sum_{m=1}^{M}
[a_m(q)\cos(m\chi)+b_m(q)\sin(m\chi)].
\]

Candidate components:

- (a_0(q)): angular mean and the closest analogue of the current radial
  profile;
- (m=2): dominant two-direction anisotropy;
- (m=4): a higher even angular pattern;
- odd-harmonic energy: a possible acquisition or positioning quality-control
  signal rather than an assumed biological signal.

If orientation is not standardized, use rotation-invariant amplitudes:

\[
A_m(q)=\sqrt{a_m(q)^2+b_m(q)^2},
\qquad
R_m(q)=\frac{A_m(q)}{a_0(q)+\varepsilon}.
\]

The compact vector could contain 20-30 basis coefficients for (a_0), 3-5
coefficients for (A_2), 3-5 for (A_4), and a small number of QC and
uncertainty summaries. The approximate total is 35-50 values, not thousands of
raw bins.

This is a future phase of the experiment. The first documentation and data
contract work should establish cake parity and uncertainty propagation before
claiming that any harmonic component carries cancer-related information.

## Controls

The following controls are required:

- rotate a cake and confirm that rotation-invariant features remain unchanged;
- shuffle or permute chi sectors and verify that angular signal disappears;
- compare cake angular mean with the existing 1D profile;
- test missing-bin and masked-pixel handling;
- measure uncertainty separately for low- and high-SNR cases;
- test whether angular features predict acquisition date, hardware, batch, or
  age more strongly than expected;
- fit every FPCA or other learned basis inside the training fold only;
- keep all folds patient-safe, including repeated measurements from one
  patient;
- compare deterministic scores and measurement intervals using the same frozen
  model and threshold.

High prediction of acquisition batch, hardware, or age is a confounding warning.
It is not evidence that the representation measures cancer biology.

## Acceptance Criteria

The experiment can proceed to model comparison only when:

1. the cake angular mean reproduces the reference radial profile;
2. q/chi axes, geometry, masking, and normalization are recorded per run;
3. detector-level and profile/cake-level Monte Carlo agree on a representative
   subset within the predefined numerical tolerance;
4. repeated draws are numerically stable at the selected draw count;
5. measurement uncertainty is reported separately from Wilson/model uncertainty;
6. no patient, specimen, or measurement leakage is introduced;
7. every representation is compared on identical patient-safe folds;
8. any feature reduction is at least as stable as the raw-profile baseline and
   does not rely on unreviewed confounders;
9. any proposed report field is reviewed as a separate contract change.

These criteria do not establish clinical performance, clinical utility, or
autonomous diagnosis.

## Limitations

- Current product preprocessing stores 1D Poisson `sigma` for SNR but does not
  yet propagate uncertainty through normalization, FPCA, symmetry, or the
  classifier.
- The current model-input schema does not retain a measurement uncertainty
  field.
- Detector-level uncertainty requires calibrated noise and geometry
  covariance; without these inputs, the result is only a partial uncertainty
  model.
- A diagonal profile covariance ignores correlations introduced by pixel
  splitting and normalization.
- A polar cake can encode acquisition artefacts as well as sample structure.
- The available cohort is small for a high-dimensional representation.
- Results from this experiment are research-draft evidence and not independent
  clinical validation.

## Reproducibility And Lineage

This branch is documentation-only. It does not add a Monte Carlo or polar-cake
CLI. The commands below reproduce the current `0.2.14-beta` DVC/MLflow product
lineage and provide the controlled starting point for the future experiment.

```bash
git switch experiment/measurement-uncertainty-polar-cake
conda activate eosproduct
git status --short --branch

# Configure the machine-local internal DVC remote once, if needed.
dvc remote add --local --default internal-h5 /path/to/controlled/aramina-dvc/remote

# Materialize the exact source archive referenced by Git.
dvc pull data/combined_archive.h5.dvc
dvc status

# Reproduce the current DVC-tracked product preprocessing, evaluation,
# train-on-all fit, model artifact, and MLflow run.
python -m aramina preprocess-train \
  --config config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_3.yaml

# Inspect the local product MLflow store.
mlflow ui \
  --backend-store-uri sqlite:///examples/outputs/mlflow/aramina_radial_profile.db \
  --port 5000

# Verify repository code and existing contracts.
conda run --no-capture-output -n eosproduct ruff check .
conda run --no-capture-output -n eosproduct pytest -q
```

The future implementation must add its own controlled experiment command and
record at least:

```text
experiment branch and source SHA
XRD-preprocessing source SHA/version
input H5 DVC pointer, DVC hash, size, and SHA256
preprocessing and integration configuration
q/chi axes and geometry metadata
noise and nuisance-parameter distributions
Monte Carlo draw count and seed policy
representation and basis fingerprint
patient-safe fold manifest
model artifact and threshold
deterministic predictions
Monte Carlo summaries
controls and acceptance results
```

MLflow must remain one lineage record for the complete data build, evaluation,
and frozen model run. DVC identifies the source H5 content; Git identifies
code, documentation, and YAML; MLflow records the run-specific parameters,
metrics, predictions, and artifacts. The source H5 remains internal.

## Scope Boundary

This commit intentionally changes documentation only. It does not modify:

- `src/` preprocessing or prediction code;
- YAML configuration or report contracts;
- tests;
- model version or model artifact;
- DVC pointer or MLflow implementation;
- product output fields.

The next implementation phase should be a controlled experiment branch change,
with source, tests, lineage artifacts, and documentation reviewed together.
