# Correlated Measurement-Uncertainty Experiment v0.2

Status: research-draft experiment. This work does not change the frozen
Aramina `0.2.14-beta` model artifact, deterministic `p_cancer`, decision
threshold, preprocessing contract, or report contracts.

## Purpose

v0.1 sampled each one-dimensional integrated-profile bin independently from
its pyFAI sigma. That construction omitted correlations created by detector
pixel splitting and shared median normalization. It produced intervals that
were not sufficiently calibrated against the detector-level reference.

v0.2 estimates only the covariance induced by the available detector-level
centered-Poisson Monte Carlo. It does not estimate empirical measurement
repeatability.

## Reference and Fast Paths

The engineering reference starts from the unchanged, baseline-corrected
detector frame. For each positive estimated-photon component, it uses:

\[
\lambda=\max(x,0),\qquad x^{(b)}=x+\operatorname{Poisson}(\lambda)-\lambda.
\]

Every draw then passes through the unchanged faulty-pixel mask, pyFAI
integration, q-range median normalization, patient aggregation, symmetry
calculation, LR1, and LR2. The detector reference is intentionally limited to
a stratified subset balanced by historical class and target-SNR quantile.

The first seeded detector-draw block fits the covariance. A disjoint second
seeded block is retained for comparison. This prevents comparing a fast model
with the same Monte Carlo realizations from which its correlation structure was
estimated.

The stochastic draws are held out, but the reference patients are not. The
pooled covariance includes fit draws from the same stratified cases used for
detector comparison. The present gate therefore tests Monte Carlo approximation
within the available reference cohort; it does not establish patient-level
covariance transfer to an independent cohort.

For each detector-reference measurement, covariance is calculated after the
exact integration and normalization path. It is converted to a correlation
matrix by its detector-MC standard deviations. These matrices are pooled over
the stratified measurements:

\[
R=\operatorname{mean}_j\left[
D_j^{-1}\operatorname{Cov}(z_j)D_j^{-1}
\right].
\]

The pooled correlation is stored in memory-bounded form:

\[
R \approx B\Lambda B^\mathsf{T}+D.
\]

For a new measurement, the model uses its own pyFAI sigma after the fixed
normalization scale, `S_i`, to transfer this structure:

\[
\Sigma_i=S_i\left(B\Lambda B^\mathsf{T}+D\right)S_i.
\]

Fast normalized-profile draws are generated as:

\[
z_i^{(b)}=z_i+B\Lambda^{1/2}u^{(b)}+D^{1/2}v^{(b)},
\qquad u,v\sim\mathcal{N}(0,I),
\]

then multiplied componentwise by `S_i` before the frozen product model scores
each draw.

This is a heteroscedastic covariance-transfer assumption: detector-MC learns
the correlation structure, while the target measurement's pyFAI sigma sets the
per-bin amplitude. It is not a repeated-measurement model and must not be
interpreted as biological or positioning repeatability.

## Explicit Scope

Included:

- centered-Poisson perturbation of the positive estimated-photon component;
- product faulty-pixel mask and pyFAI reintegration;
- product q normalization and its induced detector-MC correlation;
- pooled detector-MC correlation and measurement-specific pyFAI sigma scaling.

Excluded because the current archive contains no reviewed covariance or repeated
measurements for them:

- gain, readout, and baseline uncertainty;
- PONI geometry and calibration uncertainty shared by a calibration session;
- sample and calibrant thickness uncertainty;
- patient positioning and local-region selection uncertainty;
- biological within-breast variability and longitudinal change;
- model parameter and training-data uncertainty.

The existing three measurements at different anatomical locations are not
technical repeat measurements. They cannot identify the excluded sources.

## Provisional Research Gates

The run records these engineering gates. They are not clinical claims:

| Check | Provisional gate |
| --- | --- |
| Deterministic frozen-score parity | exact within `1e-12` |
| Threshold-crossing agreement | at least `0.95` |
| Median covariance/detector interval-width ratio | `0.80` to `1.25` |
| 500 to 1,000-draw endpoint change | at most `0.005` |

The comparison records interval widths, crossing agreement, and absolute
probability-above-threshold difference. The convergence artifact records 250,
500, and 1,000 draw checkpoints from one nested seeded stream.

## Configurations

The historical runnable v0.1 configurations remain unchanged:

```text
config/experiments/config_measurement_uncertainty_pilot_v0_1.yaml
config/experiments/config_measurement_uncertainty_v0_1.yaml
```

The new covariance configurations are separate contracts:

```text
config/experiments/config_measurement_uncertainty_pilot_v0_2.yaml
config/experiments/config_measurement_uncertainty_v0_2.yaml
```

The full v0.2 configuration uses 24 detector-reference cases when all
class-by-SNR strata are populated: 2 historical classes x 4 SNR quantiles x 3
cases. The pilot uses up to 8 cases and lower draw budgets.

```bash
git switch experiment/measurement-uncertainty-v0_2
conda activate eosproduct

aramina experiment-measurement-uncertainty \
  --config config/experiments/config_measurement_uncertainty_pilot_v0_2.yaml \
  --verbose

aramina experiment-measurement-uncertainty \
  --config config/experiments/config_measurement_uncertainty_v0_2.yaml \
  --verbose
```

## Artifacts and Lineage

Each v0.2 MLflow run stores the effective configuration and preprocessing YAML,
DVC pointer, H5 checksum, frozen model SHA256, Aramina SHA, XRD-preprocessing
SHA, reference subset manifest, detector fit profiles, covariance `npz`,
eigen-spectrum diagnostics, fast and detector score draws, summaries,
comparison, convergence, and polar-cake parity artifacts.

The H5 archive remains internal. This experiment remains decision-support
research and is not for autonomous diagnosis.
