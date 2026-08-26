# Polar Harmonic Ablation Experiment v0.1

## Status And Question

This document defines and reports a completed research experiment. It tests
whether angular information can add decision-support signal to the ordinary
radial profile while reducing the polar cake to a small, controlled feature
representation.

The ablation compares three rotation-invariant harmonic representations:

```text
A0          ordinary radial profile
A0 + A2     radial profile plus two-direction anisotropy
A0 + A2 + A4
            radial profile plus two-direction and higher-order angular structure
```

The experiment changes representation only. It does not change product
`p_cancer`, the product threshold, report contracts, or the immutable product
model artifact.

## Data And Lineage

Both pilot and full runs use the same DVC-tracked source and the same frozen
model lineage:

```text
data/combined_archive.h5
models/aramina_target_breast_risk_0_2_14-beta_98526329f40d/model.joblib
```

The planned cohort contains all `496` accepted target measurements and all
`175` target cases. The experiment does not apply a new quality filter and
does not add measurements. All measurements and cases belonging to one patient
remain in one partition. Patients with more than one historical case label are
retained; their cases are not silently collapsed or dropped.

The source H5 is identified through its DVC pointer, content hash, and MLflow
lineage. Git identifies the code, configuration, and documentation. MLflow
records each run-specific configuration, cohort manifest, fold manifest,
representation metadata, predictions, metrics, and model lineage.

## Polar Cake

Each measurement is represented as a `256 q x n_chi` polar cake. The full run
compares `n_chi` values `12`, `18`, `36`, and `72`. The pilot uses `12` and
`36` to check the pipeline before the full sweep.

The fixed axes are:

```text
radial q range:       2.0-23.0 nm^-1
normalization range:  6.7-7.1 nm^-1
harmonic q range:     2.1-12.2 nm^-1
azimuthal range:      -180 to 180 degrees
```

Normalization is applied once per measurement using the count-weighted angular
mean in the normalization range. Individual angular sectors are not normalized
separately. Missing sectors receive zero weight in the harmonic fit; they are
not replaced with physical zero intensity or silently interpolated.

The harmonic range is restricted to `2.1-12.2 nm^-1` because the detector does
not provide sufficient independent angular support for the requested harmonic
fit throughout the higher-q region. The full `256 q` cake remains available in
the cache, while the harmonic features use this common supported range.

## Harmonic Features And Compression

At each supported q value, `A0` is the count-weighted mean over the observed
angular sectors. The even angular terms are fitted to the residual on the
observed detector arc:

\[
J(q,\chi)-A_0(q)=\sum_{m\in\{2,4\}}
[a_m(q)\cos(m\chi)+b_m(q)\sin(m\chi)].
\]

The candidate cancer-prediction channels are:

\[
A_0(q)=a_0(q),\qquad
A_2(q)=\sqrt{a_2(q)^2+b_2(q)^2},\qquad
A_4(q)=\sqrt{a_4(q)^2+b_4(q)^2}.
\]

`A0` is the ordinary radial signal. `A2` describes two-direction angular
anisotropy. `A4` describes a higher-order angular pattern. Odd modes `m=1` and
`m=3` are fitted to the remaining residual as QC channels; they are not
cancer-prediction channels.

The detector does not cover a complete circle at any q value in this archive.
The fitted amplitudes are therefore partial-arc estimates, not complete
full-circle Fourier coefficients. Their angular-gap, rank, and condition-number
diagnostics are retained. This limitation is tested across angular resolutions
and must remain explicit when interpreting any apparent gain.

Every selected channel is compressed independently on the shared physical q
grid with a cubic B-spline basis. The coefficient budgets are `8`, `12`, and
`16` per channel. The primary comparison uses `36` angular sectors and `12`
coefficients per channel. The three mode sets are compared at this primary
setting; angular-sector and coefficient sweeps are robustness checks.

No basis is fitted using outer-test data. If a learned projection or scaler is
needed by the implementation, it is fitted inside the corresponding training
partition only.

## Evaluation Route

The ablation preserves the product downstream route:

```text
polar harmonic representation
-> fold-local cubic B-spline coefficients
-> LR1 probability per measurement
-> target-case aggregation
-> LR2 with the existing product correction inputs
-> target-breast p_cancer
-> thresholded decision-support metrics
```

The `LR1 -> LR2` path uses inner patient-safe OOF LR1 scores inside each outer
training partition. The threshold is then selected on that outer-training
partition at target sensitivity `0.95`. The outer-test patients are not used to
fit either logistic-regression stage or to select the threshold. This route is
required for the full evaluation; it is not a replacement for an independent
blind validation cohort.

All representations use identical measurement keys, patient assignments, and
outer fold manifests. The radial `raw100` product representation is retained
as a matched baseline. Reported metrics are sensitivity, specificity, ROC AUC,
balanced accuracy, PPV, NPV, confusion-matrix counts, and paired 95% confidence
intervals where supported.

## Planned Runs

Pilot:

```bash
aramina experiment-polar-harmonic-ablation \
  --config config/experiments/config_polar_harmonic_ablation_pilot_v0_1.yaml \
  --verbose
```

Full repeated patient-safe comparison:

```bash
aramina experiment-polar-harmonic-ablation \
  --config config/experiments/config_polar_harmonic_ablation_v0_1.yaml \
  --verbose
```

If parent statistics or MLflow finalization is interrupted after all child
runs are complete, resume without repeating preprocessing or model fitting:

```bash
aramina experiment-polar-harmonic-ablation \
  --config config/experiments/config_polar_harmonic_ablation_v0_1.yaml \
  --resume-run-folder examples/outputs/experiments/polar_harmonic_ablation/runs/<run>
```

The full run uses repeated patient-safe `5-fold x20` evaluation. The pilot uses
`2-fold x1` only for pipeline validation. Pilot metrics must not be presented
as final evidence.

## Acceptance Criteria

These are pre-specified experiment checks, not observed results.

1. Every variant uses the same `496` measurement keys, `175` target cases, and
   patient-safe fold manifest.
2. The polar axes, q ranges, chi ranges, normalization policy, and harmonic
   range match the configuration fingerprint exactly.
3. Each q bin has at least `4` observed angular sectors. Both four-column
   even-mode and odd-QC designs have rank `4` and condition number below `1e4`.
   The maximum angular gap is recorded as a limitation and is not filled by
   interpolation.
4. `A0` agrees with independent 1D integration with median relative RMSE below
   `1e-5` and maximum relative RMSE below `1e-4`.
5. The primary comparison is evaluated through inner-OOF `LR1 -> LR2` fitting,
   with no outer-test patient used in fitting or threshold selection.
6. For a representation to claim sensitivity non-inferiority against the
   matched `raw100` baseline, the lower paired 95% CI for the sensitivity
   difference must be above `-0.05`. Specificity improvement requires a point
   difference of at least `0.05` and a lower paired 95% CI above zero. ROC AUC
   non-inferiority requires a lower paired 95% CI above `-0.02`.
7. The direction of the primary effect remains present for at least three of
   the four angular resolutions in the full sweep.
8. No biological angular interpretation is accepted without a future
   acquisition-compatible chi-permutation control and usable session labels.
9. DVC and MLflow lineage artifacts are complete. No product model, product
   report, or product contract is modified by the experiment.

## Completed Full Run

```text
date:               2026-08-26
target cases:       175
patients:           164
measurements:       496
evaluation:         patient-safe 5-fold x20
primary compression: 12 coefficients per channel
MLflow run:         1869a23f11e24010a2a77bae470e7f28
status:             research-only; no product artifact changed
```

Primary results:

| angular bins | representation | sensitivity | specificity | ROC AUC |
|---:|---|---:|---:|---:|
| 12 | A0 | 0.905 | 0.302 | 0.679 |
| 12 | A0+A2 | 0.899 | 0.314 | 0.679 |
| 12 | A0+A2+A4 | 0.900 | 0.314 | 0.683 |
| 18 | A0 | 0.905 | 0.302 | 0.679 |
| 18 | A0+A2 | 0.906 | 0.320 | 0.682 |
| 18 | A0+A2+A4 | 0.899 | 0.308 | 0.680 |
| 36 | A0 | 0.905 | 0.302 | 0.679 |
| 36 | A0+A2 | 0.895 | 0.318 | 0.681 |
| 36 | A0+A2+A4 | 0.895 | 0.299 | 0.680 |
| 72 | A0 | 0.905 | 0.302 | 0.679 |
| 72 | A0+A2 | 0.899 | 0.320 | 0.681 |
| 72 | A0+A2+A4 | 0.897 | 0.296 | 0.680 |

The `A0+A2` specificity difference relative to `A0` was positive at every
angular resolution: `+0.012`, `+0.018`, `+0.016`, and `+0.018`. Paired
patient-cluster bootstrap 95% intervals were above zero, but the predefined
minimum useful improvement of `+0.05` was not reached. Sensitivity differences
ranged from `-0.011` to `+0.001`; ROC AUC differences ranged from approximately
zero to `+0.003`.

`A4` did not provide a reproducible gain. At `36` and `72` angular bins it
reduced specificity relative to `A0+A2` by `0.019` and `0.025`, respectively.
The conclusion was qualitatively unchanged across `8`, `12`, and `16` spline
coefficients per channel.

The `A0+A2` predictions were highly correlated with the `36`-bin reference:
`0.995` for `12` bins, `0.997` for `18` bins, and `0.999` for `72` bins. Mean
threshold-class disagreement was `3.0%`, `2.5%`, and `1.6%`, respectively.

The matched raw-100 baseline produced sensitivity `0.905`, specificity `0.294`,
and ROC AUC `0.687`. The compressed `A0` route retained sensitivity and slightly
increased point specificity, but ROC AUC decreased by `0.008`; its paired 95%
interval was `[-0.023, 0.006]`. The predefined ROC AUC non-inferiority margin of
`-0.02` was therefore not met.

`A0` predictions were numerically identical across all four angular grids. The
independent 1D-integration parity check was not repeated on all `496`
measurements in this run; the earlier detector-integration parity result covered
`48` reference measurements and cannot be treated as a full-cohort check.

The data do not support promotion of `A2` or `A4` into the product model. `A2`
is a stable low-dimensional research feature with a small specificity effect;
`A4` adds complexity without stable benefit. Complete-circle anisotropy cannot
be inferred because the detector missing wedge reaches `270-280` degrees at
some q bins. Chi permutation was not executed for this partial-arc estimator,
and session stress testing was unavailable because recorded session labels were
sparse or high-cardinality.

## Limitations

- The archive is retrospective and small: `175` target cases from `164`
  patients. The experiment is development evidence, not independent clinical
  validation.
- Several patients have target cases with different historical labels. A
  patient-safe split is required, but this structure limits simple case-level
  interpretation of aggregate metrics.
- The polar and harmonic representations may capture detector geometry,
  calibration session, acquisition date, thickness, positioning, or photon
  statistics rather than tissue biology.
- Partial-arc amplitudes reduce direct angle dependence but are not equivalent
  to complete-circle rotation-invariant coefficients and do not remove
  acquisition or positioning confounding.
- Harmonic amplitudes have positive noise bias, especially when sector counts
  are low. The QC and permutation controls cannot establish biological meaning
  by themselves.
- Comparing angular resolutions and coefficient budgets creates multiple
  exploratory comparisons. The primary setting is fixed in advance, while the
  remaining settings are robustness checks.
- The common harmonic range ends at `12.2 nm^-1`; the radial baseline may retain
  a different usable q range. Any performance difference must therefore be
  interpreted as a representation-and-range comparison unless the ranges are
  explicitly matched.
- A positive result requires confirmation on an independent cohort with a
  separately controlled acquisition and positioning protocol.
- The patient-cluster bootstrap estimates held-out-prediction sampling
  uncertainty conditional on repeated fitted folds; it does not replace a
  separate blind cohort.
