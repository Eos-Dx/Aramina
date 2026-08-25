# Photon-Statistical Covariance Rank Scan v0.1

## Purpose

This research experiment tests whether the failed rank-30 uncertainty model
was limited by covariance compression or by the covariance approximation
itself. It does not modify the frozen 0.2.14-beta product model, deterministic
`p_cancer`, decision threshold, report contracts, or product output.

The uncertainty scope remains deliberately narrow: photon/statistical detector
noise after the existing product integration and normalization path. The
result is not total product uncertainty and is not an independent clinical
validation.

## Controlled Reference

The scan reuses the completed detector Monte Carlo run
`e94c86d3d54c40d5b705c54d73b65935`. Its first 100 draws for each of 141
measurements estimate covariance. A separate 100-draw block for 24 target cases
remains the held-out detector reference. Reusing these exact arrays prevents a
new random detector stream from changing the rank comparison.

Source artifacts, source lineage, DVC input H5, and frozen model checksums must
match before preprocessing or scoring. The source arrays are not silently
regenerated.

## Compared Models

The pooled measurement correlation is compared at empirical rank budgets 30,
50, 75, and 100. The observed 100-bin matrix has 99 positive eigenmodes; the
rank-100 budget therefore retains the complete positive spectrum without
adding an artificial zero-variance direction. The lower-rank variants retain
the strongest eigenmodes and preserve omitted pointwise variance on the
diagonal.

The fifth variant uses full-rank Ledoit-Wolf shrinkage. Detector-MC residuals
are centered and standardized within each measurement, pooled, shrunk toward a
scaled identity target, and converted to a unit-diagonal correlation matrix.
This controls unstable off-diagonal estimates without spectral truncation.

Full covariance means the complete 100 x 100 profile-error correlation. It is
not the original detector image and does not add calibration, positioning, or
biological uncertainty.

## Propagation And Gates

Each model generates 1,000 seeded profile draws for all 175 target cases. This
is 875,000 complete patient-level propagations across the five covariance
variants. Checkpoints at 250, 500, and 1,000 draws quantify Monte Carlo endpoint
stability. Every draw passes through the unchanged frozen sequence:

```text
measurement profile
-> LR1
-> target and contralateral aggregation
-> age and symmetry route
-> LR2
-> p_cancer and fixed threshold
```

All variants are compared with the same detector-reference cases. Provisional
research gates are unchanged:

- threshold-crossing agreement at least 0.95;
- median interval-width ratio from 0.8 to 1.25;
- maximum endpoint change at the final convergence checkpoint no more than
  0.005.

Passing these gates would support the photon-statistical approximation only.
It would not establish calibration, positioning, biological, training-data,
or total product uncertainty.

## Command

Pilot integration check on two detector-reference cases:

```bash
aramina experiment-measurement-uncertainty-rank-scan \
  --config config/experiments/config_measurement_uncertainty_rank_scan_pilot_v0_1.yaml
```

Full comparison:

```bash
aramina experiment-measurement-uncertainty-rank-scan \
  --config config/experiments/config_measurement_uncertainty_rank_scan_v0_1.yaml
```

## Interpretation

If empirical rank 100 matches the detector reference while lower ranks do not,
spectral compression is the limiting factor. If rank 100 also fails, the pooled
Gaussian covariance transfer is insufficient. A difference between empirical
rank 100 and full Ledoit-Wolf identifies sensitivity to covariance estimation
rather than rank alone.

## Completed Result

The full scan was completed on 2026-08-25 for all 175 target cases. The frozen
model and DVC input archive were unchanged. MLflow run
`23ac994010c0453282a47f6d0e1ea4da` finished successfully and contains all 23
required artifacts.

| Covariance model | Variance retained | Mean 95% interval width | Threshold-crossing cases | Agreement with detector reference | Median width ratio | Final maximum endpoint change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical rank 30 | 0.611 | 0.493 | 93/175 | 0.792 | 1.225 | 0.059 |
| Empirical rank 50 | 0.741 | 0.486 | 91/175 | 0.750 | 1.180 | 0.066 |
| Empirical rank 75 | 0.886 | 0.505 | 100/175 | 0.708 | 1.247 | 0.072 |
| Empirical full positive spectrum | 1.000 | 0.510 | 98/175 | 0.750 | 1.230 | 0.062 |
| Full Ledoit-Wolf | 1.000 | 0.510 | 102/175 | 0.750 | 1.280 | 0.049 |

The held-out direct detector Monte Carlo reference contained 24 target cases.
Its median interval width was 0.441, and 8 of 24 intervals crossed the fixed
decision threshold. The full Ledoit-Wolf transfer classified 13 of these 24
intervals as crossing: seven agreed positive crossings, five additional
crossings, and one missed crossing. Thus, the pooled covariance model produced
similar population-level widths but did not preserve the patient-specific
threshold behavior.

Empirical full covariance and Ledoit-Wolf were close to each other. Their
median absolute endpoint differences were 0.0049 for the lower endpoint and
0.0058 for the upper endpoint, and their threshold-crossing status agreed for
96.6% of all 175 cases. Rank truncation therefore changes some individual
cases but does not explain the failed detector-reference agreement. The main
limitation is transfer of one pooled Gaussian covariance structure between
measurements.

None of the five models passed all provisional gates. At 1,000 draws, the
median final endpoint change ranged from 0.0067 to 0.0083; only 33-39% of cases
met the 0.005 per-case convergence criterion. The extreme maximum endpoint
change ranged from 0.049 to 0.072. The 2.5% and 97.5% endpoints therefore
remain Monte Carlo-limited at this draw count, particularly when the maximum
over 175 cases is used as the gate.

## Decision

The pooled covariance transfer is not suitable for product reporting in its
current form. The direct detector Monte Carlo path is the better provisional
source of photon-statistical uncertainty because it preserves the individual
detector measurement instead of imposing a shared covariance pattern. The
present direct reference is still limited to 24 cases and 100 held-out draws,
so it is a research signal rather than a validated product interval.

The next controlled experiment should increase direct detector draws for a
small prespecified cohort spanning low, near-threshold, and high deterministic
scores. Batched prediction is required before increasing the draw budget.
Calibration, exposure, thickness, positioning, biological, and model
uncertainty remain separate future components and must not be combined with
the photon-statistical interval without independent data.

The current pandas-based patient scorer required approximately 80 minutes for
875,000 complete propagations on this machine. This implementation is adequate
for an auditable experiment but not for routine uncertainty generation.
