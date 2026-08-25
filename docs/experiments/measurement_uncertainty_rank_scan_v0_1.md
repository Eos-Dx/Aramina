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
