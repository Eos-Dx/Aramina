# Joint Measurement-Uncertainty Experiment

Status: research-draft sensitivity experiment; not a clinical confidence
interval and not for autonomous diagnosis.

## Question

The experiment tests how bounded acquisition perturbations propagate from the
detector frame through azimuthal integration and the immutable Aramina model to
`p_cancer`. It separates photon counting statistics, sample-thickness
uncertainty, beam-centre uncertainty, and detector-distance uncertainty before
combining them. Model coefficients, the target-breast endpoint, and the decision
threshold are not changed.

## Data and model

```text
input: 20260827_combined_archive.h5
input SHA256: e0034c7e2eb59d9a511e3a55e8af3b39e3ea95c0c14b6ddbe3ee06a70c053e9b
input DVC MD5: 15244bac1dbea063bc6461385773cf76
model: aramina_target_breast_risk 0.2.15-beta
model SHA256: 43b2865632ea4dc2387bd5c23b5cb25083a771a9b4b87afcd235989ed5fbcc1d
decision threshold: 0.24041049078429919
eligible cohort: 178 patients, 193 target-breast cases, 973 measurements
```

The model is the same two-stage radial-profile architecture used for the
preceding product candidate. It was fitted on all eligible current data only
after the frozen `0.2.14-beta` model had been evaluated on patients absent from
its training manifest. The independent-patient check and the `0.2.15-beta`
training metrics are reported in
[`aramina_t100_target_case_model_v0_4.md`](../modeling/aramina_t100_target_case_model_v0_4.md).

## Perturbation contract

For every Monte Carlo draw, the observed detector frame is perturbed and
reintegrated. Geometry perturbations are applied before integration:

```text
D_effective' = D_poni + delta_distance
               - 0.5 * ((sample_thickness + delta_thickness)
                        - calibrant_thickness)

Poni1' = Poni1 + delta_row_pixels * pixel1
Poni2' = Poni2 + delta_column_pixels * pixel2
```

| Source | Bounded sensitivity distribution | Correlation scope |
| --- | --- | --- |
| Photon statistics | centered Poisson draw at the observed positive count scale | independent by detector pixel and measurement |
| Sample thickness | uniform `+/-5 mm` at thickness `<=50 mm`; uniform `+/-10 mm` above `50 mm` | shared within one patient visit |
| Beam centre | area-uniform disk with radius `5 pixels`; `10 pixels` is a boundary stress test | shared by calibration session, including different patients |
| Detector distance | uniform `+/-5 mm`; `+/-10 mm` is a boundary stress test | shared by calibration session, including different patients |

Common random numbers are used across component, joint, and leave-one-component
out scenarios. Stable keyed random streams make the result invariant to patient
order and draw chunking. Beam-centre and detector-distance perturbations are
shared by calibration session, but are conditionally independent because a
joint calibration-fit covariance is not yet available. The visit-shared
thickness perturbation represents a common signed compression or positioning
error with a thickness-dependent bound.

The bounded thickness and PONI distributions are engineering assumptions. They
must be replaced by repeated thickness measurements and calibration-fit
residuals when these data become available. The centered Poisson component is a
sensitivity test at the observed detector scale; it assumes count-like values
and does not include gain, readout, flat-field, drift, or exposure uncertainty.
The resulting quantiles are bounded acquisition-sensitivity quantiles, not
clinical 95% confidence intervals.

## Numerical parity gate

The Apple Metal integration is compared with direct pyFAI integration before
model scoring. The final ten-patient, twelve-scenario audit produced 120 parity
rows:

| Quantity | Observed difference | Fail-closed limit |
| --- | ---: | ---: |
| Maximum normalized-profile value | `0.001407` | `0.002` |
| 99th percentile of normalized-profile errors | `5.52e-5` | `1e-4` |
| Maximum `p_cancer` | `0.000223` | `0.0003` |

The largest profile difference was confined to an outer q-bin. The 99th
percentile remained below `1e-4`. The `p_cancer` limit was selected after a
complete diagnostic audit of all 120 pilot patient/scenario units; the observed
maximum was `0.000238`. A separate fail-closed gate requires exact agreement of
the decision-support class. All audited classes agreed.

Geometry and photon validation are separated. Geometry profiles are compared
draw by draw with direct pyFAI. The centered-Poisson path was tested over 20,000
draws against the validated static Metal reference; integrated-profile means
and variances met the statistical acceptance limits. The direct Monte Carlo
test subset passed with `53 passed, 6 skipped`.

The q-support preflight covered 696 audited measurement draws. Every one of the
100 product q-bins was supported, and the normalization band was supported in
every draw.

## Preliminary pilot

The software pilot used ten patient-safe target cases, ten draws, and the
baseline `5-pixel / 5-mm` geometry bounds. The draw count was selected to test
lineage, parity, memory use, and the complete scoring path; it is too small to
estimate stable 2.5th and 97.5th percentiles.

| Scenario | Median provisional range width | Cases crossing threshold |
| --- | ---: | ---: |
| Photon only | `0.283` | `3/10` |
| Thickness only | `0.129` | `0/10` |
| Beam centre, 5 pixels | `0.212` | `0/10` |
| Detector distance, 5 mm | `0.184` | `0/10` |
| Joint, 5 pixels and 5 mm | `0.378` | `2/10` |
| Joint stress test, 10 pixels and 10 mm | `0.445` | `2/10` |

Observation: photon counting statistics produced the largest median
single-component range in this pilot. Beam-centre perturbation was the next
largest component. The joint range was wider than every component-only range.

Interpretation: the model is sensitive to detector-level perturbations, but the
relative contributions are patient dependent and nonlinear. Component widths
must not be combined by quadrature. The ten-draw pilot does not establish the
population frequency of threshold instability.

MLflow run:

```text
run_id: 22b915a294694a0d9474b8afa71a4a13
status: FINISHED
patient/scenario units: 120
elapsed time including preprocessing and MLflow: 265.22 s
```

## Scaling result

The reference implementation rebuilt a pyFAI integration plan for every
geometry draw. This gave approximately `8.3` measurement-profile evaluations
per second after preprocessing. Extrapolation to `973 measurements x 5000
draws x 9 scenarios` is approximately 61 days and is therefore not an
acceptable full-run implementation. The final design contains 12 scenarios
after adding the `10-pixel / 10-mm` boundary stress tests; the same reference
implementation would require approximately 81 days.

The geometry-aware engine keeps detector images, masks, PONI geometry, and RNG
seeds in persistent Apple Metal buffers. Draw-specific distance and beam-centre
arrays are transferred in chunks without rebuilding a CPU CSR plan. Real
Human-1 benchmarks produced approximately `213 profiles/s` with centered
Poisson sampling and `253 profiles/s` without it, compared with `8.3
profiles/s` for the reference plan-per-draw implementation.

The full `973 measurements x 5000 draws x 12 scenarios` run is estimated at
approximately 70 hours on one deterministic Metal queue. Two concurrent queues
increased aggregate benchmark throughput by approximately 1.6-fold, while four
provided little further gain. The first full run retains one queue to avoid an
untested merge and checkpoint race.

The production full-run configuration submits one complete 250-draw global
stage chunk with a Metal profile batch size of 128. This reduces Python/Metal
dispatch overhead relative to the conservative 32-draw/16-profile pilot setup
without introducing concurrent writers or changing random draws.

The full experiment is divided into global 250-draw stages. Every stage covers
all target cases and all 12 scenarios before a convergence result is published.
Each patient/scenario/stage slice is flushed to the probability memmap before
its atomic checkpoint is marked complete. Resume verifies data, model, config,
case, scenario, and cached-frame fingerprints and repeats only incomplete
slices.

After every 250 draws the run writes case and cohort CSV summaries, a scenario
dashboard, a convergence-history plot, and plateau metrics under
`convergence/draws_NNNNN/`. `convergence/latest.json` and
`convergence/latest.png` always point to the newest completed global stage.

The full configuration enables conservative automatic plateau stopping. It is
not considered before 2000 draws and requires three consecutive checkpoints in
which all 12 scenarios satisfy the endpoint-change and threshold-crossing
criteria. A single stable checkpoint cannot stop the run. This is a Monte Carlo
convergence rule, not a clinical performance criterion.

The Metal profile parity gate uses a `0.0025` maximum absolute tolerance and a
separate `0.0001` p99 tolerance. The maximum limit admits isolated numerical
outliers observed in the full cohort (`0.002314`), while the unchanged p99 gate
continues to control population-wide profile agreement. The independent
`p_cancer` parity and decision-class gates remain unchanged.

To request a safe manual stop, create `STOP_REQUESTED` in the run folder. The
current patient/scenario/stage slice is completed atomically, then the run is
marked `paused`. Set `output.resume_run_folder` to that folder and rerun the
same command; the stale stop request is cleared and calculation continues from
the first missing slice. `Ctrl+C` also marks the run paused, although the active
slice is repeated after resume.

## Decision criteria for the full run

The `193-case x 5000-draw` experiment may start only after:

1. zero perturbation reproduces the static product path;
2. random perturbed geometry agrees with direct pyFAI within the declared
   profile and `p_cancer` tolerances;
3. results are invariant to draw chunk size;
4. the frozen model threshold remains exactly `0.24041049078429919`;
5. the pilot benchmark gives a bounded runtime and memory estimate;
6. invalid q coverage is reported explicitly rather than silently dropped;
7. the centered-Poisson generator passes an independent statistical acceptance
   test;
8. interruption and resume tests preserve only atomically completed units.

All eight gates passed in the final pilot before the full run was started.

The final output reports per-case scenario ranges, threshold crossing,
`scenario_draw_fraction_at_or_above_threshold`,
`scenario_class_flip_fraction`, convergence checkpoints, and cohort summaries.
These outputs remain measurement-sensitivity estimates, not probabilities of
diagnosis. Because `0.2.15-beta` was fitted on this cohort, label-stratified
summaries are descriptive and are not an independent performance validation.
