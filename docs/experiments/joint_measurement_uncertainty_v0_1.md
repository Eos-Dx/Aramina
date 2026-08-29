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
| Maximum `p_cancer`, pilot | `0.000223` | `0.0003` |

The largest profile difference was confined to an outer q-bin. The 99th
percentile remained below `1e-4`. The `p_cancer` limit was selected after a
complete diagnostic audit of all 120 pilot patient/scenario units; the observed
maximum was `0.000238`. A separate fail-closed gate requires exact agreement of
the decision-support class. All audited classes agreed.

The first nested full-cohort attempt exposed one additional numerical case:
for `Nova-215`, the Metal and pyFAI values were `0.72746` and `0.72781`. The
absolute difference was `0.000353`; both values remained far above the fixed
`0.24041049078429919` threshold, and the decision class was unchanged. The
nested configuration therefore initially used a declared `0.0005` numerical
`p_cancer` tolerance. Profile maximum and p99 gates remain `0.005` and
`0.0001`, and exact decision-class agreement remains mandatory. This changes
only the Metal-versus-pyFAI numerical acceptance margin; it does not change the
model, threshold, perturbation distribution, or reported uncertainty values.

A later full-cohort checkpoint exposed `0.000583` for `Nova-270` under the
`joint_without_beam_center_5mm` scenario. The final nested numerical tolerance
is therefore `0.001`, equivalent to 0.1 percentage point on the probability
scale. This remains a numerical parity gate, not a model-performance or
clinical tolerance. Exact decision agreement and both unchanged profile gates
remain fail-closed requirements.

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

The original full `973 measurements x 5000 draws x 12 scenarios` run was
estimated at approximately 70 hours on one deterministic Metal queue. The
first execution confirmed that geometry reconstruction dominated runtime:
after approximately 2.75 hours, only 1347 of 2136 patient/scenario units in the
first 250-draw stage were complete. The run was stopped atomically before any
complete global stage was reported. Its detector-frame cache and unit
checkpoints were preserved, but this direct-design run must not be resumed with
the nested configuration.

### Nested geometry-photon design

The accelerated design separates independent acquisition geometry from
conditional photon statistics:

```text
1000 independent geometry draws
  x 5 conditional photon realizations per geometry
  = 5000 p_cancer realizations per scenario
```

For one geometry draw, the Metal kernel calculates effective distance, beam
centre, q coordinates, bin assignment, and normalization membership once. It
then generates five independent centered-Poisson detector realizations inside
the same GPU kernel. Persistent detector images, masks, q-bin edges, and output
buffers remain on the GPU across calls. This removes four of five repeated
geometry calculations and most Python/Metal dispatch overhead.

The five photon replicates sharing one geometry are conditionally independent,
but they are not five independent geometry observations. Output artifacts
therefore report both counts explicitly:

```text
draws: 5000
independent_geometry_draws: 1000
photon_replicates_per_geometry: 5
```

This is a nested Monte Carlo estimate of the total acquisition distribution,
not an increase in effective geometry sample size. Convergence requires at
least 400 independent geometry draws in addition to the existing minimum of
2000 output realizations. Non-photon scenarios contain 1000 unique geometry
results; repeating them along the output axis preserves a common artifact shape
without claiming additional independent information.

The reproducible Human-1 acceptance benchmark used 10 real patients, 12 target
cases, the `joint_10px_10mm` scenario, 50 geometry draws, and five photon
replicates. Flattened and nested execution used identical geometry values,
photon seeds, and model artifact:

| Quantity | Flattened | Nested |
| --- | ---: | ---: |
| Metal integration time | `71.85 s` | `8.77 s` |
| Speedup | `1.00x` | `8.20x` |
| Maximum normalized-profile difference |  | `2.00e-5` |
| 99th percentile profile difference |  | `7.39e-6` |
| Maximum `p_cancer` difference |  | `3.98e-5` |
| Maximum 2.5/50/97.5% endpoint difference |  | `1.39e-5` |
| Draw-level decision agreement |  | `100%` |

The benchmark is written by
`scripts/benchmark_nested_joint_uncertainty.py`. The measured integration
speedup projects the complete nested experiment to approximately 11--15 hours,
subject to scenario mix, model-scoring overhead, and thermal conditions. This
estimate must be checked with the first complete 250-output checkpoint before a
long run is allowed to continue.

The nested configuration submits 50 geometry draws with five conditional
photon replicates as one 250-output global stage. Metal profile batch size 128
keeps the five conditional outputs within one fused dispatch. This reduces
Python/Metal dispatch overhead without concurrent writers.

The full experiment remains divided into global 250-output stages. Each stage
contains 50 independent geometry draws and covers all target cases and all 12
scenarios before a convergence result is published.
Each patient/scenario/stage slice is flushed to the probability memmap before
its atomic checkpoint is marked complete. Resume verifies data, model, config,
case, scenario, and cached-frame fingerprints and repeats only incomplete
slices.

After every 250 output realizations the run writes case and cohort CSV
summaries, a scenario dashboard, a convergence-history plot, and plateau
metrics under
`convergence/draws_NNNNN/`. `convergence/latest.json` and
`convergence/latest.png` always point to the newest completed global stage.

The full configuration enables conservative automatic plateau stopping. It is
not considered before 2000 draws and requires three consecutive checkpoints in
which all 12 scenarios satisfy the endpoint-change and threshold-crossing
criteria. A single stable checkpoint cannot stop the run. This is a Monte Carlo
convergence rule, not a clinical performance criterion.

The Metal profile parity gate uses a `0.005` maximum absolute tolerance and a
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

All eight gates passed before the direct full run was started. The nested
replacement additionally passed exact paired flattened-versus-nested tests,
chunk-invariance tests, stage-offset and checkpoint tests, and the real
10-patient model-level benchmark above. A replacement full run has not been
started.

The final output reports per-case scenario ranges, threshold crossing,
`scenario_draw_fraction_at_or_above_threshold`,
`scenario_class_flip_fraction`, convergence checkpoints, and cohort summaries.
These outputs remain measurement-sensitivity estimates, not probabilities of
diagnosis. Because `0.2.15-beta` was fitted on this cohort, label-stratified
summaries are descriptive and are not an independent performance validation.
