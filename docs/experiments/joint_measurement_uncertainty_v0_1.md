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
| Beam centre | area-uniform disk with radius `5 pixels`; `10 pixels` is a boundary stress test | shared by exact PONI-file geometry, including different patients |
| Detector distance | uniform `+/-5 mm`; `+/-10 mm` is a boundary stress test | shared by exact PONI-file geometry, including different patients |

Common random numbers are used across component, joint, and leave-one-component
out scenarios. Stable keyed random streams make the result invariant to patient
order and draw chunking. Beam-centre and detector-distance perturbations are
shared by PONI-file hash, but are conditionally independent because a joint
calibration-fit covariance is not yet available. The visit-shared
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

Geometry is now calculated only by pyFAI. For every perturbed PONI state, pyFAI
builds and warms the exact bbox/CSR engine with the measurement-specific mask.
Metal receives these immutable weights and performs only centered-Poisson
sampling and repeated integration. It does not recalculate q, PONI, mask
membership, or pixel splitting.

This separation removed the geometry-dependent discrepancy observed with the
superseded dynamic-geometry Metal kernel. For the previously limiting
`Nova-254` draw 300, the maximum normalized-profile difference decreased from
approximately `6.11e-4` to `2.83e-5`; the corresponding `p_cancer` difference
was `4.91e-6`, and the decision-support class was unchanged. A ten-patient
smoke audit of the `joint_10px_10mm` scenario gave a maximum profile difference
of `3.65e-5`, maximum p99 profile difference of `2.96e-5`, and maximum
`p_cancer` difference of `1.96e-5`. All audited decision classes agreed.

These residual differences arise from deterministic float32 accumulation in
the static Metal kernel. They are numerical implementation differences, not
measurement uncertainty. Profile, `p_cancer`, and exact decision-class gates
remain fail closed. Final numerical limits will be fixed after the complete
ten-patient, twelve-scenario audit.

## Cohort geometry contract

The current cohort contains 178 patients, 973 measurements, and 76 distinct
PONI-file geometries. One PONI file can be shared by several patients measured
on the same day. Of the 178 patients, 177 are linked to one PONI geometry and
one patient is linked to two.

One outer geometry draw is therefore a complete cohort realization, not one
scalar perturbation applied to every patient. The draw contains one beam-centre
and detector-distance perturbation for each of the 76 PONI hashes. All
measurements linked to the same hash receive the same perturbation, including
measurements from different patients. Thickness remains a patient-level or
measurement-level term according to the declared sensitivity scenario, and
photon statistics remain measurement specific.

The runner creates this cohort field before the patient loop. Patient batches
are used only to bound memory; draw index `g` retains the same physical meaning
for the complete probability matrix. `nuisance_scope_manifest.csv` records the
PONI, thickness, and photon group assigned to every measurement.

The 973 masks are all distinct, so a single CSR integration plan cannot be
shared by all measurements acquired with one PONI file. A representative
100-bin plan occupies approximately 5.13 MB in host arrays. Computational
batches should therefore follow connected PONI-patient groups while preserving
one plan per measurement.

## Nested allocation and convergence

The Monte Carlo design contains two independent sample counts:

```text
G cohort geometry draws
  x P conditional photon realizations per geometry
  = G x P p_cancer realizations per scenario
```

Increasing `P` does not increase the number of independent geometry states.
Conversely, increasing `G` does not establish convergence of the conditional
photon distribution. The frozen allocation therefore uses at least `P = 50`
measurement-specific photon realizations for every outer geometry state. One
outer state is a complete cohort field: it contains one independently sampled
perturbation for each PONI geometry, shared by all patients linked to that PONI,
plus the declared thickness field. It is not one scalar geometry applied to the
complete cohort and it is not one independent PONI error per patient.

The replacement run has a maximum of `G = 5000` cohort geometry states and
`P = 50` photon replicates. Geometry is evaluated in stages of 250 states and
may stop only after at least 2000 independent geometry states and three stable
checkpoints. Configured geometry summaries are written at `100, 250, 500, 1000,
2000, 3000, 4000, 5000`; photon summaries are written separately at `10, 20,
30, 40, 50`. `nested_axis_case_convergence.csv` and
`nested_axis_changes.csv` prevent the two sample counts from being interpreted
as one undifferentiated draw count.

At the maximum allocation, the probability cube contains 579 million float32
values, approximately 2.32 GB. It represents 250,000 `p_cancer` values per
case and scenario but only 5000 independent geometry states. Auto-stop is
therefore required; 5000 is a bounded upper limit, not a mandatory target.

The revised benchmark uses the same pyFAI-prepared CSR path as the experiment.
For one patient with five measurements, one geometry draw required 0.79 s with
two photon replicates and 1.92 s with 2000 replicates. A ten-patient,
58-measurement audit with one cohort geometry and 50 photon replicates required
8.64 s. Metal generated 2900 perturbed profiles at 336 profiles/s. Maximum
profile error against the pyFAI-prepared reference was `3.65e-5`, maximum p99
error was `2.96e-5`, maximum `p_cancer` error was `1.96e-5`, and all decision
classes agreed.

The same bounded audit was extended to 500 photon replicates. Threshold-crossing
status was unchanged from 100 through 500 replicates. Between 450 and 500, the
median interval-endpoint change was 0.00168 and p90 change was 0.00865. This
shows that 50 is a minimum per-geometry allocation rather than a stand-alone
conditional-photon convergence claim. The complete run pools 50 photon draws
over each successive geometry field and evaluates both axes at every stage.
These observations also confirm that pyFAI plan construction dominates runtime,
whereas additional Metal photon replicates are comparatively inexpensive.

The superseded dynamic-geometry run remains preserved for audit. It contains
13,259 completed atomic units and 1,500 completed output draws, but it must not
be resumed because Metal recalculated geometry in float32. The replacement run
will use a new folder and a new runner fingerprint.

To request a safe manual stop, create `STOP_REQUESTED` in the new run folder.
The active cohort stage is completed atomically, after which the run is marked
paused. Resume verifies data, model, config, case, scenario, PONI-scope, runner,
and Metal-library fingerprints before continuing.

## Decision criteria for the full run

The `193-case x up-to-5000-geometry x 50-photon` experiment may start only after:

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
