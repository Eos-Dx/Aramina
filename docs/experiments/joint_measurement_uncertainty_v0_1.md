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
order and draw chunking.

The bounded thickness and PONI distributions are engineering assumptions. They
must be replaced by repeated thickness measurements and calibration-fit
residuals when these data become available. Their quantiles are scenario
quantiles, not clinical 95% confidence intervals.

## Numerical parity gate

The Apple Metal integration is compared with direct pyFAI integration before
model scoring. The initial ten-patient pilot produced:

| Quantity | Maximum observed difference |
| --- | ---: |
| Normalized profile value | `4.4e-5` |
| `p_cancer` | `2.81e-5` |

Both values were below the predeclared fail-closed tolerances of `1e-4`.
Changing the numerical backend did not change the decision class in the parity
checks.

## Preliminary pilot

The software pilot used ten patient-safe target cases, ten draws, and the
baseline `5-pixel / 5-mm` geometry bounds. The draw count was selected to test
lineage, parity, memory use, and the complete scoring path; it is too small to
estimate stable 2.5th and 97.5th percentiles.

| Scenario | Median provisional range width | Cases crossing threshold |
| --- | ---: | ---: |
| Photon only | `0.298` | `2/10` |
| Thickness only | `0.136` | `0/10` |
| Beam centre, 5 pixels | `0.201` | `1/10` |
| Detector distance, 5 mm | `0.146` | `0/10` |
| Joint, 5 pixels and 5 mm | `0.441` | `2/10` |

Observation: photon counting statistics produced the largest median
single-component range in this pilot. Beam-centre perturbation was the next
largest component. The joint range was wider than every component-only range.

Interpretation: the model is sensitive to detector-level perturbations, but the
relative contributions are patient dependent and nonlinear. Component widths
must not be combined by quadrature. The ten-draw pilot does not establish the
population frequency of threshold instability.

MLflow run:

```text
run_id: 5ea00a4356094db8aa1681bbd6d67637
status: FINISHED
profiles evaluated: 5220
elapsed time including preprocessing and MLflow: 755.63 s
```

## Scaling result

The reference implementation rebuilt a pyFAI integration plan for every
geometry draw. This gave approximately `8.3` measurement-profile evaluations
per second after preprocessing. Extrapolation to `973 measurements x 5000
draws x 9 scenarios` is approximately 61 days and is therefore not an
acceptable full-run implementation. The final design contains 12 scenarios
after adding the `10-pixel / 10-mm` boundary stress tests; the same reference
implementation would require approximately 81 days.

The required production experiment engine keeps detector images and geometry
buffers on the Apple GPU and computes draw-specific geometry without building a
CPU CSR plan for every draw. A full run is blocked until zero-perturbation and
random-geometry parity pass for that engine.

## Decision criteria for the full run

The `193-case x 5000-draw` experiment may start only after:

1. zero perturbation reproduces the static product path;
2. random perturbed geometry agrees with direct pyFAI within the declared
   profile and `p_cancer` tolerances;
3. results are invariant to draw chunk size;
4. the frozen model threshold remains exactly `0.24041049078429919`;
5. the pilot benchmark gives a practical runtime and bounded memory use;
6. invalid q coverage is reported explicitly rather than silently dropped.

The final output will report per-case scenario ranges, threshold crossing,
class-flip fraction under each bounded scenario, convergence checkpoints, and
cohort summaries. These outputs remain measurement-sensitivity estimates, not
probabilities of diagnosis.
