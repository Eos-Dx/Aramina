# Research Experiment Configurations

These YAML files are isolated from product preprocessing, training, prediction,
and report contracts.

| Command | Full configuration | Pilot configuration |
|---|---|---|
| Measurement uncertainty | `config_measurement_uncertainty_v0_1.yaml` | `config_measurement_uncertainty_pilot_v0_1.yaml` |
| Polar-basis compression | `config_polar_basis_compression_v0_1.yaml` | `config_polar_basis_compression_pilot_v0_1.yaml` |
| Polar harmonic ablation | `config_polar_harmonic_ablation_v0_1.yaml` | `config_polar_harmonic_ablation_pilot_v0_1.yaml` |

Run the polar pilot:

```bash
aramina experiment-polar-basis-compression \
  --config config/experiments/config_polar_basis_compression_pilot_v0_1.yaml \
  --verbose
```

The pilot limits patient count but retains all three representations, all three
coefficient budgets, patient-safe folds, DVC/model lineage, and MLflow output.
The full configuration may generate missing 256 x 36 polar cakes from the
existing immutable H5 archive. It does not permit new measurements.

Run the polar harmonic ablation pilot:

```bash
aramina experiment-polar-harmonic-ablation \
  --config config/experiments/config_polar_harmonic_ablation_pilot_v0_1.yaml \
  --verbose
```

The harmonic ablation compares `A0`, `A0+A2`, and `A0+A2+A4` using cubic
B-spline compression. The full run sweeps `n_chi` values `12`, `18`, `36`, and
`72`, coefficient budgets `8`, `12`, and `16`, and repeated patient-safe
`5-fold x20` evaluation. It uses all `496` accepted target measurements and
`175` target cases. It is exploratory and does not change product artifacts or
reports.
