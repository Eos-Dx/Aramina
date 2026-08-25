# Research Experiment Configurations

These YAML files are isolated from product preprocessing, training, prediction,
and report contracts.

| Command | Full configuration | Pilot configuration |
|---|---|---|
| Measurement uncertainty | `config_measurement_uncertainty_v0_1.yaml` | `config_measurement_uncertainty_pilot_v0_1.yaml` |
| Polar-basis compression | `config_polar_basis_compression_v0_1.yaml` | `config_polar_basis_compression_pilot_v0_1.yaml` |

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
