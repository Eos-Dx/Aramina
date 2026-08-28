# Experiment Configurations

These YAML files define research-draft sensitivity experiments. They do not
change the frozen product model or prediction contract.

## Joint measurement uncertainty

```bash
python -m aramina experiment-joint-measurement-uncertainty \
  --config config/experiments/config_joint_measurement_uncertainty_pilot_v0_1.yaml
```

The pilot config uses ten patient-safe target cases and ten draws. It verifies
H5/DVC/model lineage, pyFAI and Metal parity, stable random streams, artifact
generation, and MLflow logging.

`config_joint_measurement_uncertainty_full_v0_1.yaml` defines 5000 draws for all
eligible target cases. Run it only after the geometry-aware Metal benchmark and
parity gates recorded in
[`joint_measurement_uncertainty_v0_1.md`](../../docs/experiments/joint_measurement_uncertainty_v0_1.md)
pass.

Long runs write atomic patient/scenario/250-draw-stage checkpoints and publish
convergence summaries and plots after every completed global stage.
To continue an interrupted run, set `output.resume_run_folder` to the absolute
existing run folder. Resume fails closed if data, model, config, cases,
scenarios, or cached detector frames differ.

To pause without losing a completed stage slice:

```bash
touch <run-folder>/STOP_REQUESTED
```

The full config may stop automatically only after 2000 draws and three
consecutive stable convergence checkpoints.

The bounded perturbation quantiles are engineering sensitivity ranges. They are
not clinical confidence intervals.
