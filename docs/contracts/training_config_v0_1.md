# Aramina Training Config Contract v0.3

Status: legacy research draft for tag `0.2.13-beta`.

The retained YAML filename is
`config/training/config_training_target_breast_risk_v0_1.yaml`; its declared
contract is `aramina_training_config_v0_3` and model version is
`0.2.13-beta`.

It selects the fixed target-breast logistic-regression architecture, runs
patient-safe repeated stratified 5-fold x20 evaluation, and optionally fits the
executable model on all accepted target cases. It requires preprocessing YAML
and an input-H5 SHA256 in the preprocessing artifact, but does not require DVC.

Use it with code checked out at tag `0.2.13-beta`. Current DVC-backed development
uses [training contract v0.4](training_config_v0_4.md).
