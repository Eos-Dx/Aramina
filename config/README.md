# Aramina Configuration

This directory contains only input YAML files used by Aramina commands. Every
runnable input file begins with `config_`. Product preprocessing YAMLs are
assembled from readable fragments under `preprocessing/`.

```text
prediction/                    one-patient prediction request template
preprocessing/                 product preprocessing inputs and fragments
preprocessing_and_training/    combined preprocess-and-train input
training/                      training input
```

Runnable examples live under `examples/`. Filled output-contract examples live
under `contracts/`. Canonical field definitions live under `docs/contracts/`.
