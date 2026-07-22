# Future Development Steps

Status: deferred product-development work. None of these items changes the
current research-draft model or its released artifact.

## Product Controls

- Add strict semantic freeze validation for the complete preprocessing pipeline.
- Add immutable model compatibility checks for feature schema, prediction
  contract, and XRD-preprocessing release identity.
- Add a machine-readable row-level measurement audit manifest.
- Make one release tag cover code, configs, documentation, model artifact, and
  reproducibility bundle; require a matching artifact source commit.

## Evidence And Operations

- Add MLflow tracking for one complete dataset-build and training/evaluation run.
- Revisit no-OOF LR1-to-LR2 training on a larger cohort or in an independent
  validation design.
- Revisit a separately labelled contralateral-breast training cohort only when
  an endpoint appropriate for that clinical question is available. The current
  model remains trained on biopsied target breasts and uses the contralateral
  breast only as patient-internal symmetry context.
- Implement the API service as a versioned package with automated contract
  tests; website and PDF clients must consume Aramis report contracts without
  adding clinical logic.
