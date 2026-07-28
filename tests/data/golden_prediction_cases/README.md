# Golden prediction cases

Two real one-patient H5 v0.3 fixtures supplement the three public prediction
examples and the tracked five-patient archive subset.

```text
Nova_106: bilateral measurements remain after QC
Nova_212: only one target measurement remains after QC; symmetry is unavailable
```

The files were extracted from the controlled `combined_archive.h5` without
changing detector frames, PONI data, or measurement metadata. Their SHA-256
identities and expected frozen-model outputs are stored in
`tests/data/golden_prediction_cohort_v0_1.yaml`.
