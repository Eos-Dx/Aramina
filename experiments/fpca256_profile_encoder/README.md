# FPCA256 profile encoder experiment

Research-only comparison of the current Aramina model with a lower-dimensional
LR1 profile representation. This directory does not alter product source,
configuration, contracts, prediction behavior, or promoted model artifacts.

## Question

Does replacing the raw LR1 radial profile with a 4-7 component discrete FPCA
representation preserve patient-safe sensitivity, specificity, and ROC AUC?

`scikit-learn PCA` is applied to profiles sampled on one shared, uniformly
spaced q grid. On that grid it is used as a discrete approximation to functional
PCA. PCA is fitted independently inside every outer training fold. Held-out
patients never contribute to the PCA mean, components, LR1, LR2, or threshold.

## Controlled architecture

LR2 architecture and regularization remain fixed. Two controlled profile-path
changes are evaluated: integration resolution and LR1 representation. Changing
`npt` also changes the sampled normalized profiles and therefore the Core4
symmetry values derived from those profiles; those values are recomputed inside
each npt dataset rather than copied from the npt100 input.

```text
H5 / preprocessing artifact
-> npt=256 normalized radial profile
-> raw 256 bins OR fold-local PCA with 4, 5, 6, or 7 components
-> LR1 LogisticRegression(C=0.1, class_weight="balanced")
-> measurement probabilities aggregated by mean logit for each target breast
-> LR2 profile + age + neutral-gated SK Core4
-> LogisticRegression(C=0.3, class_weight="balanced")
-> threshold calibrated on each outer-training partition for target sensitivity 0.95
```

The 0.95 value is a threshold-selection target on outer-training scores. It is
not a guarantee of 0.95 sensitivity on held-out patients.

Preserved policies:

- LR1 rows: `biopsy_only`.
- Target unit: biopsied target-breast case.
- Bilateral target cases remain in one patient-safe fold.
- Evaluation: repeated stratified 5-fold x20.
- Symmetry: Core4 recomputed from each npt dataset; unavailable symmetry is
  neutralized and its availability flag is not a learned feature.
- Train-on-all: same architecture and regularization; metrics are explicitly
  in-sample and are not independent validation.

## Cohorts

### Controlled common cohort

Exact matched measurement identities and shared patient folds:

| Artifact | Rows | Patients | Target cases |
|---|---:|---:|---:|
| npt100 common | 803 | 161 | 164 |
| npt256 common | 803 | 161 | 164 |

Encoders: `raw100`, `raw256`, `fpca256_4`, `fpca256_5`, `fpca256_6`,
`fpca256_7`.

### Full npt256 cohort

| Artifact | Rows | Patients | Target cases |
|---|---:|---:|---:|
| npt256 full | 876 | 163 | 171 |

Encoders: `raw256`, `fpca256_4`, `fpca256_5`, `fpca256_6`, `fpca256_7`.

## Immutable input lineage

The YAML pins each existing preprocessing artifact by file SHA-256, pipeline
fingerprint, source-H5 SHA-256, integration variant, npt, and PyFAI method
`bbox/csr/cython`. The experiment refuses mismatches before fitting a model.

The pinned runtime is `pyFAI==2026.5.0`. The current XRD transformer does not
pass `method` to `integrate1d`; therefore PyFAI supplies its exact signature
default `("bbox", "csr", "cython")`. The source is recorded as
`pyfai_integrate1d_default`. Before artifact evaluation or raw-H5 preprocessing,
the runner verifies both the installed distribution version and the inspected
`integrate1d` default. A version or default-method mismatch stops the run.

The pinned preprocessing artifacts were independently regenerated from the
source H5 under this pinned default. Their dataframes and resulting model
metrics matched exactly. The result lineage records the verified PyFAI version,
method, and method source for every input artifact.

Raw-H5 mode separately validates the source-H5 SHA-256 and the base main
preprocessing-config SHA-256. Its newly generated artifact SHA-256 is recorded,
not compared with a previous artifact, because serialized provenance and
timestamps may legitimately change the file bytes.

## Run from existing artifacts

```bash
cd /Users/sad/dev/Aramina_MCR
conda activate eosproduct

python -m experiments.fpca256_profile_encoder.runner \
  --config experiments/fpca256_profile_encoder/config_fpca256_profile_encoder_v0_1.yaml \
  --cohort all \
  --verbose
```

Run one mode:

```bash
python -m experiments.fpca256_profile_encoder.runner \
  --config experiments/fpca256_profile_encoder/config_fpca256_profile_encoder_v0_1.yaml \
  --cohort common

python -m experiments.fpca256_profile_encoder.runner \
  --config experiments/fpca256_profile_encoder/config_fpca256_profile_encoder_v0_1.yaml \
  --cohort full_npt256
```

## Run from H5

Raw H5 mode creates a research-only npt256 preprocessing artifact from the
current main biopsy-patient preprocessing config. It changes only
`integration.npt` in an in-memory copy and bypasses the product npt=100 contract.
It does not write to `config/` or product code.

```bash
python -m experiments.fpca256_profile_encoder.runner \
  --config experiments/fpca256_profile_encoder/config_fpca256_profile_encoder_v0_1.yaml \
  --cohort full_npt256 \
  --input-h5 /path/to/combined_archive.h5 \
  --verbose
```

## Outputs

Each cohort directory contains:

- `fold_metrics.csv`
- `fold_predictions.csv`
- `fold_manifest.csv`
- `aggregate_summary.csv`
- `aggregate_summary.yaml`
- `repeat_averaged_cross_fitted_predictions.csv`
- `paired_fold_deltas.csv`
- `paired_delta_summary.csv`
- `train_all_artifact.joblib`
- `pca_explained_variance.csv`
- `pca_basis_components.csv`
- `pca_fold_basis.joblib`
- `roc_comparison.png`
- `fpca_component_convergence.png`
- `effective_experiment_config.yaml`

The FPCA30 follow-up additionally contains a descriptive component analysis in
`results/components_10_to_30/common/component_interpretation/`:

- `fpca30_component_activity.csv`
- `fpca30_fold_component_stability.csv`
- `fpca30_active_components.png`
- `fpca30_component_interpretation.md`

Recreate this footprint after the FPCA30 sweep with:

```bash
python -m experiments.fpca256_profile_encoder.component_interpretation \
  --config experiments/fpca256_profile_encoder/config_fpca256_profile_encoder_v0_1.yaml \
  --result-folder experiments/fpca256_profile_encoder/results/components_10_to_30/common \
  --output-folder experiments/fpca256_profile_encoder/results/components_10_to_30/common/component_interpretation
```

Component activity is assessed from the LR1 coefficient per standardized
component score, while basis stability is checked across the 100 patient-safe
outer folds. Neither value establishes a biological interpretation.

`fold_manifest.csv` records every train/test target-case assignment. Bilateral
target cases remain in the same patient-safe set. Paired deltas compare each
FPCA encoder with the applicable raw baseline on the same folds. Their 2.5% and
97.5% limits are descriptive empirical quantiles, not inferential confidence
intervals: repeated folds overlap and are not independent observations. They
must not be used as a model-selection test.

## Limitations

- Small retrospective cohort.
- Repeated folds are not independent datasets.
- Train-on-all metrics are in-sample only.
- Discrete PCA is an FPCA approximation, not a continuous basis fit.
- PCA signs are arbitrary; component signs may reverse without changing model
  information.
- No product model promotion is performed.
- Direct comparisons with `raw100` are valid only in the controlled common
  cohort with matched measurements and folds.
- Research decision support only; not for autonomous diagnosis.

See [RESULTS.md](RESULTS.md) for the completed experiment results and the
non-promotion conclusion. A compact, tracked audit footprint is stored under
[`results/`](results/README.md); large executable and per-case outputs remain
generated locally under `outputs/`.
