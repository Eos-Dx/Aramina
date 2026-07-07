# Aramis v0.1-beta Implementation Plan

Status: research draft. This document is the living checklist for the current
Aramis product-development pass. Update this file whenever a task is completed
or the product decision changes.

Last updated: 2026-07-04.

## Fixed Product Decisions

- [x] Clinical scenario: Aramis v0.1-beta is decision support for women with
  BI-RADS 3/4 findings where the radiologist has already selected the target
  breast.
- [x] Output intent: report `p_cancer` and suggested BENIGN/CANCER support
  class for the target breast; requires radiologist review; not autonomous
  diagnosis.
- [x] Primary training dataset: `biopsy_patients` only. The biopsied breast is
  the training target because it is the clinically suspicious breast and has the
  BENIGN/CANCER endpoint.
- [x] `all_patients` is exploratory only: use it for label-policy and metadata
  sensitivity checks, not for the primary v0.1-beta product model.
- [x] Label policy: `BENIGN + NORMAL -> BENIGN`;
  `CANCER + ATYPICAL + PRE_CANCEROUS -> CANCER`; `NA -> exclude`.
- [x] Main validation view: honest patient-safe 70/30 x50.
- [x] Secondary validation views: stratified K-fold, LOOVM / LOOCV pooled, and
  train-all as exploratory ceiling only.
- [x] Target operating point: target sensitivity is 0.95; report specificity at
  this target.
- [x] Primary candidate for now: M1, profile evidence plus SK symmetry block.
- [x] M2 with age is an audit/comparison model, not primary v0.1-beta model.
- [x] BMI is deferred to later model versions.
- [x] Sample and calibrant thickness are QC/correction/provenance fields, not
  v0.1-beta model features.
- [x] Thickness may be revisited later as a modifier of expected symmetry or
  within-breast variability.
- [x] LR1 profile model uses LogisticRegression on normalized
  `radial_profile_data`.
- [x] LR1 measurement probabilities are aggregated by logit-average, not plain
  probability mean.
- [x] No SNR, intensity, or thickness weighting in the first aggregation rule.
- [x] Symmetry feature set is SK block for v0.1-beta.
  Current cosine fields remain audit/comparison fields.
- [x] Symmetry is interpreted as paired-breast context: the score itself is
  patient-level asymmetry, while target-specific interpretation comes from
  target-side LR1 probability and target/contralateral within-breast features.
- [x] Missing or incomplete paired-breast context must produce explicit
  reliability warnings.
- [x] Standard model joblib keeps model-needed columns plus provenance and drops
  heavy 2D/raw arrays and masks.
- [x] Per-patient report YAML contains only report-instance fields; validation
  details live in model-version metadata.
- [x] MLflow is planned after core preprocess/train/predict workflow is stable.

## Current Model Definitions

M0:

```text
profile_p_cancer_logit_average
```

M1:

```text
profile_p_cancer_logit_average
symmetry_available
sk_meanrms1
sk_weightedrms1
sk_sigma_target1
sk_sigma_contralateral1
sk_mahalanobis1
sk_meanrms2
sk_weightedrms2
sk_sigma_target2
sk_sigma_contralateral2
sk_mahalanobis2
sk_peak14_intensity
sk_mean_peak_value
sk_wasserstein_distance_mu_tc
sk_cosine_distance_full_q2
sk_wasserstein_distance_full_q2
```

M2:

```text
M1 features
age
age_available
```

## Implementation Checklist

### 1. Cleanup

- [x] Audit docs, configs, notebooks, and `src/aramis` for contradictions with
  the fixed decisions above.
- [x] Replace plain LR1 probability averaging with logit-average in training
  code and tests.
- [x] Rename or document patient-level LR1 score fields so code and docs clearly
  distinguish probability mean from logit-average probability.
- [x] Ensure docs consistently describe M1 as the current primary candidate and
  M2 as age audit/comparison.
- [x] Ensure report wording uses decision-support language only.
- [x] Ensure standard joblib output excludes heavy raw data and masks.
- [x] Check README links and remove or mark obsolete preprocessing/training
  configs.
- [x] Run `ruff check .` and `pytest -q`.

### 2. Train CLI

- [x] Keep `python -m aramis train --config <training.yaml>` as the standard
  training entrypoint.
- [x] Add `config/training/aramis_v0_1_beta_primary_train.yaml` as the primary
  v0.1-beta train config: biopsy cohort, M0/M1 only.
- [x] Ensure train YAML fully controls dataset joblib, model set, validation
  mode, target sensitivity, and output paths.
- [x] Ensure model joblib embeds training config text, config SHA256, input
  joblib SHA256, git/version metadata, feature schema, metrics, and warnings.
- [x] Ensure M0/M1/M2 candidates train from the same patient-safe splits when
  compared.
- [x] Align symmetry feature naming and target-side mapping:
  target/contralateral in product docs, train feature table, and future report
  logic; left/right remains only source-side metadata.
- [x] Ensure train summaries report ROC AUC, sensitivity, specificity,
  threshold, PPV, NPV, balanced accuracy, and confusion matrix where available.
- [x] Document train-all as exploratory ceiling only.

### 3. Predict CLI

- [x] Entry point: `python -m aramis predict --config <predict.yaml>`.
- [x] Current v0.1 input: preprocessed DataFrame joblib, trained Aramis model
  joblib, patient id, clinician-supplied `target_side`, `model_id`, and
  `selected_model`.
- [x] H5 route: one-patient H5 can be preprocessed with the prediction
  preprocessing config stored in the model artifact before scoring.
- [x] Current v0.1 output: report YAML/JSON with `p_cancer`, suggested class,
  threshold, warnings, versions, and provenance.
- [x] Report language: decision support, requires radiologist review,
  not autonomous diagnosis.
- [x] Prediction target side is supplied by predict YAML; it is not inferred from
  H5 metadata, labels, biopsy fields, or `specimen_status`.
- [x] Prediction model id is supplied by predict YAML and checked against
  `training.name` stored in the model joblib.

### 4. Report Template

- [x] Use the provided one-page EOSDx PDF as the visual reference for the future
  Aramis report.
- [x] Keep validation mode out of per-patient report YAML; store it in
  model-version metadata.
- [ ] Create report YAML template after predict schema is fixed.

## Notes For Review

- `train-all` can be shown only as an optimistic in-dataset ceiling.
- Main training should use the `biopsy_patients` YAML. The `all_patients` YAML
  remains an exploratory comparison only.
- `biopsy_patients` and exploratory `all_patients` train YAMLs were rerun after
  the 2026-07-04 logit-average and target/contralateral symmetry changes. Full
  model-grid result tables still need regeneration before broad comparison.
- Age can improve results but may act as a clinical shortcut; keep M2 separate.
- If fewer than three valid measurements per breast are available, report
  reduced redundancy. With one measurement per breast, within-breast variability
  cannot be estimated.
- If contralateral breast is unavailable, report profile-only decision support
  with reduced symmetry confidence.
