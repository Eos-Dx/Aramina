# Paired Raw100, FPCA30, and Additive Comparison

Status: research-only internal model-development evaluation. The output is
decision-support evidence requiring radiologist review, not autonomous
diagnosis or independent clinical validation.

## Run

```bash
PYTHONPATH=src python -m aramina.paired_evaluation \
  --raw100-input /Users/sad/dev/Aramina/examples/outputs/model_input/aramina_biopsy_patients_model_input_v0_1.joblib \
  --fpca256-input /Users/sad/dev/Aramina_MCR/experiments/fpca256_profile_encoder/outputs/generated/preprocessing_full_npt256.joblib \
  --output-dir outputs/paired_model_comparison
```

Defaults reproduce patient-safe repeated stratified 5-fold x20 evaluation with
seed `42`, target sensitivity `0.95`, five LR1 cross-fitting folds, four
additive meta folds, and 2,000 patient-cluster bootstrap samples.

## Paired Design

The runner intersects exact measurement identities before building cases. It
verifies matched labels, biopsy flags, ages, target-case keys, and source H5
identity. One `case_manifest.csv` and one `fold_manifest.csv` govern all models:

1. `raw100_product`: raw 100-bin LR1 plus current product LR2.
2. `fpca30_product`: fold-local FPCA30 from 256 bins plus current product LR2.
3. `additive_recalibration_full`: fold-local FPCA30/LR1 profile logit plus age
   and gated SK Core4 symmetry in the full additive architecture.

Comparisons 1 versus 2 isolate profile encoder change. Comparisons 2 versus 3
isolate final-model architecture change. Every outer split contains identical
test patients and target cases for all models. PCA, scalers, LR1, LR2, additive
scalers, and nested meta models fit without outer-test patients.

Product comparators preserve same-data outer-train LR1-to-LR2 fitting and select
their thresholds from fitted outer-train scores. The additive comparator fits
on patient-safe FPCA30/LR1 OOF meta inputs and selects its threshold from nested
full-chain outer-train OOF scores. All thresholds use the unchanged target-
sensitivity rule. Output rows record both fit and threshold provenance.

The additive model uses fixed full-model regularization recorded by the prior
raw100-based `experiment2` run and transfers it to FPCA30 without retuning.
Values were recorded at commit `543a8319108aebee420b39fbcda888234b8045a6`:
`profile_c=0.001`, `age_c=0.3`, and `symmetry_c=0.001`. No regularization is
selected from outer test cases.

## Outputs

`measurement_manifest.csv` records common-cohort inclusion and explicit row
exclusions. `fold_metrics.csv`, `fold_predictions.csv`, and
`threshold_scores.csv` preserve paired evidence. `summary.csv` reports fold
summaries and descriptive patient-bootstrap intervals. Paired fold differences
and repeat-averaged patient-cluster bootstrap intervals are in
`paired_fold_deltas.csv` and `paired_delta_summary.csv`. Both files label the
primary `encoder_effect` as FPCA30 minus raw100 and the primary
`architecture_effect` as additive FPCA30 minus product FPCA30. A secondary
`total_effect` reports additive FPCA30 minus raw100. `run_metadata.yaml`
records input hashes, manifest hashes, runtime versions, controls, and source
commit.

## Limitations

- Common-cohort construction may exclude measurements absent from either
  preprocessing artifact; exclusions remain visible in the measurement
  manifest.
- Fixed additive penalties were selected previously for raw100 on an
  overlapping T100 source cohort and are not independently confirmed for
  FPCA30 inputs.
- Product same-data LR1-to-LR2 fitting and additive OOF meta fitting are
  intentionally different deployable procedures despite identical outer folds.
- Repeated folds overlap. Bootstrap intervals describe repeat-averaged OOF
  predictions and do not establish external or prospective performance.
- No result establishes stable target sensitivity on unseen clinical data.
