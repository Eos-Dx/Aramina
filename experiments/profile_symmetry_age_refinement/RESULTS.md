# Staged Refinement Results

Research-only result. Not clinically validated and not for autonomous
diagnosis.

## Cohort and protocol

- Input: current T100 biopsy preprocessing artifact.
- Measurements: 893.
- Patients: 164.
- Target-breast cases: 175 (`76 CANCER`, `99 BENIGN`).
- Evaluation: patient-safe repeated stratified 5-fold cross-validation,
  repeated 20 times (`100` held-out folds).
- LR1, refinement blocks, and the target-sensitivity threshold are fitted from
  each training fold only.
- Target threshold sensitivity: `0.95`.
- Initial architecture comparison: LR1 `C=0.1`; current LR2, staged symmetry,
  and staged age corrections `C=0.3`.

## Held-out results

Mean +/- standard deviation across the same 100 test folds:

| Model stage | ROC AUC | PR AUC | Sensitivity | Specificity | Brier | Log loss |
|---|---:|---:|---:|---:|---:|---:|
| Current joint LR2 | 0.645 +/- 0.069 | 0.605 +/- 0.085 | 0.818 +/- 0.099 | 0.376 +/- 0.133 | 0.255 +/- 0.033 | 0.732 +/- 0.089 |
| Profile only | 0.603 +/- 0.078 | 0.552 +/- 0.090 | 0.821 +/- 0.101 | 0.326 +/- 0.107 | 0.245 +/- 0.019 | 0.686 +/- 0.044 |
| Profile -> symmetry | 0.584 +/- 0.083 | 0.552 +/- 0.088 | 0.789 +/- 0.104 | 0.317 +/- 0.114 | 0.252 +/- 0.021 | 0.702 +/- 0.056 |
| Profile -> symmetry -> age | **0.686 +/- 0.069** | **0.646 +/- 0.078** | **0.850 +/- 0.089** | 0.360 +/- 0.141 | **0.228 +/- 0.023** | **0.653 +/- 0.062** |

Relative to current joint LR2, the staged final score changes mean held-out
metrics by:

- ROC AUC: `+0.040`; higher in 83 of 100 folds.
- PR AUC: `+0.041`; higher in 81 of 100 folds.
- Sensitivity: `+0.033`.
- Specificity: `-0.016`.
- Brier score: `-0.027`; lower in 93 of 100 folds.
- Log loss: `-0.079`; lower in 95 of 100 folds.

The symmetry correction alone does not improve this cohort. Relative to the
profile score, it changes ROC AUC by `-0.018`, sensitivity by `-0.032`, and
specificity by `-0.009`. The subsequent age correction is responsible for the
observed gain: relative to the symmetry stage, ROC AUC changes by `+0.101`,
sensitivity by `+0.061`, specificity by `+0.043`, Brier score by `-0.023`,
and log loss by `-0.048`.

## Train-all description

These values are in-sample descriptions, not independent validation:

| Model stage | ROC AUC | Sensitivity | Specificity | TP | TN | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current joint LR2 | 0.865 | 0.961 | 0.495 | 73 | 49 | 3 | 50 |
| Profile only | 0.832 | 0.961 | 0.374 | 73 | 37 | 3 | 62 |
| Profile -> symmetry | 0.846 | 0.961 | 0.424 | 73 | 42 | 3 | 57 |
| Profile -> symmetry -> age | 0.843 | 0.961 | 0.414 | 73 | 41 | 3 | 58 |

## Interpretation

The proposed sequential architecture is technically viable: each block adds a
regularized correction in logit space, the age correction can depend on the
incoming risk, and an unavailable block is an exact identity transformation.

The experiment does not support promoting this architecture to the product.
Age improves held-out ranking and probability error, but the staged final
model does not improve held-out specificity at the train-selected
target-sensitivity threshold. Symmetry is not independently beneficial in the
tested order. The divergence between held-out and train-all behavior remains a
warning that 164 patients are insufficient for a reliable architecture change.

## Sequential Regularization Selection

A separate patient-safe repeated 5-fold x20 selection used
`C={0.03, 0.1, 0.3, 1.0}`. Selection was sequential and probability-first:
lower mean held-out log loss, then lower Brier score, higher ROC AUC, higher
specificity at the train-fold target-sensitivity threshold, and smaller `C`.

| Stage | Selected C | Mean held-out log loss | Mean ROC AUC | Mean sensitivity | Mean specificity |
|---|---:|---:|---:|---:|---:|
| LR1 profile | 0.10 | 0.686 | 0.603 | 0.821 | 0.326 |
| Symmetry correction | 0.03 | 0.694 | 0.588 | 0.775 | 0.346 |
| Age correction | 1.00 | 0.647 | 0.691 | 0.866 | 0.339 |

The selected tuple is therefore `LR1 C=0.10`, `symmetry C=0.03`, and
`age C=1.00`. The repeated-fold figures used to select it are not an
independent estimate of its final performance.

The selected train-all fit reaches sensitivity `0.961` (`73/76 CANCER`) at a
threshold of `0.306`, but specificity is only `0.343` (`34/99 BENIGN`), with
`65` false positives. It is therefore weaker at the intended operating point
than both the initial staged fit (`0.414` specificity) and the current product
joint LR2 (`0.495` specificity), each at the same train-all sensitivity.

This selection run resolves the regularization question for the tested grid:
the staged architecture is not being held back simply by its original fixed
regularization. It remains research-only and should not replace the current
product model.
