# Recalibrated Joint Additive Experiment

## Status

Research-only. This experiment does not modify the Aramina product model,
model artifact, prediction contract, preprocessing, or decision threshold.

## Architecture

```text
z0 = logit(LR1 profile_p_cancer)
z = z0 + alpha + delta*z0
    + age_available * g_age(age)
    + symmetry_available * g_SK(Core4)
p_cancer = sigmoid(z)
```

`delta` is regularized toward zero, so the reference profile slope is one.
The bound `delta > -0.999999` enforces a positive total profile slope. The
intercept is unpenalized. Age and symmetry availability are gates only, never
learned risk features. The primary objective is ordinary unweighted logistic
likelihood.

## Nested Evaluation

The development evaluation is patient-safe repeated stratified 5-fold x10.
For every outer split and every meta-validation fold:

1. LR1 OOF rows for meta-model training are built using meta-train patients only.
2. LR1 fits all meta-train measurements and scores the meta-validation patients.
3. Candidate regularization is evaluated on these cached full-chain folds.
4. The threshold is selected from full-chain outer-train OOF predictions only.
5. The outer test patients are scored once, after all selection is complete.

`fold_manifest.csv` is the patient-role evidence. `paired_fold_deltas.csv`
contains paired differences versus the exact legacy comparator. Fold standard
deviation is descriptive variability, not a confidence interval. A
patient-cluster confidence interval is not implemented.

## Train-All Descriptions

`current_product_exact_legacy` is the exact fitted-LR1, fitted-LR2, fitted
training-threshold procedure. It is labelled
`training_cohort_current_product_exact_not_independent`.

Each joint row is a deployed-chain description: LR1 is refitted on all training
measurements, while the meta-model is fitted on LR1-OOF rows and its threshold
comes from full-chain OOF predictions. These rows are labelled
`training_cohort_deployed_chain_not_independent`.

## Locked T100 Result

The retained run used implementation commit `543a831`, T100 model input,
patient-safe repeated 5-fold cross-validation repeated 10 times, random seed
`42`, and the expanded grid
`C={0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0}`.

The cohort contained 893 measurements from 164 patients and 175 target-breast
cases (`76 CANCER`, `99 BENIGN`). Values below are mean +/- descriptive
standard deviation across the 50 outer folds.

| Model | ROC AUC | Sensitivity | Specificity | Log loss |
|---|---:|---:|---:|---:|
| Exact current product procedure | 0.647 +/- 0.063 | 0.807 +/- 0.099 | 0.386 +/- 0.132 | 0.728 +/- 0.085 |
| Current architecture retrained on OOF LR1 | 0.685 +/- 0.063 | 0.943 +/- 0.081 | 0.196 +/- 0.111 | 0.650 +/- 0.058 |
| Recalibrated profile | 0.602 +/- 0.069 | 0.954 +/- 0.066 | 0.098 +/- 0.097 | 0.680 +/- 0.027 |
| Recalibrated profile + age | **0.689 +/- 0.059** | 0.936 +/- 0.075 | 0.205 +/- 0.102 | **0.635 +/- 0.038** |
| Recalibrated profile + symmetry | 0.585 +/- 0.069 | 0.944 +/- 0.077 | 0.107 +/- 0.095 | 0.692 +/- 0.056 |
| Recalibrated profile + age + symmetry | 0.678 +/- 0.068 | 0.926 +/- 0.086 | 0.207 +/- 0.100 | 0.647 +/- 0.056 |

Recalibration plus age improves held-out ranking and probability loss relative
to the exact current procedure, but it changes the operating point toward more
CANCER calls. Mean specificity falls from `0.386` to `0.205`. The empirical
outer-train threshold target does not guarantee 0.95 sensitivity on unseen
patients; held-out sensitivity is `0.936`.

The SK Core4 block does not improve this architecture. Relative to profile +
age, adding symmetry reduces ROC AUC from `0.689` to `0.678`, reduces
sensitivity from `0.936` to `0.926`, and changes specificity only from `0.205`
to `0.207`. Symmetry regularization selected the strongest tested shrinkage
`C=0.001` in 37 of 51 full-model selections and 44 of 51 profile + symmetry
selections, including the train-all selection.

The exact current train-all description remains ROC AUC `0.865`, sensitivity
`0.961` (`73/76`), and specificity `0.495` (`49/99`) at threshold `0.246659`.
These are training-cohort values, not independent validation. The new stacked
train-all rows use OOF-selected thresholds and a different deployed-chain
training procedure, so their in-cohort values must not be compared as if they
were the same fit.

## Conclusion

The experiment supports profile-logit recalibration as a technically coherent
architecture and confirms that age carries reproducible incremental signal.
It does not support replacing the current product model: specificity at the
intended high-sensitivity operating point is lower, and the symmetry block adds
no stable benefit. The remaining limitation is threshold transport on a small
cohort, not absence of model flexibility.

Retained aggregate evidence is in
`evidence/t100_5x10_20260731/`. Patient-level predictions, threshold scores,
and fold manifests were validated locally but are excluded from Git.

## Limitations

- The cohort remains small: 175 target-breast cases from 164 patients.
- Candidate regularization selection is nested but still high-variance on this
  cohort; grid-boundary selections are explicitly recorded and do not establish
  a precise C value.
- No external or temporally independent cohort is used.
- All currently accepted T100 target cases have age. Missing-age behavior is
  contract-tested but not clinically estimated.
- Symmetry evidence is available only for bilateral cases with valid Core4
  features. Its incremental value requires independent confirmation.
- The Git SHA in a run records the base repository commit. The run also records
  whether the experiment worktree was dirty; an uncommitted research change
  cannot be reconstructed from the SHA alone.
