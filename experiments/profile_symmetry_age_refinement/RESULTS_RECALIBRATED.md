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

## Results Status

No result is claimed for the current implementation until a new locked T100
run writes `threshold_oof_predictions.csv` together with the other outputs.
Earlier local tables predate the final output schema and the expanded C grid;
they are provisional development notes, not a current result or product-model
selection.

The next retained run should write to
`experiments/profile_symmetry_age_refinement/evidence/<run_id>/` and preserve
the complete `summary.yaml`, fold manifest, metrics, paired deltas, and
threshold-score evidence. Fold standard deviations remain descriptive
variability, not confidence intervals.

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
