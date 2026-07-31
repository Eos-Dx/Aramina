# Profile -> Symmetry -> Age Refinement Experiment

Research-only comparison of the current `GatedSymmetryLogistic` model with an
experiment-owned staged classifier. This folder does not change the Aramina
product model, product configuration, or prediction contract.

## Question

Does an explicit sequential refinement model behave differently from the
current joint LR2 model when both are evaluated on exactly the same held-out
patients?

```text
LR1 radial-profile model
    -> profile_p_cancer
    -> optional symmetry correction
    -> after_symmetry_p_cancer
    -> optional age correction
    -> final_p_cancer
```

The staged model is implemented separately in `staged_model.py` as
`StagedProfileSymmetryAgeClassifier`. The runner requires this API:

```python
StagedProfileSymmetryAgeClassifier(
    symmetry_c=0.3,
    age_c=0.3,
    random_state=42,
)
```

Methods required by the runner:

```python
fit(X, y)
predict_proba(X)
predict_stage_probabilities(X)
stage_logit_corrections(X)
```

`predict_stage_probabilities(X)` must return `profile_p_cancer`,
`after_symmetry_p_cancer`, and `final_p_cancer`.

## Offset architecture

Each later block refines the preceding evidence in logit space. It is not a
sequence of independent probabilities. Conceptually:

```text
z_profile = logit(profile_p_cancer)
z_symmetry = z_profile + symmetry_available * delta_z_symmetry
z_final = z_symmetry + age_available * delta_z_age
p_cancer = sigmoid(z_final)
```

The staged model owns the precise implementation. The intended identity
behavior is strict: when symmetry is unavailable, `after_symmetry_p_cancer`
equals `profile_p_cancer`; when age is unavailable, `final_p_cancer` equals
`after_symmetry_p_cancer`.

## Evaluation

The runner uses repeated stratified 5-fold cross-validation, repeated 20
times, for 100 patient-safe outer test folds. A patient, including bilateral
target cases, cannot occur in both training and test data for a fold.

For each outer fold:

1. LR1 is fitted only on training measurements.
2. Training LR1 scores form train patient-level features.
3. The fitted training LR1 scores the untouched test measurements.
4. Current LR2 and staged models fit only training patient features.
5. Each model/stage selects its threshold on training scores only, at target
   sensitivity `0.95`.
6. Held-out metrics use the frozen training threshold on test patients.

The same folds, LR1 policy, input features, target sensitivity, and random
seed are used for the current and staged model. Symmetry and age regularization
are fixed at `C=0.3`; LR1 remains fixed at `C=0.1`.

Run:

```bash
python experiments/profile_symmetry_age_refinement/run_experiment.py \
  --input-joblib /path/to/preprocessing_artifact.joblib \
  --output-dir outputs/profile_symmetry_age_refinement
```

## Outputs

`summary.yaml` is the compact machine-readable record. CSV files support
inspection and paired fold-level comparisons:

- `summary.csv`: mean and standard deviation across outer folds.
- `fold_metrics.csv`: one metrics row per model/stage and outer fold.
- `split_predictions.csv`: held-out target-case predictions and frozen
  training thresholds.
- `train_all_metrics.csv`: in-sample metrics after fitting all accepted cases.

Train-all metrics are descriptive only. They are not independent validation.
The completed T100 cohort comparison is summarized in
[`RESULTS.md`](RESULTS.md).

## Regularization

This first architecture comparison deliberately freezes regularization rather
than selecting it from the outer test folds:

- LR1 profile model: `C=0.1`, matching the current product experiment.
- Symmetry correction: `C=0.3`.
- Age correction: `C=0.3`.

The custom offset-logistic objective uses the same sample-weight normalization
as scikit-learn logistic regression. Each correction minimizes mean balanced
log loss plus `||beta||^2 / (2 * C * sum(sample_weight))`. Block intercepts are
not penalized. No hyperparameter is selected from held-out patients.

## Limitations

- This is a research comparison on the current small cohort, not a product
  selection or clinical-validation result.
- The first-stage LR1 is fitted within each outer training fold before scoring
  the test fold. No test patient, measurement, or target breast contributes to
  LR1, correction-block fitting, or threshold selection.
- The correction blocks receive train LR1 fitted scores, as does the current
  product LR2. This preserves a fair comparison with the current architecture,
  but it is not a fully nested comparison of all possible correction-model
  designs.
- A missing block must act as an identity transformation, not as a learned
  missingness predictor. Availability flags are gates, not risk features.
- All current T100 target cases contain age. The age-missing identity path is
  contract-tested, but its clinical performance cannot be estimated from this
  cohort.
- The fixed correction regularization isolates the proposed architecture. A
  later model-selection experiment would require nested patient-safe tuning;
  the present outer-fold results must not be used to tune `C`.
- Performance changes require confirmation on an independent cohort before any
  product-model or preprocessing change.
