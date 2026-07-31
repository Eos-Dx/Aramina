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

Select regularization before a train-all descriptive fit:

```bash
python experiments/profile_symmetry_age_refinement/select_regularization.py \
  --input-joblib /path/to/preprocessing_artifact.joblib \
  --output-dir outputs/profile_symmetry_age_refinement/regularization_selection
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

## Recalibrated Joint Additive Experiment

`recalibrated_joint_model.py` and `run_recalibrated_joint_experiment.py` are
a second, independent research experiment. They preserve the staged model and
the product code unchanged.

```text
LR1 cross-fitted profile probability
    -> z0 + alpha + delta * z0
    -> jointly fitted optional age contribution
    -> jointly fitted optional SK Core4 contribution
    -> p_cancer
```

`z0` is the LR1 profile logit. The reference calibration is therefore the LR1
identity transform: `alpha = 0`, `delta = 0`, and total profile slope `1`.
The calibration intercept is unpenalized; L2 regularization pulls `delta`
toward zero, rather than pulling the total profile slope toward zero. The age
and symmetry availability fields are gates only: they multiply their respective
contribution and are not fitted as predictors. Missing optional data therefore
produces an exact zero contribution. Age and each SK coefficient have separate
L2 strengths. Primary fitting uses ordinary unweighted logistic likelihood.

For every patient-safe outer fold, each meta validation fold is constructed as
a complete nested chain. LR1 OOF rows used to fit the meta-model are built from
meta-train patients only; LR1 is then fitted on all meta-train measurements to
score the meta-validation patients. These LR1 feature pairs are cached while
candidate penalties are evaluated. Thus a meta-validation patient cannot enter
an LR1 model used to build that fold's meta training table.

Thresholds for the research chains use only these full-chain OOF scores. Each
metric row records its threshold provenance and class-specific threshold sample
counts. `fold_manifest.csv` records outer, meta, and LR1 patient roles. The
runner makes four separately selected joint ablations and two current-model
comparators:

- `calibrated_profile`
- `profile_age`
- `profile_symmetry`
- `profile_age_symmetry`
- `current_product_exact_legacy`: exact legacy LR1/LR2/train-threshold path.
- `current_architecture_oof_retrained`: current architecture retrained from
  the same fully nested OOF inputs as the research model.

`threshold_oof_predictions.csv` preserves every score row that selected a
threshold: outer split, model/ablation, meta-fold, patient/case identifier,
label, score, frozen threshold, provenance, score kind, and all relevant C
values. The legacy fitted-score comparator uses
`legacy_fitted_outer_train_scores`; nested chains use
`nested_full_chain_oof_scores`. Train-all rows use distinct
`training_cohort_*` score kinds.

Run the strict evaluator:

```bash
python experiments/profile_symmetry_age_refinement/run_recalibrated_joint_experiment.py \
  --input-joblib /path/to/preprocessing_artifact.joblib \
  --output-dir outputs/recalibrated_joint_refinement
```

Run only the exploratory regularization selector:

```bash
python experiments/profile_symmetry_age_refinement/select_recalibrated_joint_regularization.py \
  --input-joblib /path/to/preprocessing_artifact.joblib \
  --output-dir outputs/recalibrated_joint_refinement/selection
```

The selector is not independent validation. The evaluator repeats independent
coordinate selection within each ablation and each outer training fold. Its
outer test metrics are the only held-out comparison produced by this experiment.
The default development run is repeated 5-fold x10. A 5-fold x20 run is a
separate locked confirmation run, not a source of additional tuning.

The default block-specific grid is
`C={0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0}`. A selected grid edge is
recorded in `regularization_selection.csv`; it means that the current grid did
not bracket that block's apparent optimum and is not evidence for a precise
regularization value.

The runner exposes all evaluation controls on the CLI:

```bash
python experiments/profile_symmetry_age_refinement/run_recalibrated_joint_experiment.py \
  --input-joblib /path/to/model_input.joblib \
  --output-dir experiments/profile_symmetry_age_refinement/evidence/t100_5x10 \
  --outer-splits 5 --outer-repeats 10 --inner-lr1-splits 5 --meta-splits 4 \
  --candidate-c 0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 \
  --lr1-c 0.1 --current-lr2-c 0.3 --random-state 42
```

Evidence can be written to the tracked
`experiments/profile_symmetry_age_refinement/evidence/<run_id>/` directory.
Review and deliberately add only non-sensitive, approved artifacts; do not
commit input joblib files or patient-level CSVs without data-governance approval.

The primary joint fit is ordinary unweighted logistic likelihood. There is no
balanced-weight variant in this experiment. The total profile slope is
constrained positive using `delta > -0.999999`; this preserves monotonicity of
the LR1 evidence after recalibration.

See [`RESULTS_RECALIBRATED.md`](RESULTS_RECALIBRATED.md) for the results and
limitations of the nested implementation.

## T130 Same-source Held-out Check

`t130_holdout_comparison.py` freezes the current product and both full
recalibrated procedures from T100 before opening the T130 manifest. It then
recreates one-patient H5 containers from the locked source archive, applies the
frozen product prediction preprocessing, and scores the same 22 target-breast
cases for every procedure. Patient-level predictions remain local.

```bash
PYTHONPATH=src python -m \
  experiments.profile_symmetry_age_refinement.t130_holdout_comparison
```

Aggregate evidence and limitations are recorded under
[`evidence/t130_holdout_20260731/`](evidence/t130_holdout_20260731/). This is a
patient-disjoint but same-source T130 quality-control check, not independent
external validation. Five of 17 patients contribute two target breasts.

## Staged Experiment Regularization

This first architecture comparison deliberately freezes regularization rather
than selecting it from the outer test folds:

- LR1 profile model: `C=0.1`, matching the current product experiment.
- Symmetry correction: `C=0.3`.
- Age correction: `C=0.3`.

The custom offset-logistic objective uses the same sample-weight normalization
as scikit-learn logistic regression. Each correction minimizes mean balanced
log loss plus `||beta||^2 / (2 * C * sum(sample_weight))`. Block intercepts are
not penalized. No hyperparameter is selected from held-out patients.

## Staged Experiment Limitations

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

## Staged Experiment Sequential C Selection

`select_regularization.py` is a separate model-selection run. It evaluates the
grid `C={0.03, 0.1, 0.3, 1.0}` with the same patient-safe repeated 5-fold x20
splits. Selection proceeds in architectural order:

1. LR1 `C` from held-out profile probability metrics.
2. Symmetry `C` with LR1 frozen at the selected value.
3. Age `C` with LR1 and symmetry frozen at their selected values.

The prespecified primary criterion is lower held-out log loss, followed by
lower Brier score, higher ROC AUC, higher specificity at the training-fold
target-sensitivity threshold, then smaller `C`. This favors calibrated
probabilities for a model whose final decision threshold is frozen later.

The selection folds cannot also be presented as independent validation of the
selected tuple. The runner therefore writes their summary as
`selected_configuration_reused_fold_summary`, explicitly marked as non-
independent. The selected tuple is then fitted once on all accepted cases and
described in `selected_train_all_metrics.yaml`.
