# Random Forest LR1 Screening Experiment

Research-only comparison against the frozen `0.2.12-beta` architecture.

## Fixed components

- T100 preprocessing dataframe used by the frozen artifact;
- biopsy-only LR1 profile rows and patient-safe splitting;
- repeated stratified 5-fold x20 (`100` outer held-out folds);
- threshold selected only on each outer training fold for sensitivity `>=0.95`.

## Candidate models

- released logistic LR1 (`C=0.1`) followed by unchanged logistic LR2 (`C=0.3`);
- Random Forest replacing only LR1;
- `min_samples_leaf` = `1`, `2`, `4`, and `8`; `200` trees;
  `max_features=sqrt`; `class_weight=balanced_subsample`.

The candidates are prespecified and all reported. No winner may be selected
from the outer-fold results without a new independent selection procedure.

## Important limitation

The patient-level LR2 is unchanged. This is not a candidate product artifact:
the experiment only tests whether non-linear profile learning has a signal
that warrants a later independent selection and validation procedure.

## Run

```bash
conda run -n eosproduct env PYTHONPATH=src python \
  experiments/random_forest_0_2_12/run_random_forest_lr1_experiment.py \
  --dataframe-joblib examples/outputs/preprocessing_and_training/aramis_target_breast_risk_preprocessing_and_training_20260724T125256Z_552f0758/preprocessing/dataframe.joblib \
  --output-folder outputs/random_forest_lr1_experiment
```

Outputs are `summary.yaml`, `fold_metrics.csv`, `lr1_feature_importances.csv`,
and `train_all_metrics.csv`.

## Results: 2026-07-26

Input: frozen-model T100 dataframe (`893` measurements, `164` patients,
`175` target-breast cases: `76` CANCER / `99` BENIGN). All values below are
mean +/- SD across the same `100` patient-safe held-out folds. LR2 is the same
GatedSymmetryLogistic in every row. The threshold is selected only within the
corresponding outer training fold for sensitivity `>=0.95`, then transferred
unchanged to its held-out patients.

| LR1 profile estimator | Held-out ROC AUC | Held-out sensitivity | Held-out specificity | Train-test ROC AUC gap |
|---|---:|---:|---:|---:|
| Logistic regression, released baseline | 0.645 +/- 0.069 | 0.818 +/- 0.099 | 0.376 +/- 0.133 | 0.237 +/- 0.071 |
| Random Forest, leaf 1 | 0.606 +/- 0.079 | 0.028 +/- 0.045 | 0.974 +/- 0.049 | 0.394 +/- 0.079 |
| Random Forest, leaf 2 | 0.604 +/- 0.077 | 0.070 +/- 0.072 | 0.933 +/- 0.083 | 0.396 +/- 0.077 |
| Random Forest, leaf 4 | 0.601 +/- 0.076 | 0.314 +/- 0.165 | 0.738 +/- 0.137 | 0.399 +/- 0.076 |
| Random Forest, leaf 8 | 0.592 +/- 0.081 | 0.378 +/- 0.168 | 0.688 +/- 0.143 | 0.403 +/- 0.080 |

The logistic baseline reproduces the frozen `0.2.12-beta` footprint exactly.
By contrast, the Random Forest reaches train-on-all ROC AUC `1.000` for leaf
sizes `1` to `4` and `0.995` for leaf `8`, but this does not transfer to
unseen patients. Its fold-specific training threshold is consequently too high
for new patients and held-out sensitivity collapses. Larger leaves reduce
neither the train-test separation nor the loss of sensitivity.

**Conclusion:** no Random Forest candidate is selected. It is materially less
stable than the released regularized logistic LR1 for this cohort. This is an
architecture-screening result only; it does not alter the frozen product model
or its reports.
