# Aramis Training Pipeline Classes v0.1

Status: research draft.

## Entry points

```text
run_training_from_config
  -> load_training_config
  -> resolve_training_recipe
  -> load preprocessing DataFrame artifact
  -> AramisPatientTrainingPipeline.fit
  -> write evaluation artifacts
  -> optional final model artifact
```

Combined route:

```text
run_preprocess_train_from_config
  -> run_preprocessing_artifact_from_config
  -> persist preprocessing joblib
  -> pass DataFrame in memory
  -> run_training_from_config
```

## Estimators

`PatientModelInputBuilder`:

```text
select biopsy-only LR1 rows
fit LR1 profile LogisticRegression
score target-breast measurements
logit-average measurement probabilities
create one row per biopsied target breast
calculate age, SK Core4 symmetry, and reliability fields
```

`PatientModelSetEvaluator`:

```text
repeat patient-safe stratified k-fold
fit preprocessing/model state inside each train fold
choose threshold from train-fold scores
score held-out patients only
write per-fold metrics and predictions
```

`PatientModelSetTrainer`:

```text
fit final M2Q recipe on all accepted patients
freeze threshold from train-all scores at sensitivity >=0.95
```

`AramisPatientTrainingPipeline` coordinates those estimators. Evaluation runs
before final model fitting. Bilateral biopsy patients create two target-breast
cases, but both cases always remain in the same patient fold.

## M2Q

```text
normalized target-breast profiles
-> LR1 LogisticRegression, C=0.1
-> target-breast logit-average p_cancer
-> LR2 LogisticRegression, C=0.3
   inputs: profile score + age + gated SK Core4
-> final p_cancer
```

If contralateral data are unavailable, SK features contribute zero after
training-fold scaling. Reliability reports the missing evidence. It is not a
learned risk feature.

## Artifact separation

Evaluation footprint:

```text
evaluation.joblib/json/yaml
evaluation_metrics.csv
evaluation_predictions.csv
```

Deployable research artifact:

```text
model.joblib
model_description.yaml
```

The model joblib contains no fold predictions. It stores executable estimators,
feature schema, threshold, identity, and resolved YAML snapshots needed to
reproduce preprocessing and prediction.
