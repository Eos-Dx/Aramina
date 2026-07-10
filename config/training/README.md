# Aramis Training YAML

Status: research draft.

Training is driven by YAML and sklearn-style transformers/classes in
`src/aramis/training.py`. A training run reads one preprocessing artifact
joblib, builds patient-level model inputs, trains selected model variants, and
writes a model artifact joblib plus optional JSON/YAML summaries.

Current product-clean training config:

```text
aramis_m2q_t100_primary_train_v0_1.yaml
```

Primary route:

```text
input DataFrame: examples/outputs/model_selection_m1q_v0_1/preprocessing/aramis_t100_biopsy_patients_model_input.joblib
model family: patient_m0_m1_m2_logistic_set
selected product candidate: M2Q
preprocessing: T100 biopsy-patient cohort; symmetry is optional
regularization: LR1 L2 C=0.1; LR2 L2 C=0.1
final fit: all eligible development patients
validation recorded in artifact: repeated patient-safe stratified 5-fold x20
target sensitivity: 0.95 training-fold operating point
```

Validation evidence remains separate from the final train-all artifact:

```text
primary evidence: repeated patient-safe stratified 5-fold x20
secondary evidence: patient-safe 80/20 x50 and LOOVM
train-all: final fitted candidate artifact, not validation evidence
```

Model meanings:

```text
M0: profile p_cancer only
M0Q: M0 + target measurement count
M1: profile p_cancer + same-patient SK symmetry block
M1Q: M1 + target measurement count
M2: M1 + age and age_available
M2Q: M1Q + age and age_available; product schema uses SK Core4
```

M2Q is the current primary candidate because age is a clinically meaningful
risk prior for breast cancer: older women have higher baseline risk. In current
model-selection checks, adding age improved the candidate model by roughly 3
percentage points while preserving the same preprocessing and regularization.

Risk and reliability are separate. `p_cancer` is the decision-support risk
score. Reliability fields describe whether enough valid target and
contralateral measurements support that score.

Training artifacts store:

```text
training YAML text and SHA256
prediction preprocessing YAML text and SHA256
model entries
feature schema
metrics
dataset summary
```

Full historical model grids are archived on branch:

```text
experiment/aramis-v0.1-research-state
```

The development branch intentionally does not keep runnable model-selection YAML
grids. It keeps only the product candidate training YAML and the evidence
documents that justify it.
