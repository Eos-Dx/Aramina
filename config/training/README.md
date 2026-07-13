# Aramis Training YAML

Status: research draft.

Training is driven by YAML and sklearn-style transformers/classes in
`src/aramis/training.py`. A training run reads one preprocessing artifact
joblib, builds target-breast model inputs, trains selected model variants, and
writes a model artifact joblib plus optional JSON/YAML summaries. One biopsied
breast is one historical target case; patient-safe splitters keep a patient's
cases in the same fold.

Current product-clean training config:

```text
aramis_m2q_t100_primary_train_v0_1.yaml
```

Primary route:

```text
input DataFrame: examples/outputs/model_selection_m1q_v0_1/preprocessing/aramis_t100_biopsy_patients_model_input.joblib
model family: target-breast M2Q LogisticRegression stack
selected model: M2Q
preprocessing: T100 biopsy-patient cohort; symmetry is optional
regularization: fixed LR1 L2 C=0.1; LR2 L2 C=0.3
final fit: all 175 eligible T100 target cases
validation: documented separately as outer repeated patient-safe stratified 5-fold x20
threshold: train-all operating point targeting sensitivity 0.95
```

Validation evidence remains separate from the final train-all artifact:

```text
primary evidence: nested repeated patient-safe stratified 5-fold x20
secondary evidence: 80/20 x50 and LOOVM are historical robustness checks
train-all: final fitted candidate artifact, not validation evidence
```

Model meanings:

```text
M0: profile p_cancer only
A0: age and age_available only; shortcut-risk control
M0Q: same prediction as M0 plus separate reliability reporting
M1: one profile model with gated SK Core4 refinement when symmetry is available
M1Q: same prediction as M1 plus separate reliability reporting
M2: gated M1 + age and age_available
M2Q: same prediction as M2 plus separate reliability reporting
```

M2Q is the current primary candidate because age is a clinically meaningful
risk prior for breast cancer: older women have higher baseline risk. Age-only
performance is always reported because age can dominate this small cohort.

Risk and reliability are separate. `p_cancer` is the decision-support risk
score. Reliability fields describe whether enough valid target and
contralateral measurements support that score.
Measurement counts do not enter the diagnostic model. `symmetry_available` is
only the gate that makes SK terms neutral when no contralateral breast exists.

Training artifacts store:

```text
training YAML text and SHA256
prediction preprocessing YAML text and SHA256
prediction contract YAML text and SHA256
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
