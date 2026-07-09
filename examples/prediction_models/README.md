# Aramis Prediction Model Examples

This folder stores small tracked model artifacts for `aramis predict` smoke
tests and colleague onboarding.

Current artifact:

```text
aramis_m2q_t100_train_all_c0p1.joblib
```

It is a research-draft decision-support model artifact. It contains the M2Q
model, model metadata, thresholds, and embedded prediction preprocessing config.
It is used by:

```text
examples/prediction_h5/*_predict.yaml
config/prediction/aramis_predict_from_h5_template_v0_1.yaml
```

These examples test installation, H5 reading, preprocessing, scoring, and
report generation. They are not clinical validation examples.
