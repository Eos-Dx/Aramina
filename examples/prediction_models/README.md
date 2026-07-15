# Aramis Prediction Model Examples

This folder stores the tracked model artifact used by `aramis predict` examples
and colleague onboarding.

Tracked artifacts:

```text
aramis_m2q_t100_0_2_7_beta.joblib
  current packaged M2Q artifact with embedded GFRM prediction preprocessing and
  a reproducibility record for full-H5 preprocess-train

aramis_m2q_t100_0_2_7_beta_model_description.yaml
  human-readable immutable identity, SHA256, feature schema, threshold, and
  pointers to the separately retained evaluation artifacts
```

This research-draft decision-support artifact contains the M2Q model, model
metadata, thresholds, embedded prediction preprocessing and immutable
prediction contract. It is used by:

```text
examples/prediction_h5/*_predict.yaml
```

`config/prediction/aramis_predict_from_h5_template_v0_1.yaml` and the three
real one-patient examples use this same artifact. They test installation, H5
reading, preprocessing, scoring and report generation; they are not clinical
validation examples.
