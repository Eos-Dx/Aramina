# Aramis Prediction Model Examples

This folder stores small tracked model artifacts for `aramis predict` smoke
tests and colleague onboarding.

Tracked artifacts:

```text
aramis_m2q_t100_gated_sk_core4_nested_c1_0p1_c2_0p3.joblib
  packaged product artifact with embedded GFRM preprocessing

aramis_m2q_t100_gated_sk_core4_synthetic_h5_example.joblib
  synthetic H5 smoke-test artifact with embedded raw-array preprocessing
```

Both are research-draft decision-support artifacts. Each contains the M2Q
model, model metadata, thresholds, embedded prediction preprocessing, and an
immutable prediction contract. The synthetic artifact is used by:

```text
examples/prediction_h5/*_predict.yaml
```

`config/prediction/aramis_predict_from_h5_template_v0_1.yaml` uses the product
artifact. The synthetic examples test installation, H5 reading,
preprocessing, scoring, and report generation. They are not clinical
validation examples.
