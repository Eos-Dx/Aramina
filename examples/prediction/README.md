# Prediction examples

`configs/` contains runnable one-patient prediction examples. Paths resolve from
the Aramis root and point to the tracked product artifact. In the reproducible
Docker bundle, the launcher replaces only `io.input_model_joblib_path` with the
newly trained model and writes the resolved copy to
`outputs/prediction_examples/resolved_configs/`.
