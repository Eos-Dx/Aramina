# Docker prediction examples

These three templates are for the reproducible Docker bundle only. The launcher
sets `io.input_model_joblib_path` to the model created by `install_and_train` and
writes an auditable resolved copy to `outputs/prediction_examples/resolved_configs/`.

The templates use container paths deliberately. Run them through
`predict_examples.bat`, `predict_examples.ps1`, or `predict_examples.sh`; do not
run these YAML files directly on the host.
