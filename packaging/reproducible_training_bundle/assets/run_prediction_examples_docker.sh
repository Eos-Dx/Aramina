#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="/opt/Aramis/config/prediction/prediction_examples"
OUTPUT_ROOT="/opt/Aramis/examples/outputs"
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown bundle option: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "${MODEL_PATH}" ]] || { echo "Missing --model." >&2; exit 2; }
[[ -f "${MODEL_PATH}" ]] || { echo "Missing trained model: ${MODEL_PATH}" >&2; exit 2; }

stage() {
  printf '\n=== %s ===\n' "$1"
}

run_example() {
  local name="$1"
  local template="${CONFIG_ROOT}/${name}_predict.yaml"
  local resolved_dir="${OUTPUT_ROOT}/prediction_examples/resolved_configs"
  local resolved="${resolved_dir}/${name}_predict.yaml"

  [[ -f "${template}" ]] || { echo "Missing prediction config: ${template}" >&2; exit 1; }
  mkdir -p "${resolved_dir}"
  python - "${template}" "${resolved}" "${MODEL_PATH}" "${name}" <<'PY'
from pathlib import Path
import sys

import yaml

template, resolved, model_path = map(Path, sys.argv[1:4])
name = sys.argv[4]
config = yaml.safe_load(template.read_text(encoding="utf-8"))
config["io"]["input_model_joblib_path"] = str(model_path)
# Resolved example configs are written under outputs/. Keep the fixture H5
# anchored at its mounted project location instead of resolving relative to that
# output directory or the installed package.
config["io"]["input_h5_path"] = f"/opt/Aramis/examples/prediction_h5/{name}_one_patient.h5"
resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

  stage "Predict ${name} fixture"
  python -m aramis predict --config "${resolved}"
}

stage "Verify trained model"
python - "${MODEL_PATH}" <<'PY'
from pathlib import Path
import joblib

model_path = Path(__import__("sys").argv[1])
joblib.load(model_path)
print(f"model={model_path}")
PY

run_example cancer
run_example benign
run_example atypical

stage "Prediction examples completed"
printf 'Reports: %s\n' "${OUTPUT_ROOT}/prediction_examples"
