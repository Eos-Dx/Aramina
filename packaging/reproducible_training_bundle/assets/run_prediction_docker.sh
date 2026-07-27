#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=""
INPUT_H5_PATH=""
MODEL_PATH=""
OUTPUT_FOLDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --input-h5)
      INPUT_H5_PATH="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output-folder)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

for pair in \
  "CONFIG_PATH:${CONFIG_PATH}" \
  "INPUT_H5_PATH:${INPUT_H5_PATH}" \
  "MODEL_PATH:${MODEL_PATH}" \
  "OUTPUT_FOLDER:${OUTPUT_FOLDER}"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  [[ -n "${value}" ]] || { echo "Missing --${name,,}." >&2; exit 2; }
done

[[ -f "${CONFIG_PATH}" ]] || { echo "Missing prediction config: ${CONFIG_PATH}" >&2; exit 1; }
[[ -f "${INPUT_H5_PATH}" ]] || { echo "Missing input H5: ${INPUT_H5_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}" ]] || { echo "Missing model joblib: ${MODEL_PATH}" >&2; exit 1; }
mkdir -p "${OUTPUT_FOLDER}"

RESOLVED_CONFIG="${OUTPUT_FOLDER}/prediction_request_resolved.yaml"
python - "${CONFIG_PATH}" "${RESOLVED_CONFIG}" "${INPUT_H5_PATH}" "${MODEL_PATH}" "${OUTPUT_FOLDER}" <<'PY'
from pathlib import Path
import sys

import yaml

source, target, input_h5, model, output = (
    Path(value).expanduser().resolve() for value in sys.argv[1:]
)
config = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit(f"Prediction config must be a YAML mapping: {source}")
config.setdefault("io", {})
config["io"]["input_h5_path"] = str(input_h5)
config["io"].pop("input_dataframe_joblib_path", None)
config["io"]["input_model_joblib_path"] = str(model)
config["io"]["output_folder"] = str(output)
target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

printf '\n=== Run external H5 prediction ===\n'
python -m aramina predict --config "${RESOLVED_CONFIG}" --verbose

REPORT_PATH="$(find "${OUTPUT_FOLDER}" -maxdepth 1 -name '*_external_report.yaml' -type f -print -quit)"
[[ -n "${REPORT_PATH}" ]] || { echo "External report YAML was not created." >&2; exit 1; }
printf '\n=== External report ===\n'
cat "${REPORT_PATH}"
printf 'Reports: %s\n' "${OUTPUT_FOLDER}"
