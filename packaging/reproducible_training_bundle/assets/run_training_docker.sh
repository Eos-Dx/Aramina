#!/usr/bin/env bash
set -euo pipefail

stage() {
  printf '\n=== %s ===\n' "$1"
}

stage "Verify installed packages"
python - <<'PY'
import aramis
import pyFAI
import xrd_preprocessing

print(f"aramis={aramis.__file__}")
print(f"xrd_preprocessing={xrd_preprocessing.__file__}")
print(f"pyfai={pyFAI.version}")
PY

stage "Run preprocessing and training"
rm -rf examples/outputs/workflows
python -m aramis preprocess-train \
  --config config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml \
  --verbose

MODEL_PATH="$(find examples/outputs/workflows -name model.joblib -print -quit)"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "No generated model.joblib was found." >&2
  exit 1
fi

stage "Compare generated model with reference"
python scripts/compare_model_artifacts.py \
  --reference examples/prediction_models/aramis_m2q_t100_0_2_7_beta.joblib \
  --candidate "${MODEL_PATH}"

stage "Bundle completed"
printf 'Generated model: %s\n' "${MODEL_PATH}"
