#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="/opt/aramis-bundle-config"
DEFAULT_WORKFLOW_CONFIG="${CONFIG_ROOT}/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml"
WORKFLOW_CONFIG="${DEFAULT_WORKFLOW_CONFIG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workflow-config)
      WORKFLOW_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown bundle option: $1" >&2
      exit 2
      ;;
  esac
done

case "${WORKFLOW_CONFIG}" in
  "${CONFIG_ROOT}"/workflows/*.yaml|"${CONFIG_ROOT}"/workflows/*.yml) ;;
  *)
    echo "Workflow config must be under ${CONFIG_ROOT}/workflows/: ${WORKFLOW_CONFIG}" >&2
    exit 2
    ;;
esac

[[ -f "${WORKFLOW_CONFIG}" ]] || {
  echo "Missing workflow config: ${WORKFLOW_CONFIG}" >&2
  exit 2
}

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
  --config "${WORKFLOW_CONFIG}" \
  --verbose

MODEL_PATH="$(find examples/outputs/workflows -name model.joblib -print -quit)"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "No generated model.joblib was found." >&2
  exit 1
fi

if [[ "${WORKFLOW_CONFIG}" == "${DEFAULT_WORKFLOW_CONFIG}" ]]; then
  stage "Compare generated model with reference"
  python scripts/compare_model_artifacts.py \
    --reference examples/prediction_models/aramis_m2q_t100_0_2_7_beta.joblib \
    --candidate "${MODEL_PATH}"
else
  stage "Custom workflow completed"
  echo "Reference comparison skipped because a non-baseline workflow was selected."
fi

stage "Bundle completed"
printf 'Generated model: %s\n' "${MODEL_PATH}"
