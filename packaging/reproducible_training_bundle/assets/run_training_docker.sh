#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="/opt/Aramis/config"
DEFAULT_PREPROCESS_TRAIN_CONFIG="${CONFIG_ROOT}/preprocessing_and_training/aramis_target_breast_risk_preprocessing_and_training_v0_1.yaml"
PREPROCESS_TRAIN_CONFIG="${DEFAULT_PREPROCESS_TRAIN_CONFIG}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preprocess-train-config)
      PREPROCESS_TRAIN_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown bundle option: $1" >&2
      exit 2
      ;;
  esac
done

case "${PREPROCESS_TRAIN_CONFIG}" in
  "${CONFIG_ROOT}"/preprocessing_and_training/*.yaml|"${CONFIG_ROOT}"/preprocessing_and_training/*.yml) ;;
  *)
    echo "Preprocessing-and-training config must be under ${CONFIG_ROOT}/preprocessing_and_training/: ${PREPROCESS_TRAIN_CONFIG}" >&2
    exit 2
    ;;
esac

[[ -f "${PREPROCESS_TRAIN_CONFIG}" ]] || {
  echo "Missing preprocess-train config: ${PREPROCESS_TRAIN_CONFIG}" >&2
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
rm -rf examples/outputs/preprocessing_and_training
python -m aramis preprocess-train \
  --config "${PREPROCESS_TRAIN_CONFIG}" \
  --verbose

MODEL_PATH="$(find examples/outputs/preprocessing_and_training -name model.joblib -print -quit)"
if [[ -z "${MODEL_PATH}" ]]; then
  echo "No generated model.joblib was found." >&2
  exit 1
fi

if [[ "${PREPROCESS_TRAIN_CONFIG}" == "${DEFAULT_PREPROCESS_TRAIN_CONFIG}" ]]; then
  stage "Compare generated model with reference"
  python scripts/compare_model_artifacts.py \
    --reference models/aramis_target_breast_risk_0_2_8-beta_509f84b2a745/model.joblib \
    --candidate "${MODEL_PATH}"
else
  stage "Custom preprocess-train completed"
  echo "Reference comparison skipped because a non-baseline config was selected."
fi

stage "Bundle completed"
printf 'Generated model: %s\n' "${MODEL_PATH}"
