#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-${HOME}/dev/eosproduct}"
ENV_NAME="${ENV_NAME:-eosproduct}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${TARGET_ROOT}/data/combined_archive.h5}"
ARAMIS_ROOT="${TARGET_ROOT}/Aramis"
AGBH_CONFIG_PATH="${AGBH_CONFIG_PATH:-${ARAMIS_ROOT}/docs/meta/aramis_preprocessing_v0_1_config.json}"
ALL_PATIENTS_CONFIG_PATH="${ALL_PATIENTS_CONFIG_PATH:-${ARAMIS_ROOT}/config/preprocessing/aramis_all_patients_model_input_v0_1.yaml}"
BIOPSY_PATIENTS_CONFIG_PATH="${BIOPSY_PATIENTS_CONFIG_PATH:-${ARAMIS_ROOT}/config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml}"
ALL_PATIENTS_PORT="${ALL_PATIENTS_PORT:-27181}"
BIOPSY_PATIENTS_PORT="${BIOPSY_PATIENTS_PORT:-27182}"

launch_terminal() {
  local title="$1"
  local command="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript <<OSA
tell application "Terminal"
  do script "${command}"
  set custom title of front window to "${title}"
end tell
OSA
  else
    echo "${title}:"
    echo "${command}"
  fi
}

ALL_PATIENTS_URL="http://127.0.0.1:${ALL_PATIENTS_PORT}"
BIOPSY_PATIENTS_URL="http://127.0.0.1:${BIOPSY_PATIENTS_PORT}"

ALL_PATIENTS_CMD="echo 'Aramis all-patients model input: ${ALL_PATIENTS_URL}' && cd '${ARAMIS_ROOT}' && conda run --no-capture-output -n '${ENV_NAME}' python -m marimo run --host 127.0.0.1 --port '${ALL_PATIENTS_PORT}' --no-token examples/aramis_dataframe_all_patients_v0_1.py -- --aramis-preprocessing-config-path '${ALL_PATIENTS_CONFIG_PATH}'"
BIOPSY_PATIENTS_CMD="echo 'Aramis biopsy-patients model input: ${BIOPSY_PATIENTS_URL}' && cd '${ARAMIS_ROOT}' && conda run --no-capture-output -n '${ENV_NAME}' python -m marimo run --host 127.0.0.1 --port '${BIOPSY_PATIENTS_PORT}' --no-token examples/aramis_dataframe_biopsy_patients_v0_1.py -- --aramis-preprocessing-config-path '${BIOPSY_PATIENTS_CONFIG_PATH}'"

launch_terminal "Aramis all-patients" "${ALL_PATIENTS_CMD}"
launch_terminal "Aramis biopsy-patients" "${BIOPSY_PATIENTS_CMD}"
