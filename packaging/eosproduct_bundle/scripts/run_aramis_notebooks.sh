#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-${HOME}/dev/eosproduct}"
ENV_NAME="${ENV_NAME:-eosproduct}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${TARGET_ROOT}/data/combined_archive.h5}"
ARAMIS_ROOT="${TARGET_ROOT}/Aramis"
AGBH_CONFIG_PATH="${AGBH_CONFIG_PATH:-${ARAMIS_ROOT}/docs/meta/aramis_preprocessing_v0_1_config.json}"
ONE_TO_ONE_CONFIG_PATH="${ONE_TO_ONE_CONFIG_PATH:-${ARAMIS_ROOT}/config/preprocessing/aramis_one_to_one_max_v0_1.yaml}"
ONE_TO_MANY_CONFIG_PATH="${ONE_TO_MANY_CONFIG_PATH:-${ARAMIS_ROOT}/config/preprocessing/aramis_one_to_many_max_v0_1.yaml}"
ONE_TO_ONE_PORT="${ONE_TO_ONE_PORT:-27181}"
ONE_TO_MANY_PORT="${ONE_TO_MANY_PORT:-27182}"

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

ONE_TO_ONE_URL="http://127.0.0.1:${ONE_TO_ONE_PORT}"
ONE_TO_MANY_URL="http://127.0.0.1:${ONE_TO_MANY_PORT}"

ONE_TO_ONE_CMD="echo 'Aramis one-to-one: ${ONE_TO_ONE_URL}' && cd '${ARAMIS_ROOT}' && conda run --no-capture-output -n '${ENV_NAME}' python -m marimo run --host 127.0.0.1 --port '${ONE_TO_ONE_PORT}' --no-token examples/aramis_dataframe_one_to_one_v0_1.py -- --aramis-preprocessing-config-path '${ONE_TO_ONE_CONFIG_PATH}'"
ONE_TO_MANY_CMD="echo 'Aramis one-to-many: ${ONE_TO_MANY_URL}' && cd '${ARAMIS_ROOT}' && conda run --no-capture-output -n '${ENV_NAME}' python -m marimo run --host 127.0.0.1 --port '${ONE_TO_MANY_PORT}' --no-token examples/aramis_dataframe_one_to_many_v0_1.py -- --aramis-preprocessing-config-path '${ONE_TO_MANY_CONFIG_PATH}'"

launch_terminal "Aramis one-to-one" "${ONE_TO_ONE_CMD}"
launch_terminal "Aramis one-to-many" "${ONE_TO_MANY_CMD}"
