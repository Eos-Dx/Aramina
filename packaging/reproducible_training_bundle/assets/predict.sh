#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${BUNDLE_DIR}/bundle_manifest.json"
LOG_DIR="${BUNDLE_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/predict_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

CONFIG=""
INPUT_H5=""
MODEL_PATH=""
OUTPUT_FOLDER=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --input-h5) INPUT_H5="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --output-folder) OUTPUT_FOLDER="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

for value in CONFIG INPUT_H5 MODEL_PATH OUTPUT_FOLDER; do
  [[ -n "${!value}" ]] || { echo "Missing --${value,,}." >&2; exit 2; }
done
for path in "${CONFIG}" "${INPUT_H5}" "${MODEL_PATH}"; do
  [[ -f "${path}" ]] || { echo "Missing file: ${path}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_FOLDER}"

read_manifest() {
  python3 - "${MANIFEST}" "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)[sys.argv[2]])
PY
}
sha256_file() { command -v sha256sum >/dev/null 2>&1 && sha256sum "$1" | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'; }
stage() { printf '\n=== %s ===\n' "$1"; }

case "$(uname -m)" in
  x86_64|amd64) IMAGE_ARCH="amd64" ;;
  arm64|aarch64) IMAGE_ARCH="arm64" ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac
IMAGE_TAG="$(read_manifest "image_${IMAGE_ARCH}_tag")"
IMAGE_PLATFORM="$(read_manifest "image_${IMAGE_ARCH}_platform")"
IMAGE_ARCHIVE="${BUNDLE_DIR}/$(read_manifest "image_${IMAGE_ARCH}_archive")"
EXPECTED_IMAGE="$(read_manifest "image_${IMAGE_ARCH}_archive_sha256")"

command -v docker >/dev/null || { echo "Docker is required. Run install_and_train first." >&2; exit 1; }
docker version >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  stage "Load validated Linux runtime image"
  [[ "$(sha256_file "${IMAGE_ARCHIVE}")" == "${EXPECTED_IMAGE}" ]] || { echo "Docker image SHA256 mismatch." >&2; exit 1; }
  docker load --input "${IMAGE_ARCHIVE}"
fi

CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd -P)/$(basename "${CONFIG}")"
INPUT_H5="$(cd "$(dirname "${INPUT_H5}")" && pwd -P)/$(basename "${INPUT_H5}")"
MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd -P)/$(basename "${MODEL_PATH}")"
OUTPUT_FOLDER="$(mkdir -p "${OUTPUT_FOLDER}" && cd "${OUTPUT_FOLDER}" && pwd -P)"

stage "Run external H5 prediction"
docker run --rm --platform "${IMAGE_PLATFORM}" \
  --mount "type=bind,src=$(dirname "${CONFIG}"),dst=/opt/aramina-user-config,readonly" \
  --mount "type=bind,src=$(dirname "${INPUT_H5}"),dst=/opt/aramina-user-input,readonly" \
  --mount "type=bind,src=$(dirname "${MODEL_PATH}"),dst=/opt/aramina-user-model,readonly" \
  --mount "type=bind,src=${OUTPUT_FOLDER},dst=/opt/aramina-user-output" \
  "${IMAGE_TAG}" \
  bash /opt/aramina-bundle/run_prediction_docker.sh \
    --config "/opt/aramina-user-config/$(basename "${CONFIG}")" \
    --input-h5 "/opt/aramina-user-input/$(basename "${INPUT_H5}")" \
    --model "/opt/aramina-user-model/$(basename "${MODEL_PATH}")" \
    --output-folder /opt/aramina-user-output

printf 'Reports: %s\n' "${OUTPUT_FOLDER}"
printf 'Log saved to: %s\n' "${LOG_PATH}"
