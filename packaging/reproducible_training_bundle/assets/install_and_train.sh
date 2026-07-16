#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${BUNDLE_DIR}/bundle_manifest.json"
DATA_DIR="${BUNDLE_DIR}/data"
OUTPUT_DIR="${BUNDLE_DIR}/outputs"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/install_and_train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

DEFAULT_PREPROCESS_TRAIN_CONFIG="config/preprocess_train/aramis_biopsy_patients_primary_preprocess_train_v0_1.yaml"
PREPROCESS_TRAIN_CONFIG="${DEFAULT_PREPROCESS_TRAIN_CONFIG}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preprocess-train-config)
      PREPROCESS_TRAIN_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "${PREPROCESS_TRAIN_CONFIG}" in
  config/preprocess_train/*.yaml|config/preprocess_train/*.yml) ;;
  *) echo "Preprocess-train config must be under config/preprocess_train/: ${PREPROCESS_TRAIN_CONFIG}" >&2; exit 2 ;;
esac
[[ -f "${BUNDLE_DIR}/${PREPROCESS_TRAIN_CONFIG}" ]] || {
  echo "Missing preprocess-train config: ${BUNDLE_DIR}/${PREPROCESS_TRAIN_CONFIG}" >&2
  exit 2
}
CONTAINER_PREPROCESS_TRAIN_CONFIG="/opt/Aramis/${PREPROCESS_TRAIN_CONFIG}"

read_manifest() {
  python3 - "${MANIFEST}" "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)[sys.argv[2]])
PY
}

stage() { printf '\n=== %s ===\n' "$1"; }

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

case "$(uname -m)" in
  x86_64|amd64) IMAGE_ARCH="amd64" ;;
  arm64|aarch64) IMAGE_ARCH="arm64" ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

IMAGE_TAG="$(read_manifest "image_${IMAGE_ARCH}_tag")"
IMAGE_PLATFORM="$(read_manifest "image_${IMAGE_ARCH}_platform")"
IMAGE_ARCHIVE="${BUNDLE_DIR}/$(read_manifest "image_${IMAGE_ARCH}_archive")"
EXPECTED_H5="$(read_manifest h5_sha256)"
EXPECTED_IMAGE="$(read_manifest "image_${IMAGE_ARCH}_archive_sha256")"

command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
docker version >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }

stage "Verify bundled H5"
ACTUAL_H5="$(sha256_file "${DATA_DIR}/combined_archive.h5")"
[[ "${ACTUAL_H5}" == "${EXPECTED_H5}" ]] || { echo "H5 SHA256 mismatch." >&2; exit 1; }

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  stage "Load validated Linux runtime image"
  ACTUAL_IMAGE="$(sha256_file "${IMAGE_ARCHIVE}")"
  [[ "${ACTUAL_IMAGE}" == "${EXPECTED_IMAGE}" ]] || { echo "Docker image SHA256 mismatch." >&2; exit 1; }
  docker load --input "${IMAGE_ARCHIVE}"
fi

stage "Run Linux preprocessing and training"
mkdir -p "${OUTPUT_DIR}"
docker run --rm --platform "${IMAGE_PLATFORM}" \
  --mount "type=bind,src=${DATA_DIR},dst=/opt/data,readonly" \
  --mount "type=bind,src=${BUNDLE_DIR}/config,dst=/opt/Aramis/config,readonly" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/opt/Aramis/examples/outputs" \
  "${IMAGE_TAG}" \
  bash /opt/aramis-bundle/run_training_docker.sh --preprocess-train-config "${CONTAINER_PREPROCESS_TRAIN_CONFIG}"

printf 'Log saved to: %s\n' "${LOG_PATH}"
