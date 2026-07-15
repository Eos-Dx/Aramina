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

read_manifest() {
  python3 - "${MANIFEST}" "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)[sys.argv[2]])
PY
}

stage() { printf '\n=== %s ===\n' "$1"; }

IMAGE_TAG="$(read_manifest image_tag)"
IMAGE_ARCHIVE="${BUNDLE_DIR}/$(read_manifest image_archive)"
EXPECTED_H5="$(read_manifest h5_sha256)"
EXPECTED_IMAGE="$(read_manifest image_archive_sha256)"

command -v docker >/dev/null || { echo "Docker is required." >&2; exit 1; }
docker version >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }

stage "Verify bundled H5"
ACTUAL_H5="$(sha256sum "${DATA_DIR}/combined_archive.h5" | awk '{print $1}')"
[[ "${ACTUAL_H5}" == "${EXPECTED_H5}" ]] || { echo "H5 SHA256 mismatch." >&2; exit 1; }

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  stage "Load validated Linux runtime image"
  ACTUAL_IMAGE="$(sha256sum "${IMAGE_ARCHIVE}" | awk '{print $1}')"
  [[ "${ACTUAL_IMAGE}" == "${EXPECTED_IMAGE}" ]] || { echo "Docker image SHA256 mismatch." >&2; exit 1; }
  docker load --input "${IMAGE_ARCHIVE}"
fi

stage "Run Linux preprocessing and training"
mkdir -p "${OUTPUT_DIR}"
docker run --rm \
  --mount "type=bind,src=${DATA_DIR},dst=/opt/data,readonly" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/opt/Aramis/examples/outputs" \
  "${IMAGE_TAG}" \
  bash /opt/aramis-bundle/run_training_docker.sh

printf 'Log saved to: %s\n' "${LOG_PATH}"
