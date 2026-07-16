#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${BUNDLE_DIR}/bundle_manifest.json"
CONFIG_DIR="${BUNDLE_DIR}/config"
EXAMPLE_H5_DIR="${BUNDLE_DIR}/examples/prediction_h5"
OUTPUT_DIR="${BUNDLE_DIR}/outputs"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/predict_examples_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1

MODEL_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

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

find_latest_model() {
  python3 - "${OUTPUT_DIR}/preprocess_train" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
models = list(root.rglob("model.joblib")) if root.exists() else []
if not models:
    raise SystemExit(1)
print(max(models, key=lambda path: path.stat().st_mtime))
PY
}

case "$(uname -m)" in
  x86_64|amd64) IMAGE_ARCH="amd64" ;;
  arm64|aarch64) IMAGE_ARCH="arm64" ;;
  *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

IMAGE_TAG="$(read_manifest "image_${IMAGE_ARCH}_tag")"
IMAGE_PLATFORM="$(read_manifest "image_${IMAGE_ARCH}_platform")"
IMAGE_ARCHIVE="${BUNDLE_DIR}/$(read_manifest "image_${IMAGE_ARCH}_archive")"
EXPECTED_IMAGE="$(read_manifest "image_${IMAGE_ARCH}_archive_sha256")"

command -v docker >/dev/null || {
  echo "Docker is required. Run install_and_train first to install and start Docker Desktop." >&2
  exit 1
}
docker version >/dev/null || {
  echo "Docker Linux engine is not running. Start Docker Desktop, then rerun this script." >&2
  exit 1
}

if [[ -z "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(find_latest_model)" || {
    echo "No trained model found under outputs/preprocess_train/. Run install_and_train first or pass --model." >&2
    exit 1
  }
fi
MODEL_PATH="$(cd "$(dirname "${MODEL_PATH}")" && pwd -P)/$(basename "${MODEL_PATH}")"
OUTPUT_ROOT="$(cd "${OUTPUT_DIR}" && pwd -P)"
case "${MODEL_PATH}" in
  "${OUTPUT_ROOT}"/*) ;;
  *) echo "Model must be inside bundle outputs/: ${MODEL_PATH}" >&2; exit 2 ;;
esac
CONTAINER_MODEL="/opt/Aramis/examples/outputs/${MODEL_PATH#"${OUTPUT_ROOT}"/}"

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  stage "Load validated Linux runtime image"
  ACTUAL_IMAGE="$(sha256_file "${IMAGE_ARCHIVE}")"
  [[ "${ACTUAL_IMAGE}" == "${EXPECTED_IMAGE}" ]] || { echo "Docker image SHA256 mismatch." >&2; exit 1; }
  docker load --input "${IMAGE_ARCHIVE}"
fi

stage "Run three prediction fixtures"
docker run --rm --platform "${IMAGE_PLATFORM}" \
  --mount "type=bind,src=${CONFIG_DIR},dst=/opt/Aramis/config,readonly" \
  --mount "type=bind,src=${EXAMPLE_H5_DIR},dst=/opt/Aramis/examples/prediction_h5,readonly" \
  --mount "type=bind,src=${OUTPUT_DIR},dst=/opt/Aramis/examples/outputs" \
  "${IMAGE_TAG}" \
  bash /opt/aramis-bundle/run_prediction_examples_docker.sh --model "${CONTAINER_MODEL}"

printf 'Reports: %s\n' "${OUTPUT_DIR}/prediction_examples"
printf 'Log saved to: %s\n' "${LOG_PATH}"
