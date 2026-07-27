#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_H5=""
OUTPUT_FOLDER="${SCRIPT_DIR}/outputs"
PORT="${ARAMINA_DEMO_PORT:-8501}"
API_PORT="${ARAMINA_API_PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-h5) SOURCE_H5="$2"; shift 2 ;;
    --output-folder) OUTPUT_FOLDER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${SOURCE_H5}" ]] || { echo "--source-h5 is required." >&2; exit 2; }
[[ -f "${SOURCE_H5}" ]] || { echo "Missing source H5: ${SOURCE_H5}" >&2; exit 2; }
command -v docker >/dev/null || { echo "Docker Desktop is required." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

case "$(uname -m)" in
  arm64|aarch64)
    API_IMAGE="eosdx/aramina-prediction-api:0.2.12-beta-arm64"
    API_ARCHIVE="${SCRIPT_DIR}/aramina_prediction_api_linux_arm64_0_2_12_beta.tar"
    PLATFORM_IMAGE="eosdx/araminavisor-demo:0.2.12-beta-arm64"
    PLATFORM_ARCHIVE="${SCRIPT_DIR}/araminavisor_demo_linux_arm64_0_2_12_beta.tar"
    ;;
  x86_64|amd64)
    API_IMAGE="eosdx/aramina-prediction-api:0.2.12-beta-amd64"
    API_ARCHIVE="${SCRIPT_DIR}/aramina_prediction_api_linux_amd64_0_2_12_beta.tar"
    PLATFORM_IMAGE="eosdx/araminavisor-demo:0.2.12-beta-amd64"
    PLATFORM_ARCHIVE="${SCRIPT_DIR}/araminavisor_demo_linux_amd64_0_2_12_beta.tar"
    ;;
  *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

for archive in "${API_ARCHIVE}" "${PLATFORM_ARCHIVE}"; do
  [[ -f "${archive}" ]] || { echo "Missing image archive: ${archive}" >&2; exit 1; }
done
if ! docker image inspect "${API_IMAGE}" >/dev/null 2>&1; then docker load --input "${API_ARCHIVE}"; fi
if ! docker image inspect "${PLATFORM_IMAGE}" >/dev/null 2>&1; then docker load --input "${PLATFORM_ARCHIVE}"; fi

SOURCE_H5="$(cd "$(dirname "${SOURCE_H5}")" && pwd)/$(basename "${SOURCE_H5}")"
mkdir -p "${OUTPUT_FOLDER}"
OUTPUT_FOLDER="$(cd "${OUTPUT_FOLDER}" && pwd)"

docker network inspect aramina-demo-network >/dev/null 2>&1 || docker network create aramina-demo-network >/dev/null
docker rm --force aramina-demo aramina-demo-api >/dev/null 2>&1 || true
docker run --detach \
  --name aramina-demo-api \
  --network aramina-demo-network \
  --publish "${API_PORT}:8000" \
  "${API_IMAGE}" >/dev/null
docker run --detach \
  --name aramina-demo \
  --network aramina-demo-network \
  --publish "${PORT}:8501" \
  --env ARAMINA_PREDICTION_API_URL=http://aramina-demo-api:8000 \
  --env ARAMINA_DEMO_OUTPUT_ROOT=/opt/araminavisor/app/static/reports \
  --env ARAMINA_MODEL_TEST_ARTIFACT_PATH=/opt/araminavisor/static/model_test/aramina_mri_or_biopsy_held_out_t130.joblib \
  --env STREAMLIT_SERVER_ENABLE_STATIC_SERVING=true \
  --volume "${SOURCE_H5}:/data/source_archive.h5:ro" \
  --volume "${OUTPUT_FOLDER}:/opt/araminavisor/app/static/reports" \
  "${PLATFORM_IMAGE}" >/dev/null

echo "Aramina browser demonstrator: http://127.0.0.1:${PORT}"
echo "Aramina prediction API: http://127.0.0.1:${API_PORT}/docs"
echo "Host report folder: ${OUTPUT_FOLDER}"
