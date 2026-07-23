#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_H5=""
OUTPUT_FOLDER="${SCRIPT_DIR}/outputs"
PORT="${ARAMIS_DEMO_PORT:-8501}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-h5) SOURCE_H5="$2"; shift 2 ;;
    --output-folder) OUTPUT_FOLDER="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -n "${SOURCE_H5}" ]] || { echo "--source-h5 is required." >&2; exit 2; }
[[ -f "${SOURCE_H5}" ]] || { echo "Missing source H5: ${SOURCE_H5}" >&2; exit 2; }
command -v docker >/dev/null || { echo "Docker Desktop is required." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

case "$(uname -m)" in
  arm64|aarch64)
    IMAGE_TAG="eosdx/aramis-demo:0.2.11-beta-arm64"
    IMAGE_ARCHIVE="${SCRIPT_DIR}/aramis_demo_linux_arm64_0_2_11_beta.tar"
    ;;
  x86_64|amd64)
    IMAGE_TAG="eosdx/aramis-demo:0.2.11-beta-amd64"
    IMAGE_ARCHIVE="${SCRIPT_DIR}/aramis_demo_linux_amd64_0_2_11_beta.tar"
    ;;
  *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  docker load --input "${IMAGE_ARCHIVE}"
fi

SOURCE_H5="$(cd "$(dirname "${SOURCE_H5}")" && pwd)/$(basename "${SOURCE_H5}")"
mkdir -p "${OUTPUT_FOLDER}"
OUTPUT_FOLDER="$(cd "${OUTPUT_FOLDER}" && pwd)"

docker rm --force aramis-demo >/dev/null 2>&1 || true
docker run --detach \
  --name aramis-demo \
  --publish "${PORT}:8501" \
  --env ARAMIS_DEMO_OUTPUT_ROOT=/opt/aramis-demo/static/reports \
  --volume "${SOURCE_H5}:/data/source_archive.h5:ro" \
  --volume "${OUTPUT_FOLDER}:/opt/aramis-demo/static/reports" \
  "${IMAGE_TAG}" >/dev/null

echo "Aramis browser demonstrator: http://127.0.0.1:${PORT}"
echo "Host report folder: ${OUTPUT_FOLDER}"
