#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${ARAMINA_API_PORT:-8000}"

case "$(uname -m)" in
  arm64|aarch64)
    IMAGE_TAG="eosdx/aramina-prediction-api:0.2.12-beta-arm64"
    IMAGE_ARCHIVE="${SCRIPT_DIR}/aramina_prediction_api_linux_arm64_0_2_12_beta.tar"
    ;;
  x86_64|amd64)
    IMAGE_TAG="eosdx/aramina-prediction-api:0.2.12-beta-amd64"
    IMAGE_ARCHIVE="${SCRIPT_DIR}/aramina_prediction_api_linux_amd64_0_2_12_beta.tar"
    ;;
  *)
    echo "Unsupported CPU architecture: $(uname -m)" >&2
    exit 1
    ;;
esac

command -v docker >/dev/null || { echo "Docker Desktop is required." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  docker load --input "${IMAGE_ARCHIVE}"
fi

docker rm --force aramina-prediction-api >/dev/null 2>&1 || true
docker run --detach \
  --name aramina-prediction-api \
  --publish "${PORT}:8000" \
  "${IMAGE_TAG}" >/dev/null

echo "Aramina prediction API: http://127.0.0.1:${PORT}"
echo "OpenAPI documentation: http://127.0.0.1:${PORT}/docs"
