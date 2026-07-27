#!/usr/bin/env bash
set -euo pipefail
docker rm --force aramina-prediction-api >/dev/null 2>&1 || true
echo "Aramina prediction API stopped."
