#!/usr/bin/env bash
set -euo pipefail
docker rm --force aramis-prediction-api >/dev/null 2>&1 || true
echo "Aramis prediction API stopped."
