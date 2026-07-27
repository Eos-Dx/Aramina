#!/usr/bin/env bash
set -euo pipefail
docker rm --force aramina-demo aramina-demo-api >/dev/null 2>&1 || true
echo "Aramina browser demonstrator and local API stopped."
