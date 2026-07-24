#!/usr/bin/env bash
set -euo pipefail
docker rm --force aramis-demo aramis-demo-api >/dev/null 2>&1 || true
echo "Aramis browser demonstrator and local API stopped."
