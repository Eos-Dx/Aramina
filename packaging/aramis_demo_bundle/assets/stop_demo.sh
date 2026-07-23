#!/usr/bin/env bash
set -euo pipefail
docker rm --force aramis-demo >/dev/null 2>&1 || true
echo "Aramis browser demonstrator stopped."
