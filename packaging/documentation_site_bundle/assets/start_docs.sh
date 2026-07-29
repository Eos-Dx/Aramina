#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PATH="${SCRIPT_DIR}/site/index.html"
[[ -f "${SITE_PATH}" ]] || { echo "Missing documentation site: ${SITE_PATH}" >&2; exit 1; }

if command -v open >/dev/null 2>&1; then
  open "${SITE_PATH}"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${SITE_PATH}"
else
  printf 'Open this file in a browser: %s\n' "${SITE_PATH}"
fi
