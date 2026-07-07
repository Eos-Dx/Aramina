#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-${HOME}/dev/eosproduct}"
SCOPE="${2:-all}"
ENV_NAME="${ENV_NAME:-eosproduct}"

run_xrd() {
  echo "Testing XRD-preprocessing"
  (cd "${TARGET_ROOT}/XRD-preprocessing" && conda run -n "${ENV_NAME}" python -m ruff check .)
  (cd "${TARGET_ROOT}/XRD-preprocessing" && conda run -n "${ENV_NAME}" pytest -q)
}

run_aramis() {
  echo "Testing Aramis"
  (cd "${TARGET_ROOT}/Aramis" && conda run -n "${ENV_NAME}" python -m ruff check .)
  (cd "${TARGET_ROOT}/Aramis" && conda run -n "${ENV_NAME}" pytest -q)
  (cd "${TARGET_ROOT}/Aramis" && conda run -n "${ENV_NAME}" python -m marimo check examples/aramis_dataframe_all_patients_v0_1.py)
  (cd "${TARGET_ROOT}/Aramis" && conda run -n "${ENV_NAME}" python -m marimo check examples/aramis_dataframe_biopsy_patients_v0_1.py)
}

case "${SCOPE}" in
  xrd) run_xrd ;;
  aramis) run_aramis ;;
  all) run_xrd; run_aramis ;;
  *) echo "Usage: $0 [target_root] [all|xrd|aramis]" >&2; exit 2 ;;
esac
