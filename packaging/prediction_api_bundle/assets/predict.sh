#!/usr/bin/env bash
set -euo pipefail

INPUT_H5=""
REQUEST_JSON=""
OUTPUT_JSON=""
API_URL="${ARAMIS_API_URL:-http://127.0.0.1:8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-h5) INPUT_H5="$2"; shift 2 ;;
    --request-json) REQUEST_JSON="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --api-url) API_URL="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -f "${INPUT_H5}" ]] || { echo "Missing --input-h5 file." >&2; exit 2; }
[[ -f "${REQUEST_JSON}" ]] || { echo "Missing --request-json file." >&2; exit 2; }
[[ -n "${OUTPUT_JSON}" ]] || { echo "Missing --output-json path." >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT_JSON}")"

curl --fail-with-body --silent --show-error \
  --request POST "${API_URL}/predict" \
  --form "input_h5=@${INPUT_H5};type=application/x-hdf5" \
  --form "request_json=<${REQUEST_JSON}" \
  --output "${OUTPUT_JSON}"

echo "Prediction response: ${OUTPUT_JSON}"
