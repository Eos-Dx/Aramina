#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-${HOME}/dev/eosproduct}"
ENV_NAME="${ENV_NAME:-eosproduct}"
ARAMIS_ROOT="${TARGET_ROOT}/Aramis"

cd "${ARAMIS_ROOT}"

for config in \
  examples/prediction_h5/benign_predict.yaml \
  examples/prediction_h5/cancer_predict.yaml \
  examples/prediction_h5/atypical_predict.yaml
do
  echo "--- ${config}"
  conda run --no-capture-output -n "${ENV_NAME}" python -m aramis predict --config "${config}"
done

echo "Prediction reports: ${ARAMIS_ROOT}/examples/outputs/prediction_h5_examples"
