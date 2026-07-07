#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m aramis preprocess \
  --config config/preprocessing/aramis_all_patients_model_input_v0_1.yaml

python -m aramis preprocess \
  --config config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml
