#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m aramis preprocess \
  --config config/preprocessing/aramis_one_to_one_max_v0_1.yaml

python -m aramis preprocess \
  --config config/preprocessing/aramis_one_to_one_biopsy_max_v0_1.yaml

python -m aramis preprocess \
  --config config/preprocessing/aramis_one_to_many_max_v0_1.yaml

python -m aramis preprocess \
  --config config/preprocessing/aramis_one_to_many_biopsy_max_v0_1.yaml
