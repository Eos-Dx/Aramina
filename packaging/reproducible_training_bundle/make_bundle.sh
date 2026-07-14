#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_H5="${SOURCE_H5:-${ARAMIS_ROOT}/../eos_play/jupyter_notebooks/Clinical_trials/data/product-aramis-data/combined_archive.h5}"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
BUNDLE_NAME="aramis_reproducible_training_bundle_0_2_6_beta"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
XRD_COMMIT="4489183472e2d84c99b64ca8a6d42696710bd362"

if [[ ! -f "${SOURCE_H5}" ]]; then
  echo "Missing source H5: ${SOURCE_H5}" >&2
  exit 1
fi

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p "${WORK_DIR}/data" "${DIST_DIR}"
cp "${SOURCE_H5}" "${WORK_DIR}/data/combined_archive.h5"
cp "${SCRIPT_DIR}/assets/install_and_train.bat" "${WORK_DIR}/install_and_train.bat"
cp "${SCRIPT_DIR}/assets/install_and_train.ps1" "${WORK_DIR}/install_and_train.ps1"
cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
cp "${ARAMIS_ROOT}/environment.yml" "${WORK_DIR}/environment.yml"

python - "${WORK_DIR}/bundle_manifest.json" "$(git -C "${ARAMIS_ROOT}" rev-parse HEAD)" "${XRD_COMMIT}" <<'PY'
import json
import sys

path, aramis_commit, xrd_commit = sys.argv[1:]
payload = {
    "contract": "aramis_reproducible_training_bundle_v0_1",
    "environment_name": "aramis_repro_0_2_6",
    "aramis_repository": "https://github.com/Eos-Dx/Aramis.git",
    "aramis_commit": aramis_commit,
    "xrd_preprocessing_repository": "https://github.com/Eos-Dx/XRD-preprocessing.git",
    "xrd_preprocessing_commit": xrd_commit,
    "workflow_config": "config/workflows/aramis_biopsy_patients_primary_workflow_v0_1.yaml",
    "reference_model_relative_path": "examples/prediction_models/aramis_m2q_t100_0_2_6_beta.joblib",
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

(
  cd "${DIST_DIR}"
  zip -qry "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
