#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEV_ROOT="$(cd "${ARAMIS_ROOT}/.." && pwd)"
XRD_ROOT="${XRD_ROOT:-${DEV_ROOT}/XRD-preprocessing}"
CONTAINER_ROOT="${CONTAINER_ROOT:-${DEV_ROOT}/container}"
DATA_H5="${DATA_H5:-${ARAMIS_ROOT}/tests/data/aramis_real_h5_subset_20260128_5_patients.h5}"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
BUNDLE_NAME="${BUNDLE_NAME:-eosproduct_onboarding_bundle}"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.tar.gz"

copy_repo() {
  local src="$1"
  local dst="$2"
  if [[ ! -d "${src}" ]]; then
    echo "Missing repo: ${src}" >&2
    exit 1
  fi
  mkdir -p "${dst}"
  rsync -a --delete \
    --exclude ".git" \
    --exclude ".idea" \
    --exclude ".DS_Store" \
    --exclude ".pytest_cache" \
    --exclude ".ruff_cache" \
    --exclude "__pycache__" \
    --exclude "__marimo__" \
    --exclude "*.pyc" \
    --exclude "*.egg-info" \
    --exclude "dist" \
    --exclude "reports" \
    --exclude "examples/outputs" \
    "${src}/" "${dst}/"
}

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p "${WORK_DIR}/repos" "${WORK_DIR}/data" "${WORK_DIR}/docs" "${DIST_DIR}"

copy_repo "${XRD_ROOT}" "${WORK_DIR}/repos/XRD-preprocessing"
copy_repo "${ARAMIS_ROOT}" "${WORK_DIR}/repos/Aramis"
copy_repo "${CONTAINER_ROOT}" "${WORK_DIR}/repos/container"
mkdir -p "${WORK_DIR}/repos/Bremen"
touch "${WORK_DIR}/repos/Bremen/.gitkeep"

if [[ -f "${DATA_H5}" ]]; then
  cp "${DATA_H5}" "${WORK_DIR}/data/combined_archive.h5"
else
  echo "Data H5 not found, bundle will contain empty data/: ${DATA_H5}" >&2
fi

cp "${SCRIPT_DIR}/scripts/install.sh" "${WORK_DIR}/install.sh"
cp "${SCRIPT_DIR}/scripts/install.ps1" "${WORK_DIR}/install.ps1"
cp "${SCRIPT_DIR}/scripts/run_tests.sh" "${WORK_DIR}/run_tests.sh"
cp "${SCRIPT_DIR}/scripts/run_tests.ps1" "${WORK_DIR}/run_tests.ps1"
cp "${SCRIPT_DIR}/scripts/run_aramis_notebooks.sh" "${WORK_DIR}/run_aramis_notebooks.sh"
cp "${SCRIPT_DIR}/scripts/run_aramis_notebooks.ps1" "${WORK_DIR}/run_aramis_notebooks.ps1"
cp "${SCRIPT_DIR}/scripts/run_aramis_prediction_examples.sh" "${WORK_DIR}/run_aramis_prediction_examples.sh"
cp "${SCRIPT_DIR}/scripts/run_aramis_prediction_examples.ps1" "${WORK_DIR}/run_aramis_prediction_examples.ps1"
cp "${SCRIPT_DIR}/environment.yml" "${WORK_DIR}/environment.yml"
cp "${SCRIPT_DIR}/docs/INSTALL.md" "${WORK_DIR}/docs/INSTALL.md"
cp "${SCRIPT_DIR}/docs/INSTALL.md" "${WORK_DIR}/README.md"
chmod +x \
  "${WORK_DIR}/install.sh" \
  "${WORK_DIR}/run_tests.sh" \
  "${WORK_DIR}/run_aramis_notebooks.sh" \
  "${WORK_DIR}/run_aramis_prediction_examples.sh"

(
  cd "${DIST_DIR}"
  tar -czf "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
