#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XRD_ROOT="${XRD_ROOT:-${ARAMIS_ROOT}/../XRD-preprocessing}"
SOURCE_H5="${SOURCE_H5:-${ARAMIS_ROOT}/data/combined_archive.h5}"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
BUNDLE_NAME="aramis_docker_training_bundle_0_2_20_beta"
AMD64_IMAGE_TAG="eosdx/aramis-training:0.2.20-beta-amd64"
AMD64_IMAGE_ARCHIVE="aramis_training_linux_amd64_0_2_20_beta.tar"
ARM64_IMAGE_TAG="eosdx/aramis-training:0.2.20-beta-arm64"
ARM64_IMAGE_ARCHIVE="aramis_training_linux_arm64_0_2_20_beta.tar"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
BUILD_CONTEXT="$(mktemp -d)"

cleanup() { rm -rf "${BUILD_CONTEXT}"; }
trap cleanup EXIT

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }
[[ -f "${SOURCE_H5}" ]] || { echo "Missing source H5: ${SOURCE_H5}" >&2; exit 1; }
[[ -d "${XRD_ROOT}/.git" ]] || { echo "Missing XRD-preprocessing checkout: ${XRD_ROOT}" >&2; exit 1; }

ARAMIS_COMMIT="$(git -C "${ARAMIS_ROOT}" rev-parse HEAD)"
XRD_COMMIT="$(git -C "${XRD_ROOT}" rev-parse HEAD)"

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p \
  "${WORK_DIR}/data" \
  "${WORK_DIR}/config/preprocessing_and_training" \
  "${WORK_DIR}/config/prediction_examples" \
  "${WORK_DIR}/config/training" \
  "${WORK_DIR}/config/preprocessing/exclusions" \
  "${WORK_DIR}/config/preprocessing/outputs" \
  "${WORK_DIR}/config/preprocessing/shared" \
  "${WORK_DIR}/examples/prediction_h5" \
  "${DIST_DIR}"
cp "${SOURCE_H5}" "${WORK_DIR}/data/combined_archive.h5"
cp "${ARAMIS_ROOT}/config/preprocessing_and_training/aramis_biopsy_patients_primary_preprocessing_and_training_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing_and_training/"
cp "${ARAMIS_ROOT}/config/training/aramis_m2q_t100_primary_train_v0_1.yaml" \
  "${WORK_DIR}/config/training/"
cp "${ARAMIS_ROOT}/config/prediction_examples/"*_predict.yaml \
  "${WORK_DIR}/config/prediction_examples/"
cp "${ARAMIS_ROOT}/examples/prediction_h5/"*_one_patient.h5 \
  "${WORK_DIR}/examples/prediction_h5/"
cp "${ARAMIS_ROOT}/config/preprocessing/aramis_biopsy_patients_model_input_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/"
cp "${ARAMIS_ROOT}/config/preprocessing/aramis_prediction_patient_model_input_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/"
cp "${ARAMIS_ROOT}/config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/exclusions/"
cp "${ARAMIS_ROOT}/config/preprocessing/outputs/model_input_output_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/outputs/"
cp "${ARAMIS_ROOT}/config/preprocessing/outputs/prediction_model_input_output_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/outputs/"
cp "${ARAMIS_ROOT}/config/preprocessing/shared/aramis_pipeline_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/shared/"
sed \
  's#input_h5_path: ./data/combined_archive.h5#input_h5_path: /opt/data/combined_archive.h5#' \
  "${ARAMIS_ROOT}/config/preprocessing/shared/aramis_policy_v0_1.yaml" \
  > "${WORK_DIR}/config/preprocessing/shared/aramis_policy_v0_1.yaml"
cp "${SCRIPT_DIR}/assets/install_and_train.bat" "${WORK_DIR}/install_and_train.bat"
cp "${SCRIPT_DIR}/assets/install_and_train.ps1" "${WORK_DIR}/install_and_train.ps1"
cp "${SCRIPT_DIR}/assets/install_and_train.sh" "${WORK_DIR}/install_and_train.sh"
cp "${SCRIPT_DIR}/assets/predict_examples.bat" "${WORK_DIR}/predict_examples.bat"
cp "${SCRIPT_DIR}/assets/predict_examples.ps1" "${WORK_DIR}/predict_examples.ps1"
cp "${SCRIPT_DIR}/assets/predict_examples.sh" "${WORK_DIR}/predict_examples.sh"
cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
chmod +x "${WORK_DIR}/install_and_train.sh"
chmod +x "${WORK_DIR}/predict_examples.sh"

rsync -a \
  --exclude '.git' \
  --exclude 'dist' \
  --exclude 'examples/outputs' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "${ARAMIS_ROOT}/" "${BUILD_CONTEXT}/Aramis/"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "${XRD_ROOT}/" "${BUILD_CONTEXT}/XRD-preprocessing/"
cp "${SCRIPT_DIR}/assets/Dockerfile" "${BUILD_CONTEXT}/Dockerfile"
cp "${SCRIPT_DIR}/assets/run_training_docker.sh" "${BUILD_CONTEXT}/run_training_docker.sh"
cp "${SCRIPT_DIR}/assets/run_prediction_examples_docker.sh" "${BUILD_CONTEXT}/run_prediction_examples_docker.sh"

docker buildx build --platform linux/amd64 --load --tag "${AMD64_IMAGE_TAG}" "${BUILD_CONTEXT}"
docker save --output "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}"
docker buildx build --platform linux/arm64 --load --tag "${ARM64_IMAGE_TAG}" "${BUILD_CONTEXT}"
docker save --output "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" "${ARM64_IMAGE_TAG}"

python - "${WORK_DIR}/bundle_manifest.json" "${ARAMIS_COMMIT}" "${XRD_COMMIT}" "${SOURCE_H5}" "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" <<'PY'
from hashlib import sha256
import json
import sys

path, aramis_commit, xrd_commit, h5_path, amd64_image_path, arm64_image_path = sys.argv[1:]

def digest(source):
    result = sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

payload = {
    "contract": "aramis_docker_reproducible_training_bundle_v0_1",
    "aramis_commit": aramis_commit,
    "xrd_preprocessing_commit": xrd_commit,
    "h5_sha256": digest(h5_path),
    "image_amd64_tag": "eosdx/aramis-training:0.2.20-beta-amd64",
    "image_amd64_platform": "linux/amd64",
    "image_amd64_archive": "aramis_training_linux_amd64_0_2_20_beta.tar",
    "image_amd64_archive_sha256": digest(amd64_image_path),
    "image_arm64_tag": "eosdx/aramis-training:0.2.20-beta-arm64",
    "image_arm64_platform": "linux/arm64",
    "image_arm64_archive": "aramis_training_linux_arm64_0_2_20_beta.tar",
    "image_arm64_archive_sha256": digest(arm64_image_path),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

(
  cd "${DIST_DIR}"
  zip -qry "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
