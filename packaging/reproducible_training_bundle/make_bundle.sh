#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMINA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XRD_ROOT="${XRD_ROOT:-${ARAMINA_ROOT}/../XRD-preprocessing}"
EXPECTED_XRD_COMMIT="18ddac4be429e612ac82f8e81605d98399acee02"
SOURCE_H5="${SOURCE_H5:-${ARAMINA_ROOT}/../eos_play/jupyter_notebooks/Clinical_trials/data/product-aramina-data/combined_archive.h5}"
DIST_DIR="${DIST_DIR:-${ARAMINA_ROOT}/dist}"
BUNDLE_NAME="aramina_docker_training_bundle_0_2_12_beta"
AMD64_IMAGE_TAG="eosdx/aramina-training:0.2.12-beta-amd64"
AMD64_IMAGE_ARCHIVE="aramina_training_linux_amd64_0_2_12_beta.tar"
ARM64_IMAGE_TAG="eosdx/aramina-training:0.2.12-beta-arm64"
ARM64_IMAGE_ARCHIVE="aramina_training_linux_arm64_0_2_12_beta.tar"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
BUILD_CONTEXT="$(mktemp -d)"

cleanup() { rm -rf "${BUILD_CONTEXT}"; }
trap cleanup EXIT

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }
[[ -f "${SOURCE_H5}" ]] || { echo "Missing source H5: ${SOURCE_H5}" >&2; exit 1; }
[[ -d "${XRD_ROOT}/.git" ]] || { echo "Missing XRD-preprocessing checkout: ${XRD_ROOT}" >&2; exit 1; }

ARAMINA_COMMIT="$(git -C "${ARAMINA_ROOT}" rev-parse HEAD)"
XRD_COMMIT="$(git -C "${XRD_ROOT}" rev-parse HEAD)"
[[ "${XRD_COMMIT}" == "${EXPECTED_XRD_COMMIT}" ]] || {
  echo "XRD-preprocessing must be checked out at ${EXPECTED_XRD_COMMIT}; got ${XRD_COMMIT}." >&2
  exit 1
}
[[ -z "$(git -C "${XRD_ROOT}" status --porcelain)" ]] || {
  echo "XRD-preprocessing checkout must be clean before bundle creation." >&2
  exit 1
}

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p \
  "${WORK_DIR}/data" \
  "${WORK_DIR}/config/preprocessing_and_training" \
  "${WORK_DIR}/examples/prediction/configs" \
  "${WORK_DIR}/config/training" \
  "${WORK_DIR}/config/preprocessing/exclusions" \
  "${WORK_DIR}/config/preprocessing/schema" \
  "${WORK_DIR}/config/preprocessing/shared" \
  "${WORK_DIR}/examples/prediction_h5" \
  "${DIST_DIR}"
cp "${SOURCE_H5}" "${WORK_DIR}/data/combined_archive.h5"
cp "${ARAMINA_ROOT}/config/preprocessing_and_training/config_preprocess_and_train_target_breast_risk_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing_and_training/"
cp "${ARAMINA_ROOT}/config/training/config_training_target_breast_risk_v0_1.yaml" \
  "${WORK_DIR}/config/training/"
cp "${ARAMINA_ROOT}/examples/prediction/configs/"config_predict_*_example.yaml \
  "${WORK_DIR}/examples/prediction/configs/"
cp "${ARAMINA_ROOT}/examples/prediction_h5/"*_one_patient.h5 \
  "${WORK_DIR}/examples/prediction_h5/"
cp "${ARAMINA_ROOT}/config/preprocessing/config_preprocessing_biopsy_patients_v0_2.yaml" \
  "${WORK_DIR}/config/preprocessing/"
cp "${ARAMINA_ROOT}/config/preprocessing/config_preprocessing_prediction_patient_v0_2.yaml" \
  "${WORK_DIR}/config/preprocessing/"
cp "${ARAMINA_ROOT}/config/preprocessing/exclusions/agbh_quality_exclusions_t100_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/exclusions/"
cp "${ARAMINA_ROOT}/config/preprocessing/schema/model_input_columns_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/schema/"
cp "${ARAMINA_ROOT}/config/preprocessing/schema/prediction_input_columns_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/schema/"
cp "${ARAMINA_ROOT}/config/preprocessing/shared/aramina_pipeline_v0_1.yaml" \
  "${WORK_DIR}/config/preprocessing/shared/"
sed \
  's#input_h5_path: ./data/combined_archive.h5#input_h5_path: /opt/data/combined_archive.h5#' \
  "${ARAMINA_ROOT}/config/preprocessing/shared/aramina_policy_v0_1.yaml" \
  > "${WORK_DIR}/config/preprocessing/shared/aramina_policy_v0_1.yaml"
cp "${SCRIPT_DIR}/assets/install_and_train.bat" "${WORK_DIR}/install_and_train.bat"
cp "${SCRIPT_DIR}/assets/install_and_train.ps1" "${WORK_DIR}/install_and_train.ps1"
cp "${SCRIPT_DIR}/assets/install_and_train.sh" "${WORK_DIR}/install_and_train.sh"
cp "${SCRIPT_DIR}/assets/predict_examples.bat" "${WORK_DIR}/predict_examples.bat"
cp "${SCRIPT_DIR}/assets/predict_examples.ps1" "${WORK_DIR}/predict_examples.ps1"
cp "${SCRIPT_DIR}/assets/predict_examples.sh" "${WORK_DIR}/predict_examples.sh"
cp "${SCRIPT_DIR}/assets/predict.bat" "${WORK_DIR}/predict.bat"
cp "${SCRIPT_DIR}/assets/predict.ps1" "${WORK_DIR}/predict.ps1"
cp "${SCRIPT_DIR}/assets/predict.sh" "${WORK_DIR}/predict.sh"
cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
chmod +x "${WORK_DIR}/install_and_train.sh"
chmod +x "${WORK_DIR}/predict_examples.sh"
chmod +x "${WORK_DIR}/predict.sh"

rsync -a \
  --exclude '.git' \
  --exclude 'dist' \
  --exclude 'examples/outputs' \
  --exclude 'demo' \
  --exclude 'demo_outputs' \
  --exclude 'data' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "${ARAMINA_ROOT}/" "${BUILD_CONTEXT}/Aramina/"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  "${XRD_ROOT}/" "${BUILD_CONTEXT}/XRD-preprocessing/"
cp "${SCRIPT_DIR}/assets/Dockerfile" "${BUILD_CONTEXT}/Dockerfile"
cp "${SCRIPT_DIR}/assets/run_training_docker.sh" "${BUILD_CONTEXT}/run_training_docker.sh"
cp "${SCRIPT_DIR}/assets/run_prediction_examples_docker.sh" "${BUILD_CONTEXT}/run_prediction_examples_docker.sh"
cp "${SCRIPT_DIR}/assets/run_prediction_docker.sh" "${BUILD_CONTEXT}/run_prediction_docker.sh"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${AMD64_IMAGE_TAG}" \
  --build-arg "XRD_PREPROCESSING_GIT_COMMIT=${XRD_COMMIT}" \
  --build-arg "XRD_PREPROCESSING_REQUESTED_REVISION=${XRD_COMMIT}" \
  "${BUILD_CONTEXT}"
docker save --output "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}"
docker buildx build \
  --platform linux/arm64 \
  --load \
  --tag "${ARM64_IMAGE_TAG}" \
  --build-arg "XRD_PREPROCESSING_GIT_COMMIT=${XRD_COMMIT}" \
  --build-arg "XRD_PREPROCESSING_REQUESTED_REVISION=${XRD_COMMIT}" \
  "${BUILD_CONTEXT}"
docker save --output "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" "${ARM64_IMAGE_TAG}"

python - "${WORK_DIR}/bundle_manifest.json" "${ARAMINA_COMMIT}" "${XRD_COMMIT}" "${SOURCE_H5}" "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" <<'PY'
from hashlib import sha256
import json
import sys

path, aramina_commit, xrd_commit, h5_path, amd64_image_path, arm64_image_path = sys.argv[1:]

def digest(source):
    result = sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

payload = {
    "contract": "aramina_docker_reproducible_training_bundle_v0_2",
    "aramina_commit": aramina_commit,
    "xrd_preprocessing_release_tag": "v0.1.8-beta",
    "xrd_preprocessing_commit": xrd_commit,
    "h5_sha256": digest(h5_path),
    "reference_model_id": "aramina_target_breast_risk_0_2_12-beta_9bb911189af6",
    "reference_model_version": "0.2.12-beta",
    "image_amd64_tag": "eosdx/aramina-training:0.2.12-beta-amd64",
    "image_amd64_platform": "linux/amd64",
    "image_amd64_archive": "aramina_training_linux_amd64_0_2_12_beta.tar",
    "image_amd64_archive_sha256": digest(amd64_image_path),
    "image_arm64_tag": "eosdx/aramina-training:0.2.12-beta-arm64",
    "image_arm64_platform": "linux/arm64",
    "image_arm64_archive": "aramina_training_linux_arm64_0_2_12_beta.tar",
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
