#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMINA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XRD_ROOT="${XRD_ROOT:-${ARAMINA_ROOT}/../XRD-preprocessing}"
SOURCE_H5="${SOURCE_H5:-${ARAMINA_ROOT}/../eos_play/jupyter_notebooks/Clinical_trials/data/product-aramina-data/combined_archive.h5}"
DIST_DIR="${DIST_DIR:-${ARAMINA_ROOT}/dist}"
MODEL_VERSION="0.2.12-beta"
BUILD_CONTEXT="$(mktemp -d)"

cleanup() { rm -rf "${BUILD_CONTEXT}"; }
trap cleanup EXIT

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }
[[ -f "${SOURCE_H5}" ]] || { echo "Missing source H5: ${SOURCE_H5}" >&2; exit 1; }
[[ -e "${XRD_ROOT}/.git" ]] || { echo "Missing XRD-preprocessing checkout: ${XRD_ROOT}" >&2; exit 1; }

ARAMINA_COMMIT="$(git -C "${ARAMINA_ROOT}" rev-parse HEAD)"
XRD_COMMIT="$(git -C "${XRD_ROOT}" rev-parse HEAD)"
[[ "${ARAMINA_COMMIT}" == "$(git -C "${ARAMINA_ROOT}" rev-parse main)" ]] || {
  echo "Aramina must be checked out at its local main commit." >&2
  exit 1
}
[[ "${XRD_COMMIT}" == "$(git -C "${XRD_ROOT}" rev-parse main)" ]] || {
  echo "XRD-preprocessing must be checked out at its local main commit." >&2
  exit 1
}
[[ -z "$(git -C "${ARAMINA_ROOT}" status --porcelain)" ]] || {
  echo "Aramina checkout must be clean before bundle creation." >&2
  exit 1
}
[[ -z "$(git -C "${XRD_ROOT}" status --porcelain)" ]] || {
  echo "XRD-preprocessing checkout must be clean before bundle creation." >&2
  exit 1
}

ARAMINA_SHORT="${ARAMINA_COMMIT:0:12}"
XRD_SHORT="${XRD_COMMIT:0:12}"
BUNDLE_NAME="aramina_reproducible_runtime_bundle_0_2_12_beta_${ARAMINA_SHORT}_${XRD_SHORT}"
AMD64_IMAGE_TAG="eosdx/aramina-training:${MODEL_VERSION}-${ARAMINA_SHORT}-amd64"
AMD64_IMAGE_ARCHIVE="aramina_training_linux_amd64.tar"
ARM64_IMAGE_TAG="eosdx/aramina-training:${MODEL_VERSION}-${ARAMINA_SHORT}-arm64"
ARM64_IMAGE_ARCHIVE="aramina_training_linux_arm64.tar"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"

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
  "${WORK_DIR}/source/Aramina" \
  "${WORK_DIR}/source/XRD-preprocessing" \
  "${DIST_DIR}"
cp "${SOURCE_H5}" "${WORK_DIR}/data/combined_archive.h5"
git -C "${ARAMINA_ROOT}" archive "${ARAMINA_COMMIT}" | tar -x -C "${WORK_DIR}/source/Aramina"
git -C "${XRD_ROOT}" archive "${XRD_COMMIT}" | tar -x -C "${WORK_DIR}/source/XRD-preprocessing"
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

cp -a "${WORK_DIR}/source/Aramina" "${BUILD_CONTEXT}/Aramina"
cp -a "${WORK_DIR}/source/XRD-preprocessing" "${BUILD_CONTEXT}/XRD-preprocessing"
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

python - "${WORK_DIR}/bundle_manifest.json" "${ARAMINA_COMMIT}" "${XRD_COMMIT}" "${SOURCE_H5}" "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}" "${ARM64_IMAGE_TAG}" <<'PY'
from hashlib import sha256
import json
import sys

(
    path,
    aramina_commit,
    xrd_commit,
    h5_path,
    amd64_image_path,
    arm64_image_path,
    amd64_image_tag,
    arm64_image_tag,
) = sys.argv[1:]

def digest(source):
    result = sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

payload = {
    "contract": "aramina_reproducible_runtime_bundle_v0_3",
    "source": {
        "aramina": {
            "branch": "main",
            "commit": aramina_commit,
            "remote": "https://github.com/Eos-Dx/Aramina.git",
            "snapshot": "source/Aramina",
        },
        "xrd_preprocessing": {
            "branch": "main",
            "commit": xrd_commit,
            "remote": "https://github.com/Eos-Dx/XRD-preprocessing.git",
            "snapshot": "source/XRD-preprocessing",
        },
    },
    "h5_sha256": digest(h5_path),
    "reference_model_id": "aramina_target_breast_risk_0_2_12-beta_9bb911189af6",
    "reference_model_version": "0.2.12-beta",
    "image_amd64_tag": amd64_image_tag,
    "image_amd64_platform": "linux/amd64",
    "image_amd64_archive": "aramina_training_linux_amd64_0_2_12_beta.tar",
    "image_amd64_archive_sha256": digest(amd64_image_path),
    "image_arm64_tag": arm64_image_tag,
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
