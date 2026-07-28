#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMINA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${DIST_DIR:-${ARAMINA_ROOT}/dist}"
BUNDLE_NAME="aramina_prediction_api_bundle_0_2_12_beta"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
MODEL_ID="aramina_target_breast_risk_0_2_12-beta_9bb911189af6"
MODEL_VERSION="0.2.12-beta"
XRD_RELEASE_TAG="v0.1.8-beta"
XRD_COMMIT="18ddac4be429e612ac82f8e81605d98399acee02"
AMD64_IMAGE_TAG="eosdx/aramina-prediction-api:${MODEL_VERSION}-amd64"
ARM64_IMAGE_TAG="eosdx/aramina-prediction-api:${MODEL_VERSION}-arm64"
AMD64_IMAGE_ARCHIVE="aramina_prediction_api_linux_amd64_0_2_12_beta.tar"
ARM64_IMAGE_ARCHIVE="aramina_prediction_api_linux_arm64_0_2_12_beta.tar"

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }

for path in \
  "${ARAMINA_ROOT}/src/aramina/prediction_api.py" \
  "${SCRIPT_DIR}/Dockerfile" \
  "${ARAMINA_ROOT}/models/${MODEL_ID}/model.joblib"; do
  [[ -f "${path}" ]] || { echo "Missing API bundle input: ${path}" >&2; exit 1; }
done

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p \
  "${WORK_DIR}/contracts" \
  "${WORK_DIR}/examples/h5" \
  "${WORK_DIR}/examples/requests" \
  "${WORK_DIR}/examples/direct_cli_config" \
  "${DIST_DIR}"

cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
cp "${SCRIPT_DIR}/assets/start_api.sh" "${WORK_DIR}/start_api.sh"
cp "${SCRIPT_DIR}/assets/start_api.ps1" "${WORK_DIR}/start_api.ps1"
cp "${SCRIPT_DIR}/assets/stop_api.sh" "${WORK_DIR}/stop_api.sh"
cp "${SCRIPT_DIR}/assets/predict.sh" "${WORK_DIR}/predict.sh"
cp "${SCRIPT_DIR}/assets/predict.ps1" "${WORK_DIR}/predict.ps1"
cp "${SCRIPT_DIR}/assets/API_CONTRACT.md" "${WORK_DIR}/contracts/API_CONTRACT.md"
cp "${SCRIPT_DIR}/assets/openapi.yaml" "${WORK_DIR}/contracts/openapi.yaml"
cp "${ARAMINA_ROOT}/docs/contracts/prediction_config_v0_1.md" \
  "${WORK_DIR}/contracts/direct_cli_prediction_config_v0_1.md"
cp "${ARAMINA_ROOT}/docs/modeling/prediction_pipeline_v0_1.md" \
  "${WORK_DIR}/contracts/prediction_pipeline_v0_1.md"
cp "${ARAMINA_ROOT}/docs/modeling/internal_clinical_report_content_v0_9.md" \
  "${WORK_DIR}/contracts/internal_report_v0_9.md"

cp "${ARAMINA_ROOT}/examples/prediction_h5/"*_one_patient.h5 "${WORK_DIR}/examples/h5/"
cp "${ARAMINA_ROOT}/examples/prediction_h5/README.md" "${WORK_DIR}/examples/h5/README.md"
cp "${ARAMINA_ROOT}/config/prediction/config_predict_from_h5_template_v0_1.yaml" \
  "${WORK_DIR}/examples/direct_cli_config/config_predict_from_h5_template_v0_1.yaml"
cp "${SCRIPT_DIR}/assets/requests/"*.json "${WORK_DIR}/examples/requests/"
chmod +x "${WORK_DIR}/start_api.sh" "${WORK_DIR}/stop_api.sh" "${WORK_DIR}/predict.sh"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${AMD64_IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "MODEL_ID=${MODEL_ID}" \
  "${ARAMINA_ROOT}"
docker save --output "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}"

docker buildx build \
  --platform linux/arm64 \
  --load \
  --tag "${ARM64_IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --build-arg "MODEL_ID=${MODEL_ID}" \
  "${ARAMINA_ROOT}"
docker save --output "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" "${ARM64_IMAGE_TAG}"

python - \
  "${WORK_DIR}/bundle_manifest.yaml" \
  "${ARAMINA_ROOT}" \
  "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" \
  "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" \
  "${MODEL_ID}" "${MODEL_VERSION}" "${XRD_RELEASE_TAG}" "${XRD_COMMIT}" <<'PY'
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

(
    target,
    root,
    amd64_archive,
    arm64_archive,
    model_id,
    model_version,
    xrd_release_tag,
    xrd_commit,
) = sys.argv[1:]
target, root, amd64_archive, arm64_archive = map(Path, (target, root, amd64_archive, arm64_archive))
model = root / "models" / model_id / "model.joblib"
service = root / "src/aramina/prediction_api.py"
dockerfile = root / "packaging/prediction_api_bundle/Dockerfile"

def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

commit = subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
).strip()

target.write_text(
    "\n".join(
        [
            "contract: aramina_prediction_api_bundle_v0_2",
            "model_name: aramina_target_breast_risk",
            "model_version: " + model_version,
            "api_contract: v0.1",
            "aramina_commit: " + commit,
            "xrd_preprocessing_release_tag: " + xrd_release_tag,
            "xrd_preprocessing_commit: " + xrd_commit,
            "model_joblib_sha256: " + digest(model),
            "model_service_app_sha256: " + digest(service),
            "model_service_dockerfile_sha256: " + digest(dockerfile),
            "images:",
            "  amd64:",
            "    tag: eosdx/aramina-prediction-api:" + model_version + "-amd64",
            "    platform: linux/amd64",
            "    archive: " + amd64_archive.name,
            "    sha256: " + digest(amd64_archive),
            "  arm64:",
            "    tag: eosdx/aramina-prediction-api:" + model_version + "-arm64",
            "    platform: linux/arm64",
            "    archive: " + arm64_archive.name,
            "    sha256: " + digest(arm64_archive),
            "",
        ]
    ),
    encoding="utf-8",
)
PY

(
  cd "${DIST_DIR}"
  zip -qry "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
