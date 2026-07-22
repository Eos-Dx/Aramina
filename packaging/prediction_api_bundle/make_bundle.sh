#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
BUNDLE_NAME="aramis_prediction_api_bundle_0_2_10_beta"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
AMD64_IMAGE_TAG="eosdx/aramis-prediction-api:0.2.10-beta-amd64"
ARM64_IMAGE_TAG="eosdx/aramis-prediction-api:0.2.10-beta-arm64"
AMD64_IMAGE_ARCHIVE="aramis_prediction_api_linux_amd64_0_2_10_beta.tar"
ARM64_IMAGE_ARCHIVE="aramis_prediction_api_linux_arm64_0_2_10_beta.tar"

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Linux engine is not running." >&2; exit 1; }

for path in \
  "${ARAMIS_ROOT}/demo/model_service/app.py" \
  "${ARAMIS_ROOT}/demo/model_service/Dockerfile" \
  "${ARAMIS_ROOT}/models/aramis_target_breast_risk_0_2_10-beta_ccad65e77adb/model.joblib"; do
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
cp "${ARAMIS_ROOT}/docs/contracts/prediction_config_v0_1.md" \
  "${WORK_DIR}/contracts/direct_cli_prediction_config_v0_1.md"
cp "${ARAMIS_ROOT}/docs/modeling/prediction_pipeline_v0_1.md" \
  "${WORK_DIR}/contracts/prediction_pipeline_v0_1.md"
cp "${ARAMIS_ROOT}/docs/modeling/internal_clinical_report_content_v0_5.md" \
  "${WORK_DIR}/contracts/internal_report_v0_5.md"

cp "${ARAMIS_ROOT}/examples/prediction_h5/"*_one_patient.h5 "${WORK_DIR}/examples/h5/"
cp "${ARAMIS_ROOT}/examples/prediction_h5/README.md" "${WORK_DIR}/examples/h5/README.md"
cp "${ARAMIS_ROOT}/config/prediction/config_predict_from_h5_template_v0_1.yaml" \
  "${WORK_DIR}/examples/direct_cli_config/config_predict_from_h5_template_v0_1.yaml"
cp "${SCRIPT_DIR}/assets/requests/"*.json "${WORK_DIR}/examples/requests/"
chmod +x "${WORK_DIR}/start_api.sh" "${WORK_DIR}/stop_api.sh" "${WORK_DIR}/predict.sh"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${AMD64_IMAGE_TAG}" \
  --file "${ARAMIS_ROOT}/demo/model_service/Dockerfile" \
  "${ARAMIS_ROOT}"
docker save --output "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}"

docker buildx build \
  --platform linux/arm64 \
  --load \
  --tag "${ARM64_IMAGE_TAG}" \
  --file "${ARAMIS_ROOT}/demo/model_service/Dockerfile" \
  "${ARAMIS_ROOT}"
docker save --output "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" "${ARM64_IMAGE_TAG}"

python - \
  "${WORK_DIR}/bundle_manifest.yaml" \
  "${ARAMIS_ROOT}" \
  "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" \
  "${WORK_DIR}/${ARM64_IMAGE_ARCHIVE}" <<'PY'
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

target, root, amd64_archive, arm64_archive = map(Path, sys.argv[1:])
model = root / "models/aramis_target_breast_risk_0_2_10-beta_ccad65e77adb/model.joblib"
service = root / "demo/model_service/app.py"
dockerfile = root / "demo/model_service/Dockerfile"

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
            "contract: aramis_prediction_api_bundle_v0_1",
            "model_name: aramis_target_breast_risk",
            "model_version: 0.2.10-beta",
            "api_contract: v0.1",
            "aramis_commit: " + commit,
            "model_joblib_sha256: " + digest(model),
            "model_service_app_sha256: " + digest(service),
            "model_service_dockerfile_sha256: " + digest(dockerfile),
            "images:",
            "  amd64:",
            "    tag: eosdx/aramis-prediction-api:0.2.10-beta-amd64",
            "    platform: linux/amd64",
            "    archive: " + amd64_archive.name,
            "    sha256: " + digest(amd64_archive),
            "  arm64:",
            "    tag: eosdx/aramis-prediction-api:0.2.10-beta-arm64",
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
