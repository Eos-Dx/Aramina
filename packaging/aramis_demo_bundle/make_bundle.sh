#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ARAMISVISOR_ROOT="${ARAMISVISOR_ROOT:-${ARAMIS_ROOT}/../Aramisvisor-demo}"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
MODEL_ID="aramis_target_breast_risk_0_2_12-beta_f8af641a2e49"
MODEL_VERSION="0.2.12-beta"
MODEL_TEST_DIR="${MODEL_TEST_DIR:-${ARAMISVISOR_ROOT}/model_test/artifacts}"
BUNDLE_NAME="aramis_demo_bundle_0_2_12_beta"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
API_AMD64_TAG="eosdx/aramis-prediction-api:${MODEL_VERSION}-amd64"
API_ARM64_TAG="eosdx/aramis-prediction-api:${MODEL_VERSION}-arm64"
PLATFORM_AMD64_TAG="eosdx/aramisvisor-demo:${MODEL_VERSION}-amd64"
PLATFORM_ARM64_TAG="eosdx/aramisvisor-demo:${MODEL_VERSION}-arm64"
BUILD_CONTEXT="$(mktemp -d)"

cleanup() { rm -rf "${BUILD_CONTEXT}"; }
trap cleanup EXIT

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

for path in \
  "${ARAMIS_ROOT}/models/${MODEL_ID}/model.joblib" \
  "${ARAMIS_ROOT}/packaging/prediction_api_bundle/Dockerfile" \
  "${ARAMISVISOR_ROOT}/app/Dockerfile" \
  "${ARAMISVISOR_ROOT}/app/streamlit_app.py" \
  "${ARAMISVISOR_ROOT}/app/pdf_report.py" \
  "${ARAMISVISOR_ROOT}/app/assets/training_patient_ids.json" \
  "${MODEL_TEST_DIR}/aramis_mri_or_biopsy_held_out_t130.joblib" \
  "${MODEL_TEST_DIR}/aramis_mri_or_biopsy_held_out_t130.manifest.json"; do
  [[ -f "${path}" ]] || { echo "Missing demo bundle input: ${path}" >&2; exit 1; }
done

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p \
  "${WORK_DIR}/contracts" \
  "${WORK_DIR}/examples/h5" \
  "${WORK_DIR}/examples/requests" \
  "${WORK_DIR}/examples/direct_cli_config" \
  "${DIST_DIR}"
cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
cp "${SCRIPT_DIR}/assets/start_demo.sh" "${WORK_DIR}/start_demo.sh"
cp "${SCRIPT_DIR}/assets/start_demo.ps1" "${WORK_DIR}/start_demo.ps1"
cp "${SCRIPT_DIR}/assets/stop_demo.sh" "${WORK_DIR}/stop_demo.sh"
cp "${SCRIPT_DIR}/assets/stop_demo.ps1" "${WORK_DIR}/stop_demo.ps1"
cp "${ARAMIS_ROOT}/examples/prediction_h5/"*_one_patient.h5 "${WORK_DIR}/examples/h5/"
cp "${ARAMIS_ROOT}/examples/prediction_h5/README.md" "${WORK_DIR}/examples/h5/README.md"
cp "${ARAMIS_ROOT}/packaging/prediction_api_bundle/assets/requests/"*.json "${WORK_DIR}/examples/requests/"
cp "${ARAMIS_ROOT}/config/prediction/config_predict_from_h5_template_v0_1.yaml" \
  "${WORK_DIR}/examples/direct_cli_config/"
cp "${ARAMIS_ROOT}/packaging/prediction_api_bundle/assets/API_CONTRACT.md" \
  "${WORK_DIR}/contracts/API_CONTRACT.md"
cp "${ARAMIS_ROOT}/packaging/prediction_api_bundle/assets/openapi.yaml" \
  "${WORK_DIR}/contracts/openapi.yaml"
cp "${ARAMIS_ROOT}/docs/contracts/prediction_config_v0_1.md" \
  "${WORK_DIR}/contracts/direct_cli_prediction_config_v0_1.md"
cp "${ARAMIS_ROOT}/docs/modeling/prediction_pipeline_v0_1.md" \
  "${WORK_DIR}/contracts/prediction_pipeline_v0_1.md"
chmod +x "${WORK_DIR}/start_demo.sh" "${WORK_DIR}/stop_demo.sh"

mkdir -p "${BUILD_CONTEXT}/Aramisvisor-demo/static/model_test"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'outputs' \
  --exclude 'model_test/artifacts' \
  "${ARAMISVISOR_ROOT}/" "${BUILD_CONTEXT}/Aramisvisor-demo/"
cp "${MODEL_TEST_DIR}/aramis_mri_or_biopsy_held_out_t130.joblib" \
  "${BUILD_CONTEXT}/Aramisvisor-demo/static/model_test/"
cp "${MODEL_TEST_DIR}/aramis_mri_or_biopsy_held_out_t130.manifest.json" \
  "${BUILD_CONTEXT}/Aramisvisor-demo/static/model_test/"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${API_AMD64_TAG}" \
  --file "${ARAMIS_ROOT}/packaging/prediction_api_bundle/Dockerfile" \
  --build-arg "MODEL_ID=${MODEL_ID}" \
  "${ARAMIS_ROOT}"
docker save --output "${WORK_DIR}/aramis_prediction_api_linux_amd64_0_2_12_beta.tar" "${API_AMD64_TAG}"

docker buildx build \
  --platform linux/arm64 \
  --load \
  --tag "${API_ARM64_TAG}" \
  --file "${ARAMIS_ROOT}/packaging/prediction_api_bundle/Dockerfile" \
  --build-arg "MODEL_ID=${MODEL_ID}" \
  "${ARAMIS_ROOT}"
docker save --output "${WORK_DIR}/aramis_prediction_api_linux_arm64_0_2_12_beta.tar" "${API_ARM64_TAG}"

docker buildx build \
  --platform linux/amd64 \
  --load \
  --tag "${PLATFORM_AMD64_TAG}" \
  --file "${BUILD_CONTEXT}/Aramisvisor-demo/app/Dockerfile" \
  "${BUILD_CONTEXT}/Aramisvisor-demo"
docker save --output "${WORK_DIR}/aramisvisor_demo_linux_amd64_0_2_12_beta.tar" "${PLATFORM_AMD64_TAG}"

docker buildx build \
  --platform linux/arm64 \
  --load \
  --tag "${PLATFORM_ARM64_TAG}" \
  --file "${BUILD_CONTEXT}/Aramisvisor-demo/app/Dockerfile" \
  "${BUILD_CONTEXT}/Aramisvisor-demo"
docker save --output "${WORK_DIR}/aramisvisor_demo_linux_arm64_0_2_12_beta.tar" "${PLATFORM_ARM64_TAG}"

python - \
  "${WORK_DIR}/bundle_manifest.yaml" \
  "${ARAMIS_ROOT}" \
  "${ARAMISVISOR_ROOT}" \
  "${WORK_DIR}" \
  "${MODEL_ID}" "${MODEL_VERSION}" <<'PY'
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

target = Path(sys.argv[1])
aramis_root = Path(sys.argv[2])
demo_root = Path(sys.argv[3])
bundle_root = Path(sys.argv[4])
model_id = sys.argv[5]
model_version = sys.argv[6]

def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()

model = aramis_root / "models" / model_id / "model.joblib"
test_artifact = demo_root / "model_test" / "artifacts" / "aramis_mri_or_biopsy_held_out_t130.joblib"
entries = [
    ("api_amd64", "eosdx/aramis-prediction-api:" + model_version + "-amd64", "linux/amd64", "aramis_prediction_api_linux_amd64_0_2_12_beta.tar"),
    ("api_arm64", "eosdx/aramis-prediction-api:" + model_version + "-arm64", "linux/arm64", "aramis_prediction_api_linux_arm64_0_2_12_beta.tar"),
    ("platform_amd64", "eosdx/aramisvisor-demo:" + model_version + "-amd64", "linux/amd64", "aramisvisor_demo_linux_amd64_0_2_12_beta.tar"),
    ("platform_arm64", "eosdx/aramisvisor-demo:" + model_version + "-arm64", "linux/arm64", "aramisvisor_demo_linux_arm64_0_2_12_beta.tar"),
]
lines = [
    "contract: aramis_demo_bundle_v0_2",
    "model_id: " + model_id,
    "model_version: " + model_version,
    "model_joblib_sha256: " + digest(model),
    "aramis_commit: " + git_sha(aramis_root),
    "aramisvisor_commit: " + git_sha(demo_root),
    "model_test_artifact_sha256: " + digest(test_artifact),
    "images:",
]
for name, tag, platform, archive in entries:
    path = bundle_root / archive
    lines.extend(
        [
            "  " + name + ":",
            "    tag: " + tag,
            "    platform: " + platform,
            "    archive: " + archive,
            "    sha256: " + digest(path),
        ]
    )
target.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

(
  cd "${DIST_DIR}"
  zip -qry "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
