#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMIS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${DIST_DIR:-${ARAMIS_ROOT}/dist}"
BUNDLE_NAME="aramis_demo_bundle_0_2_11_beta"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
AMD64_IMAGE_TAG="eosdx/aramis-demo:0.2.11-beta-amd64"
ARM64_IMAGE_TAG="eosdx/aramis-demo:0.2.11-beta-arm64"
AMD64_IMAGE_ARCHIVE="aramis_demo_linux_amd64_0_2_11_beta.tar"
ARM64_IMAGE_ARCHIVE="aramis_demo_linux_arm64_0_2_11_beta.tar"
MODEL_PATH="${ARAMIS_ROOT}/models/aramis_target_breast_risk_0_2_11-beta_d531ea38c5dc/model.joblib"

command -v docker >/dev/null || { echo "Docker is required to build this bundle." >&2; exit 1; }
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

for path in \
  "${SCRIPT_DIR}/assets/Dockerfile" \
  "${ARAMIS_ROOT}/demo/platform/streamlit_app.py" \
  "${ARAMIS_ROOT}/demo/platform/pdf_report.py" \
  "${ARAMIS_ROOT}/demo/platform/assets/training_patient_ids.json" \
  "${MODEL_PATH}"; do
  [[ -f "${path}" ]] || { echo "Missing demo bundle input: ${path}" >&2; exit 1; }
done

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p "${WORK_DIR}/fixtures" "${DIST_DIR}"
cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
cp "${SCRIPT_DIR}/assets/start_demo.sh" "${WORK_DIR}/start_demo.sh"
cp "${SCRIPT_DIR}/assets/start_demo.ps1" "${WORK_DIR}/start_demo.ps1"
cp "${SCRIPT_DIR}/assets/stop_demo.sh" "${WORK_DIR}/stop_demo.sh"
cp "${ARAMIS_ROOT}/examples/prediction_h5/"*_one_patient.h5 "${WORK_DIR}/fixtures/"
cp "${ARAMIS_ROOT}/examples/prediction_h5/README.md" "${WORK_DIR}/fixtures/README.md"
chmod +x "${WORK_DIR}/start_demo.sh" "${WORK_DIR}/stop_demo.sh"

docker buildx build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=eosdx/aramis-training:0.2.11-beta-amd64 \
  --load \
  --tag "${AMD64_IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/assets/Dockerfile" \
  "${ARAMIS_ROOT}"
docker save --output "${WORK_DIR}/${AMD64_IMAGE_ARCHIVE}" "${AMD64_IMAGE_TAG}"

docker buildx build \
  --platform linux/arm64 \
  --build-arg BASE_IMAGE=eosdx/aramis-training:0.2.11-beta-arm64 \
  --load \
  --tag "${ARM64_IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/assets/Dockerfile" \
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
model = root / "models/aramis_target_breast_risk_0_2_11-beta_d531ea38c5dc/model.joblib"
app = root / "demo/platform/streamlit_app.py"
pdf = root / "demo/platform/pdf_report.py"
training_patient_ids = root / "demo/platform/assets/training_patient_ids.json"
dockerfile = root / "packaging/aramis_demo_bundle/assets/Dockerfile"

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
            "contract: aramis_demo_bundle_v0_1",
            "model_name: aramis_target_breast_risk",
            "model_version: 0.2.11-beta",
            "aramis_commit: " + commit,
            "model_joblib_sha256: " + digest(model),
            "streamlit_app_sha256: " + digest(app),
            "pdf_report_sha256: " + digest(pdf),
            "training_patient_ids_sha256: " + digest(training_patient_ids),
            "dockerfile_sha256: " + digest(dockerfile),
            "images:",
            "  amd64:",
            "    tag: eosdx/aramis-demo:0.2.11-beta-amd64",
            "    platform: linux/amd64",
            "    archive: " + amd64_archive.name,
            "    sha256: " + digest(amd64_archive),
            "  arm64:",
            "    tag: eosdx/aramis-demo:0.2.11-beta-arm64",
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
