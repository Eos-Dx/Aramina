#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARAMINA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XRD_ROOT="${XRD_ROOT:-${ARAMINA_ROOT}/../XRD-preprocessing}"
DIST_DIR="${DIST_DIR:-${ARAMINA_ROOT}/dist}"

[[ -e "${XRD_ROOT}/.git" ]] || { echo "Missing XRD-preprocessing checkout: ${XRD_ROOT}" >&2; exit 1; }
[[ -z "$(git -C "${ARAMINA_ROOT}" status --porcelain)" ]] || { echo "Aramina checkout must be clean." >&2; exit 1; }
[[ -z "$(git -C "${XRD_ROOT}" status --porcelain)" ]] || { echo "XRD-preprocessing checkout must be clean." >&2; exit 1; }

ARAMINA_COMMIT="$(git -C "${ARAMINA_ROOT}" rev-parse HEAD)"
XRD_COMMIT="$(git -C "${XRD_ROOT}" rev-parse HEAD)"
[[ "${ARAMINA_COMMIT}" == "$(git -C "${ARAMINA_ROOT}" rev-parse main)" ]] || { echo "Aramina must be at main." >&2; exit 1; }
[[ "${XRD_COMMIT}" == "$(git -C "${XRD_ROOT}" rev-parse main)" ]] || { echo "XRD-preprocessing must be at main." >&2; exit 1; }

BUNDLE_NAME="aramina_documentation_site_main_${ARAMINA_COMMIT:0:12}_${XRD_COMMIT:0:12}"
WORK_DIR="${DIST_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${DIST_DIR}/${BUNDLE_NAME}.zip"
BUILD_DIR="$(mktemp -d)"
cleanup() { rm -rf "${BUILD_DIR}"; }
trap cleanup EXIT

rm -rf "${WORK_DIR}" "${ARCHIVE_PATH}"
mkdir -p "${WORK_DIR}/source/Aramina" "${WORK_DIR}/source/XRD-preprocessing"
git -C "${ARAMINA_ROOT}" archive "${ARAMINA_COMMIT}" | tar -x -C "${WORK_DIR}/source/Aramina"
git -C "${XRD_ROOT}" archive "${XRD_COMMIT}" | tar -x -C "${WORK_DIR}/source/XRD-preprocessing"

python "${SCRIPT_DIR}/build_static_site.py" \
  --aramina-root "${WORK_DIR}/source/Aramina" \
  --xrd-root "${WORK_DIR}/source/XRD-preprocessing" \
  --output "${WORK_DIR}/site" \
  --aramina-commit "${ARAMINA_COMMIT}" \
  --xrd-commit "${XRD_COMMIT}"

cp "${SCRIPT_DIR}/assets/README.md" "${WORK_DIR}/README.md"
cp "${SCRIPT_DIR}/assets/start_docs.sh" "${WORK_DIR}/start_docs.sh"
cp "${SCRIPT_DIR}/assets/start_docs.ps1" "${WORK_DIR}/start_docs.ps1"
chmod +x "${WORK_DIR}/start_docs.sh"

python - "${WORK_DIR}/bundle_manifest.json" "${ARAMINA_COMMIT}" "${XRD_COMMIT}" <<'PY'
import json
import sys

path, aramina_commit, xrd_commit = sys.argv[1:]
payload = {
    "contract": "aramina_offline_documentation_bundle_v0_1",
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
    "site": "site/index.html",
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

(
  cd "${DIST_DIR}"
  zip -qry "${ARCHIVE_PATH}" "${BUNDLE_NAME}"
)

echo "${ARCHIVE_PATH}"
