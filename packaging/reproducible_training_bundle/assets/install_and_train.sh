#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="${1:-}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${BUNDLE_DIR}/bundle_manifest.json"
if [[ -z "${WORKSPACE}" ]]; then
  WORKSPACE="${BUNDLE_DIR}/workspace"
fi

stage() {
  printf '\n=== %s ===\n' "$1"
}

fail() {
  local status=$?
  printf '\nBundle failed with exit code %s. Log: %s\n' "${status}" "${LOG_PATH}" >&2
  exit "${status}"
}

manifest_python="$(command -v python3 || command -v python || true)"
if [[ -z "${manifest_python}" ]]; then
  echo "Python 3 is required to read bundle_manifest.json." >&2
  exit 1
fi

read_manifest() {
  "${manifest_python}" - "${MANIFEST}" "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)[sys.argv[2]]
print(value)
PY
}

ENV_NAME="$(read_manifest environment_name)"
ARAMIS_REPOSITORY="$(read_manifest aramis_repository)"
ARAMIS_COMMIT="$(read_manifest aramis_commit)"
XRD_REPOSITORY="$(read_manifest xrd_preprocessing_repository)"
XRD_COMMIT="$(read_manifest xrd_preprocessing_commit)"
WORKFLOW_CONFIG="$(read_manifest workflow_config)"
REFERENCE_MODEL="$(read_manifest reference_model_relative_path)"

LOG_DIR="${WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/install_and_train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_PATH}") 2>&1
trap fail ERR

find_conda() {
  command -v conda || true
}

install_miniforge() {
  local prefix="${HOME}/miniforge3"
  local os_name arch url installer
  os_name="$(uname -s)"
  arch="$(uname -m)"
  case "${os_name}:${arch}" in
    Darwin:arm64) url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh" ;;
    Darwin:x86_64) url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh" ;;
    Linux:x86_64) url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" ;;
    Linux:aarch64) url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" ;;
    *) echo "Unsupported platform for Miniforge: ${os_name} ${arch}" >&2; return 1 ;;
  esac
  installer="${TMPDIR:-/tmp}/Miniforge3-${arch}.sh"
  stage "Installing Miniforge" >&2
  curl -L "${url}" -o "${installer}" >&2
  bash "${installer}" -b -p "${prefix}" >&2
  printf '%s\n' "${prefix}/bin/conda"
}

ensure_conda() {
  local conda_path
  conda_path="$(find_conda)"
  if [[ -n "${conda_path}" ]]; then
    printf '%s\n' "${conda_path}"
    return
  fi
  if [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    printf '%s\n' "${HOME}/miniforge3/bin/conda"
    return
  fi
  install_miniforge
}

ensure_git() {
  if command -v git >/dev/null 2>&1; then
    return
  fi
  stage "Installing Git"
  if command -v brew >/dev/null 2>&1; then
    brew install git
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y git
  else
    echo "Git is required. Install Git, then rerun install_and_train.sh." >&2
    return 1
  fi
}

link_bundle_data() {
  local source="${BUNDLE_DIR}/data"
  local destination="${WORKSPACE}/data"
  [[ -f "${source}/combined_archive.h5" ]] || {
    echo "Missing bundled H5 input: ${source}/combined_archive.h5" >&2
    return 1
  }
  if [[ -L "${destination}" ]]; then
    if [[ "$(cd "${destination}" && pwd -P)" == "$(cd "${source}" && pwd -P)" ]]; then
      stage "Reuse bundle data link"
      return
    fi
    echo "Workspace data link targets a different directory: ${destination}" >&2
    return 1
  fi
  if [[ -e "${destination}" ]]; then
    echo "Workspace data path already exists and is not a bundle data link: ${destination}" >&2
    return 1
  fi
  stage "Link bundled H5 input"
  ln -s "${source}" "${destination}"
}

sync_repository() {
  local repository="$1" path="$2" commit="$3" name="$4"
  if [[ -d "${path}" ]]; then
    [[ -d "${path}/.git" ]] || { echo "${name} path exists but is not a Git checkout: ${path}" >&2; return 1; }
    stage "Fetch ${name}"
    git -C "${path}" fetch --tags --prune origin
  else
    stage "Clone ${name}"
    git clone "${repository}" "${path}"
  fi
  stage "Checkout ${name} ${commit}"
  git -C "${path}" checkout --detach "${commit}"
  git -C "${path}" reset --hard "${commit}"
}

stage "Aramis reproducible training bundle"
echo "Workspace: ${WORKSPACE}"
echo "Log: ${LOG_PATH}"
CONDA="$(ensure_conda)"
ensure_git
ARAMIS_REPO="${WORKSPACE}/Aramis"
XRD_REPO="${WORKSPACE}/XRD-preprocessing"

link_bundle_data
sync_repository "${ARAMIS_REPOSITORY}" "${ARAMIS_REPO}" "${ARAMIS_COMMIT}" "Aramis"
sync_repository "${XRD_REPOSITORY}" "${XRD_REPO}" "${XRD_COMMIT}" "XRD-preprocessing"

if "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  stage "Reuse conda environment ${ENV_NAME}"
else
  stage "Create conda environment ${ENV_NAME}"
  "${CONDA}" env create -n "${ENV_NAME}" -f "${BUNDLE_DIR}/environment.yml"
fi

stage "Install XRD-preprocessing from selected commit"
"${CONDA}" run --no-capture-output -n "${ENV_NAME}" python -m pip install -e "${XRD_REPO}[dev]"
stage "Install Aramis from selected commit"
"${CONDA}" run --no-capture-output -n "${ENV_NAME}" python -m pip install --no-deps -e "${ARAMIS_REPO}[dev]"
stage "Verify Python imports"
"${CONDA}" run --no-capture-output -n "${ENV_NAME}" python -c "import aramis, xrd_preprocessing; print('aramis', aramis.__file__); print('xrd_preprocessing', xrd_preprocessing.__file__)"

stage "Remove prior generated workflow outputs"
rm -rf "${ARAMIS_REPO}/examples/outputs/workflows"
stage "Run preprocessing and training"
(
  cd "${ARAMIS_REPO}"
  "${CONDA}" run --no-capture-output -n "${ENV_NAME}" python -m aramis preprocess-train --config "${WORKFLOW_CONFIG}" --verbose
)

LATEST_MODEL="$(find "${ARAMIS_REPO}/examples/outputs/workflows" -name model.joblib -exec ls -t {} + | head -1)"
[[ -n "${LATEST_MODEL}" ]] || { echo "No generated model.joblib was found." >&2; exit 1; }
stage "Compare generated model with reference"
"${CONDA}" run --no-capture-output -n "${ENV_NAME}" python "${ARAMIS_REPO}/scripts/compare_model_artifacts.py" --reference "${ARAMIS_REPO}/${REFERENCE_MODEL}" --candidate "${LATEST_MODEL}"

stage "Bundle completed"
echo "Generated model: ${LATEST_MODEL}"
echo "Log saved to: ${LOG_PATH}"
