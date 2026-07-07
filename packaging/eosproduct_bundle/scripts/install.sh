#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-eosproduct}"
AUTO_RUN="${AUTO_RUN:-0}"
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="${HOME}/dev/${ENV_NAME}"

ask_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  local answer
  local suffix
  if [[ "${AUTO_RUN}" == "1" ]]; then
    [[ "${default}" =~ ^[Yy]$|^[Yy][Ee][Ss]$ ]]
    return
  fi
  if [[ "${default}" =~ ^[Yy]$|^[Yy][Ee][Ss]$ ]]; then
    suffix="Y/n"
  else
    suffix="y/N"
  fi
  read -r -p "${prompt} [${suffix}] " answer || true
  answer="${answer:-${default}}"
  [[ "${answer}" =~ ^[Yy]$|^[Yy][Ee][Ss]$ ]]
}

ask_value() {
  local prompt="$1"
  local default="$2"
  local answer
  if [[ "${AUTO_RUN}" == "1" ]]; then
    echo "${default}"
    return
  fi
  read -r -p "${prompt} [${default}] " answer || true
  answer="${answer:-${default}}"
  if [[ "${answer}" =~ ^[Yy]$|^[Yy][Ee][Ss]$ ]]; then
    answer="${default}"
  fi
  echo "${answer}"
}

copy_or_update_repo() {
  local name="$1"
  local url="$2"
  local target="$3"
  local required_path="${4:-}"
  local ref="${5:-}"
  local fallback="${BUNDLE_DIR}/repos/${name}"
  if [[ "${USE_GIT}" == "yes" ]]; then
    if [[ -d "${target}/.git" ]]; then
      git -C "${target}" fetch --tags origin || {
        echo "Git fetch failed for ${name}; keeping existing checkout."
      }
      if [[ -n "${ref}" ]]; then
        git -C "${target}" checkout "${ref}" || git -C "${target}" checkout -B "${ref}" "origin/${ref}" || {
          echo "Git checkout failed for ${name}; keeping existing checkout."
        }
      else
        git -C "${target}" pull --ff-only || {
          echo "Git update failed for ${name}; keeping existing checkout."
        }
      fi
    else
      rm -rf "${target}"
      if [[ -n "${ref}" ]]; then
        git clone --branch "${ref}" "${url}" "${target}" || true
      else
        git clone "${url}" "${target}" || true
      fi
      if [[ ! -d "${target}/.git" ]]; then
        echo "Git clone failed for ${name}; using bundled fallback."
      fi
    fi
    if [[ -d "${target}/.git" && ( -z "${required_path}" || -e "${target}/${required_path}" ) ]]; then
      return
    fi
    if [[ -d "${target}/.git" ]]; then
      echo "Git checkout for ${name} lacks ${required_path}; using bundled fallback."
    fi
  fi
  rm -rf "${target}"
  mkdir -p "$(dirname "${target}")"
  cp -R "${fallback}" "${target}"
}

ensure_git() {
  if command -v git >/dev/null 2>&1; then
    return 0
  fi
  if ! ask_yes_no "git not found. Install git if possible?" "y"; then
    return 1
  fi
  if command -v brew >/dev/null 2>&1; then
    brew install git
    return 0
  fi
  if [[ "$(uname -s)" == "Darwin" ]] && command -v xcode-select >/dev/null 2>&1; then
    xcode-select --install || true
    echo "Apple Command Line Tools installer started. Rerun install.sh after it finishes."
    exit 1
  fi
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y git
    return 0
  fi
  echo "Could not install git automatically. Using bundled repository fallback." >&2
  return 1
}

run_in_terminal() {
  local title="$1"
  local command="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript <<OSA
tell application "Terminal"
  do script "${command}"
  set custom title of front window to "${title}"
end tell
OSA
  else
    echo "${title}:"
    echo "${command}"
  fi
}

install_miniforge() {
  local prefix="${HOME}/miniforge3"
  local arch
  local url
  local installer
  arch="$(uname -m)"
  case "${arch}" in
    arm64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
      ;;
    x86_64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
      ;;
    *)
      echo "Unsupported macOS architecture for automatic Miniforge install: ${arch}" >&2
      return 1
      ;;
  esac
  if [[ -x "${prefix}/bin/conda" ]]; then
    export PATH="${prefix}/bin:${PATH}"
    return 0
  fi
  installer="${TMPDIR:-/tmp}/Miniforge3-${arch}.sh"
  echo "Downloading Miniforge: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${installer}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${installer}" "${url}"
  else
    echo "Neither curl nor wget is available. Cannot download Miniforge." >&2
    return 1
  fi
  bash "${installer}" -b -p "${prefix}"
  export PATH="${prefix}/bin:${PATH}"
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if [[ -x "${HOME}/miniforge3/bin/conda" ]]; then
    export PATH="${HOME}/miniforge3/bin:${PATH}"
    return 0
  fi
  if [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    export PATH="${HOME}/miniconda3/bin:${PATH}"
    return 0
  fi
  if ask_yes_no "conda not found. Install Miniforge to ~/miniforge3?" "y"; then
    install_miniforge
    return 0
  fi
  echo "conda is required. Install Miniforge/Miniconda and rerun install.sh." >&2
  return 1
}

TARGET_ROOT="$(ask_value "Target root" "${DEFAULT_TARGET}")"
mkdir -p "${TARGET_ROOT}"

USE_GIT="no"
ensure_git || true
if command -v git >/dev/null 2>&1 && ask_yes_no "Use git to clone/update repos?" "y"; then
  USE_GIT="yes"
fi

copy_or_update_repo "XRD-preprocessing" "https://github.com/Eos-Dx/XRD-preprocessing.git" "${TARGET_ROOT}/XRD-preprocessing" "src/xrd_preprocessing/configs/preprocessing_branch_config_template.yaml" "v0.1.6-beta"
copy_or_update_repo "Aramis" "https://github.com/Eos-Dx/Aramis.git" "${TARGET_ROOT}/Aramis" "examples/aramis_dataframe_biopsy_patients_v0_1.py" "0.1.4-beta"
copy_or_update_repo "container" "https://github.com/Eos-Dx/container.git" "${TARGET_ROOT}/container" "pyproject.toml" "feat/v0_3-eoscan-session-container"
mkdir -p "${TARGET_ROOT}/Bremen"

mkdir -p "${TARGET_ROOT}/data"
if [[ -f "${BUNDLE_DIR}/data/combined_archive.h5" ]]; then
  cp "${BUNDLE_DIR}/data/combined_archive.h5" "${TARGET_ROOT}/data/combined_archive.h5"
fi

if ask_yes_no "Create/update conda env ${ENV_NAME}?" "y"; then
  ensure_conda
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda env update -n "${ENV_NAME}" -f "${BUNDLE_DIR}/environment.yml"
  else
    conda env create -n "${ENV_NAME}" -f "${BUNDLE_DIR}/environment.yml"
  fi
  conda run -n "${ENV_NAME}" python -m pip install -e "${TARGET_ROOT}/container"
  conda run -n "${ENV_NAME}" python -m pip install -e "${TARGET_ROOT}/XRD-preprocessing[dev]"
  conda run -n "${ENV_NAME}" python -m pip install --no-deps -e "${TARGET_ROOT}/Aramis[dev]"
  conda run -n "${ENV_NAME}" python -c "import aramis, xrd_preprocessing; print('imports ok'); print('xrd_preprocessing', xrd_preprocessing.__file__); print('aramis', aramis.__file__)"
fi

echo "Ready: ${TARGET_ROOT}"
echo "Run tests: ${BUNDLE_DIR}/run_tests.sh ${TARGET_ROOT}"
echo "Run notebooks: ${BUNDLE_DIR}/run_aramis_notebooks.sh ${TARGET_ROOT}"

if [[ "${AUTO_RUN}" == "1" ]]; then
  run_in_terminal "EOS product tests" "ENV_NAME='${ENV_NAME}' '${BUNDLE_DIR}/run_tests.sh' '${TARGET_ROOT}' all"
  "${BUNDLE_DIR}/run_aramis_notebooks.sh" "${TARGET_ROOT}"
  exit 0
fi

if ask_yes_no "Run XRD-preprocessing and Aramis tests now?" "n"; then
  run_in_terminal "EOS product tests" "ENV_NAME='${ENV_NAME}' '${BUNDLE_DIR}/run_tests.sh' '${TARGET_ROOT}' all"
fi
if ask_yes_no "Launch Aramis marimo notebooks now?" "n"; then
  "${BUNDLE_DIR}/run_aramis_notebooks.sh" "${TARGET_ROOT}"
fi
