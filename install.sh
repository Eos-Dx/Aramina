#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-eosproduct}"
RUN_EXAMPLE="${RUN_EXAMPLE:-1}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ask_yes_no() {
  local prompt="$1"
  local default="${2:-y}"
  local answer
  local suffix
  if [[ "${default}" =~ ^[Yy] ]]; then
    suffix="Y/n"
  else
    suffix="y/N"
  fi
  read -r -p "${prompt} [${suffix}] " answer || true
  answer="${answer:-${default}}"
  [[ "${answer}" =~ ^[Yy]$|^[Yy][Ee][Ss]$ ]]
}

install_miniforge() {
  local prefix="${HOME}/miniforge3"
  local os_name
  local arch
  local url
  local installer

  os_name="$(uname -s)"

  arch="$(uname -m)"
  case "${os_name}:${arch}" in
    Darwin:arm64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
      ;;
    Darwin:x86_64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
      ;;
    Linux:x86_64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
      ;;
    Linux:aarch64)
      url="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh"
      ;;
    *)
      echo "Unsupported platform for automatic Miniforge install: ${os_name} ${arch}" >&2
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
    echo "Neither curl nor wget is available. Install Miniforge manually." >&2
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

ensure_conda
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${REPO_DIR}/environment.yml"
else
  conda env create -n "${ENV_NAME}" -f "${REPO_DIR}/environment.yml"
fi

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install -e "${REPO_DIR}[dev]"

conda run --no-capture-output -n "${ENV_NAME}" \
  python -c "import aramina, xrd_preprocessing; print('imports ok'); print('aramina', aramina.__file__); print('xrd_preprocessing', xrd_preprocessing.__file__)"

if [[ "${RUN_EXAMPLE}" == "1" ]]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m aramina predict --config "${REPO_DIR}/examples/prediction/configs/config_predict_cancer_example.yaml"
fi

cat <<EOF

Ready.

Activate:
  conda activate ${ENV_NAME}

Run prediction example:
  cd "${REPO_DIR}"
  python -m aramina predict --config examples/prediction/configs/config_predict_cancer_example.yaml

Run all prediction examples:
  for f in examples/prediction/configs/config_predict_*_example.yaml; do python -m aramina predict --config "\$f"; done
EOF
