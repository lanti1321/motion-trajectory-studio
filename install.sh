#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python_mm() {
  "$1" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

supported_python() {
  local ver
  ver="$(python_mm "$1" 2>/dev/null)" || return 1
  case "${ver}" in
    3.12|3.13) return 0 ;;
    *) return 1 ;;
  esac
}

find_python() {
  local candidate
  if [[ -n "${MOTION_STUDIO_PYTHON:-}" ]]; then
    printf '%s\n' "${MOTION_STUDIO_PYTHON}"
    return
  fi
  for candidate in python3.12 python3.13; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "$(command -v "${candidate}")"
      return
    fi
  done
  for candidate in \
      "${HOME}/miniconda3/envs/lerobot/bin/python" \
      "${HOME}/miniconda3/envs/"*/bin/python \
      "${HOME}/anaconda3/envs/"*/bin/python; do
    if [[ -x "${candidate}" ]] && supported_python "${candidate}"; then
      printf '%s\n' "${candidate}"
      return
    fi
  done
  if command -v python3 >/dev/null 2>&1 && supported_python "$(command -v python3)"; then
    command -v python3
    return
  fi
  echo "Motion Trajectory Studio needs Python 3.12 or 3.13." >&2
  echo "conda base is currently $(python3 --version 2>/dev/null || true), which cannot install the locked wheels (numpy==2.5.2)." >&2
  echo "Install Python 3.12, or point install.sh at one:" >&2
  echo "  MOTION_STUDIO_PYTHON=/path/to/python3.12 bash install.sh" >&2
  exit 1
}

PYTHON="$(find_python)"
if ! supported_python "${PYTHON}"; then
  echo "Refusing ${PYTHON} ($(python_mm "${PYTHON}")). Use Python 3.12 or 3.13." >&2
  echo "  MOTION_STUDIO_PYTHON=/path/to/python3.12 bash install.sh" >&2
  exit 1
fi

echo "Using ${PYTHON} ($(python_mm "${PYTHON}"))"
rm -rf "${ROOT}/.venv"
"${PYTHON}" -m venv "${ROOT}/.venv"
# Tsinghua/tuna often lags new numpy/scipy wheels. Use official PyPI unless overridden.
INDEX_URL="${MOTION_STUDIO_PIP_INDEX:-https://pypi.org/simple}"
PIP=("${ROOT}/.venv/bin/python" -m pip)
"${PIP[@]}" install --upgrade pip -i "${INDEX_URL}"
"${PIP[@]}" install -r "${ROOT}/requirements-lock.txt" -i "${INDEX_URL}"
"${PIP[@]}" install setuptools wheel -i "${INDEX_URL}"
"${PIP[@]}" install --no-deps --no-build-isolation -e "${ROOT}"

echo "Installed Motion Trajectory Studio in ${ROOT}/.venv"
echo "Start with: bash ${ROOT}/run.sh"
