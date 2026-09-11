#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MOTION_STUDIO_PYTHON:-$(command -v python3)}"

"${PYTHON}" -m venv "${ROOT}/.venv"
"${ROOT}/.venv/bin/python" -m pip install --upgrade pip
if ! "${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements-lock.txt"; then
  echo "Pinned versions unavailable; installing from requirements.txt"
  "${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements.txt"
fi
if ! "${ROOT}/.venv/bin/python" -m pip install --no-deps --no-build-isolation -e "${ROOT}"; then
  echo "${ROOT}" > "${ROOT}/.venv/lib/python3."*/site-packages/motion_trajectory_studio.pth
fi

echo "Installed Motion Trajectory Studio in ${ROOT}/.venv"
