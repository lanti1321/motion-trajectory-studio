#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MOTION_STUDIO_PYTHON:-$(command -v python3)}"

"${PYTHON}" -m venv "${ROOT}/.venv"
"${ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${ROOT}/.venv/bin/python" -m pip install -r "${ROOT}/requirements-lock.txt"
"${ROOT}/.venv/bin/python" -m pip install --no-deps -e "${ROOT}"

echo "Installed Motion Trajectory Studio in ${ROOT}/.venv"
