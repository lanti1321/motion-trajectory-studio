#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOTION_STUDIO_LANG=en
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${MOTION_STUDIO_PYTHON:-$(command -v python3)}"
fi

exec "${PYTHON}" "${ROOT}/app.py" "$@"
